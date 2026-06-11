//! Minimal safetensors reader for model2vec models, which store exactly one
//! tensor: `embeddings`, shape [vocab, dim], as F32 or I8 (the quantized
//! form; see quantize.zig).
//!
//! Format: 8-byte little-endian header length, JSON header mapping tensor
//! names to dtype/shape/offsets, then the raw tensor data. We copy the tensor
//! out element-by-element instead of casting the file bytes, which sidesteps
//! both alignment and endianness.

const std = @import("std");

pub const Error = error{
    TruncatedFile,
    BadHeader,
    MissingEmbeddings,
    UnsupportedDtype,
    BadShape,
} || std.mem.Allocator.Error;

/// rows * cols values, row-major. Owned by the allocator passed to parse.
/// I8 values are the model2vec quantized form, stored without a scale: the
/// global scale factor cancels under the L2 normalization every potion model
/// applies, so pooling sums the raw values (matching the reference).
pub const Matrix = union(enum) {
    f32_data: []f32,
    i8_data: []i8,
    /// TurboQuant-style rotated 4-bit rows with one scale per row; see tq.zig.
    tq4_data: Tq4,

    pub const Tq4 = struct {
        /// rows * cols/2 bytes, two signed nibbles per byte.
        packed_data: []u8,
        scales: []f32,
    };

    pub fn deinit(self: Matrix, allocator: std.mem.Allocator) void {
        switch (self) {
            .f32_data => |d| allocator.free(d),
            .i8_data => |d| allocator.free(d),
            .tq4_data => |t| {
                allocator.free(t.packed_data);
                allocator.free(t.scales);
            },
        }
    }
};

pub const Embeddings = struct {
    matrix: Matrix,
    rows: usize,
    cols: usize,
};

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) Error!Embeddings {
    if (bytes.len < 8) return error.TruncatedFile;
    const header_len = std.mem.readInt(u64, bytes[0..8], .little);
    if (8 + header_len > bytes.len) return error.TruncatedFile;
    const header_bytes = bytes[8 .. 8 + header_len];

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const header = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), header_bytes, .{}) catch
        return error.BadHeader;
    if (header != .object) return error.BadHeader;

    const tensor = header.object.get("embeddings") orelse {
        if (header.object.get("embeddings_tq4")) |tq| {
            return parseTq4(allocator, bytes, header, tq, header_len);
        }
        return error.MissingEmbeddings;
    };
    if (tensor != .object) return error.BadHeader;

    const dtype = stringField(tensor, "dtype") orelse return error.BadHeader;
    const is_f32 = std.mem.eql(u8, dtype, "F32");
    if (!is_f32 and !std.mem.eql(u8, dtype, "I8")) return error.UnsupportedDtype;

    const shape = tensor.object.get("shape") orelse return error.BadHeader;
    if (shape != .array or shape.array.items.len != 2) return error.BadShape;
    const rows = intField(shape.array.items[0]) orelse return error.BadShape;
    const cols = intField(shape.array.items[1]) orelse return error.BadShape;

    const offsets = tensor.object.get("data_offsets") orelse return error.BadHeader;
    if (offsets != .array or offsets.array.items.len != 2) return error.BadHeader;
    const begin = intField(offsets.array.items[0]) orelse return error.BadHeader;
    const end = intField(offsets.array.items[1]) orelse return error.BadHeader;

    const data_start = 8 + header_len;
    if (end < begin or data_start + end > bytes.len) return error.TruncatedFile;
    const raw = bytes[data_start + begin .. data_start + end];
    const elem_size: usize = if (is_f32) 4 else 1;
    if (raw.len != rows * cols * elem_size) return error.BadShape;

    if (is_f32) {
        const data = try allocator.alloc(f32, rows * cols);
        errdefer allocator.free(data);
        for (data, 0..) |*v, i| {
            v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        }
        return .{ .matrix = .{ .f32_data = data }, .rows = rows, .cols = cols };
    }

    const data = try allocator.alloc(i8, rows * cols);
    errdefer allocator.free(data);
    for (data, raw) |*v, b| v.* = @bitCast(b);
    return .{ .matrix = .{ .i8_data = data }, .rows = rows, .cols = cols };
}

fn parseTq4(allocator: std.mem.Allocator, bytes: []const u8, header: std.json.Value, tq: std.json.Value, header_len: u64) Error!Embeddings {
    if (tq != .object) return error.BadHeader;
    const dtype = stringField(tq, "dtype") orelse return error.BadHeader;
    if (!std.mem.eql(u8, dtype, "U8")) return error.UnsupportedDtype;

    const shape = tq.object.get("shape") orelse return error.BadHeader;
    if (shape != .array or shape.array.items.len != 2) return error.BadShape;
    const rows = intField(shape.array.items[0]) orelse return error.BadShape;
    const half_cols = intField(shape.array.items[1]) orelse return error.BadShape;

    const data_start = 8 + header_len;
    const packed_raw = tensorRegion(bytes, tq, data_start) orelse return error.TruncatedFile;
    if (packed_raw.len != rows * half_cols) return error.BadShape;

    const scales_tensor = header.object.get("scales") orelse return error.BadHeader;
    if (scales_tensor != .object) return error.BadHeader;
    const scales_dtype = stringField(scales_tensor, "dtype") orelse return error.BadHeader;
    if (!std.mem.eql(u8, scales_dtype, "F32")) return error.UnsupportedDtype;
    const scales_raw = tensorRegion(bytes, scales_tensor, data_start) orelse return error.TruncatedFile;
    if (scales_raw.len != rows * 4) return error.BadShape;

    const packed_data = try allocator.dupe(u8, packed_raw);
    errdefer allocator.free(packed_data);
    const scales = try allocator.alloc(f32, rows);
    errdefer allocator.free(scales);
    for (scales, 0..) |*s, i| {
        s.* = @bitCast(std.mem.readInt(u32, scales_raw[i * 4 ..][0..4], .little));
    }

    return .{
        .matrix = .{ .tq4_data = .{ .packed_data = packed_data, .scales = scales } },
        .rows = rows,
        .cols = half_cols * 2,
    };
}

fn tensorRegion(bytes: []const u8, tensor: std.json.Value, data_start: u64) ?[]const u8 {
    const offsets = tensor.object.get("data_offsets") orelse return null;
    if (offsets != .array or offsets.array.items.len != 2) return null;
    const begin = intField(offsets.array.items[0]) orelse return null;
    const end = intField(offsets.array.items[1]) orelse return null;
    if (end < begin or data_start + end > bytes.len) return null;
    return bytes[data_start + begin .. data_start + end];
}

fn stringField(obj: std.json.Value, key: []const u8) ?[]const u8 {
    const v = obj.object.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn intField(v: std.json.Value) ?usize {
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

const testing = std.testing;

fn buildFixture(a: std.mem.Allocator, header: []const u8, floats: []const f32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try out.appendSlice(a, &len_buf);
    try out.appendSlice(a, header);
    for (floats) |f| {
        var fb: [4]u8 = undefined;
        std.mem.writeInt(u32, &fb, @bitCast(f), .little);
        try out.appendSlice(a, &fb);
    }
    return out.items;
}

test "parse reads a two-row tensor" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const bytes = try buildFixture(arena.allocator(),
        \\{"embeddings":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}
    , &.{ 1.0, 2.0, 3.0, -1.5, 0.25, 0.0 });

    const emb = try parse(testing.allocator, bytes);
    defer emb.matrix.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), emb.rows);
    try testing.expectEqual(@as(usize, 3), emb.cols);
    try testing.expectEqual(@as(f32, -1.5), emb.matrix.f32_data[3]);
}

test "parse reads an i8 tensor" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    var out: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    const header =
        \\{"embeddings":{"dtype":"I8","shape":[2,2],"data_offsets":[0,4]}}
    ;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try out.appendSlice(arena.allocator(), &len_buf);
    try out.appendSlice(arena.allocator(), header);
    try out.appendSlice(arena.allocator(), &[_]u8{ 0x7F, 0x81, 0x00, 0x05 }); // 127, -127, 0, 5

    const emb = try parse(testing.allocator, out.items);
    defer emb.matrix.deinit(testing.allocator);

    try testing.expectEqual(@as(i8, 127), emb.matrix.i8_data[0]);
    try testing.expectEqual(@as(i8, -127), emb.matrix.i8_data[1]);
    try testing.expectEqual(@as(i8, 5), emb.matrix.i8_data[3]);
}

test "parse rejects wrong dtype and missing tensor" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const f16_fixture = try buildFixture(arena.allocator(),
        \\{"embeddings":{"dtype":"F16","shape":[1,2],"data_offsets":[0,4]}}
    , &.{1.0});
    try testing.expectError(error.UnsupportedDtype, parse(testing.allocator, f16_fixture));

    const wrong_name = try buildFixture(arena.allocator(),
        \\{"weights":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}
    , &.{1.0});
    try testing.expectError(error.MissingEmbeddings, parse(testing.allocator, wrong_name));
}
