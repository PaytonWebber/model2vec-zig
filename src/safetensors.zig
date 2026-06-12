//! Minimal safetensors reader for model2vec models, which store exactly one
//! tensor: `embeddings`, shape [vocab, dim], as F32 or I8 (the quantized
//! form; see quantize.zig), or the tq4 pair `embeddings_tq4` + `scales`.
//!
//! Format: 8-byte little-endian header length, JSON header mapping tensor
//! names to dtype/shape/offsets, then the raw tensor data.
//!
//! With `.borrow = true`, on little-endian targets the returned matrix
//! points directly into the input bytes when alignment allows, so loading
//! copies nothing. Otherwise (and always on big-endian targets) the tensor
//! is copied out element-by-element, which sidesteps both alignment and
//! endianness.

const std = @import("std");

const native_endian = @import("builtin").cpu.arch.endian();

pub const Error = error{
    TruncatedFile,
    BadHeader,
    MissingEmbeddings,
    UnsupportedDtype,
    UnsupportedTq4Version,
    BadShape,
} || std.mem.Allocator.Error;

/// rows * cols values, row-major. Owned by the allocator passed to parse
/// unless `Embeddings.borrowed` is set, in which case the slices point into
/// the parsed bytes. I8 values are the model2vec quantized form, stored
/// without a scale: the global scale factor cancels under the L2
/// normalization every potion model applies, so pooling sums the raw values
/// (matching the reference).
pub const Matrix = union(enum) {
    f32_data: []const f32,
    i8_data: []const i8,
    /// TurboQuant-style rotated 4-bit rows with one scale per row; see tq.zig.
    tq4_data: Tq4,

    pub const Tq4 = struct {
        /// rows * cols/2 bytes, two signed nibbles per byte.
        packed_data: []const u8,
        scales: []const f32,
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
    /// True when the matrix points into the bytes given to parse, which must
    /// then outlive the matrix.
    borrowed: bool = false,

    pub fn deinit(self: Embeddings, allocator: std.mem.Allocator) void {
        if (self.borrowed) return;
        self.matrix.deinit(allocator);
    }
};

pub const Options = struct {
    /// Allow the matrix to reference the parsed bytes instead of copying.
    borrow: bool = false,
};

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8, opts: Options) Error!Embeddings {
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
            return parseTq4(allocator, bytes, header, tq, header_len, opts);
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

    const data_start = 8 + header_len;
    if (data_start > bytes.len) return error.TruncatedFile;
    const raw = tensorRegion(bytes, tensor, data_start) orelse return error.TruncatedFile;
    const elem_size: usize = if (is_f32) 4 else 1;
    if (raw.len != rows * cols * elem_size) return error.BadShape;

    if (is_f32) {
        if (borrowable(opts) and isAlignedFor(raw, f32)) {
            const data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw));
            return .{ .matrix = .{ .f32_data = data }, .rows = rows, .cols = cols, .borrowed = true };
        }
        const data = try allocator.alloc(f32, rows * cols);
        errdefer allocator.free(data);
        for (data, 0..) |*v, i| {
            v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        }
        return .{ .matrix = .{ .f32_data = data }, .rows = rows, .cols = cols };
    }

    if (borrowable(opts)) {
        const data: []const i8 = @ptrCast(raw);
        return .{ .matrix = .{ .i8_data = data }, .rows = rows, .cols = cols, .borrowed = true };
    }
    const data = try allocator.alloc(i8, rows * cols);
    errdefer allocator.free(data);
    for (data, raw) |*v, b| v.* = @bitCast(b);
    return .{ .matrix = .{ .i8_data = data }, .rows = rows, .cols = cols };
}

fn parseTq4(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    header: std.json.Value,
    tq: std.json.Value,
    header_len: u64,
    opts: Options,
) Error!Embeddings {
    // tq4 is this repo's format; the version field exists so a future layout
    // change is an error instead of silently wrong vectors.
    const meta = header.object.get("__metadata__") orelse return error.UnsupportedTq4Version;
    if (meta != .object) return error.UnsupportedTq4Version;
    const version = stringField(meta, "tq4_version") orelse return error.UnsupportedTq4Version;
    if (!std.mem.eql(u8, version, "1")) return error.UnsupportedTq4Version;

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

    if (borrowable(opts) and isAlignedFor(scales_raw, f32)) {
        return .{
            .matrix = .{ .tq4_data = .{
                .packed_data = packed_raw,
                .scales = @alignCast(std.mem.bytesAsSlice(f32, scales_raw)),
            } },
            .rows = rows,
            .cols = half_cols * 2,
            .borrowed = true,
        };
    }

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

fn borrowable(opts: Options) bool {
    return opts.borrow and native_endian == .little;
}

fn isAlignedFor(raw: []const u8, comptime T: type) bool {
    return std.mem.isAligned(@intFromPtr(raw.ptr), @alignOf(T));
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

    const emb = try parse(testing.allocator, bytes, .{});
    defer emb.deinit(testing.allocator);

    try testing.expect(!emb.borrowed);
    try testing.expectEqual(@as(usize, 2), emb.rows);
    try testing.expectEqual(@as(usize, 3), emb.cols);
    try testing.expectEqual(@as(f32, -1.5), emb.matrix.f32_data[3]);
}

test "parse borrows an aligned f32 tensor without copying" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Header padded to a multiple of 8 so the data region is f32-aligned
    // whenever the buffer itself is.
    const header =
        \\{"embeddings":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}
    ++ "       ";
    comptime std.debug.assert(header.len % 8 == 0);
    const unaligned = try buildFixture(a, header, &.{ 0.5, -2.0 });
    const bytes = try a.alignedAlloc(u8, .of(u64), unaligned.len);
    @memcpy(bytes, unaligned);

    const emb = try parse(testing.allocator, bytes, .{ .borrow = true });
    defer emb.deinit(testing.allocator);

    try testing.expect(emb.borrowed);
    try testing.expectEqual(@as(f32, 0.5), emb.matrix.f32_data[0]);
    try testing.expectEqual(@as(f32, -2.0), emb.matrix.f32_data[1]);
    try testing.expectEqual(
        @intFromPtr(bytes.ptr) + 8 + header.len,
        @intFromPtr(emb.matrix.f32_data.ptr),
    );
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

    const emb = try parse(testing.allocator, out.items, .{ .borrow = true });
    defer emb.deinit(testing.allocator);

    try testing.expect(emb.borrowed);
    try testing.expectEqual(@as(i8, 127), emb.matrix.i8_data[0]);
    try testing.expectEqual(@as(i8, -127), emb.matrix.i8_data[1]);
    try testing.expectEqual(@as(i8, 5), emb.matrix.i8_data[3]);
}

fn buildTq4Fixture(a: std.mem.Allocator, header: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try out.appendSlice(a, &len_buf);
    try out.appendSlice(a, header);
    // Two rows of dim 4: packed nibbles, then one f32 scale per row.
    try out.appendSlice(a, &[_]u8{ 0x21, 0x43, 0xFE, 0x10 });
    for ([_]f32{ 0.5, 2.0 }) |f| {
        var fb: [4]u8 = undefined;
        std.mem.writeInt(u32, &fb, @bitCast(f), .little);
        try out.appendSlice(a, &fb);
    }
    return out.items;
}

test "parse reads a tq4 tensor pair" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const bytes = try buildTq4Fixture(arena.allocator(),
        \\{"__metadata__":{"tq4_version":"1"},"embeddings_tq4":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"scales":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}
    );

    const emb = try parse(testing.allocator, bytes, .{});
    defer emb.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), emb.rows);
    try testing.expectEqual(@as(usize, 4), emb.cols);
    try testing.expectEqual(@as(u8, 0x21), emb.matrix.tq4_data.packed_data[0]);
    try testing.expectEqual(@as(f32, 0.5), emb.matrix.tq4_data.scales[0]);
    try testing.expectEqual(@as(f32, 2.0), emb.matrix.tq4_data.scales[1]);
}

test "parse rejects tq4 without a known version" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const unversioned = try buildTq4Fixture(a,
        \\{"embeddings_tq4":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"scales":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}
    );
    try testing.expectError(error.UnsupportedTq4Version, parse(testing.allocator, unversioned, .{}));

    const future = try buildTq4Fixture(a,
        \\{"__metadata__":{"tq4_version":"2"},"embeddings_tq4":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"scales":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}
    );
    try testing.expectError(error.UnsupportedTq4Version, parse(testing.allocator, future, .{}));
}

test "parse rejects wrong dtype and missing tensor" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const f16_fixture = try buildFixture(arena.allocator(),
        \\{"embeddings":{"dtype":"F16","shape":[1,2],"data_offsets":[0,4]}}
    , &.{1.0});
    try testing.expectError(error.UnsupportedDtype, parse(testing.allocator, f16_fixture, .{}));

    const wrong_name = try buildFixture(arena.allocator(),
        \\{"weights":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}
    , &.{1.0});
    try testing.expectError(error.MissingEmbeddings, parse(testing.allocator, wrong_name, .{}));
}
