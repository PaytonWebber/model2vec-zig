//! Minimal safetensors reader for model2vec models, which store exactly one
//! tensor: `embeddings`, shape [vocab, dim].
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

pub const Embeddings = struct {
    /// rows * cols values, row-major. Owned by the allocator passed to parse.
    data: []f32,
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

    const tensor = header.object.get("embeddings") orelse return error.MissingEmbeddings;
    if (tensor != .object) return error.BadHeader;

    const dtype = stringField(tensor, "dtype") orelse return error.BadHeader;
    if (!std.mem.eql(u8, dtype, "F32")) return error.UnsupportedDtype;

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
    if (raw.len != rows * cols * 4) return error.BadShape;

    const data = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(data);
    for (data, 0..) |*v, i| {
        v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
    }

    return .{ .data = data, .rows = rows, .cols = cols };
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
    defer testing.allocator.free(emb.data);

    try testing.expectEqual(@as(usize, 2), emb.rows);
    try testing.expectEqual(@as(usize, 3), emb.cols);
    try testing.expectEqual(@as(f32, -1.5), emb.data[3]);
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
