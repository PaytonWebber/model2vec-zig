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
    // Subtraction, not `8 + header_len > bytes.len`: a near-max header_len
    // would wrap the addition and pass the check.
    if (header_len > bytes.len - 8) return error.TruncatedFile;
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
    const raw = tensorRegion(bytes, tensor, data_start) orelse return error.TruncatedFile;
    const elem_size: usize = if (is_f32) 4 else 1;
    if (raw.len != checkedSize(rows, cols, elem_size) orelse return error.BadShape)
        return error.BadShape;

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
    if (packed_raw.len != checkedSize(rows, half_cols, 1) orelse return error.BadShape)
        return error.BadShape;

    const scales_tensor = header.object.get("scales") orelse return error.BadHeader;
    if (scales_tensor != .object) return error.BadHeader;
    const scales_dtype = stringField(scales_tensor, "dtype") orelse return error.BadHeader;
    if (!std.mem.eql(u8, scales_dtype, "F32")) return error.UnsupportedDtype;
    const scales_raw = tensorRegion(bytes, scales_tensor, data_start) orelse return error.TruncatedFile;
    if (scales_raw.len != checkedSize(rows, 4, 1) orelse return error.BadShape)
        return error.BadShape;
    const cols = std.math.mul(usize, half_cols, 2) catch return error.BadShape;

    if (borrowable(opts) and isAlignedFor(scales_raw, f32)) {
        return .{
            .matrix = .{ .tq4_data = .{
                .packed_data = packed_raw,
                .scales = @alignCast(std.mem.bytesAsSlice(f32, scales_raw)),
            } },
            .rows = rows,
            .cols = cols,
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

/// rows * cols * elem_size, or null on overflow.
fn checkedSize(rows: usize, cols: usize, elem_size: usize) ?usize {
    const cells = std.math.mul(usize, rows, cols) catch return null;
    return std.math.mul(usize, cells, elem_size) catch return null;
}

fn tensorRegion(bytes: []const u8, tensor: std.json.Value, data_start: u64) ?[]const u8 {
    const offsets = tensor.object.get("data_offsets") orelse return null;
    if (offsets != .array or offsets.array.items.len != 2) return null;
    const begin = intField(offsets.array.items[0]) orelse return null;
    const end = intField(offsets.array.items[1]) orelse return null;
    // Subtraction, not `data_start + end > bytes.len`: huge offsets would
    // wrap the addition and pass the check.
    if (end < begin or data_start > bytes.len or end > bytes.len - data_start) return null;
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

/// Parse must either error or return a matrix whose every claimed byte is
/// readable and sized consistently with rows * cols. Crashes, leaks, and
/// out-of-bounds slices are bugs; errors are expected outcomes.
fn checkParseInvariants(input: []const u8) !void {
    for ([_]Options{ .{}, .{ .borrow = true } }) |opts| {
        const emb = parse(testing.allocator, input, opts) catch continue;
        defer emb.deinit(testing.allocator);

        switch (emb.matrix) {
            .f32_data => |d| {
                try testing.expectEqual(emb.rows * emb.cols, d.len);
                if (d.len > 0) std.mem.doNotOptimizeAway(d[0] + d[d.len - 1]);
            },
            .i8_data => |d| {
                try testing.expectEqual(emb.rows * emb.cols, d.len);
                if (d.len > 0) std.mem.doNotOptimizeAway(d[0] +% d[d.len - 1]);
            },
            .tq4_data => |t| {
                try testing.expectEqual(emb.rows * emb.cols / 2, t.packed_data.len);
                try testing.expectEqual(emb.rows, t.scales.len);
                if (t.packed_data.len > 0) std.mem.doNotOptimizeAway(t.packed_data[t.packed_data.len - 1]);
                if (t.scales.len > 0) std.mem.doNotOptimizeAway(t.scales[t.scales.len - 1]);
            },
        }
    }
}

// Coverage-guided fuzzing of the same invariants: `zig build test --fuzz`.
// (The 0.16.0 test runner fails to compile in fuzz mode; this entry point
// works on Zig versions with the fixed runner and runs as a smoke test
// otherwise.) The framed mode wraps fuzzer-chosen header bytes in a valid
// length prefix so coverage reaches the JSON and offset logic instead of
// stopping at the length check.
test "fuzz parse" {
    try std.testing.fuzz({}, fuzzParse, .{});
}

fn fuzzParse(_: void, smith: *std.testing.Smith) anyerror!void {
    @disableInstrumentation();
    var buf: [4096]u8 = undefined;

    const framed = smith.value(bool);
    var input: []const u8 = undefined;
    if (framed) {
        const header_len = smith.slice(buf[8..2048]);
        std.mem.writeInt(u64, buf[0..8], header_len, .little);
        const data_len = smith.slice(buf[8 + header_len ..]);
        input = buf[0 .. 8 + header_len + data_len];
    } else {
        input = buf[0..smith.slice(&buf)];
    }

    try checkParseInvariants(input);
}

// Deterministic randomized harness over the same invariants, run on every
// `zig build test`: raw byte soup, length-framed soup, and valid fixtures
// with a few bytes corrupted (which reaches the offset and shape logic that
// random bytes never parse far enough to touch).
test "parse survives random and corrupted inputs" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const fixtures = [_][]const u8{
        try buildFixture(a,
            \\{"embeddings":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}
        , &.{ 1.0, 2.0, 3.0, -1.5, 0.25, 0.0 }),
        try buildTq4Fixture(a,
            \\{"__metadata__":{"tq4_version":"1"},"embeddings_tq4":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"scales":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}
        ),
    };

    var prng = std.Random.DefaultPrng.init(0x5af37e45_0f2e11);
    const rand = prng.random();
    var buf: [512]u8 = undefined;

    for (0..20_000) |_| {
        switch (rand.enumValue(enum { raw, framed, corrupted })) {
            .raw => {
                const len = rand.uintAtMost(usize, buf.len);
                rand.bytes(buf[0..len]);
                try checkParseInvariants(buf[0..len]);
            },
            .framed => {
                const header_len = rand.uintAtMost(usize, 200);
                const data_len = rand.uintAtMost(usize, buf.len - 8 - 200);
                std.mem.writeInt(u64, buf[0..8], header_len, .little);
                rand.bytes(buf[8 .. 8 + header_len + data_len]);
                try checkParseInvariants(buf[0 .. 8 + header_len + data_len]);
            },
            .corrupted => {
                const fixture = fixtures[rand.uintLessThan(usize, fixtures.len)];
                const input = buf[0..fixture.len];
                @memcpy(input, fixture);
                for (0..1 + rand.uintLessThan(usize, 16)) |_| {
                    input[rand.uintLessThan(usize, input.len)] = rand.int(u8);
                }
                try checkParseInvariants(input);
            },
        }
    }
}

test "parse rejects overflowing lengths, offsets, and shapes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // header_len near maxInt(u64): `8 + header_len` wraps, so the bounds
    // check must subtract instead.
    var wrap_len: [16]u8 = undefined;
    std.mem.writeInt(u64, wrap_len[0..8], std.math.maxInt(u64) - 3, .little);
    @memset(wrap_len[8..], 0);
    try testing.expectError(error.TruncatedFile, parse(testing.allocator, &wrap_len, .{}));

    // data_offsets far past the end of the file.
    const wrap_offsets = try buildFixture(a,
        \\{"embeddings":{"dtype":"F32","shape":[1,1],"data_offsets":[9223372036854775800,9223372036854775804]}}
    , &.{1.0});
    try testing.expectError(error.TruncatedFile, parse(testing.allocator, wrap_offsets, .{}));

    // shape product overflows usize: 2^62 * 4 == 0 mod 2^64, which would
    // match an empty data region without the checked multiply.
    const wrap_shape = try buildFixture(a,
        \\{"embeddings":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,0]}}
    , &.{});
    try testing.expectError(error.BadShape, parse(testing.allocator, wrap_shape, .{}));
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
