//! TurboQuant-style 4-bit quantization of the embedding matrix.
//!
//! Following the TurboQuant recipe (and quantal's use of it): rotate every
//! row by a fixed random orthonormal matrix, which concentrates coordinates
//! toward N(0, 1/d) so a uniform per-row scalar quantizer is near-optimal,
//! then store each coordinate as a signed nibble with one f32 scale per row.
//!
//! The rotation is applied once, at quantization time, and never stored:
//! cosine similarity is rotation-invariant and queries are pooled from the
//! same rotated matrix, so the runtime never needs to know the basis
//! changed. The only consequence is that vectors from a tq4 model are not
//! comparable with vectors from the same model quantized differently;
//! consumers that persist vectors should key them to a model fingerprint.
//!
//! Why 4-bit and not 3: measured on potion-retrieval-32M, 4-bit keeps row
//! reconstruction at ~0.993 cosine (3-bit: ~0.973) for one extra bit per
//! coordinate, and nibbles pack two per byte without bit-spanning. Retrieval
//! ranks were unchanged at either width, but downstream cosine thresholds
//! (duplicate detection, relevance floors) appreciate the tighter noise.

const std = @import("std");

/// Quantize an f32 matrix to rotated signed nibbles plus per-row scales.
/// `dim` must be even (two coordinates per packed byte).
pub const Quantized = struct {
    /// rows * dim/2 bytes; low nibble holds the even coordinate.
    packed_data: []u8,
    /// One scale per row.
    scales: []f32,
};

pub fn quantize(a: std.mem.Allocator, data: []const f32, rows: usize, dim: usize, seed: u64) !Quantized {
    std.debug.assert(dim % 2 == 0);
    std.debug.assert(data.len == rows * dim);

    const rotation = try rotationMatrix(a, dim, seed);
    defer a.free(rotation);

    const packed_data = try a.alloc(u8, rows * dim / 2);
    errdefer a.free(packed_data);
    const scales = try a.alloc(f32, rows);
    errdefer a.free(scales);

    const rotated = try a.alloc(f32, dim);
    defer a.free(rotated);

    for (0..rows) |r| {
        const row = data[r * dim ..][0..dim];
        applyRotation(rotation, dim, row, rotated);

        var max_abs: f32 = 0;
        for (rotated) |v| max_abs = @max(max_abs, @abs(v));
        const scale = if (max_abs > 0) max_abs / 8.0 else 1.0;
        scales[r] = scale;

        const out = packed_data[r * dim / 2 ..][0 .. dim / 2];
        var i: usize = 0;
        while (i < dim) : (i += 2) {
            const lo = quantizeCoord(rotated[i], scale);
            const hi = quantizeCoord(rotated[i + 1], scale);
            out[i / 2] = (@as(u8, @bitCast(@as(i8, hi))) << 4) | (@as(u8, @bitCast(@as(i8, lo))) & 0xF);
        }
    }

    return .{ .packed_data = packed_data, .scales = scales };
}

fn quantizeCoord(v: f32, scale: f32) i4 {
    const r = roundHalfToEven(v / scale);
    return @intFromFloat(std.math.clamp(r, -8.0, 7.0));
}

fn roundHalfToEven(x: f32) f32 {
    const fl = @floor(x);
    const frac = x - fl;
    if (frac > 0.5) return fl + 1;
    if (frac < 0.5) return fl;
    return if (@mod(fl, 2) == 0) fl else fl + 1;
}

/// Decode one packed byte into two signed coordinates.
pub inline fn unpackByte(b: u8) struct { lo: i8, hi: i8 } {
    const lo4: i4 = @bitCast(@as(u4, @truncate(b)));
    const hi4: i4 = @bitCast(@as(u4, @truncate(b >> 4)));
    return .{ .lo = lo4, .hi = hi4 };
}

/// A fixed orthonormal matrix from seeded Gaussian rows, orthonormalized with
/// modified Gram-Schmidt: the TurboQuant preconditioning rotation. Used only
/// at quantization time, so the O(d^3) construction cost is irrelevant.
pub fn rotationMatrix(a: std.mem.Allocator, dim: usize, seed: u64) ![]f32 {
    const rows = try a.alloc(f32, dim * dim);
    errdefer a.free(rows);

    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    for (0..dim) |i| {
        const row = rows[i * dim ..][0..dim];
        while (true) {
            for (row) |*v| v.* = @floatCast(rand.floatNorm(f64));
            // Project out all previous rows.
            for (0..i) |j| {
                const prev = rows[j * dim ..][0..dim];
                var d: f64 = 0;
                for (row, prev) |x, p| d += @as(f64, x) * p;
                for (row, prev) |*x, p| x.* -= @floatCast(d * p);
            }
            var norm_sq: f64 = 0;
            for (row) |x| norm_sq += @as(f64, x) * x;
            if (norm_sq > 1e-6) {
                const inv: f32 = @floatCast(1.0 / @sqrt(norm_sq));
                for (row) |*x| x.* *= inv;
                break;
            }
            // Degenerate draw: redo this row.
        }
    }
    return rows;
}

fn applyRotation(rotation: []const f32, dim: usize, x: []const f32, out: []f32) void {
    for (0..dim) |i| {
        const row = rotation[i * dim ..][0..dim];
        var sum: f32 = 0;
        for (row, x) |r, v| sum += r * v;
        out[i] = sum;
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "rotationMatrix is orthonormal and preserves norms" {
    const dim = 16;
    const rot = try rotationMatrix(testing.allocator, dim, 7);
    defer testing.allocator.free(rot);

    // Rows are unit length and mutually orthogonal.
    for (0..dim) |i| {
        const ri = rot[i * dim ..][0..dim];
        for (0..dim) |j| {
            const rj = rot[j * dim ..][0..dim];
            var d: f64 = 0;
            for (ri, rj) |x, y| d += @as(f64, x) * y;
            const want: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(want, d, 1e-5);
        }
    }
}

test "unpackByte round-trips signed nibbles" {
    var v: i8 = -8;
    while (v <= 7) : (v += 1) {
        const lo: i4 = @intCast(v);
        const hi: i4 = @intCast(@divTrunc(v, 2)); // [-4, 3], always in range
        const b = (@as(u8, @bitCast(@as(i8, hi))) << 4) | (@as(u8, @bitCast(@as(i8, lo))) & 0xF);
        const got = unpackByte(b);
        try testing.expectEqual(v, got.lo);
        try testing.expectEqual(@as(i8, @divTrunc(v, 2)), got.hi);
    }
}

test "quantize reconstructs rows with high cosine" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const rows = 32;
    const dim = 64;
    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();
    const data = try a.alloc(f32, rows * dim);
    for (data) |*v| v.* = @floatCast(rand.floatNorm(f64));

    const q = try quantize(a, data, rows, dim, 42);
    const rotation = try rotationMatrix(a, dim, 42);
    const rotated = try a.alloc(f32, dim);

    for (0..rows) |r| {
        applyRotation(rotation, dim, data[r * dim ..][0..dim], rotated);
        const packed_row = q.packed_data[r * dim / 2 ..][0 .. dim / 2];
        const scale = q.scales[r];

        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (packed_row, 0..) |b, i| {
            const pair = unpackByte(b);
            const deq_lo = @as(f32, @floatFromInt(pair.lo)) * scale;
            const deq_hi = @as(f32, @floatFromInt(pair.hi)) * scale;
            dot += @as(f64, rotated[i * 2]) * deq_lo + @as(f64, rotated[i * 2 + 1]) * deq_hi;
            na += @as(f64, rotated[i * 2]) * rotated[i * 2] + @as(f64, rotated[i * 2 + 1]) * rotated[i * 2 + 1];
            nb += @as(f64, deq_lo) * deq_lo + @as(f64, deq_hi) * deq_hi;
        }
        const cos = dot / (@sqrt(na) * @sqrt(nb));
        try testing.expect(cos > 0.97);
    }
}
