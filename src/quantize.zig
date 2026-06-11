//! Quantize a model2vec f32 safetensors file to i8, matching the reference
//! implementation's scheme exactly: one global scale of max(|x|)/127,
//! round-half-to-even (numpy's rint), clip to [-127, 127]. The scale is not
//! stored; it cancels under the L2 normalization the models apply, so
//! inference pools the raw i8 values.
//!
//!     m2v-quantize model.safetensors model.i8.safetensors

const std = @import("std");
const safetensors = @import("safetensors.zig");

pub fn main(init: std.process.Init) !void {
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args_it.deinit();
    _ = args_it.next(); // program name
    const in_path = args_it.next() orelse return usage();
    const out_path = args_it.next() orelse return usage();

    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = init.io;

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, in_path, a, .limited(2 * 1024 * 1024 * 1024));
    const emb = try safetensors.parse(a, bytes);
    const data = switch (emb.matrix) {
        .f32_data => |d| d,
        // Idempotent: re-running over an already-quantized model (a restored
        // CI cache, a second local run) is a no-op, not an error.
        .i8_data => {
            std.debug.print("{s} is already i8, nothing to do\n", .{in_path});
            return;
        },
    };

    const out = try quantize(a, data);
    try writeI8Safetensors(a, io, out_path, out, emb.rows, emb.cols);
    std.debug.print("wrote {s}: [{d}, {d}] i8 ({d} bytes from {d})\n", .{
        out_path, emb.rows, emb.cols, out.len, data.len * 4,
    });
}

pub fn quantize(a: std.mem.Allocator, data: []const f32) ![]i8 {
    var max_abs: f32 = 0;
    for (data) |v| max_abs = @max(max_abs, @abs(v));
    const scale = max_abs / 127.0;

    const out = try a.alloc(i8, data.len);
    for (data, out) |v, *q| {
        // The reference casts to f16 first, divides at higher precision with
        // an f16 destination, then rounds; matching the exact order keeps the
        // output byte-identical to a Python-quantized model.
        const v16: f16 = @floatCast(v);
        const scaled: f16 = @floatCast(@as(f64, v16) / @as(f64, scale));
        const r = roundHalfToEven(@floatCast(scaled));
        q.* = @intFromFloat(std.math.clamp(r, -127.0, 127.0));
    }
    return out;
}

/// numpy.rint semantics; Zig's @round rounds half away from zero.
fn roundHalfToEven(x: f32) f32 {
    const fl = @floor(x);
    const frac = x - fl;
    if (frac > 0.5) return fl + 1;
    if (frac < 0.5) return fl;
    return if (@mod(fl, 2) == 0) fl else fl + 1;
}

fn writeI8Safetensors(a: std.mem.Allocator, io: std.Io, path: []const u8, data: []const i8, rows: usize, cols: usize) !void {
    const header = try std.fmt.allocPrint(
        a,
        "{{\"embeddings\":{{\"dtype\":\"I8\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}}}",
        .{ rows, cols, data.len },
    );
    // The reference pads headers to an 8-byte boundary with spaces.
    const padded_len = (header.len + 7) / 8 * 8;

    var out: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, padded_len, .little);
    try out.appendSlice(a, &len_buf);
    try out.appendSlice(a, header);
    try out.appendNTimes(a, ' ', padded_len - header.len);
    try out.appendSlice(a, @ptrCast(data));

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
}

fn usage() error{BadUsage} {
    std.debug.print("usage: m2v-quantize <in.safetensors> <out.safetensors>\n", .{});
    return error.BadUsage;
}

const testing = std.testing;

test "quantize matches the reference scheme" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    // max abs 12.7 -> scale 0.1
    const data = [_]f32{ 12.7, -12.7, 0.0, 0.05, -0.05, 0.15, 1.0 };
    const q = try quantize(arena.allocator(), &data);

    try testing.expectEqual(@as(i8, 127), q[0]);
    try testing.expectEqual(@as(i8, -127), q[1]);
    try testing.expectEqual(@as(i8, 0), q[2]);
    // 0.5 rounds to even: 0; -0.5 likewise; 1.5 rounds to 2.
    try testing.expectEqual(@as(i8, 0), q[3]);
    try testing.expectEqual(@as(i8, 0), q[4]);
    try testing.expectEqual(@as(i8, 2), q[5]);
    try testing.expectEqual(@as(i8, 10), q[6]);
}
