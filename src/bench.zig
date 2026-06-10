//! Embedding throughput on a real model: `zig build bench` (needs
//! scripts/fetch-model.sh first).

const std = @import("std");
const m2v = @import("root.zig");

const text = "the daemon owns the store; per-process stores corrupt the snapshot on concurrent writes";
const iterations = 50_000;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var model = try m2v.Model.load(gpa, io, "models/potion-base-8M");
    defer model.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const out = try gpa.alloc(f32, model.dim);
    defer gpa.free(out);

    // Warm up, then measure.
    for (0..1000) |_| {
        try model.embedInto(arena.allocator(), text, out);
        _ = arena.reset(.retain_capacity);
    }

    const start = std.Io.Timestamp.now(io, .awake);
    for (0..iterations) |_| {
        try model.embedInto(arena.allocator(), text, out);
        _ = arena.reset(.retain_capacity);
    }
    const elapsed_ns = std.Io.Timestamp.now(io, .awake).nanoseconds - start.nanoseconds;

    var ids: std.ArrayList(u32) = .empty;
    try model.tok.encode(arena.allocator(), text, &ids);

    const ns_per: f64 = @as(f64, @floatFromInt(elapsed_ns)) / iterations;
    std.debug.print("{d} embeds of a {d}-token text: {d:.1} us/embed, {d:.0} embeds/s\n", .{
        iterations,
        ids.items.len,
        ns_per / 1000.0,
        1e9 / ns_per,
    });
}
