const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("model2vec", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);

    const fmt = b.addFmt(.{
        .paths = &.{ "build.zig", "src" },
        .check = true,
    });

    const test_step = b.step("test", "Run tests (set M2V_MODEL_DIR for the parity test)");
    test_step.dependOn(&run_tests.step);

    const check_step = b.step("check", "Run tests and format checks");
    check_step.dependOn(test_step);
    check_step.dependOn(&fmt.step);

    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    const run_bench = b.addRunArtifact(bench);
    b.step("bench", "Measure embedding throughput (needs a fetched model)").dependOn(&run_bench.step);
}
