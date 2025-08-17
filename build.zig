const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("bigfloat", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .strip = true,
            .imports = &.{.{ .name = "bigfloat", .module = mod }},
        }),
    });
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&b.addInstallArtifact(bench_exe, .{}).step);
    bench_step.dependOn(&b.addRunArtifact(bench_exe).step);
}
