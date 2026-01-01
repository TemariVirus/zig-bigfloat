const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const bigfloat_mod = b.addModule("bigfloat", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const options = b.addOptions();
    options.addOption([32]u8, "test_seed", try getTestSeed(b));
    options.addOption(bool, "run_slow_tests", b.option(
        bool,
        "run_slow_tests",
        "Whether to run slow tests.",
    ) orelse false);
    const options_mod = options.createModule();

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "options", .module = options_mod }},
        }),
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
            .imports = &.{.{ .name = "bigfloat", .module = bigfloat_mod }},
        }),
    });
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&b.addInstallArtifact(bench_exe, .{}).step);
    bench_step.dependOn(&b.addRunArtifact(bench_exe).step);
}

fn getTestSeed(b: *std.Build) ![32]u8 {
    var seed = [_]u8{0} ** 32;
    const is_root = b.pkg_hash.len == 0;
    if (!is_root) return seed;

    const seed_hex = b.option(
        []const u8,
        "test_seed",
        "Seed to use for random tests. Defaults to the current commit hash",
    ) orelse b.run(&.{ "git", "rev-parse", "--verify", "HEAD" });
    if (seed_hex.len < 40) {
        return error.SeedTooShort;
    }

    _ = try std.fmt.hexToBytes(&seed, seed_hex[0..40]);
    return seed;
}
