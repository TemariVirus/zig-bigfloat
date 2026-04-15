const std = @import("std");

var test_options_mod: *std.Build.Module = undefined;

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const bigfloat_mod = b.addModule("bigfloat", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_options = b.addOptions();
    test_options.addOption([32]u8, "test_seed", try getTestSeed(b));
    test_options.addOption(bool, "run_slow_tests", b.option(
        bool,
        "run_slow_tests",
        "Whether to run slow tests.",
    ) orelse false);
    test_options_mod = test_options.createModule();

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "options", .module = test_options_mod }},
        }),
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    testCross(b, optimize);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "bigfloat", .module = bigfloat_mod }},
        }),
    });
    const bench_asm = bench_exe.getEmittedAsm();

    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&b.addInstallArtifact(bench_exe, .{}).step);
    bench_step.dependOn(&b.addInstallFile(bench_asm, "bench.S").step);
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

fn testCross(b: *std.Build, optimize: std.builtin.OptimizeMode) void {
    b.enable_qemu = true;
    b.enable_wasmtime = true;

    const test_step = b.step("test-cross", "Run unit tests on multiple architectures");
    for ([_]std.Target.Query{
        .{ .cpu_arch = .arm, .cpu_model = .baseline },
        .{ .cpu_arch = .aarch64, .cpu_model = .baseline },
        .{ .cpu_arch = .riscv32, .cpu_model = .baseline },
        .{ .cpu_arch = .riscv64, .cpu_model = .baseline },
        .{ .cpu_arch = .wasm32, .cpu_model = .baseline, .os_tag = .wasi },
        // TODO
        // Disable due to weird type mismatch error
        // .{ .cpu_arch = .wasm64, .cpu_model = .baseline, .os_tag = .wasi },
        .{ .cpu_arch = .x86, .cpu_model = .baseline },
        .{ .cpu_arch = .x86_64, .cpu_model = .baseline },

        // Apple
        .{ .cpu_arch = .aarch64, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 } },
        .{ .cpu_arch = .aarch64, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m4 } },
        .{ .cpu_arch = .aarch64, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_s4 } },
        .{ .cpu_arch = .aarch64, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_s5 } },
        // Snapdragon
        .{ .cpu_arch = .aarch64, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.kryo } },
        .{ .cpu_arch = .aarch64, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.oryon_1 } },

        // Intel
        .{ .cpu_arch = .x86, .cpu_model = .{ .explicit = &std.Target.x86.cpu.pentium_m } },
        .{ .cpu_arch = .x86_64, .cpu_model = .{ .explicit = &std.Target.x86.cpu.sandybridge } },
        .{ .cpu_arch = .x86_64, .cpu_model = .{ .explicit = &std.Target.x86.cpu.skylake_avx512 } },
        .{ .cpu_arch = .x86_64, .cpu_model = .{ .explicit = &std.Target.x86.cpu.arrowlake } },
        // AMD
        .{ .cpu_arch = .x86_64, .cpu_model = .{ .explicit = &std.Target.x86.cpu.znver1 } },
        .{ .cpu_arch = .x86_64, .cpu_model = .{ .explicit = &std.Target.x86.cpu.znver3 } },
        // Generic
        .{ .cpu_arch = .x86_64, .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64 } },
        .{ .cpu_arch = .x86_64, .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v4 } },
    }) |target_query| {
        const unit_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/root.zig"),
                .target = b.resolveTargetQuery(target_query),
                // LLVM takes forever to compile for WASI in debug mode
                .optimize = if (target_query.os_tag == .wasi) .ReleaseSmall else optimize,
                .imports = &.{.{ .name = "options", .module = test_options_mod }},
            }),
        });
        const run_unit_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_unit_tests.step);
    }
}
