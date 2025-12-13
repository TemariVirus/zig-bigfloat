const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

// TODO: replace with std.testing.fuzz when it's ready
fn fuzz(
    context: anytype,
    comptime testOne: fn (context: @TypeOf(context), rng: std.Random) anyerror!void,
    iters: u64,
) anyerror!void {
    if (!@import("options").run_slow_tests) return error.SkipZigTest;

    var rng: std.Random.ChaCha = .init(@import("options").test_seed);
    for (0..iters) |_| {
        try testOne(context, rng.random());
    }
}

const TestOp = enum {
    // Unary
    abs,
    neg,
    inv,
    exp2,
    log2,

    // Binary
    add,
    sub,
    mul,
    pow,
};
fn Context(BF: type, comptime op: TestOp) type {
    return struct {
        const F = @FieldType(BF, "significand");
        const Int = std.meta.Int(.signed, @typeInfo(F).float.bits);
        const arg_count = switch (op) {
            .abs, .neg, .inv, .exp2, .log2 => 1,
            .add, .sub, .mul, .pow => 2,
        };

        fn isEquivalent(a: F, b: F) bool {
            if (std.math.isNan(a)) {
                return std.math.isNan(b);
            } else {
                return a == b;
            }
        }

        fn isApproxEquivalent(a: F, b: F) bool {
            if (std.math.isNan(a)) {
                return std.math.isNan(b);
            } else {
                // Bitcast so that inifities are properly handled
                // (I <3 IEEE754)
                const ai: Int = @bitCast(a);
                const bi: Int = @bitCast(b);
                // Use tolerance of 20-21 ulp
                return @abs(ai -% bi) <= 20;
            }
        }

        fn expect(args: [arg_count]F, expected: F, actual: BF) !void {
            const actual_f = actual.toFloat(F);
            const check = switch (op) {
                .abs, .neg, .inv, .add, .sub, .mul => isEquivalent,
                .exp2, .log2, .pow => isApproxEquivalent,
            };
            if (check(expected, actual_f)) return;

            std.debug.print("BigFloat type: {}\nexpected {t}(", .{
                @TypeOf(actual),
                op,
            });
            std.debug.print("{e}", .{args[0]});
            for (args[1..]) |arg| {
                std.debug.print(", {e}", .{arg});
            }
            std.debug.print(") = {e}, found {e}\n", .{
                expected,
                actual_f,
            });
            return error.UnexpectedTestResult;
        }

        /// Returns an array of random floats where each bit pattern is equally likely.
        fn randomFloats(rng: std.Random) [arg_count]F {
            var floats: [arg_count]F = undefined;
            rng.bytes(@ptrCast(&floats));
            return floats;
        }

        fn applyF(args: [arg_count]F) F {
            return switch (op) {
                .abs => @abs(args[0]),
                .neg => -args[0],
                .inv => 1.0 / args[0],
                .exp2 => @exp2(args[0]),
                .log2 => @log2(args[0]),
                .add => args[0] + args[1],
                .sub => args[0] - args[1],
                .mul => args[0] * args[1],
                .pow => std.math.pow(F, args[0], args[1]),
            };
        }

        fn applyBF(args: [arg_count]BF) BF {
            return switch (op) {
                .abs => args[0].abs(),
                .neg => args[0].neg(),
                .inv => args[0].inv(),
                .exp2 => args[0].exp2(),
                .log2 => args[0].log2(),
                .add => args[0].add(args[1]),
                .sub => args[0].sub(args[1]),
                .mul => args[0].mul(args[1]),
                .pow => args[0].pow(args[1]),
            };
        }

        fn containsSubnormal(fs: [arg_count]F, expected: F) bool {
            const epsilon = std.math.floatMin(F);
            // Subnormal numbers can't always be represented exactly
            for (fs) |f| {
                if (f != 0 and @abs(f) < epsilon) return true;
            }
            return expected != 0 and @abs(expected) < epsilon;
        }

        fn testOne(_: @This(), rng: std.Random) !void {
            const fs, const expected = while (true) {
                const fs = randomFloats(rng);
                const expected = applyF(fs);
                if (containsSubnormal(fs, expected)) continue;
                if (F == f128 and op == .inv) {
                    // https://codeberg.org/ziglang/zig/issues/30179
                    // f128 division rounds to 0 when the result should be subnormal
                    if (@abs(fs[0]) > 1.0 / std.math.floatMin(f128)) continue;
                }
                break .{ fs, expected };
            };

            var bfs: [arg_count]BF = undefined;
            for (0..arg_count) |i| {
                bfs[i] = BF.initExact(fs[i]).?;
            }
            const actual = applyBF(bfs);
            try expect(fs, expected, actual);
        }
    };
}

const FUZZ_ITERS = 420_069;

test "fuzz abs" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .abs);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz neg" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .neg);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz inv" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .inv);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz add" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .add);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz sub" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .sub);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz mul" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .mul);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz exp2" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .exp2);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz log2" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f80),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .log2);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz pow" {
    inline for (.{
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
    }) |BF| {
        const Ctx = Context(BF, .pow);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}
