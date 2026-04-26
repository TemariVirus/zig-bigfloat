const std = @import("std");
const math = std.math;
const testing = std.testing;

const BigFloat = @import("../root.zig").BigFloat;
const utils = @import("../test_utils.zig");

// TODO: replace with std.testing.fuzz when it's ready
fn fuzz(
    context: anytype,
    comptime testOne: fn (context: @TypeOf(context), rng: std.Random) anyerror!void,
    iters: u64,
) anyerror!void {
    if (!@import("options").run_slow_tests) return error.SkipZigTest;

    var rng: std.Random.ChaCha = .init(@import("options").test_seed);
    var i: u64 = 0;
    while (i < iters) : (i += 1) {
        try testOne(context, rng.random());
    }
}

const TestOp = enum {
    // Unary
    inv,
    exp2,
    log2,

    // Binary
    add,
    sub,
    mul,
    div,
    pow,
};
fn Context(BF: type, comptime op: TestOp) type {
    return struct {
        const F = @FieldType(BF, "significand");
        const Int = @Int(.signed, @typeInfo(F).float.bits);
        const arg_count = switch (op) {
            .inv, .exp2, .log2 => 1,
            .add, .sub, .mul, .div, .pow => 2,
        };

        fn isEquivalent(a: F, b: F) bool {
            if (math.isNan(a)) {
                return math.isNan(b);
            } else {
                return a == b;
            }
        }

        fn isApproxEquivalent(a: F, b: F) bool {
            if (math.isNan(a)) {
                return math.isNan(b);
            } else {
                // @exp2 and @log2 have poor accuracy on f128 as
                // they convert them to f64s before evaluation
                if (F == f128) {
                    return math.approxEqRel(F, a, b, 1e-13);
                }
                if (op == .pow) {
                    // Our pow isn't very accurate :(
                    return math.approxEqRel(F, a, b, switch (F) {
                        f16, f32 => 1e-3,
                        f64, f80, f128 => 5e-5,
                        else => unreachable,
                    });
                }
                return math.approxEqRel(F, a, b, switch (F) {
                    f16 => 1e-3,
                    f32 => 1e-6,
                    f64 => 1e-11,
                    f80 => 1e-16,
                    f128 => 1e-27,
                    else => unreachable,
                });
            }
        }

        fn expect(args: [arg_count]F, expected: F, actual: BF) !void {
            const actual_f = actual.toFloat(F);
            const check = switch (op) {
                .inv, .add, .sub, .mul, .div => isEquivalent,
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
                .inv => 1.0 / args[0],
                .exp2 => @import("../exp2.zig").exp2(args[0], true),
                .log2 => @import("../log2.zig").log2(args[0]),
                .add => args[0] + args[1],
                .sub => args[0] - args[1],
                .mul => args[0] * args[1],
                .div => args[0] / args[1],
                .pow => math.pow(F, args[0], args[1]),
            };
        }

        fn applyBF(args: [arg_count]BF) BF {
            return switch (op) {
                .inv => args[0].inv(),
                .exp2 => args[0].exp2(),
                .log2 => args[0].log2(),
                .add => args[0].add(args[1]),
                .sub => args[0].sub(args[1]),
                .mul => args[0].mul(args[1]),
                .div => args[0].div(args[1]),
                .pow => args[0].pow(args[1]),
            };
        }

        fn containsSubnormal(fs: [arg_count]F, expected: F) bool {
            // Subnormal numbers can't always be represented exactly
            for (fs) |f| {
                if (math.isFinite(f) and !math.isNormal(f) and f != 0) return true;
                const f2: f64 = @floatCast(f);
                switch (op) {
                    .exp2, .log2 => if (math.isFinite(f2) and !math.isNormal(f2)) return true,
                    else => {},
                }
            }

            // f128 is cast to f64 for @exp2 and @log2
            const expected_rounded = switch (F) {
                f128 => @as(f64, @floatCast(expected)),
                else => expected,
            };
            return switch (op) {
                .exp2, .log2 => math.isFinite(expected) and !math.isNormal(expected_rounded),
                else => math.isFinite(expected) and !math.isNormal(expected) and expected != 0,
            };
        }

        fn testOne(_: @This(), rng: std.Random) !void {
            const fs, const expected = while (true) {
                const fs = randomFloats(rng);
                const expected = applyF(fs);
                if (containsSubnormal(fs, expected)) continue;
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

const FUZZ_ITERS = 69_420;

test "fuzz inv" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
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
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .mul);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz div" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
        utils.EmulatedFloat(f128),
    }) |BF| {
        const Ctx = Context(BF, .div);
        try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS);
    }
}

test "fuzz exp2" {
    inline for (.{
        utils.EmulatedFloat(f16),
        utils.EmulatedFloat(f32),
        utils.EmulatedFloat(f64),
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

fn ParseContext(BF: type) type {
    return struct {
        fn isApproxEquivalent(a: BF, b: BF) bool {
            if (a.isNan()) {
                return b.isNan();
            }
            return BF.approxEqRel(a, b, switch (@FieldType(BF, "significand")) {
                f16 => 1e-3,
                f32 => 1e-6,
                f64 => 1e-15,
                f80 => 1e-18,
                f128 => 1e-33,
                else => unreachable,
            });
        }

        fn expect(expected: BF, actual: BF) !void {
            // Our parser and formatter are not exact, so parsing roundtrips with a small error
            if (isApproxEquivalent(expected, actual)) return;

            std.debug.print("BigFloat type: {}\nexpected {e}, found {e}\n", .{
                @TypeOf(actual),
                expected,
                actual,
            });
            return error.UnexpectedTestResult;
        }

        /// Returns a random float evenly distributed in the range [1, 2) or (-2, -1].
        fn randomFloat(F: type, rng: std.Random) F {
            const C = @Int(.unsigned, @typeInfo(F).float.bits);

            // Mantissa
            var repr: C = rng.int(@Int(.unsigned, math.floatMantissaBits(F)));
            // Explicit bit is always 1
            if (math.floatMantissaBits(F) != math.floatFractionalBits(F)) {
                repr |= @as(C, 1) << math.floatFractionalBits(F);
            }
            // Exponent is always 0
            repr |= math.floatExponentMax(F) << math.floatMantissaBits(F);
            // Sign
            repr |= @as(C, rng.int(u1)) << (@typeInfo(F).float.bits - 1);

            return @bitCast(repr);
        }

        fn testOne(_: @This(), rng: std.Random) !void {
            const expected = BF{
                .significand = randomFloat(@FieldType(BF, "significand"), rng),
                .exponent = rng.int(@FieldType(BF, "exponent")),
            };
            var buf: [128]u8 = undefined;
            var w: std.Io.Writer = .fixed(&buf);
            try w.print("{e}", .{expected});
            const actual: BF = try .parse(w.buffered());
            try expect(expected, actual);
        }
    };
}

test "fuzz parse" {
    inline for (&.{ f32, f64, f80, f128 }) |s| {
        inline for (&.{ i23, i128 }) |e| {
            const F = BigFloat(.{
                .Significand = s,
                .Exponent = e,
                .bake_render = true,
            });
            const Ctx = ParseContext(F);
            try fuzz(Ctx{}, Ctx.testOne, FUZZ_ITERS / 4);
        }
    }
}
