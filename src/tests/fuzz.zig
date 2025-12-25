const std = @import("std");
const math = std.math;
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

// Remove after this bugfix is merged
// https://codeberg.org/ziglang/zig/pulls/30176
/// Returns x raised to the power of y (x^y).
pub fn pow(comptime T: type, x: T, y: T) T {
    if (@typeInfo(T) == .int) {
        return math.powi(T, x, y) catch unreachable;
    }

    if (T != f32 and T != f64) {
        @compileError("pow not implemented for " ++ @typeName(T));
    }

    // pow(x, +-0) = 1      for any x not a signaling nan
    if (y == 0) {
        if (math.isSignalNan(x)) {
            @branchHint(.unlikely);
            return math.nan(T);
        }
        return 1;
    }

    // pow(1, y) = 1        for any y
    if (x == 1) {
        return 1;
    }

    // pow(nan, y) = nan    for any y != 0
    // pow(x, nan) = nan    for any x != 1
    if (math.isNan(x) or math.isNan(y)) {
        @branchHint(.unlikely);
        return math.nan(T);
    }

    // pow(x, 1) = x        for any x
    if (y == 1) {
        return x;
    }

    if (x == 0) {
        if (y < 0) {
            // pow(+-0, y) = +-inf  for y an odd integer
            if (isOddInteger(y)) {
                return math.copysign(math.inf(T), x);
            }
            // pow(+-0, y) = +inf   for y an even integer
            else {
                return math.inf(T);
            }
        } else {
            if (isOddInteger(y)) {
                return x;
            } else {
                return 0;
            }
        }
    }

    if (math.isInf(y)) {
        // pow(-1, inf) = 1
        if (x == -1) {
            return 1.0;
        }
        // pow(x, +inf) = +0    for |x| < 1
        // pow(x, -inf) = +0    for |x| > 1
        else if ((@abs(x) < 1) == math.isPositiveInf(y)) {
            return 0;
        }
        // pow(x, -inf) = +inf  for |x| < 1
        // pow(x, +inf) = +inf  for |x| > 1
        else {
            return math.inf(T);
        }
    }

    if (math.isInf(x)) {
        if (math.isNegativeInf(x)) {
            return pow(T, 1 / x, -y);
        }
        // pow(+inf, y) = +0    for y < 0
        else if (y < 0) {
            return 0;
        }
        // pow(+inf, y) = +0    for y > 0
        else if (y > 0) {
            return math.inf(T);
        }
    }

    // special case sqrt
    if (y == 0.5) {
        return @sqrt(x);
    }

    if (y == -0.5) {
        return 1 / @sqrt(x);
    }

    const r1 = math.modf(@abs(y));
    var yi = r1.ipart;
    var yf = r1.fpart;

    if (yf != 0 and x < 0) {
        return math.nan(T);
    }
    if (yi >= 1 << (@typeInfo(T).float.bits - 1)) {
        // yi is a large even int, so the result is always positive
        // and the sign of x doesn't matter
        return @exp(y * @log(@abs(x)));
    }

    // a = a1 * 2^ae
    var a1: T = 1.0;
    var ae: i32 = 0;

    // a *= x^yf
    if (yf != 0) {
        if (yf > 0.5) {
            yf -= 1;
            yi += 1;
        }
        a1 = @exp(yf * @log(x));
    }

    // a *= x^yi
    const r2 = math.frexp(x);
    var xe = r2.exponent;
    var x1 = r2.significand;

    var i = @as(std.meta.Int(.signed, @typeInfo(T).float.bits), @intFromFloat(yi));
    while (i != 0) : (i >>= 1) {
        const overflow_shift = math.floatExponentBits(T) + 1;
        if (xe < -(1 << overflow_shift) or (1 << overflow_shift) < xe) {
            // catch xe before it overflows the left shift below
            // Since i != 0 it has at least one bit still set, so ae will accumulate xe
            // on at least one more iteration, ae += xe is a lower bound on ae
            // the lower bound on ae exceeds the size of a float exp
            // so the final call to Ldexp will produce under/overflow (0/Inf)
            ae += xe;
            break;
        }
        if (i & 1 == 1) {
            a1 *= x1;
            ae += xe;
        }
        x1 *= x1;
        xe <<= 1;
        if (x1 < 0.5) {
            x1 += x1;
            xe -= 1;
        }
    }

    // a *= a1 * 2^ae
    if (y < 0) {
        a1 = 1 / a1;
        ae = -ae;
    }

    return math.scalbn(a1, ae);
}
fn isOddInteger(x: f64) bool {
    if (@abs(x) >= 1 << 53) {
        // From https://golang.org/src/math/pow.go
        // 1 << 53 is the largest exact integer in the float64 format.
        // Any number outside this range will be truncated before the decimal point and therefore will always be
        // an even integer.
        // Without this check and if x overflows i64 the @intFromFloat(r.ipart) conversion below will panic
        return false;
    }
    const r = math.modf(x);
    return r.fpart == 0.0 and @as(i64, @intFromFloat(r.ipart)) & 1 == 1;
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
        const Int = std.meta.Int(.signed, @typeInfo(F).float.bits);
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
                    return math.approxEqRel(F, a, b, 5e-5);
                }
                return math.approxEqRel(F, a, b, switch (F) {
                    f16 => 1e-3,
                    f32 => 1e-6,
                    f64 => 1e-13,
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
                .exp2 => @exp2(args[0]),
                .log2 => @log2(args[0]),
                .add => args[0] + args[1],
                .sub => args[0] - args[1],
                .mul => args[0] * args[1],
                .div => args[0] / args[1],
                .pow => pow(F, args[0], args[1]),
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
                // TODO: f128 division rounds to 0 when the result should be subnormal
                // https://codeberg.org/ziglang/zig/issues/30179
                .div => math.isFinite(expected) and !math.isNormal(expected_rounded),
                .exp2, .log2 => math.isFinite(expected) and !math.isNormal(expected_rounded),
                else => math.isFinite(expected) and !math.isNormal(expected) and expected != 0,
            };
        }

        fn testOne(_: @This(), rng: std.Random) !void {
            const fs, const expected = while (true) {
                const fs = randomFloats(rng);
                const expected = applyF(fs);
                if (containsSubnormal(fs, expected)) continue;
                if (F == f128 and op == .inv) {
                    // TODO: f128 division rounds to 0 when the result should be subnormal
                    // https://codeberg.org/ziglang/zig/issues/30179
                    if (@abs(fs[0]) > 1.0 / math.floatMin(f128)) continue;
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
