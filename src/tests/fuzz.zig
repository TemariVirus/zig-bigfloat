const std = @import("std");
const testing = std.testing;

const BigFloat = @import("../root.zig").BigFloat;

// TODO: replace with std.testing.fuzz when it's ready
fn fuzz(
    context: anytype,
    comptime testOne: fn (context: @TypeOf(context), rng: std.Random) anyerror!void,
    iters: u64,
) anyerror!void {
    var rng: std.Random.ChaCha = .init(@import("options").test_seed);
    for (0..iters) |_| {
        try testOne(context, rng.random());
    }
}

const TestOp = enum {
    add,
    sub,
    mul,
    pow,
};
fn Context(BF: type, comptime op: TestOp) type {
    return struct {
        fn isEquivalent(T: type, a: T, b: T) bool {
            if (std.math.isNan(a)) {
                return std.math.isNan(b);
            } else {
                return a == b;
            }
        }

        fn expectEquivalent(lhs: anytype, rhs: anytype, expected: anytype, actual: anytype) !void {
            const F = @TypeOf(expected);
            const actual_f = actual.toFloat(F);
            if (isEquivalent(F, expected, actual_f)) return;

            std.debug.print("BigFloat type: {}\nexpected {e} {s} {e} = {e}, found {e}\n", .{
                @TypeOf(actual),
                lhs,
                switch (op) {
                    .add => "+",
                    .sub => "-",
                    .mul => "*",
                    .pow => "^",
                },
                rhs,
                expected,
                actual_f,
            });
            return error.TestExpectedEquivalent;
        }

        /// Returns a random float where each bit pattern is equally likely.
        fn randomFloat(comptime F: type, rng: std.Random) F {
            const float_len = @typeInfo(F).float.bits / 8;
            var buf: [float_len]u8 = undefined;
            rng.bytes(&buf);
            return std.mem.bytesToValue(F, &buf);
        }

        fn testBinaryOp(_: @This(), rng: std.Random) !void {
            const f1, const f2, const expected = while (true) {
                const F = @FieldType(BF, "significand");
                const epsilon = std.math.floatMin(F);

                const f1 = randomFloat(F, rng);
                const f2 = randomFloat(F, rng);
                const expected = switch (op) {
                    .add => f1 + f2,
                    .sub => f1 - f2,
                    .mul => f1 * f2,
                    .pow => std.math.pow(F, f1, f2),
                };
                // Denormal numbers can't always be represented exactly
                if ((f1 != 0 and @abs(f1) < epsilon) or
                    (f2 != 0 and @abs(f2) < epsilon) or
                    (expected != 0 and @abs(expected) < epsilon)) continue;
                break .{ f1, f2, expected };
            };

            const bf1 = BF.initExact(f1).?;
            const bf2 = BF.initExact(f2).?;
            const actual = switch (op) {
                .add => bf1.add(bf2),
                .sub => bf1.sub(bf2),
                .mul => bf1.mul(bf2),
                .pow => bf1.pow(bf2),
            };

            try expectEquivalent(f1, f2, expected, actual);
        }
    };
}

const FUZZ_ITERS = 420_069;

test "fuzz add" {
    if (!@import("options").run_slow_tests) return error.SkipZigTest;

    inline for (.{
        BigFloat(.{ .Significand = f16, .Exponent = i5 }),
        BigFloat(.{ .Significand = f32, .Exponent = i8 }),
        BigFloat(.{ .Significand = f64, .Exponent = i11 }),
        BigFloat(.{ .Significand = f80, .Exponent = i15 }),
        BigFloat(.{ .Significand = f128, .Exponent = i15 }),
    }) |BF| {
        const Ctx = Context(BF, .add);
        try fuzz(Ctx{}, Ctx.testBinaryOp, FUZZ_ITERS);
    }
}

test "fuzz sub" {
    if (!@import("options").run_slow_tests) return error.SkipZigTest;

    inline for (.{
        BigFloat(.{ .Significand = f16, .Exponent = i5 }),
        BigFloat(.{ .Significand = f32, .Exponent = i8 }),
        BigFloat(.{ .Significand = f64, .Exponent = i11 }),
        BigFloat(.{ .Significand = f80, .Exponent = i15 }),
        BigFloat(.{ .Significand = f128, .Exponent = i15 }),
    }) |BF| {
        const Ctx = Context(BF, .sub);
        try fuzz(Ctx{}, Ctx.testBinaryOp, FUZZ_ITERS);
    }
}

test "fuzz mul" {
    if (!@import("options").run_slow_tests) return error.SkipZigTest;

    inline for (.{
        BigFloat(.{ .Significand = f16, .Exponent = i5 }),
        BigFloat(.{ .Significand = f32, .Exponent = i8 }),
        BigFloat(.{ .Significand = f64, .Exponent = i11 }),
        BigFloat(.{ .Significand = f80, .Exponent = i15 }),
        BigFloat(.{ .Significand = f128, .Exponent = i15 }),
    }) |BF| {
        const Ctx = Context(BF, .mul);
        try fuzz(Ctx{}, Ctx.testBinaryOp, FUZZ_ITERS);
    }
}

test "fuzz pow" {
    if (!@import("options").run_slow_tests) return error.SkipZigTest;

    inline for (.{
        BigFloat(.{ .Significand = f32, .Exponent = i8 }),
        BigFloat(.{ .Significand = f64, .Exponent = i11 }),
    }) |BF| {
        const Ctx = Context(BF, .pow);
        try fuzz(Ctx{}, Ctx.testBinaryOp, FUZZ_ITERS);
    }
}
