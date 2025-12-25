const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "abs" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(
            F.init(123),
            F.init(123).abs(),
        );
        try testing.expectEqual(
            F.init(123),
            F.init(-123).abs(),
        );
        try testing.expectEqual(
            F.init(0),
            F.init(0).abs(),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.abs(),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.neg().abs(),
        );
        try testing.expect(F.nan.abs().isNan());
    }
}

test "neg" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(
            F.init(-123),
            F.init(123).neg(),
        );
        try testing.expectEqual(
            F.init(123),
            F.init(-123).neg(),
        );
        try testing.expectEqual(
            F.init(-0.0),
            F.init(0).neg(),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.inf.neg(),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.neg().neg(),
        );
        try testing.expect(F.nan.neg().isNan());
    }
}

fn testInv(F: type, x: @FieldType(F, "significand")) !void {
    const fr = std.math.frexp(x);
    const ans = F.normalize(.{
        .significand = 1 / fr.significand,
        .exponent = @intCast(-fr.exponent),
    });
    try utils.expectBitwiseEqual(ans, F.init(x).inv());
}

test "inv" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i11, i16, i19, i32 })) |F| {
        try testInv(F, 2);
        try testInv(F, -2);
        try testInv(F, 0.25);
        try testInv(F, -0.25);
        try testInv(F, 2.134321e56);
        try testInv(F, -3.188498107e-33);

        try testing.expectEqual(
            F{
                .significand = 2 / F.max_value.significand,
                .exponent = -1 - F.max_value.exponent,
            },
            F.max_value.inv(),
        );
        try testing.expectEqual(
            F{
                .significand = -2.0 / F.max_value.significand,
                .exponent = -1 - F.max_value.exponent,
            },
            F.max_value.neg().inv(),
        );
        try testing.expectEqual(
            F.inf,
            F.min_value.inv(),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.min_value.neg().inv(),
        );

        try testing.expectEqual(
            F.init(0),
            F.inf.inv(),
        );
        try testing.expectEqual(
            F.init(-0.0),
            F.inf.neg().inv(),
        );
        try testing.expectEqual(
            F.inf,
            F.init(0).inv(),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.init(-0.0).inv(),
        );
        try testing.expect(F.nan.inv().isNan());
    }
}
