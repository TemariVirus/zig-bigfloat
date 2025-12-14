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

test "inv" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i11, i16, i19, i32 })) |F| {
        try testing.expectEqual(
            F.init(0.5),
            F.init(2).inv(),
        );
        try testing.expectEqual(
            F.init(-0.5),
            F.init(-2).inv(),
        );
        try testing.expectEqual(
            F.init(4),
            F.init(0.25).inv(),
        );
        try testing.expectEqual(
            F.init(-4),
            F.init(-0.25).inv(),
        );
        try utils.expectApproxEqRel(
            F.init(4.6853308382384842767325064973825399e-57),
            F.init(2.134321e56).inv(),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(-3.1362728358050739153222272871181604e32),
            F.init(-3.188498107e-33).inv(),
            utils.f64_error_tolerance,
        );

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
