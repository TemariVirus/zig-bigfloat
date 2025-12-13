const std = @import("std");
const math = std.math;
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "pow" {
    const large_power_tolerance = 1e-7;
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F{ .significand = 1, .exponent = 100_000_000 },
            try utils.expectCanonicalPassthrough(F.init(2).pow(.init(100_000_000))),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = -100_000_000 },
            try utils.expectCanonicalPassthrough(F.init(2).pow(.init(-100_000_000))),
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.1099157202316952388898221715929617, .exponent = 56 },
            try utils.expectCanonicalPassthrough(F.init(23.4).pow(.init(12.345))),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.8019386188912608618888586275231166, .exponent = -57 },
            try utils.expectCanonicalPassthrough(F.init(23.4).pow(.init(-12.345))),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.5745848124494259913414545428268009, .exponent = 561535380 },
            try utils.expectCanonicalPassthrough(F.init(23.4).pow(.init(123456789))),
            large_power_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.2701761024157203962625228005634512, .exponent = -561535381 },
            try utils.expectCanonicalPassthrough(F.init(23.4).pow(.init(-123456789))),
            large_power_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.0124222013774393757001601268900093, .exponent = 0 },
            try utils.expectCanonicalPassthrough(F.init(1.000_000_000_1).pow(.init(123456789.01234))),
            large_power_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.9754604326919372321867791758592745, .exponent = -1 },
            try utils.expectCanonicalPassthrough(F.init(1.000_000_000_1).pow(.init(-123456789.01234))),
            large_power_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(1.0 / 1.23),
            try utils.expectCanonicalPassthrough(F.init(1.23).pow(.init(-1))),
            utils.f64_error_tolerance,
        );

        const max_exp = math.maxInt(@FieldType(F, "exponent"));
        const min_exp = math.minInt(@FieldType(F, "exponent"));
        const is_even = @typeInfo(@FieldType(F, "exponent")).int.bits - 1 >
            1 + math.floatFractionalBits(@FieldType(F, "significand"));
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.init(100).pow(.init(max_exp))),
        );
        try testing.expectEqual(
            if (is_even) F.inf else F.inf.neg(),
            try utils.expectCanonicalPassthrough(F.init(-100).pow(.init(max_exp))),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(100).pow(.init(-max_exp))),
        );
        try utils.expectBitwiseEqual(
            if (is_even) F.init(0) else F.init(-0.0),
            try utils.expectCanonicalPassthrough(F.init(-100).pow(.init(-max_exp))),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.max_value.neg().pow(.init(2))),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            try utils.expectCanonicalPassthrough(F.max_value.neg().pow(.init(-212389))),
        );
        if (math.floatFractionalBits(@FieldType(F, "significand")) + 1 >= @typeInfo(@FieldType(F, "exponent")).int.bits) {
            try testing.expectEqual(
                F{ .significand = 1, .exponent = max_exp },
                try utils.expectCanonicalPassthrough(F.init(2).pow(.init(max_exp))),
            );
            try testing.expectEqual(
                F{ .significand = 1, .exponent = min_exp + 1 },
                try utils.expectCanonicalPassthrough(F.init(2).pow(.init(min_exp + 1))),
            );
        }
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.min_value.pow(.init(2))),
        );
        try testing.expectEqual(
            F.min_value,
            try utils.expectCanonicalPassthrough(F.min_value.pow(.init(1))),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.min_value.pow(.init(-1))),
        );
    }
}

test "pow special" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        // pow(x, +-0)    = 1
        try testing.expectEqual(
            F.init(1),
            F.init(4).pow(.init(0)),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(-7).pow(.init(-0.0)),
        );
        try testing.expectEqual(
            F.init(1),
            F.nan.pow(.init(-0.0)),
        );
        // pow(+-0, y)    = +-inf  for y an odd integer < 0
        try testing.expectEqual(
            F.inf,
            F.init(0).pow(.init(-1)),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.init(-0.0).pow(.init(-5)),
        );
        // pow(+-0, -inf) = +inf
        try testing.expectEqual(
            F.inf,
            F.init(0).pow(F.inf.neg()),
        );
        try testing.expectEqual(
            F.inf,
            F.init(-0.0).pow(F.inf.neg()),
        );
        // pow(+-0, +inf) = +0
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(F.inf),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(F.inf),
        );
        // pow(+-0, y)    = +-0    for finite y > 0 an odd integer
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(3)),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).pow(.init(5)),
        );
        // pow(-1, +-inf) = 1
        try testing.expectEqual(
            F.init(1),
            F.init(-1).pow(F.inf),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(-1).pow(F.inf.neg()),
        );
        // pow(+1, y)     = 1
        try testing.expectEqual(
            F.init(1),
            F.init(1).pow(.init(4)),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1).pow(.init(-7)),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1).pow(F.inf.neg()),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1).pow(F.nan),
        );
        // pow(x, +inf)   = +0     for −1 < x < 1
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0.2).pow(F.inf),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.2).pow(F.inf),
        );
        // pow(x, +inf)   = +inf   for x < −1 or for 1 < x (including +-inf)
        try testing.expectEqual(
            F.inf,
            F.init(1.2).pow(F.inf),
        );
        try testing.expectEqual(
            F.inf,
            F.init(-1.2).pow(F.inf),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.pow(F.inf),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.neg().pow(F.inf),
        );
        // pow(x, −inf)   = +inf   for −1 < x < 1
        try testing.expectEqual(
            F.inf,
            F.init(0.2).pow(F.inf.neg()),
        );
        try testing.expectEqual(
            F.inf,
            F.init(-0.2).pow(F.inf.neg()),
        );
        // pow(x, −inf)   = +0     for x < −1 or for 1 < x (including +-inf)
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(1.2).pow(F.inf.neg()),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-1.2).pow(F.inf.neg()),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.inf.pow(F.inf.neg()),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.inf.neg().pow(F.inf.neg()),
        );
        // pow(+inf, y)   = +0     for a number y < 0
        try utils.expectBitwiseEqual(
            F.init(0),
            F.inf.pow(.init(-2)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.inf.pow(.init(-0.2)),
        );
        // pow(+inf, y)   = +inf   for a number y > 0
        try testing.expectEqual(
            F.inf,
            F.inf.pow(.init(2)),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.pow(.init(0.2)),
        );
        // pow(−inf, y)   = −0     for finite y < 0 an odd integer
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.inf.neg().pow(.init(-3)),
        );
        // pow(−inf, y)   = −inf   for finite y > 0 an odd integer
        try testing.expectEqual(
            F.inf.neg(),
            F.inf.neg().pow(.init(5)),
        );
        // pow(−inf, y)   = +0     for finite y < 0 and not an odd integer
        try utils.expectBitwiseEqual(
            F.init(0),
            F.inf.neg().pow(.init(-2)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.inf.neg().pow(.init(-5.2)),
        );
        // pow(−inf, y)   = +inf   for finite y > 0 and not an odd integer
        try testing.expectEqual(
            F.inf,
            F.inf.neg().pow(.init(4)),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.neg().pow(.init(0.5)),
        );
        // pow(+-0, y)    = +inf   for finite y < 0 and not an odd integer
        try testing.expectEqual(
            F.inf,
            F.init(0).pow(.init(-2)),
        );
        try testing.expectEqual(
            F.inf,
            F.init(-0.0).pow(.init(-2)),
        );
        try testing.expectEqual(
            F.inf,
            F.init(0).pow(.init(-5.2)),
        );
        try testing.expectEqual(
            F.inf,
            F.init(-0.0).pow(.init(-0.5)),
        );
        // pow(+-0, y)    = +0     for finite y > 0 and not an odd integer
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(2)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.init(2.0)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(5.2)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.init(0.5)),
        );
        // pow(x, y)      = nan    for finite x < 0 and finite non-integer y
        try testing.expect(
            F.init(-1).pow(.init(1.2)).isNan(),
        );
        try testing.expect(
            F.init(-12.4).pow(.init(-78.5)).isNan(),
        );
        // pow(x, 1)      = x
        try testing.expectEqual(
            F.init(45),
            F.init(45).pow(.init(1)),
        );
        try testing.expectEqual(
            F.init(-45),
            F.init(-45).pow(.init(1)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(1)),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).pow(.init(1)),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.pow(.init(1)),
        );
        try testing.expect(F.nan.pow(.init(1)).isNan());
        // pow(nan, y)    = nan    for y != +-0
        try testing.expect(F.nan.pow(.init(5)).isNan());
        // pow(x, nan)    = nan    for x != 1
        try testing.expect(F.init(5).pow(F.nan).isNan());
    }
}

test "powi" {
    const large_power_tolerance = 1e-7;
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        // Normal
        try testing.expectEqual(
            F{ .significand = 1, .exponent = 100_000_000 },
            try utils.expectCanonicalPassthrough(F.init(2).powi(100_000_000)),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = -100_000_000 },
            try utils.expectCanonicalPassthrough(F.init(2).powi(-100_000_000)),
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.4961341053179219085744446147145936, .exponent = 54 },
            try utils.expectCanonicalPassthrough(F.init(23.4).powi(12)),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.3367785634263105096278138105491006, .exponent = -55 },
            try utils.expectCanonicalPassthrough(F.init(23.4).powi(-12)),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.5745848124494259913414545428268009, .exponent = 561535380 },
            try utils.expectCanonicalPassthrough(F.init(23.4).powi(123456789)),
            large_power_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.2701761024157203962625228005634512, .exponent = -561535381 },
            try utils.expectCanonicalPassthrough(F.init(23.4).powi(-123456789)),
            large_power_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.0124222013761900467037236039862069, .exponent = 0 },
            try utils.expectCanonicalPassthrough(F.init(1.000_000_000_1).powi(123456789)),
            large_power_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.9754604326943749503606006445672171, .exponent = -1 },
            try utils.expectCanonicalPassthrough(F.init(1.000_000_000_1).powi(-123456789)),
            large_power_tolerance,
        );

        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).powi(1)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).powi(100_000_000)),
        );
        try testing.expectEqual(
            F.init(1.23),
            try utils.expectCanonicalPassthrough(F.init(1.23).powi(1)),
        );
        try testing.expectEqual(
            F.init(1.0 / 1.23),
            try utils.expectCanonicalPassthrough(F.init(1.23).powi(-1)),
        );

        const min_exp = math.minInt(@FieldType(F, "exponent"));
        const max_exp = math.maxInt(@FieldType(F, "exponent"));
        try testing.expectEqual(
            F.max_value.neg().inv(),
            try utils.expectCanonicalPassthrough(F.max_value.neg().powi(-1)),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = max_exp },
            try utils.expectCanonicalPassthrough(F.init(2).powi(max_exp)),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = min_exp },
            try utils.expectCanonicalPassthrough(F.init(2).powi(min_exp)),
        );
        try testing.expectEqual(
            F.min_value,
            try utils.expectCanonicalPassthrough(F.min_value.powi(1)),
        );

        // Overflow/underflow
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.init(100).powi(max_exp)),
        );
        try testing.expectEqual(
            F.inf.neg(),
            try utils.expectCanonicalPassthrough(F.init(-100).powi(max_exp)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(100).powi(-max_exp)),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            try utils.expectCanonicalPassthrough(F.init(-100).powi(-max_exp)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.max_value.neg().powi(2)),
        );
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.min_value.powi(2)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.min_value.powi(-1)),
        );
    }
}

test "powi special" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        // powi(x, 0)    = 1
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-1.2).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(0).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.inf.powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.nan.powi(0)),
        );
        // powi(+-0, n)  = +-inf  for odd n < 0
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0).powi(-1),
        );
        try utils.expectBitwiseEqual(
            F.inf.neg(),
            F.init(-0.0).powi(-123),
        );
        // powi(+-0, n)  = +inf   for even n < 0
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0).powi(-2),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-0.0).powi(-67890),
        );
        // powi(+-0, n)  = +0     for even n > 0
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).powi(2),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).powi(67890),
        );
        // powi(+-0, n)  = +-0    for odd n > 0
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).powi(3),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).powi(67891),
        );
        // powi(+inf, n) = +inf   for n > 0
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.powi(3)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.powi(123456)),
        );
        // powi(−inf, n) = −inf   for odd n > 0
        try testing.expectEqual(
            F.inf.neg(),
            try utils.expectCanonicalPassthrough(F.inf.neg().powi(5)),
        );
        // powi(−inf, n) = +inf   for even n > 0
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.neg().powi(2)),
        );
        // powi(+inf, n) = +0     for n < 0
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.inf.powi(-1)),
        );
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.inf.powi(-1234)),
        );
        // powi(−inf, n) = −0     for odd n < 0
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            try utils.expectCanonicalPassthrough(F.inf.neg().powi(-123)),
        );
        // powi(−inf, n) = +0     for even n < 0
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.inf.neg().powi(-4)),
        );
        // powi(nan, n)  = nan    for n != 0
        try testing.expect(F.nan.powi(1).isNan());
        try testing.expect(F.nan.powi(-123).isNan());
    }
}
