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
        const is_even = @typeInfo(@FieldType(F, "exponent")).int.bits > math.floatMantissaBits(@FieldType(F, "significand"));
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.init(100).pow(.init(max_exp))),
        );
        try testing.expectEqual(
            if (is_even) F.inf else F.minus_inf,
            try utils.expectCanonicalPassthrough(F.init(-100).pow(.init(max_exp))),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(100).pow(.init(-max_exp))),
        );
        try utils.expectBitwiseEqual(
            if (is_even) F.init(0.0) else F.init(-0.0),
            try utils.expectCanonicalPassthrough(F.init(-100).pow(.init(-max_exp))),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.min_value.pow(.init(2))),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            try utils.expectCanonicalPassthrough(F.min_value.pow(.init(-212389))),
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
            try utils.expectCanonicalPassthrough(F.epsilon.pow(.init(2))),
        );
        try testing.expectEqual(
            F.epsilon,
            try utils.expectCanonicalPassthrough(F.epsilon.pow(.init(1))),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.epsilon.pow(.init(-1))),
        );

        // Special cases
        // nan^y = nan
        try testing.expect(F.nan.pow(.init(1.23)).isNan());
        try testing.expect(F.nan.pow(.init(0)).isNan());
        try testing.expect(F.nan.pow(.init(-1)).isNan());

        // x^nan = nan
        try testing.expect(F.init(1.23).pow(.nan).isNan());
        try testing.expect(F.init(0).pow(.nan).isNan());
        try testing.expect(F.init(-0.0).pow(.nan).isNan());
        try testing.expect(F.init(-1).pow(.nan).isNan());

        // x^0 = 1
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1.23).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-1.23).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1.23e123).pow(.init(-0.0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(0).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(0).pow(.init(-0.0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-0.0).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-0.0).pow(.init(-0.0))),
        );

        // 1^y = 1
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).pow(.init(1))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).pow(.init(-0.0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).pow(.init(100_000_000))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).pow(.init(-12.3876))),
        );

        // -1^+-inf = 1
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-1).pow(.inf)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-1).pow(.minus_inf)),
        );

        // x^1 = x
        try testing.expectEqual(
            F.init(-1.2),
            try utils.expectCanonicalPassthrough(F.init(-1.2).pow(.init(1))),
        );
        try testing.expectEqual(
            F.init(1.233e-12),
            try utils.expectCanonicalPassthrough(F.init(1.233e-12).pow(.init(1))),
        );
        try testing.expectEqual(
            F.max_value,
            try utils.expectCanonicalPassthrough(F.max_value.pow(.init(1))),
        );
        try testing.expectEqual(
            F.epsilon,
            try utils.expectCanonicalPassthrough(F.epsilon.pow(.init(1))),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.pow(.init(1))),
        );
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.pow(.init(1))),
        );

        // +-0^+inf = +0
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.inf),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.inf),
        );

        // +-0^-inf = +inf
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0).pow(.minus_inf),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-0.0).pow(.minus_inf),
        );

        // -0^y = nan for finite non-integer y
        try testing.expect(F.init(-0.0).pow(.init(1.5)).isNan());
        try testing.expect(F.init(-0.0).pow(.init(-313.23)).isNan());
        try testing.expect(F.init(-0.0).pow(.init(0.0123)).isNan());

        // x^y = nan for x < 0 and finite non-integer y
        try testing.expect(F.init(-1).pow(.init(1.5)).isNan());
        try testing.expect(F.init(-4.654e12).pow(.init(-313.23)).isNan());
        try testing.expect(F.init(-1.2).pow(.init(0.0123)).isNan());

        // +0^y = +0 when y > 0, +inf when y < 0
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(1.875)),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0).pow(.init(-1)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(187432)),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0).pow(.init(-1493874.321)),
        );

        // -0^y = +0^y when y is an even integer
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.init(2)),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-0.0).pow(.init(-2)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.init(187432)),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-0.0).pow(.init(-1493874)),
        );

        // -0^y = -(+0^y) when y is an odd integer
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).pow(.init(1)),
        );
        try utils.expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).pow(.init(-1)),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).pow(.init(187431)),
        );
        try utils.expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).pow(.init(-1493873)),
        );

        // x^+inf = +inf when |x| > 1
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(1.2).pow(.inf),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-1.00001).pow(.inf),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(1e30).pow(.inf),
        );

        // x^+inf = +0 when |x| < 1
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0.8).pow(.inf),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.99999).pow(.inf),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(1e-30).pow(.inf),
        );

        // x^-inf = +0 when |x| > 1
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(1.2).pow(.minus_inf),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-1.00001).pow(.minus_inf),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(1e30).pow(.minus_inf),
        );

        // x^-inf = +inf when |x| < 1
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0.8).pow(.minus_inf),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-0.99999).pow(.minus_inf),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(1e-30).pow(.minus_inf),
        );

        // +inf^y = +inf when y > 0, +0 when y < 0
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.pow(.init(1.321))),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.pow(.init(18937210))),
        );
        try utils.expectBitwiseEqual(
            F.init(0.0),
            F.inf.pow(.init(-1)),
        );
        try utils.expectBitwiseEqual(
            F.init(0.0),
            F.inf.pow(.init(-1421987.413)),
        );

        // -inf^y = +inf^y when y is an even integer
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.pow(.init(2))),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.pow(.init(12309874))),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.minus_inf.pow(.init(-2)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.minus_inf.pow(.init(-123098)),
        );

        // -inf^y = -(+inf^y) when y is an odd integer
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.pow(.init(1))),
        );
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.pow(.init(123099))),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.pow(.init(-1)),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.pow(.init(-1230987)),
        );
    }
}

test "powi" {
    const large_power_tolerance = 1e-7;
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
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
            try utils.expectCanonicalPassthrough(F.init(1).powi(0)),
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
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1.23).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-1.23).powi(0)),
        );

        const max_exp = math.maxInt(@FieldType(F, "exponent"));
        const min_exp = math.minInt(@FieldType(F, "exponent"));
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.init(100).powi(max_exp)),
        );
        try testing.expectEqual(
            F.minus_inf,
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
            try utils.expectCanonicalPassthrough(F.min_value.powi(2)),
        );
        try testing.expectEqual(
            F.min_value.inv(),
            try utils.expectCanonicalPassthrough(F.min_value.powi(-1)),
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
            F.init(0),
            try utils.expectCanonicalPassthrough(F.epsilon.powi(2)),
        );
        try testing.expectEqual(
            F.epsilon,
            try utils.expectCanonicalPassthrough(F.epsilon.powi(1)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.epsilon.powi(-1)),
        );

        // Special cases
        // nan^y = nan
        try testing.expect(F.nan.powi(123).isNan());
        try testing.expect(F.nan.powi(0).isNan());

        // x^0 = 1
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

        // 1^y = 1
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).powi(1)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).powi(-123876)),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).powi(3981)),
        );

        // x^1 = x
        try testing.expectEqual(
            F.init(-1.2),
            try utils.expectCanonicalPassthrough(F.init(-1.2).powi(1)),
        );
        try testing.expectEqual(
            F.init(1.233e-12),
            try utils.expectCanonicalPassthrough(F.init(1.233e-12).powi(1)),
        );
        try testing.expectEqual(
            F.max_value,
            try utils.expectCanonicalPassthrough(F.max_value.powi(1)),
        );
        try testing.expectEqual(
            F.epsilon,
            try utils.expectCanonicalPassthrough(F.epsilon.powi(1)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.powi(1)),
        );
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.powi(1)),
        );

        // +0^y = +0 when y > 0, +inf when y < 0
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).powi(1),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0).powi(-1),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(0).powi(187432),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(0).powi(-1493874),
        );

        // -0^y = +0^y when y is even
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).powi(2),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-0.0).powi(-2),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).powi(187432),
        );
        try utils.expectBitwiseEqual(
            F.inf,
            F.init(-0.0).powi(-1493874),
        );

        // -0^y = -(+0^y) when y is odd
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).powi(1),
        );
        try utils.expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).powi(-1),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).powi(187431),
        );
        try utils.expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).powi(-1493873),
        );

        // +inf^y = +inf when y > 0, +0 when y < 0
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.powi(1)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.powi(18937210)),
        );
        try utils.expectBitwiseEqual(
            F.init(0.0),
            F.inf.powi(-1),
        );
        try utils.expectBitwiseEqual(
            F.init(0.0),
            F.inf.powi(-1421987),
        );

        // -inf^y = +inf^y when y is even
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.powi(2)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.powi(12309874)),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.minus_inf.powi(-2),
        );
        try utils.expectBitwiseEqual(
            F.init(0),
            F.minus_inf.powi(-123098),
        );

        // -inf^y = -(+inf^y) when y is odd
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.powi(1)),
        );
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.powi(123099)),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.powi(-1),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.powi(-1230987),
        );
    }
}
