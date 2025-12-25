const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "div" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i23, i64 })) |F| {
        try testing.expectEqual(
            F.init(0),
            F.init(0).div(F.init(1)),
        );
        try utils.expectApproxEqRel(
            F.init(0.2857142857142857142857142857142857),
            F.init(1).div(F.init(3.5)),
            utils.f64_error_tolerance,
        );
        try testing.expectEqual(
            F.init(7),
            F.init(161).div(F.init(23)),
        );
        try testing.expectEqual(
            F.init(2.1666666666666666666666666666666667),
            F.init(3.25).div(F.init(1.5)),
        );
        try utils.expectApproxEqRel(
            F.init(-0.09967585089141004862236628849270665),
            F.init(123).div(F.init(-1234)),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(0.1839539007092198581560283687943262),
            F.init(-0.83).div(F.init(-4.512)),
            utils.f64_error_tolerance,
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1e38).div(F.init(1e38)),
        );
        try utils.expectApproxEqRel(
            F{ .significand = 1.6158503035655503650357438344334976, .exponent = -2047 },
            F.init(0.6e-308).div(F.init(0.6e308)),
            utils.f64_error_tolerance,
        );

        try testing.expectEqual(
            F.inf,
            F.max_value.div(F.min_value),
        );
        try testing.expectEqual(
            F.init(0),
            F.min_value.div(F.max_value),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.min_value.div(F.max_value.neg()),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.div(F.init(1)),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.inf.neg().div(F.init(1e30)),
        );

        // Special cases
        // x / nan = nan
        // nan / y = nan
        try testing.expect(F.init(1.23).div(F.nan).isNan());
        try testing.expect(F.nan.div(F.init(-0.123)).isNan());
        try testing.expect(F.inf.div(F.nan).isNan());
        try testing.expect(F.nan.div(F.inf.neg()).isNan());
        try testing.expect(F.nan.div(F.nan).isNan());

        // +-inf / +-inf = nan
        try testing.expect(F.inf.div(F.inf).isNan());
        try testing.expect(F.inf.div(F.inf.neg()).isNan());
        try testing.expect(F.inf.neg().div(F.inf).isNan());
        try testing.expect(F.inf.neg().div(F.inf.neg()).isNan());

        // 0 / 0 = nan
        try testing.expect(F.init(0).div(F.init(0)).isNan());
        try testing.expect(F.init(0).div(F.init(-0.0)).isNan());
        try testing.expect(F.init(-0.0).div(F.init(0)).isNan());
        try testing.expect(F.init(-0.0).div(F.init(-0.0)).isNan());

        //+-x / inf = +-0 for finite x
        try utils.expectBitwiseEqual(F.init(0), F.init(1.23).div(F.inf));
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(-123).div(F.inf));
        try utils.expectBitwiseEqual(F.init(0), F.init(0).div(F.inf));
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(-0.0).div(F.inf));

        //+-x / -inf = -+0 for finite x
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(1.23).div(F.inf.neg()));
        try utils.expectBitwiseEqual(F.init(0), F.init(-123).div(F.inf.neg()));
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(0).div(F.inf.neg()));
        try utils.expectBitwiseEqual(F.init(0), F.init(-0.0).div(F.inf.neg()));

        //+-x / 0 = +-inf  for x != 0
        try testing.expectEqual(F.inf, F.init(1.23).div(F.init(0)));
        try testing.expectEqual(F.inf.neg(), F.inf.neg().div(F.init(0)));
        try testing.expectEqual(F.inf.neg(), F.inf.div(F.init(-0.0)));
        try testing.expectEqual(F.inf, F.init(-123).div(F.init(-0.0)));
    }
}
