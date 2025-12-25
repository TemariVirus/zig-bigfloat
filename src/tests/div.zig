const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

fn testDiv(F: type, lhs: @FieldType(F, "significand"), rhs: @FieldType(F, "significand")) !void {
    const _l = std.math.frexp(lhs);
    const _r = std.math.frexp(rhs);
    const ans = F.normalize(.{
        .significand = _l.significand / _r.significand,
        .exponent = @intCast(_l.exponent - _r.exponent),
    });
    try utils.expectBitwiseEqual(ans, F.init(lhs).div(F.init(rhs)));
}

test "div" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i23, i64 })) |F| {
        try testing.expectEqual(
            F.init(0),
            F.init(0).div(F.init(1)),
        );
        try testDiv(F, 1, 3.5);
        try testDiv(F, 161, 23);
        try testDiv(F, 3.25, 1.5);
        try testDiv(F, 123, -1234);
        try testDiv(F, -0.83, -4.512);
        try testing.expectEqual(
            F.init(1),
            F.init(1e38).div(F.init(1e38)),
        );
        try testDiv(F, 0.6e-308, 0.6e308);

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
