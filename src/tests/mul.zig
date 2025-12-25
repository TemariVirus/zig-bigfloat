const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

fn testMul(F: type, lhs: @FieldType(F, "significand"), rhs: @FieldType(F, "significand")) !void {
    const _l = std.math.frexp(lhs);
    const _r = std.math.frexp(rhs);
    const ans = F.normalize(.{
        .significand = _l.significand * _r.significand,
        .exponent = @intCast(_l.exponent + _r.exponent),
    });
    try utils.expectBitwiseEqual(ans, F.init(lhs).mul(F.init(rhs)));
}

test "mul" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i23, i64 })) |F| {
        try utils.expectBitwiseEqual(F.init(0), F.init(0).mul(F.init(0)));
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(-0.0).mul(F.init(0)));
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(0).mul(F.init(-0.0)));
        try utils.expectBitwiseEqual(F.init(0.0), F.init(-0.0).mul(F.init(-0.0)));

        try testing.expectEqual(
            F.init(0),
            F.init(1).mul(F.init(0)),
        );
        try testing.expectEqual(
            F.init(3.5),
            F.init(1).mul(F.init(3.5)),
        );
        try testMul(F, 123, 321);
        try testMul(F, 1.5, 3.25);
        try testMul(F, 123, -1234);
        try testMul(F, -0.83, -4.512);
        try testMul(F, 1e38, 1e-38);
        try testMul(F, 0.6e308, 0.6e308);

        try testing.expectEqual(
            F.init(0),
            F.min_value.mul(F.min_value),
        );
        try utils.expectBitwiseEqual(
            F.init(-0.0),
            F.min_value.mul(F.min_value.neg()),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.inf.mul(F.inf.neg()),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.mul(F.inf),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.mul(F.init(1)),
        );

        // Special cases
        // x * nan = nan
        // nan * y = nan
        try testing.expect(F.init(1.23).mul(F.nan).isNan());
        try testing.expect(F.nan.mul(F.init(-0.123)).isNan());
        try testing.expect(F.inf.mul(F.nan).isNan());
        try testing.expect(F.nan.mul(F.inf.neg()).isNan());
        try testing.expect(F.nan.mul(F.nan).isNan());

        // +-inf * +-0 = nan
        // +-0 * +-inf = nan
        try testing.expect(F.inf.mul(F.init(0)).isNan());
        try testing.expect(F.inf.mul(F.init(-0.0)).isNan());
        try testing.expect(F.inf.neg().mul(F.init(0)).isNan());
        try testing.expect(F.inf.neg().mul(F.init(-0.0)).isNan());
    }
}
