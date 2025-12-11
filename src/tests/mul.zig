const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "mul" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i23, i64 })) |F| {
        try utils.expectBitwiseEqual(F.init(0), F.init(0).mul(F.init(0)));
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(-0.0).mul(F.init(0)));
        try utils.expectBitwiseEqual(F.init(-0.0), F.init(0).mul(F.init(-0.0)));
        try utils.expectBitwiseEqual(F.init(0.0), F.init(-0.0).mul(F.init(-0.0)));

        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(1).mul(F.init(0))),
        );
        try testing.expectEqual(
            F.init(3.5),
            try utils.expectCanonicalPassthrough(F.init(1).mul(F.init(3.5))),
        );
        try testing.expectEqual(
            F.init(39483),
            try utils.expectCanonicalPassthrough(F.init(123).mul(F.init(321))),
        );
        try testing.expectEqual(
            F.init(4.875),
            try utils.expectCanonicalPassthrough(F.init(1.5).mul(F.init(3.25))),
        );
        try testing.expectEqual(
            F.init(-151782),
            try utils.expectCanonicalPassthrough(F.init(123).mul(F.init(-1234))),
        );
        try utils.expectApproxEqRel(
            F.init(3.74496),
            try utils.expectCanonicalPassthrough(F.init(-0.83).mul(F.init(-4.512))),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1e38).mul(F.init(1e-38))),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F{ .significand = 0.89117166164618254333829281056332, .exponent = 2045 },
            try utils.expectCanonicalPassthrough(F.init(0.6e308).mul(F.init(0.6e308))),
            utils.f64_error_tolerance,
        );

        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.inf.mul(F.minus_inf)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.mul(F.inf)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.mul(F.init(1))),
        );

        // Special cases
        // x * nan = nan
        // nan * y = nan
        try testing.expect(F.init(1.23).mul(F.nan).isNan());
        try testing.expect(F.nan.mul(F.init(-0.123)).isNan());
        try testing.expect(F.inf.mul(F.nan).isNan());
        try testing.expect(F.nan.mul(F.minus_inf).isNan());
        try testing.expect(F.nan.mul(F.nan).isNan());

        // +-inf * +-0 = nan
        // +-0 * +-inf = nan
        try testing.expect(F.inf.mul(F.init(0)).isNan());
        try testing.expect(F.inf.mul(F.init(-0.0)).isNan());
        try testing.expect(F.minus_inf.mul(F.init(0)).isNan());
        try testing.expect(F.minus_inf.mul(F.init(-0.0)).isNan());
    }
}
