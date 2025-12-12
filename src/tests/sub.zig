const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "sub" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{i11})) |F| {
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(0).sub(F.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(1).sub(F.init(0))),
        );
        try testing.expectEqual(
            F.init(-198),
            try utils.expectCanonicalPassthrough(F.init(123).sub(F.init(321))),
        );
        try testing.expectEqual(
            F.init(246),
            try utils.expectCanonicalPassthrough(F.init(123).sub(F.init(-123))),
        );
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(123).sub(F.init(123))),
        );
        try testing.expectEqual(
            F.init(-1.75),
            try utils.expectCanonicalPassthrough(F.init(1.5).sub(F.init(3.25))),
        );
        try testing.expectEqual(
            F.init(1e38),
            try utils.expectCanonicalPassthrough(F.init(1e38).sub(F.init(1e-38))),
        );
        try utils.expectApproxEqRel(
            F.init(1e36),
            try utils.expectCanonicalPassthrough(F.init(1e38).sub(F.init(0.99e38))),
            utils.f64_error_tolerance,
        );

        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.max_value.sub(F.max_value)),
        );
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.max_value.neg().sub(F.max_value.neg())),
        );
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.max_value.neg().sub(F.max_value)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.max_value.sub(F.max_value.neg())),
        );

        // Only valid when exponent is i11
        try testing.expect(!F.init(0.9e308).isInf());
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.init(0.9e308).sub(F.init(-0.9e308))),
        );
        try testing.expectEqual(
            F.init(0.9e308),
            try utils.expectCanonicalPassthrough(F.init(0.9e308).sub(F.init(0.9e-308))),
        );

        // Special cases
        // x - nan = nan
        try testing.expect(F.init(0).sub(F.nan).isNan());
        try testing.expect(F.init(-1.32e2).sub(F.nan).isNan());
        try testing.expect(F.inf.sub(F.nan).isNan());
        try testing.expect(F.minus_inf.sub(F.nan).isNan());

        // nan - y = nan
        try testing.expect(F.nan.sub(F.init(0)).isNan());
        try testing.expect(F.nan.sub(F.init(1)).isNan());
        try testing.expect(F.nan.sub(F.init(-0.123)).isNan());
        try testing.expect(F.nan.sub(F.nan).isNan());

        // +inf - +inf = nan
        // -inf - -inf = nan
        try testing.expect(F.inf.sub(F.inf).isNan());
        try testing.expect(F.minus_inf.sub(F.minus_inf).isNan());

        // x - +-inf = -+inf for finite x
        // +-inf - y = +-inf for finite y
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.init(-12e32).sub(F.inf)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.init(12).sub(F.minus_inf)),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.sub(F.init(-12e32))),
        );
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.minus_inf.sub(F.init(12))),
        );
    }
}
