const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

fn testSub(F: type, lhs: @FieldType(F, "significand"), rhs: @FieldType(F, "significand")) !void {
    try utils.expectBitwiseEqual(
        F.init(lhs - rhs),
        F.init(lhs).sub(F.init(rhs)),
    );
}

test "sub" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i11, i32 })) |F| {
        try testing.expectEqual(
            F.init(0),
            F.init(0).sub(F.init(0)),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1).sub(F.init(0)),
        );
        try testing.expectEqual(
            F.init(0),
            F.init(123).sub(F.init(123)),
        );
        try testSub(F, 123, 321);
        try testSub(F, 123, -123);
        try testSub(F, 1.5, 3.25);
        try testSub(F, 1e38, 1e-38);
        try testSub(F, 1e38, 0.99e38);
        try testSub(F, 0.9e308, 0.9e-308);
        if (@FieldType(F, "significand") == i11) {
            try testSub(F, 0.9e308, -0.9e308);
        }

        try testing.expectEqual(
            F.init(0),
            F.max_value.sub(F.max_value),
        );
        try testing.expectEqual(
            F.init(0),
            F.max_value.neg().sub(F.max_value.neg()),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.max_value.neg().sub(F.max_value),
        );
        try testing.expectEqual(
            F.inf,
            F.max_value.sub(F.max_value.neg()),
        );

        // Special cases
        // x - nan = nan
        try testing.expect(F.init(0).sub(F.nan).isNan());
        try testing.expect(F.init(-1.32e2).sub(F.nan).isNan());
        try testing.expect(F.inf.sub(F.nan).isNan());
        try testing.expect(F.inf.neg().sub(F.nan).isNan());

        // nan - y = nan
        try testing.expect(F.nan.sub(F.init(0)).isNan());
        try testing.expect(F.nan.sub(F.init(1)).isNan());
        try testing.expect(F.nan.sub(F.init(-0.123)).isNan());
        try testing.expect(F.nan.sub(F.nan).isNan());

        // +inf - +inf = nan
        // -inf - -inf = nan
        try testing.expect(F.inf.sub(F.inf).isNan());
        try testing.expect(F.inf.neg().sub(F.inf.neg()).isNan());

        // x - +-inf = -+inf for finite x
        // +-inf - y = +-inf for finite y
        try testing.expectEqual(
            F.inf.neg(),
            F.init(-12e32).sub(F.inf),
        );
        try testing.expectEqual(
            F.inf,
            F.init(12).sub(F.inf.neg()),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.sub(F.init(-12e32)),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.inf.neg().sub(F.init(12)),
        );
    }
}
