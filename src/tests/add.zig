const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

fn testAdd(F: type, lhs: @FieldType(F, "significand"), rhs: @FieldType(F, "significand")) !void {
    try utils.expectBitwiseEqual(
        F.init(lhs + rhs),
        F.init(lhs).add(F.init(rhs)),
    );
}

test "add" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i11, i32 })) |F| {
        try testing.expectEqual(
            F.init(0),
            F.init(0).add(F.init(0)),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1).add(F.init(0)),
        );
        try testing.expectEqual(
            F.init(0),
            F.init(123).add(F.init(-123)),
        );
        try testAdd(F, 123, 321);
        try testAdd(F, 1.5, 3.25);
        try testAdd(F, 1e38, 1e-38);
        try testAdd(F, 1e38, -0.99e38);
        try testAdd(F, 0.9e308, 0.9e-308);
        if (@FieldType(F, "significand") == i11) {
            try testAdd(F, 0.9e308, 0.9e308);
        }

        try testing.expectEqual(
            F.inf,
            F.max_value.add(F.max_value),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.max_value.neg().add(F.max_value.neg()),
        );
        try testing.expectEqual(
            F.init(0),
            F.max_value.neg().add(F.max_value),
        );
        try testing.expectEqual(
            F.init(0),
            F.max_value.add(F.max_value.neg()),
        );

        // Special cases
        // x + nan = nan
        try testing.expect(F.init(0).add(F.nan).isNan());
        try testing.expect(F.init(-1.32e2).add(F.nan).isNan());
        try testing.expect(F.inf.add(F.nan).isNan());
        try testing.expect(F.inf.neg().add(F.nan).isNan());

        // nan + y = nan
        try testing.expect(F.nan.add(F.init(0)).isNan());
        try testing.expect(F.nan.add(F.init(1)).isNan());
        try testing.expect(F.nan.add(F.init(-0.123)).isNan());
        try testing.expect(F.nan.add(F.nan).isNan());

        // +inf + -inf = nan
        // -inf + +inf = nan
        try testing.expect(F.inf.add(F.inf.neg()).isNan());
        try testing.expect(F.inf.neg().add(F.inf).isNan());

        // x + +-inf = +-inf for finite x
        // +-inf + y = +-inf for finite y
        try testing.expectEqual(
            F.inf,
            F.init(-12e32).add(F.inf),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.init(12).add(F.inf.neg()),
        );
        try testing.expectEqual(
            F.inf,
            F.init(-12e32).add(F.inf),
        );
        try testing.expectEqual(
            F.inf.neg(),
            F.init(12).add(F.inf.neg()),
        );
    }
}
