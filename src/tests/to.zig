const std = @import("std");
const math = std.math;
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "toFloat" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i16, i19, i32 })) |F| {
        const S = @FieldType(F, "significand");

        try testing.expectEqual(@as(S, 0), F.init(0).toFloat(S));
        try testing.expect(math.isNegativeZero(F.init(-0.0).toFloat(S)));
        try testing.expectEqual(@as(S, 1), F.init(1).toFloat(S));
        try testing.expectEqual(@as(S, -521.122), F.init(-521.122).toFloat(S));
        try testing.expectEqual(@as(S, 1e23), F.init(1e23).toFloat(S));
        try testing.expectEqual(@as(S, 1e-23), F.init(1e-23).toFloat(S));
        try testing.expectEqual(@as(S, -1e-45), F.init(-1e-45).toFloat(S));

        try testing.expectEqual(math.inf(S), F.inf.toFloat(S));
        try testing.expectEqual(-math.inf(S), F.minus_inf.toFloat(S));
        try testing.expect(math.isNan(F.nan.toFloat(S)));

        try testing.expectEqual(math.inf(S), F.max_value.toFloat(S));
        try testing.expectEqual(-math.inf(S), F.min_value.toFloat(S));
        try testing.expectEqual(@as(S, 0), F.epsilon.toFloat(S));
    }
}
