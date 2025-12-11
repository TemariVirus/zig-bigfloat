const std = @import("std");
const math = std.math;
const testing = std.testing;

const BigFloat = @import("../root.zig").BigFloat;

test "sign" {
    inline for (.{
        BigFloat(.{ .Significand = f32, .Exponent = i8 }),
        BigFloat(.{ .Significand = f32, .Exponent = i32 }),
        BigFloat(.{ .Significand = f64, .Exponent = i16 }),
        BigFloat(.{ .Significand = f128, .Exponent = i32 }),
    }) |F| {
        try testing.expectEqual(1, F.init(123).sign());
        try testing.expectEqual(0, F.init(0).sign());
        try testing.expectEqual(-1, F.init(-123).sign());
        try testing.expectEqual(0, F.init(math.nan(f32)).sign());
        try testing.expectEqual(1, F.init(math.inf(f32)).sign());
    }
}
