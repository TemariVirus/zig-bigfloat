const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "isNan" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.nan.isNan());
        try testing.expect(!F.inf.isNan());
        try testing.expect(!F.minus_inf.isNan());
        try testing.expect(!F.init(0).isNan());
        try testing.expect(!F.init(123).isNan());
    }
}
