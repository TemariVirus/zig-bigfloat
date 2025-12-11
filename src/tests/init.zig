const std = @import("std");
const math = std.math;
const testing = std.testing;

const BigFloat = @import("../root.zig").BigFloat;
const utils = @import("../test_utils.zig");

test "init" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        const S = @FieldType(F, "significand");
        const E = @FieldType(F, "exponent");

        try testing.expectEqual(F{
            .significand = 1,
            .exponent = 0,
        }, F.init(1));
        try testing.expectEqual(F{
            .significand = -123.0 / 64.0,
            .exponent = 6,
        }, F.init(@as(i32, -123)));
        try testing.expectEqual(F{
            .significand = 0.0043 * 256.0,
            .exponent = -8,
        }, F.init(0.0043));
        try testing.expectEqual(F{
            .significand = 0,
            .exponent = 0,
        }, F.init(0));
        try testing.expectEqual(F{
            .significand = -0.0,
            .exponent = 0,
        }, F.init(-0.0));

        @setEvalBranchQuota(10_000);
        try testing.expectEqual(
            if (comptime utils.fitsInt(E, math.floatExponentMin(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S),
                }
            else
                F.init(0),
            F.init(math.floatMin(S)),
        );
        try testing.expectEqual(
            if (comptime utils.fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S) - math.floatFractionalBits(S),
                }
            else
                F.init(0),
            F.init(math.floatTrueMin(S)),
        );

        try testing.expectEqual(F{
            .significand = math.inf(S),
            .exponent = 0,
        }, F.init(math.inf(S)));
        try testing.expectEqual(F{
            .significand = -math.inf(S),
            .exponent = 0,
        }, F.init(-math.inf(S)));
        try testing.expect(math.isNan(
            F.init(math.nan(S)).significand,
        ));
    }

    const Small = BigFloat(.{
        .Significand = f16,
        .Exponent = i5,
        .bake_render = false,
    });
    try testing.expectEqual(Small{
        .significand = 1,
        .exponent = 0,
    }, Small.init(9.99999e-1));
    try testing.expectEqual(Small.inf, Small.init(65536));
    try testing.expectEqual(Small.max_value, Small.init(65504));
    try testing.expectEqual(Small{
        .significand = 1.9990234375,
        .exponent = 12,
    }, Small.init(8189));
    try testing.expectEqual(Small{
        .significand = 1,
        .exponent = 13,
    }, Small.init(8190));
    try testing.expectEqual(Small{
        .significand = 1,
        .exponent = 13,
    }, Small.init(8191));
}

test "initExact" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        const S = @FieldType(F, "significand");
        const E = @FieldType(F, "exponent");

        try testing.expectEqual(F{
            .significand = 1,
            .exponent = 0,
        }, F.initExact(1));
        try testing.expectEqual(F{
            .significand = -123.0 / 64.0,
            .exponent = 6,
        }, F.initExact(@as(i32, -123)));
        try testing.expectEqual(F{
            .significand = @as(f32, 0.0043) * 256.0,
            .exponent = -8,
        }, F.initExact(@as(f32, 0.0043)));
        try testing.expectEqual(F{
            .significand = 0,
            .exponent = 0,
        }, F.initExact(0));
        try testing.expectEqual(F{
            .significand = -0.0,
            .exponent = 0,
        }, F.initExact(-0.0));

        if (S != f128) {
            try testing.expectEqual(null, F.initExact(0.12));
            try testing.expectEqual(null, F.initExact(0.0043));
            try testing.expectEqual(null, F.initExact(@as(comptime_int, @intFromFloat(1.23e100))));
        }

        @setEvalBranchQuota(10_000);
        try testing.expectEqual(
            if (comptime utils.fitsInt(E, math.floatExponentMin(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S),
                }
            else
                null,
            F.initExact(math.floatMin(S)),
        );
        try testing.expectEqual(
            if (comptime utils.fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S) - math.floatFractionalBits(S),
                }
            else
                null,
            F.initExact(math.floatTrueMin(S)),
        );

        try testing.expectEqual(F{
            .significand = math.inf(S),
            .exponent = 0,
        }, F.initExact(math.inf(S)));
        try testing.expectEqual(F{
            .significand = -math.inf(S),
            .exponent = 0,
        }, F.initExact(-math.inf(S)));
        try testing.expect(math.isNan(
            F.initExact(math.nan(S)).?.significand,
        ));
    }

    const Small = BigFloat(.{
        .Significand = f16,
        .Exponent = i5,
        .bake_render = false,
    });
    try testing.expectEqual(null, Small.initExact(9.99999e-1));
    try testing.expectEqual(null, Small.initExact(1023.25));
    try testing.expectEqual(Small{
        .significand = 1.9990234375,
        .exponent = 9,
    }, Small.initExact(1023.5));
    try testing.expectEqual(null, Small.initExact(1023.75));

    try testing.expectEqual(null, Small.initExact(8186));
    try testing.expectEqual(null, Small.initExact(8187));
    try testing.expectEqual(Small{
        .significand = 1.9990234375,
        .exponent = 12,
    }, Small.initExact(8188));
    try testing.expectEqual(null, Small.initExact(8189));
    try testing.expectEqual(null, Small.initExact(8190));
    try testing.expectEqual(Small{
        .significand = 1,
        .exponent = 13,
    }, Small.initExact(8192));

    try testing.expectEqual(Small.max_value, Small.initExact(65504));
    try testing.expectEqual(null, Small.initExact(65536));
}
