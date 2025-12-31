const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "eql" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.init(123).eql(F.init(123)));
        try testing.expect(!F.init(123).eql(F.init(122)));
        try testing.expect(F.init(0).eql(F.init(-0.0)));
        try testing.expect(!F.init(0).eql(F.init(123)));
        try testing.expect(F.inf.eql(F.inf));
        try testing.expect(!F.inf.eql(F.max_value));
        try testing.expect(!F.inf.eql(F.inf.neg()));
        try testing.expect(!F.nan.eql(F.nan));
    }
}

test "approxEqRel" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        // Exactly equal
        try testing.expect(F.init(123).approxEqRel(F.init(123), 1e-6));
        try testing.expect(!F.init(123).approxEqRel(F.init(122), 1e-6));
        try testing.expect(F.init(0).approxEqRel(F.init(-0.0), 1e-6));
        try testing.expect(F.inf.approxEqRel(.inf, 1e-6));
        try testing.expect(F.inf.approxEqRel(.max_value, 1e-6));
        try testing.expect(F.inf.approxEqRel(F.inf.neg(), 1e-6));
        try testing.expect(!F.nan.approxEqRel(.nan, 1e-6));

        // Almost equal
        try testing.expect(!F.init(1).approxEqRel(F.init(0), 1e-6));
        try testing.expect(F.init(1).approxEqRel(
            F.init(1 - (1.0 / 1024.0)),
            1.0 / 1024.0,
        ));
        try testing.expect(F.init(1).approxEqRel(
            F.init(1 - 1e2),
            1e2,
        ));
        try testing.expect(!F.init(1).approxEqRel(
            F.init(1 - 1e-5),
            1e-6,
        ));
    }
}

test "gt" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(!F.init(0).gt(F.init(0)));
        try testing.expect(!F.init(-0.0).gt(F.init(0)));
        try testing.expect(!F.init(0).gt(F.init(-0.0)));
        try testing.expect(!F.init(-0.0).gt(F.init(-0.0)));

        try testing.expect(F.init(123).gt(F.init(122)));
        try testing.expect(!F.init(123).gt(F.init(123)));
        try testing.expect(!F.init(123).gt(F.init(124)));
        try testing.expect(F.init(123).gt(F.init(12)));
        try testing.expect(!F.init(12).gt(F.init(123)));
        try testing.expect(!F.init(1.23e-4).gt(F.init(1)));

        try testing.expect(F.init(123).gt(F.init(-123)));
        try testing.expect(F.init(12).gt(F.init(-123)));
        try testing.expect(F.init(123).gt(F.init(-12)));
        try testing.expect(!F.init(-123).gt(F.init(123)));
        try testing.expect(!F.init(-12).gt(F.init(123)));
        try testing.expect(!F.init(-123).gt(F.init(12)));

        try testing.expect(!F.init(-123).gt(F.init(-122)));
        try testing.expect(!F.init(-123).gt(F.init(-123)));
        try testing.expect(F.init(-123).gt(F.init(-124)));
        try testing.expect(!F.init(-123).gt(F.init(-12)));
        try testing.expect(F.init(-12).gt(F.init(-123)));

        try testing.expect(!F.init(0).gt(F.max_value));
        try testing.expect(F.max_value.gt(F.init(0)));
        try testing.expect(!F.init(0).gt(F.min_value));
        try testing.expect(F.min_value.gt(F.init(0)));
        try testing.expect(F.inf.gt(F.max_value));
        try testing.expect(!F.max_value.gt(F.inf));
        try testing.expect(!F.nan.gt(F.init(2)));
        try testing.expect(!F.init(2).gt(F.nan));

        try testing.expect(!F.inf.gt(F.inf));
        try testing.expect(F.inf.gt(F.inf.neg()));
        try testing.expect(!F.inf.neg().gt(F.inf));
        try testing.expect(!F.inf.gt(F.nan));
        try testing.expect(!F.nan.gt(F.inf));
        try testing.expect(!F.nan.gt(F.nan));
    }
}

test "lt" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(!F.init(0).lt(F.init(0)));
        try testing.expect(!F.init(-0.0).lt(F.init(0)));
        try testing.expect(!F.init(0).lt(F.init(-0.0)));
        try testing.expect(!F.init(-0.0).lt(F.init(-0.0)));

        try testing.expect(!F.init(123).lt(F.init(122)));
        try testing.expect(!F.init(123).lt(F.init(123)));
        try testing.expect(F.init(123).lt(F.init(124)));
        try testing.expect(!F.init(123).lt(F.init(12)));
        try testing.expect(F.init(12).lt(F.init(123)));
        try testing.expect(!F.init(1).lt(F.init(1.23e-4)));

        try testing.expect(!F.init(123).lt(F.init(-123)));
        try testing.expect(!F.init(12).lt(F.init(-123)));
        try testing.expect(!F.init(123).lt(F.init(-12)));
        try testing.expect(F.init(-123).lt(F.init(123)));
        try testing.expect(F.init(-12).lt(F.init(123)));
        try testing.expect(F.init(-123).lt(F.init(12)));

        try testing.expect(F.init(-123).lt(F.init(-122)));
        try testing.expect(!F.init(-123).lt(F.init(-123)));
        try testing.expect(!F.init(-123).lt(F.init(-124)));
        try testing.expect(F.init(-123).lt(F.init(-12)));
        try testing.expect(!F.init(-12).lt(F.init(-123)));

        try testing.expect(F.init(0).lt(F.max_value));
        try testing.expect(!F.max_value.lt(F.init(0)));
        try testing.expect(F.init(0).lt(F.min_value));
        try testing.expect(!F.min_value.lt(F.init(0)));
        try testing.expect(!F.inf.lt(F.max_value));
        try testing.expect(F.max_value.lt(F.inf));
        try testing.expect(!F.nan.lt(F.init(2)));
        try testing.expect(!F.init(2).lt(F.nan));

        try testing.expect(!F.inf.lt(F.inf));
        try testing.expect(!F.inf.lt(F.inf.neg()));
        try testing.expect(F.inf.neg().lt(F.inf));
        try testing.expect(!F.inf.lt(F.nan));
        try testing.expect(!F.nan.lt(F.inf));
        try testing.expect(!F.nan.lt(F.nan));
    }
}

test "gte" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.init(0).gte(F.init(0)));
        try testing.expect(F.init(-0.0).gte(F.init(0)));
        try testing.expect(F.init(0).gte(F.init(-0.0)));
        try testing.expect(F.init(-0.0).gte(F.init(-0.0)));

        try testing.expect(F.init(123).gte(F.init(122)));
        try testing.expect(F.init(123).gte(F.init(123)));
        try testing.expect(!F.init(123).gte(F.init(124)));
        try testing.expect(F.init(123).gte(F.init(12)));
        try testing.expect(!F.init(12).gte(F.init(123)));
        try testing.expect(!F.init(1.23e-4).gte(F.init(1)));

        try testing.expect(F.init(123).gte(F.init(-123)));
        try testing.expect(F.init(12).gte(F.init(-123)));
        try testing.expect(F.init(123).gte(F.init(-12)));
        try testing.expect(!F.init(-123).gte(F.init(123)));
        try testing.expect(!F.init(-12).gte(F.init(123)));
        try testing.expect(!F.init(-123).gte(F.init(12)));

        try testing.expect(!F.init(-123).gte(F.init(-122)));
        try testing.expect(F.init(-123).gte(F.init(-123)));
        try testing.expect(F.init(-123).gte(F.init(-124)));
        try testing.expect(!F.init(-123).gte(F.init(-12)));
        try testing.expect(F.init(-12).gte(F.init(-123)));

        try testing.expect(!F.init(0).gte(F.max_value));
        try testing.expect(F.max_value.gte(F.init(0)));
        try testing.expect(!F.init(0).gte(F.min_value));
        try testing.expect(F.min_value.gte(F.init(0)));
        try testing.expect(F.inf.gte(F.max_value));
        try testing.expect(!F.max_value.gte(F.inf));
        try testing.expect(!F.nan.gte(F.init(2)));
        try testing.expect(!F.init(2).gte(F.nan));

        try testing.expect(F.inf.gte(F.inf));
        try testing.expect(F.inf.gte(F.inf.neg()));
        try testing.expect(!F.inf.neg().gte(F.inf));
        try testing.expect(!F.inf.gte(F.nan));
        try testing.expect(!F.nan.gte(F.inf));
        try testing.expect(!F.nan.gte(F.nan));
    }
}

test "lte" {
    inline for (utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.init(0).lte(F.init(0)));
        try testing.expect(F.init(-0.0).lte(F.init(0)));
        try testing.expect(F.init(0).lte(F.init(-0.0)));
        try testing.expect(F.init(-0.0).lte(F.init(-0.0)));

        try testing.expect(!F.init(123).lte(F.init(122)));
        try testing.expect(F.init(123).lte(F.init(123)));
        try testing.expect(F.init(123).lte(F.init(124)));
        try testing.expect(!F.init(123).lte(F.init(12)));
        try testing.expect(F.init(12).lte(F.init(123)));
        try testing.expect(!F.init(1).lte(F.init(1.23e-4)));

        try testing.expect(!F.init(123).lte(F.init(-123)));
        try testing.expect(!F.init(12).lte(F.init(-123)));
        try testing.expect(!F.init(123).lte(F.init(-12)));
        try testing.expect(F.init(-123).lte(F.init(123)));
        try testing.expect(F.init(-12).lte(F.init(123)));
        try testing.expect(F.init(-123).lte(F.init(12)));

        try testing.expect(F.init(-123).lte(F.init(-122)));
        try testing.expect(F.init(-123).lte(F.init(-123)));
        try testing.expect(!F.init(-123).lte(F.init(-124)));
        try testing.expect(F.init(-123).lte(F.init(-12)));
        try testing.expect(!F.init(-12).lte(F.init(-123)));

        try testing.expect(F.init(0).lte(F.max_value));
        try testing.expect(!F.max_value.lte(F.init(0)));
        try testing.expect(F.init(0).lte(F.min_value));
        try testing.expect(!F.min_value.lte(F.init(0)));
        try testing.expect(!F.inf.lte(F.max_value));
        try testing.expect(F.max_value.lte(F.inf));
        try testing.expect(!F.nan.lte(F.init(2)));
        try testing.expect(!F.init(2).lte(F.nan));

        try testing.expect(F.inf.lte(F.inf));
        try testing.expect(!F.inf.lte(F.inf.neg()));
        try testing.expect(F.inf.neg().lte(F.inf));
        try testing.expect(!F.inf.lte(F.nan));
        try testing.expect(!F.nan.lte(F.inf));
        try testing.expect(!F.nan.lte(F.nan));
    }
}
