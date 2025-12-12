const std = @import("std");
const testing = std.testing;

const BigFloat = @import("../root.zig").BigFloat;
const utils = @import("../test_utils.zig");

test "log2" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(1).log2()),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(2).log2()),
        );
        try testing.expectEqual(
            F.init(-1),
            try utils.expectCanonicalPassthrough(F.init(1.0 / 2.0).log2()),
        );
        try testing.expectEqual(
            F.init(20),
            try utils.expectCanonicalPassthrough(F.init(1024 * 1024).log2()),
        );
        try testing.expectEqual(
            F.init(-20),
            try utils.expectCanonicalPassthrough(F.init(1.0 / 1024.0 / 1024.0).log2()),
        );

        try utils.expectApproxEqRel(
            F.init(6.9477830262554195105713484746171828),
            try utils.expectCanonicalPassthrough(F.init(123.45).log2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(-3.0180012584066675330396098138509877),
            try utils.expectCanonicalPassthrough(F.init(0.12345).log2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(15171.544267666148357902627905870537),
            try utils.expectCanonicalPassthrough(F.init(1.23e4567).log2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(-15170.946951035019327544869763085553),
            try utils.expectCanonicalPassthrough(F.init(1.23e-4567).log2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(123456789.30392683648069481483070962),
            try utils.expectCanonicalPassthrough(
                (F{ .significand = 1.2345, .exponent = 123456789 }).log2(),
            ),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(-123456788.69607316351930518516929038),
            try utils.expectCanonicalPassthrough(
                (F{ .significand = 1.2345, .exponent = -123456789 }).log2(),
            ),
            utils.f64_error_tolerance,
        );

        // < 0 => nan
        try testing.expect(F.init(-1).log2().isNan());
        try testing.expect(F.minus_inf.log2().isNan());
        try testing.expect(F.min_value.neg().log2().isNan());

        // -0, 0 => -inf
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.init(-0.0).log2()),
        );
        try testing.expectEqual(
            F.minus_inf,
            try utils.expectCanonicalPassthrough(F.init(0.0).log2()),
        );

        // +inf => +inf
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.log2()),
        );

        // nan => nan
        try testing.expect(F.nan.log2().isNan());
    }

    // 65504 is the max finite value for f16
    const Small = BigFloat(.{
        .Significand = f16,
        .Exponent = i32,
    });
    const f16_error_tolerance = 9.765625e-4; // 2^-10
    try utils.expectApproxEqRel(
        Small.init(0.2992080183872788182197666346168540),
        try utils.expectCanonicalPassthrough(Small.init(1.23).log2()),
        f16_error_tolerance,
    );
    try testing.expectEqual(
        Small.init(123456),
        try utils.expectCanonicalPassthrough(
            (Small{ .significand = 1, .exponent = 123456 }).log2(),
        ),
    );
    try testing.expectEqual(
        Small.init(-123456),
        try utils.expectCanonicalPassthrough(
            (Small{ .significand = 1, .exponent = -123456 }).log2(),
        ),
    );
    try utils.expectApproxEqRel(
        Small.init(12345678.304006068589101766689691059),
        try utils.expectCanonicalPassthrough(
            (Small{ .significand = 1.2345678, .exponent = 12345678 }).log2(),
        ),
        f16_error_tolerance,
    );
    try utils.expectApproxEqRel(
        Small.init(2147483647.9992953870234106272584284),
        try utils.expectCanonicalPassthrough(Small.max_value.log2()),
        f16_error_tolerance,
    );
    try testing.expectEqual(
        Small.init(-2147483648),
        try utils.expectCanonicalPassthrough(Small.min_value.log2()),
    );

    // f64 goes up to around 2^1024 before hitting inf
    const Big = BigFloat(.{
        .Significand = f64,
        .Exponent = i1030,
    });
    try utils.expectApproxEqRel(
        Big.init(0.2986583155645151788790713924919448),
        try utils.expectCanonicalPassthrough(Big.init(1.23).log2()),
        utils.f64_error_tolerance,
    );
    try testing.expectEqual(
        Big.init(123456),
        try utils.expectCanonicalPassthrough(
            (Big{ .significand = 1, .exponent = 123456 }).log2(),
        ),
    );
    try testing.expectEqual(
        Big.init(-123456),
        try utils.expectCanonicalPassthrough(
            (Big{ .significand = 1, .exponent = -123456 }).log2(),
        ),
    );
    try utils.expectApproxEqRel(
        Big.init(12345678.304006068589101766689691059),
        try utils.expectCanonicalPassthrough(
            (Big{ .significand = 1.2345678, .exponent = 12345678 }).log2(),
        ),
        utils.f64_error_tolerance,
    );
    try utils.expectApproxEqRel(
        Big.init(5.7526180315594109047337766105248791e309),
        try utils.expectCanonicalPassthrough(Big.max_value.log2()),
        utils.f64_error_tolerance,
    );
    try utils.expectApproxEqRel(
        Big.init(-5.7526180315594109047337766105248791e309),
        try utils.expectCanonicalPassthrough(Big.min_value.log2()),
        utils.f64_error_tolerance,
    );
}
