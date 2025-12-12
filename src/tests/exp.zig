const std = @import("std");
const testing = std.testing;

const utils = @import("../test_utils.zig");

test "exp2" {
    inline for (utils.bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F.init(2),
            try utils.expectCanonicalPassthrough(F.init(1).exp2()),
        );
        try testing.expectEqual(
            F.init(1.0 / 2.0),
            try utils.expectCanonicalPassthrough(F.init(-1).exp2()),
        );
        try testing.expectEqual(
            F.init(1024),
            try utils.expectCanonicalPassthrough(F.init(10).exp2()),
        );
        try testing.expectEqual(
            F.init(1.0 / 1024.0),
            try utils.expectCanonicalPassthrough(F.init(-10).exp2()),
        );
        try utils.expectApproxEqRel(
            F.init(2.3456698984637576073197579763422596),
            try utils.expectCanonicalPassthrough(F.init(1.23).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(1.9830380770415906313713607977912150e-4),
            try utils.expectCanonicalPassthrough(F.init(-12.3).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            switch (@FieldType(F, "significand")) {
                f64 => F{ .significand = 1.3195078889668167666275307021103743, .exponent = 907374182 },
                f80 => F{ .significand = 1.3195079107941892571016437098436364, .exponent = 907374182 },
                f128 => F{ .significand = 1.3195079107728942593740019523158827, .exponent = 907374182 },
                else => unreachable,
            },
            try utils.expectCanonicalPassthrough(F.init(9.073741824e8).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            switch (@FieldType(F, "significand")) {
                f64 => F{ .significand = 1.7411010690456652445660984990257473, .exponent = -937374183 },
                f80 => F{ .significand = 1.7411011265781988192919530481853739, .exponent = -937374183 },
                f128 => F{ .significand = 1.7411011265922482782725399850457871, .exponent = -937374183 },
                else => unreachable,
            },
            try utils.expectCanonicalPassthrough(F.init(-9.373741822e8).exp2()),
            utils.f64_error_tolerance,
        );

        try utils.expectApproxEqRel(
            F.init(1.4142135623730950488016887242096981),
            try utils.expectCanonicalPassthrough(F.init(0.5).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(1.0892989912812542821268342891053001),
            try utils.expectCanonicalPassthrough(F.init(0.1234).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(1.0000000000000022250024495974269185),
            try utils.expectCanonicalPassthrough(F.init(3.21e-15).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(0.50034669373129031626878431965192960),
            try utils.expectCanonicalPassthrough(F.init(-0.999).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(0.87055056329612413913627001747974610),
            try utils.expectCanonicalPassthrough(F.init(-0.2).exp2()),
            utils.f64_error_tolerance,
        );
        try utils.expectApproxEqRel(
            F.init(0.99999999999999783738079665297297308),
            try utils.expectCanonicalPassthrough(F.init(-3.12e-15).exp2()),
            utils.f64_error_tolerance,
        );

        // Only valid when E is i64 or smaller
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.init(1e19).exp2()),
        );
        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.init(-1e19).exp2()),
        );

        try testing.expectEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.max_value.neg().exp2()),
        );
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.max_value.exp2()),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.min_value.exp2()),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.min_value.neg().exp2()),
        );

        // -0, 0 => 1
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(-0.0).exp2()),
        );
        try testing.expectEqual(
            F.init(1),
            try utils.expectCanonicalPassthrough(F.init(0).exp2()),
        );

        // +inf => +inf
        try testing.expectEqual(
            F.inf,
            try utils.expectCanonicalPassthrough(F.inf.exp2()),
        );

        // -inf => 0
        try utils.expectBitwiseEqual(
            F.init(0),
            try utils.expectCanonicalPassthrough(F.minus_inf.exp2()),
        );

        // nan => nan
        try testing.expect(F.nan.exp2().isNan());
    }
}
