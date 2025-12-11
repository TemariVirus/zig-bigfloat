const std = @import("std");

const BigFloat = @import("root.zig").BigFloat;

/// The default relative error tolerance for operations on f64 significands.
pub const f64_error_tolerance = 2.220446049250313e-14; // 10 ^ (-log_10(2^52) + 2)

/// Returns whether `value` fits in the range of `Int`.
pub fn fitsInt(Int: type, value: anytype) bool {
    return value >= std.math.minInt(Int) and value <= std.math.maxInt(Int);
}

/// Returns the cartesian product of `BigFloat`s using the significand types
/// in `ss` and exponent types in `es`.
pub fn bigFloatTypes(ss: []const type, es: []const type) [ss.len * es.len]type {
    var types: [ss.len * es.len]type = undefined;
    for (ss, 0..) |s, i| {
        for (es, 0..) |e, j| {
            types[i * es.len + j] = BigFloat(.{
                .Significand = s,
                .Exponent = e,
            });
        }
    }
    return types;
}

/// Returns a `BigFloat` that has the same largest finite value as `F`.
/// The smallest positive value of the returned type is 1/4 the smallest
/// positive normal value of `F`.
pub fn EmulatedFloat(F: type) type {
    return BigFloat(.{
        .Significand = F,
        .Exponent = std.meta.Int(.signed, std.math.floatExponentBits(F)),
    });
}

/// Tests if a `BigFloat` is in canonical form, and returns it if it is.
pub fn expectCanonicalPassthrough(actual: anytype) !@TypeOf(actual) {
    try std.testing.expect(actual.isCanonical());
    return actual;
}

/// Tests if 2 `BigFloat`s are approximately equal within a relative tolerance.
pub fn expectApproxEqRel(expected: anytype, actual: anytype, tolerance: comptime_float) !void {
    if (expected.approxEqRel(actual, tolerance)) {
        return;
    }

    // Add initial space to prevent newline from being stripped
    const fmt = std.fmt.comptimePrint(" \n" ++
        \\  expected {{e:.{0d}}}
        \\     found {{e:.{0d}}}
        \\  expected error ±{{e:.{0d}}}%
        \\    actual error ±{{e:.{0d}}}%
        \\
    , .{@TypeOf(expected).Decimal.maxDigitCount - 1});
    const abs_diff = expected.sub(actual).abs();
    const scale = expected.max(actual).powi(-1);
    std.debug.print(fmt, .{
        expected,
        actual,
        tolerance * 100,
        abs_diff.mul(scale).mul(.init(100)),
    });
    return error.TestExpectedApproxEqual;
}

/// Tests if the bit representation of 2 `BigFloat`s are equal.
pub fn expectBitwiseEqual(expected: anytype, actual: anytype) !void {
    const S = @FieldType(@TypeOf(expected), "significand");
    const Bits = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
    const expected_bits: Bits = @bitCast(expected.significand);
    const actual_bits: Bits = @bitCast(actual.significand);

    if (expected_bits != actual_bits) {
        std.debug.print(
            "expected {e}, found {e}\n",
            .{ expected.significand, actual.significand },
        );
        return error.TestExpectedEqual;
    }
    try std.testing.expectEqual(expected.exponent, actual.exponent);
}
