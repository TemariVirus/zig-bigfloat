const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

/// Represents a floating-point number as `0.abs(mantissa) * 2^exponent`.
/// `0.abs(mantissa)` is in the interval `[0.5, 1)`.
/// The sign of `mantissa` is the sign of the number.
///
/// Special cases:
///  - `0     => mantissa = 0,         exponent = 0`
///  - `+-inf => mantissa = +-1,       exponent = 0`
///  - `nan   => mantissa = minInt(M), exponent = 0`
pub fn BigFloat(M: type, E: type) type {
    switch (@typeInfo(M)) {
        .int => |info| {
            assert(info.signedness == .signed);
            assert(info.bits >= 3);
        },
        else => @compileError("mantissa must be a signed int"),
    }
    switch (@typeInfo(E)) {
        .int => |info| assert(info.signedness == .signed),
        else => @compileError("exponent must be a signed int"),
    }

    // Using a packed struct increases performance by 44% to 170%;
    return packed struct {
        mantissa: M,
        exponent: E,

        const Self = @This();
        const sign_mask = @as(M, 1) << (mant_bits - 1);
        const mant_bits = @typeInfo(M).int.bits;
        const MantMask = std.meta.Int(.unsigned, mant_bits);
        /// Mantissa value that represents 0.5.
        const mantissa_half = @as(M, 1) << (mant_bits - 2);
        const max_exponent = math.maxInt(E);
        const min_exponent = math.minInt(E);

        // zig fmt: off
        pub const zero: Self =     .{ .mantissa = 0,               .exponent = 0 };
        pub const inf: Self =      .{ .mantissa = 1,               .exponent = 0 };
        pub const minusInf: Self = .{ .mantissa = -1,              .exponent = 0 };
        pub const nan: Self =      .{ .mantissa = math.minInt(M),  .exponent = 0 };
        /// Largest value smaller than `inf`.
        pub const maxValue: Self = .{ .mantissa = math.maxInt(M),  .exponent = math.maxInt(E) };
        /// Smallest value larger than `minusInf`.
        pub const minValue: Self = .{ .mantissa = -math.maxInt(M), .exponent = math.maxInt(E) };
        /// Smallest value larger than `zero`.
        pub const epsilon: Self =  .{ .mantissa = mantissa_half,   .exponent = math.minInt(E) };
        // zig fmt: on

        pub fn from(x: anytype) Self {
            const T = @TypeOf(x);
            switch (@typeInfo(T)) {
                .int, .comptime_int => {
                    if (x == 0) return zero;

                    // Zig ints go up to 65,536 bits, so using i32 is always safe
                    const exponent: i32 = @intCast(1 + math.log2(@abs(x)));
                    const x_bits: u16 = switch (T) {
                        comptime_int => @intCast(exponent),
                        else => @typeInfo(T).int.bits,
                    };
                    if (exponent > max_exponent) return if (x > 0) inf else minusInf;

                    const shift = exponent - @as(i32, mant_bits) + 1;
                    const mantissa: M = @truncate(math.shr(
                        if (mant_bits > x_bits) M else std.meta.Int(.signed, x_bits),
                        x,
                        shift,
                    ));
                    return .{
                        .mantissa = mantissa,
                        .exponent = @intCast(exponent),
                    };
                },
                .float, .comptime_float => {
                    const fr = math.frexp(switch (T) {
                        // comptime_float internally is a f128; this preserves precision.
                        comptime_float => @as(f128, x),
                        else => x,
                    });
                    const F = @TypeOf(fr.significand);
                    if (math.isNan(fr.significand)) return nan;
                    if (math.isInf(fr.significand) or fr.exponent > max_exponent) {
                        return if (fr.significand > 0) inf else minusInf;
                    }
                    if (fr.significand == 0 or fr.exponent < min_exponent) return zero;

                    const x_bits: comptime_int = @typeInfo(F).float.bits;
                    const Int: type = std.meta.Int(.unsigned, x_bits);
                    const MantInt: type = std.meta.Int(.unsigned, math.floatFractionalBits(F));

                    const v: Int = @bitCast(fr.significand);
                    const positive: bool = (v >> (x_bits - 1)) == 0;
                    const m: MantInt = @truncate(v);
                    const shift = math.floatFractionalBits(F) - @as(i32, mant_bits) + 2;
                    const mantissa: MantMask = @truncate(math.shr(
                        if (mant_bits > math.floatFractionalBits(F)) MantMask else MantInt,
                        m,
                        shift,
                    ) | (@as(MantMask, 1) << (mant_bits - 2)));

                    return .{
                        .mantissa = if (positive) @bitCast(mantissa) else -@as(M, @bitCast(mantissa)),
                        .exponent = @intCast(fr.exponent),
                    };
                },
                else => @compileError("x must be an int or float"),
            }
        }

        fn mantMask(self: Self) MantMask {
            return @bitCast(self.mantissa);
        }

        pub fn sign(self: Self) M {
            return math.sign(self.mantissa);
        }

        pub fn isInf(self: Self) bool {
            return @abs(self.mantissa) == 1;
        }

        pub fn isNan(self: Self) bool {
            if (self.mantissa == nan.mantissa) return true;

            const top_2_bits = self.mantMask() >> (mant_bits - 2);
            return (top_2_bits == 0b00 or top_2_bits == 0b11) and
                @abs(self.mantissa) > 1 and
                self.mantissa != -mantissa_half;
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            if (lhs.isNan() or rhs.isNan()) return nan;
            if (lhs.isInf()) {
                if (!rhs.isInf()) return lhs;
                const same_sign = (lhs.mantissa & sign_mask) == (rhs.mantissa & sign_mask);
                return if (same_sign) lhs else nan;
            }
            if (rhs.isInf()) return rhs;
            return if (lhs.exponent < rhs.exponent)
                @call(.always_inline, add2, .{ rhs, lhs })
            else
                @call(.always_inline, add2, .{ lhs, rhs });
        }

        fn add2(lhs: Self, rhs: Self) Self {
            assert(lhs.exponent >= rhs.exponent);
            assert(!lhs.isNan() and !rhs.isNan());
            assert(!lhs.isInf() and !rhs.isInf());

            const exp_diff = lhs.exponent - rhs.exponent;
            // The exponent difference is too large, we can just return lhs
            if (exp_diff >= mant_bits - 1) return lhs;

            const normalized_rhs = rhs.mantissa >> @intCast(exp_diff);
            const m, const overflow = @addWithOverflow(lhs.mantissa, normalized_rhs);
            if (overflow == 1) {
                if (lhs.exponent == math.maxInt(E)) {
                    return if (lhs.mantissa > 0) inf else minusInf;
                }
                const sign_bit = ~(m >> 1) & sign_mask;
                const unsigned = (m >> 1) & ~sign_mask;
                return .{
                    .mantissa = sign_bit | unsigned,
                    .exponent = lhs.exponent + 1,
                };
            }
            if (@abs(m) >= mantissa_half) {
                return .{
                    .mantissa = m,
                    .exponent = lhs.exponent,
                };
            }
            if (m == 0) return zero;

            const exp_offset = @clz(@abs(m)) - 1;
            assert(exp_offset > 0);
            const ExpInt = std.meta.Int(.signed, @max(@typeInfo(E).int.bits, @typeInfo(@TypeOf(exp_offset)).int.bits + 1));
            const new_exponent, const overflow2 = @subWithOverflow(@as(ExpInt, lhs.exponent), @as(ExpInt, exp_offset));
            return if (overflow2 == 1 or new_exponent < min_exponent)
                zero
            else
                .{
                    .mantissa = m << @intCast(exp_offset),
                    .exponent = @intCast(new_exponent),
                };
        }
    };
}

const testing = std.testing;

fn bigFloatTypes(ms: []const type, es: []const type) [ms.len * es.len]type {
    var types: [ms.len * es.len]type = undefined;
    for (ms, 0..) |s, i| {
        for (es, 0..) |e, j| {
            types[i * es.len + j] = BigFloat(s, e);
        }
    }
    return types;
}

test "from" {
    inline for (bigFloatTypes(&.{ i32, i64 }, &.{ i8, i16, i19, i32 })) |F| {
        const m_bits = @typeInfo(@FieldType(F, "mantissa")).int.bits;
        try testing.expectEqualDeep(F{
            .mantissa = 1 << (m_bits - 2),
            .exponent = 1,
        }, F.from(1));
        try testing.expectEqual(F{
            .mantissa = -(123 << (m_bits - 2 - math.log2(123))),
            .exponent = math.log2(123) + 1,
        }, F.from(@as(i32, -123)));
        try testing.expectEqual(F{
            .mantissa = 0x4673_81D7_DBF4_87FC >> @max(0, 64 - m_bits),
            .exponent = -7,
        }, F.from(0.0043));
        try testing.expectEqual(F{
            // 0.0043 = 0x1.19ce075f6fd22p-8
            //          0x1.19ce075f6fd22p-8
            // (0x19ce075f6fd22 << 10) | 0x4000_0000_0000_0000 = 0x4673_81D7_DBF4_8800
            .mantissa = 0x4673_81D7_DBF4_8800 >> @max(0, 64 - m_bits),
            .exponent = -7,
        }, F.from(@as(f64, 0.0043)));
        try testing.expectEqual(F{
            .mantissa = 0,
            .exponent = 0,
        }, F.from(0));
        try testing.expectEqual(F{
            .mantissa = 0,
            .exponent = 0,
        }, F.from(0.0));
        try testing.expectEqual(F.inf, F.from(math.inf(f64)));
        try testing.expectEqual(F.nan, F.from(math.nan(f64)));
    }
}

test "add" {
    inline for (bigFloatTypes(&.{ i64, i80 }, &.{i11})) |F| {
        try testing.expectEqualDeep(F.from(0), F.from(0).add(.from(0)));
        try testing.expectEqualDeep(F.from(1), F.from(1).add(.from(0)));
        try testing.expectEqualDeep(F.from(444), F.from(123).add(.from(321)));
        try testing.expectEqualDeep(F.from(0), F.from(123).add(.from(-123)));
        try testing.expectEqualDeep(F.from(4.75), F.from(1.5).add(.from(3.25)));
        try testing.expectEqualDeep(F.from(1e38), F.from(1e38).add(.from(1e-38)));
        {
            const expected = F.from(1e36);
            const actual = F.from(1e38).add(.from(-0.99e38));
            try testing.expectEqual(expected.exponent, actual.exponent);
            try testing.expect(math.approxEqRel(
                f128,
                @floatFromInt(expected.mantissa),
                @floatFromInt(actual.mantissa),
                2.220446049250313e-14, // 10 ^ (-log_10(2^52) + 2)
            ));
        }

        try testing.expect(!F.from(0.6e308).isInf());
        try testing.expectEqualDeep(F.inf, F.from(0.6e308).add(.from(0.6e308)));
        try testing.expectEqualDeep(F.minusInf, F.from(12).add(F.minusInf));
        try testing.expect(F.inf.add(F.minusInf).isNan());
        try testing.expect(F.nan.add(.from(2)).isNan());
    }
}
