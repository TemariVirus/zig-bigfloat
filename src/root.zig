const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const testing = std.testing;

const exp2_128 = @import("exp2_128.zig").exp2_128;

/// Represents a floating-point number as `significand * 2^exponent`.
/// `abs(significand)` is in the interval `[0.5, 1)`.
///
/// Special cases:
///  - `+-0   => significand = +-0,   exponent = 0`
///  - `+-inf => significand = +-inf, exponent = 0`
///  - `nan   => significand = nan,   exponent = 0`
pub fn BigFloat(S: type, E: type) type {
    assert(@typeInfo(S) == .float);
    switch (@typeInfo(E)) {
        .int => |info| assert(info.signedness == .signed),
        else => @compileError("exponent must be a signed int"),
    }

    // Using a packed struct increases performance by 45% to 140%;
    return packed struct {
        /// The significand, normalized to the range `[0.5, 1)`.
        /// `normalize` must be called after modifying this field directly.
        significand: S,
        /// The base-2 exponent. `normalize` must be called after modifying this field directly.
        exponent: E,

        const Self = @This();
        const max_exponent = math.maxInt(E);
        const min_exponent = math.minInt(E);

        // zig fmt: off
        pub const zero: Self =      .{ .significand = 0,                        .exponent = 0 };
        pub const minus_zero: Self = .{ .significand = -0.0,                     .exponent = 0 };
        pub const inf: Self =       .{ .significand = math.inf(S),              .exponent = 0 };
        pub const minus_inf: Self =  .{ .significand = -math.inf(S),             .exponent = 0 };
        pub const nan: Self =       .{ .significand = math.nan(S),              .exponent = 0 };
        /// Largest value smaller than `inf`.
        pub const max_value: Self =  .{ .significand = math.nextAfter(S, 1, 0),  .exponent = math.maxInt(E) };
        /// Smallest value larger than `minus_inf`.
        pub const min_value: Self =  .{ .significand = math.nextAfter(S, -1, 0), .exponent = math.maxInt(E) };
        /// Smallest value larger than `zero`.
        pub const epsilon: Self =   .{ .significand = 0.5,                      .exponent = math.minInt(E) };
        // zig fmt: on

        pub fn from(x: anytype) Self {
            const T = @TypeOf(x);
            switch (@typeInfo(T)) {
                .int, .comptime_int => {
                    if (x == 0) return zero;

                    // Zig ints go up to 65,535 bits, so using i32 is always safe
                    const exponent: i32 = @intCast(1 + math.log2(@abs(x)));
                    if (exponent > max_exponent) return if (x > 0) inf else minus_inf;

                    // Bit shift to ensure x fits in the range of S
                    const shift = @max(0, exponent - math.floatFractionalBits(S) - 1);
                    const significand: S = @floatFromInt(x >> @intCast(shift));
                    return .{
                        .significand = math.ldexp(significand, shift - exponent),
                        .exponent = @intCast(exponent),
                    };
                },
                .float, .comptime_float => {
                    const fr = math.frexp(switch (T) {
                        // comptime_float internally is a f128; this preserves precision.
                        comptime_float => @as(f128, x),
                        else => x,
                    });
                    if (math.isNan(fr.significand)) return nan;
                    if (fr.exponent < min_exponent) return if (fr.significand > 0) zero else minus_zero;
                    if (fr.exponent > max_exponent) return if (fr.significand > 0) inf else minus_inf;
                    return .{
                        .significand = math.lossyCast(S, fr.significand),
                        .exponent = @intCast(fr.exponent),
                    };
                },
                else => @compileError("x must be an int or float"),
            }
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            // Handle special cases
            if (self.isInf() or self.isNan() or self.eql(zero)) {
                return std.fmt.formatType(self.significand, fmt, options, writer, std.options.fmt_max_depth);
            }

            const mode: std.fmt.format_float.Format =
                comptime if (fmt.len == 0 or std.mem.eql(u8, fmt, "e"))
                    .scientific
                else if (std.mem.eql(u8, fmt, "d"))
                    .decimal
                else
                    std.fmt.invalidFmtError(fmt, self);

            // significand is normalized, we don't have to deal with subnormal numbers
            assert(@abs(self.significand) >= 0.5 and @abs(self.significand) < 1.0);

            const s10, const e10 = blk: {
                const log10_2 = 0.301029995663981195213738894724493027;
                const log2_10 = 3.321928094887362347870319429489390176;
                const e: f128 = @floatFromInt(self.exponent);
                const e10 = e * log10_2;
                var e10_floor: f128 = @floor(e10);

                // Compute `e10 - e10_floor` with high precision
                const e10_diff = diff_blk: {
                    const log10_2_2e192: u192 = 1889595908185821346144366738539203194586237552713217407397; // log10(2) * 2^192
                    const m_bits = math.floatMantissaBits(f128);
                    const e_bits = math.floatExponentBits(f128);

                    const bias: i32 = @intCast((@as(u32, 1) << (e_bits - 1)) - 1);
                    const exponent: i32 = @intCast(
                        (@as(u128, @bitCast(e)) >> m_bits) &
                            ((@as(u128, 1) << e_bits) - 1),
                    );
                    const mantissa = @as(u128, @bitCast(e)) &
                        ((@as(u128, 1) << m_bits) - 1) |
                        (@as(u128, 1) << m_bits);

                    const e10_384 = math.mulWide(u192, mantissa, log10_2_2e192);
                    const shift: u9 = @intCast(192 - 128 + m_bits - exponent + bias);
                    const e10_frac_m: u128 = @truncate(e10_384 >> shift);
                    const diff = math.ldexp(@as(f128, @floatFromInt(e10_frac_m)), -128);
                    break :diff_blk if (e > 0) diff else 1 - diff;
                };

                var s10: f128 = self.significand * exp2_128(e10_diff * log2_10);
                if (@abs(s10) < 1) {
                    s10 *= 10;
                    e10_floor -= 1;
                }
                break :blk .{ s10, @as(E, @intFromFloat(e10_floor)) };
            };
            var buf: [std.fmt.format_float.min_buffer_size + @as(u16, math.log10_int(@typeInfo(E).int.bits))]u8 = undefined;
            switch (mode) {
                .scientific => {
                    const str = try std.fmt.formatFloat(
                        &buf,
                        @as(S, @floatCast(s10)),
                        .{ .mode = .scientific, .precision = options.precision },
                    );
                    const n = std.fmt.formatIntBuf(buf[str.len - 1 ..], e10, 10, .lower, .{});
                    try std.fmt.formatBuf(buf[0 .. str.len - 1 + n], options, writer);
                },
                .decimal => {
                    @panic("TODO");
                    // const str = try std.fmt.formatFloat(
                    //     &buf,
                    //     @as(S, @floatCast(s10)),
                    //     .{ .mode = .decimal, .precision = options.precision },
                    // );
                },
            }
        }

        pub fn sign(self: Self) S {
            return math.sign(self.significand);
        }

        pub fn isInf(self: Self) bool {
            return math.isInf(self.significand);
        }

        pub fn isNan(self: Self) bool {
            return math.isNan(self.significand);
        }

        pub fn eql(lhs: Self, rhs: Self) bool {
            return lhs.significand == rhs.significand and lhs.exponent == rhs.exponent;
        }

        /// Performs an approximate comparison of `lhs` and `rhs`.
        /// Returns true if the absolute difference between them is less or equal than
        /// `max(|lhs|, |rhs|) * tolerance`, where `tolerance` is a positive number greater
        /// than zero.
        ///
        /// NaN values are never considered equal to any value.
        pub fn approxEqRel(lhs: Self, rhs: Self, tolerance: S) bool {
            assert(tolerance > 0);

            // Fast paths.
            if (lhs.eql(rhs)) return true;
            if (lhs.isInf() or rhs.isInf()) return false;
            if (lhs.isNan() or rhs.isNan()) return false;

            // lhs and rhs must be finite and non-zero.
            const lhs_abs = lhs.abs();
            const rhs_abs = rhs.abs();
            const abs_max = if (lhs.gt(rhs)) lhs_abs else rhs_abs;
            return !lhs.add(rhs.neg()).abs().gt(abs_max.mul(.from(tolerance)));
        }

        pub fn gt(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand > rhs.significand;
            }
            const exp_cmp = if (lhs.sign() == 1.0) lhs.exponent > rhs.exponent else lhs.exponent < rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand > rhs.significand);
        }

        pub fn lt(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand < rhs.significand;
            }
            const exp_cmp = if (lhs.sign() == 1.0) lhs.exponent < rhs.exponent else lhs.exponent > rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand < rhs.significand);
        }

        pub fn abs(self: Self) Self {
            return .{
                .significand = @abs(self.significand),
                .exponent = self.exponent,
            };
        }

        pub fn neg(self: Self) Self {
            return .{
                .significand = -self.significand,
                .exponent = self.exponent,
            };
        }

        /// Returns e where `x = s * 2^e` and `abs(s)` is in the interval `[0.5, 1)`.
        ///
        /// Asserts that `x` is finite and non-zero.
        fn floatExponent(x: S) i32 {
            assert(math.isFinite(x));
            assert(x != 0);

            const Int: type = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
            const MantInt: type = std.meta.Int(.unsigned, math.floatMantissaBits(S));
            const ExpInt = std.meta.Int(.unsigned, math.floatExponentBits(S));
            const bias: comptime_int = (1 << (math.floatExponentBits(S) - 1)) - 2;
            const ones_place: comptime_int = math.floatMantissaBits(S) - math.floatFractionalBits(S);

            const v: Int = @bitCast(x);
            const m: MantInt = @truncate(v);
            const e: ExpInt = @truncate(v >> math.floatMantissaBits(S));

            return switch (e) {
                // subnormal
                0 => math.floatExponentMin(S) - @as(i32, @clz(m)) + ones_place,
                // normal
                else => @as(i32, e) - bias,
            };
        }

        /// Returns x * 2^n.
        /// Asserts that `x` is finite and in the interval `[0.5, 1)`.
        /// Asserts that `n` is non-positive and greater than `1 - floatExponentMax(S)`.
        fn ldexpFast(x: S, n: i32) S {
            assert(!math.isNan(x));
            assert(0.5 <= @abs(x) and @abs(x) < 1.0);
            assert(n <= 0);
            assert(n > 1 - math.floatExponentMax(S));

            const SBits = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
            const mantissa_bits = math.floatMantissaBits(S);
            const repr = @as(SBits, @bitCast(x));
            return @as(S, @bitCast(repr - (@as(SBits, @intCast(-n)) << mantissa_bits)));
        }

        /// Normalizes the significand and exponent of `x` so that the significand is in the
        /// interval `[0.5, 1)`, or returns one of the special cases for zero, infinity, or NaN.
        /// `-0` is normalized to `0`.
        ///
        /// `normalize` must be called after modifying the significand or exponent of `x` directly.
        pub fn normalize(x: Self) Self {
            if (math.isNan(x.significand)) return nan;
            if (math.isInf(x.significand)) {
                return if (x.significand > 0) inf else minus_inf;
            }
            return normalizeFinite(x);
        }

        /// Performs the same function as `normalize`, but asserts that `x.significand` is finite.
        pub fn normalizeFinite(x: Self) Self {
            assert(!math.isNan(x.significand));
            assert(!math.isInf(x.significand));

            if (x.significand == 0) return zero;

            const exp_offset = floatExponent(x.significand);
            const ExpInt = std.meta.Int(.signed, @max(@typeInfo(E).int.bits, @typeInfo(@TypeOf(exp_offset)).int.bits) + 1);
            const new_exponent = @as(ExpInt, x.exponent) + @as(ExpInt, exp_offset);
            if (new_exponent > max_exponent) {
                return if (x.significand > 0) inf else minus_inf;
            }
            if (new_exponent < min_exponent) return zero;
            return .{
                .significand = math.ldexp(x.significand, -exp_offset),
                .exponent = @intCast(new_exponent),
            };
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            if (lhs.isNan() or rhs.isNan()) return nan;
            if (lhs.isInf()) {
                if (!rhs.isInf()) return lhs;
                const same_sign = math.signbit(lhs.significand) == math.signbit(rhs.significand);
                return if (same_sign) lhs else nan;
            }
            if (rhs.isInf()) return rhs;
            if (lhs.significand == 0) return rhs;
            if (rhs.significand == 0) return lhs;

            return if (lhs.exponent < rhs.exponent)
                add2(rhs, lhs)
            else
                add2(lhs, rhs);
        }

        fn add2(lhs: Self, rhs: Self) Self {
            assert(lhs.exponent >= rhs.exponent);
            assert(!lhs.isNan() and !rhs.isNan());
            assert(!lhs.isInf() and !rhs.isInf());
            assert(lhs.significand != 0 and rhs.significand != 0);

            const exp_diff = lhs.exponent - rhs.exponent;
            // The exponent difference is too large, we can just return lhs
            if (exp_diff > math.floatFractionalBits(S)) return lhs;

            const normalized_rhs = ldexpFast(rhs.significand, @intCast(-exp_diff));
            return normalizeFinite(.{
                .significand = lhs.significand + normalized_rhs,
                .exponent = lhs.exponent,
            });
        }

        pub fn sub(lhs: Self, rhs: Self) Self {
            if (lhs.isNan() or rhs.isNan()) return nan;
            if (lhs.isInf()) {
                if (!rhs.isInf()) return lhs;
                const same_sign = math.signbit(lhs.significand) == math.signbit(rhs.significand);
                return if (same_sign) nan else lhs;
            }
            if (rhs.isInf()) return rhs.neg();
            if (lhs.significand == 0) return rhs.neg();
            if (rhs.significand == 0) return lhs;

            return if (lhs.exponent < rhs.exponent)
                sub2(rhs, lhs).neg()
            else
                sub2(lhs, rhs);
        }

        fn sub2(lhs: Self, rhs: Self) Self {
            assert(lhs.exponent >= rhs.exponent);
            assert(!lhs.isNan() and !rhs.isNan());
            assert(!lhs.isInf() and !rhs.isInf());
            assert(lhs.significand != 0 and rhs.significand != 0);

            const exp_diff = lhs.exponent - rhs.exponent;
            // The exponent difference is too large, we can just return lhs
            if (exp_diff > math.floatFractionalBits(S)) return lhs;

            const normalized_rhs = ldexpFast(rhs.significand, @intCast(-exp_diff));
            return normalizeFinite(.{
                .significand = lhs.significand - normalized_rhs,
                .exponent = lhs.exponent,
            });
        }

        pub fn mul(lhs: Self, rhs: Self) Self {
            const significand = lhs.significand * rhs.significand;
            if (math.isNan(significand)) return nan;
            if (math.isInf(significand)) {
                return .{
                    .significand = significand,
                    .exponent = 0,
                };
            }
            if (significand == 0) return zero;

            const ExpInt = std.meta.Int(.signed, @max(32, @typeInfo(E).int.bits) + 2);
            const exp_offset = floatExponent(significand);
            const exponent = @as(ExpInt, lhs.exponent) + @as(ExpInt, rhs.exponent) + exp_offset;
            if (exponent > max_exponent) return if (significand > 0) inf else minus_inf;
            if (exponent < min_exponent) return zero;
            return .{
                .significand = math.ldexp(significand, -exp_offset),
                .exponent = @intCast(exponent),
            };
        }
    };
}

test {
    testing.refAllDecls(@This());
}

const f64_error_tolerance = 2.220446049250313e-14; // 10 ^ (-log_10(2^52) + 2)

fn fitsInt(Int: type, value: anytype) bool {
    return value >= math.minInt(Int) and value <= math.maxInt(Int);
}

fn bigFloatTypes(ss: []const type, es: []const type) [ss.len * es.len]type {
    var types: [ss.len * es.len]type = undefined;
    for (ss, 0..) |s, i| {
        for (es, 0..) |e, j| {
            types[i * es.len + j] = BigFloat(s, e);
        }
    }
    return types;
}

test "from" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        const S = @FieldType(F, "significand");
        const E = @FieldType(F, "exponent");

        try testing.expectEqual(F{
            .significand = 0.5,
            .exponent = 1,
        }, F.from(1));
        try testing.expectEqual(F{
            .significand = -123.0 / 128.0,
            .exponent = 7,
        }, F.from(@as(i32, -123)));
        try testing.expectEqual(F{
            .significand = 0.0043 * 128.0,
            .exponent = -7,
        }, F.from(0.0043));
        try testing.expectEqual(F{
            .significand = 0,
            .exponent = 0,
        }, F.from(0));

        @setEvalBranchQuota(10_000);
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) + 1))
                F{
                    .significand = 0.5,
                    .exponent = math.floatExponentMin(S) + 1,
                }
            else
                F.zero,
            F.from(math.floatMin(S)),
        );
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S) + 1))
                F{
                    .significand = 0.5,
                    .exponent = math.floatExponentMin(S) - math.floatFractionalBits(S) + 1,
                }
            else
                F.zero,
            F.from(math.floatTrueMin(S)),
        );

        try testing.expectEqual(F{
            .significand = math.inf(S),
            .exponent = 0,
        }, F.from(math.inf(S)));
        try testing.expect(math.isNan(
            F.from(math.nan(S)).significand,
        ));
    }
}

test "format" {
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i53, i64 })) |F| {
        try testing.expectFmt("0e0", "{e}", .{F.zero});
        try testing.expectFmt("-0e0", "{e}", .{F.from(-0.0)});
        try testing.expectFmt("inf", "{e}", .{F.inf});
        try testing.expectFmt("-inf", "{e}", .{F.minus_inf});
        try testing.expectFmt("nan", "{e}", .{F.nan});
        try testing.expectFmt("1.2345e4", "{e}", .{F.from(12345)});
        try testing.expectFmt(
            "-7.629816727e35",
            "{e:.9}",
            .{F.from(-762981672689762158671378613432987234.123)},
        );
        try testing.expectFmt(
            "     6.1267e-23     ",
            "{e:^20.4}",
            .{F.from(6.1267346318123e-23)},
        );
        try testing.expectFmt("6.969e69696969696969", "{e:.3}", .{F{
            .significand = 0.59682029048932636742444910978537,
            .exponent = 231528321764878,
        }});

        try testing.expectFmt("0", "{d}", .{F.zero});
        try testing.expectFmt("-0", "{d}", .{F.from(-0.0)});
        try testing.expectFmt("inf", "{d}", .{F.inf});
        try testing.expectFmt("-inf", "{d}", .{F.minus_inf});
        try testing.expectFmt("nan", "{d}", .{F.nan});
        // try testing.expectFmt("     12345     ", "{d:^15}", .{F.from(12345)});
        // try testing.expectFmt(
        //     "-762981672489762158671378613432987234.12",
        //     "{d:.2}",
        //     .{F.from(-762981672489762158671378613432987234.123)},
        // );
        // try testing.expectFmt(
        //     "0.00000000000000000000006126734632",
        //     "{d:.32}",
        //     .{F.from(6.1267346318123e-23)},
        // );
    }
}

test "sign" {
    inline for (.{
        BigFloat(f32, i8),
        BigFloat(f32, i32),
        BigFloat(f64, i16),
        BigFloat(f128, i32),
    }) |F| {
        try testing.expectEqual(1, F.from(123).sign());
        try testing.expectEqual(0, F.from(0).sign());
        try testing.expectEqual(-1, F.from(-123).sign());
        try testing.expectEqual(0, F.from(math.nan(f32)).sign());
        try testing.expectEqual(1, F.from(math.inf(f32)).sign());
    }
}

test "isInf" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(true, F.inf.isInf());
        try testing.expectEqual(true, F.minus_inf.isInf());
        try testing.expectEqual(false, F.from(0).isInf());
        try testing.expectEqual(false, F.from(123).isInf());
        try testing.expectEqual(false, F.nan.isInf());
    }
}

test "isNan" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(true, F.nan.isNan());
        try testing.expectEqual(false, F.inf.isNan());
        try testing.expectEqual(false, F.minus_inf.isNan());
        try testing.expectEqual(false, F.from(0).isNan());
        try testing.expectEqual(false, F.from(123).isNan());
    }
}

test "eql" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.from(123).eql(F.from(123)));
        try testing.expect(!F.from(123).eql(F.from(122)));
        try testing.expect(F.from(0).eql(F.from(-0.0)));
        try testing.expect(F.inf.eql(F.inf));
        try testing.expect(!F.inf.eql(F.max_value));
        try testing.expect(!F.inf.eql(F.minus_inf));
        try testing.expect(!F.nan.eql(F.nan));
    }
}

test "approxEqRel" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        // Exactly equal
        try testing.expect(F.from(123).approxEqRel(.from(123), 1e-6));
        try testing.expect(!F.from(123).approxEqRel(.from(122), 1e-6));
        try testing.expect(F.from(0).approxEqRel(.from(-0.0), 1e-6));
        try testing.expect(F.inf.approxEqRel(.inf, 1e-6));
        try testing.expect(!F.inf.approxEqRel(.max_value, 1e-6));
        try testing.expect(!F.inf.approxEqRel(.minus_inf, 1e-6));
        try testing.expect(!F.nan.approxEqRel(.nan, 1e-6));

        // Almost equal
        try testing.expect(!F.from(1).approxEqRel(.from(0), 1e-6));
        try testing.expect(F.from(1).approxEqRel(
            .from(1 - (1.0 / 1024.0)),
            1.0 / 1024.0,
        ));
        try testing.expect(F.from(1).approxEqRel(
            .from(1 - 1e2),
            1e2,
        ));
        try testing.expect(!F.from(1).approxEqRel(
            .from(1 - 1e-5),
            1e-6,
        ));
    }
}

test "gt" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(false, F.from(0).gt(F.from(0)));
        try testing.expectEqual(false, F.from(-0.0).gt(F.from(0)));
        try testing.expectEqual(false, F.from(0).gt(F.from(-0.0)));
        try testing.expectEqual(false, F.from(-0.0).gt(F.from(-0.0)));

        try testing.expectEqual(true, F.from(123).gt(F.from(122)));
        try testing.expectEqual(false, F.from(123).gt(F.from(123)));
        try testing.expectEqual(false, F.from(123).gt(F.from(124)));
        try testing.expectEqual(true, F.from(123).gt(F.from(12)));
        try testing.expectEqual(false, F.from(12).gt(F.from(123)));

        try testing.expectEqual(true, F.from(123).gt(F.from(-123)));
        try testing.expectEqual(true, F.from(12).gt(F.from(-123)));
        try testing.expectEqual(true, F.from(123).gt(F.from(-12)));
        try testing.expectEqual(false, F.from(-123).gt(F.from(123)));
        try testing.expectEqual(false, F.from(-12).gt(F.from(123)));
        try testing.expectEqual(false, F.from(-123).gt(F.from(12)));

        try testing.expectEqual(false, F.from(-123).gt(F.from(-122)));
        try testing.expectEqual(false, F.from(-123).gt(F.from(-123)));
        try testing.expectEqual(true, F.from(-123).gt(F.from(-124)));
        try testing.expectEqual(false, F.from(-123).gt(F.from(-12)));
        try testing.expectEqual(true, F.from(-12).gt(F.from(-123)));

        try testing.expectEqual(false, F.inf.gt(F.inf));
        try testing.expectEqual(true, F.inf.gt(F.minus_inf));
        try testing.expectEqual(false, F.minus_inf.gt(F.inf));
        try testing.expectEqual(false, F.nan.gt(F.nan));
    }
}

test "lt" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(false, F.from(0).lt(F.from(0)));
        try testing.expectEqual(false, F.from(-0.0).lt(F.from(0)));
        try testing.expectEqual(false, F.from(0).lt(F.from(-0.0)));
        try testing.expectEqual(false, F.from(-0.0).lt(F.from(-0.0)));

        try testing.expectEqual(false, F.from(123).lt(F.from(122)));
        try testing.expectEqual(false, F.from(123).lt(F.from(123)));
        try testing.expectEqual(true, F.from(123).lt(F.from(124)));
        try testing.expectEqual(false, F.from(123).lt(F.from(12)));
        try testing.expectEqual(true, F.from(12).lt(F.from(123)));

        try testing.expectEqual(false, F.from(123).lt(F.from(-123)));
        try testing.expectEqual(false, F.from(12).lt(F.from(-123)));
        try testing.expectEqual(false, F.from(123).lt(F.from(-12)));
        try testing.expectEqual(true, F.from(-123).lt(F.from(123)));
        try testing.expectEqual(true, F.from(-12).lt(F.from(123)));
        try testing.expectEqual(true, F.from(-123).lt(F.from(12)));

        try testing.expectEqual(true, F.from(-123).lt(F.from(-122)));
        try testing.expectEqual(false, F.from(-123).lt(F.from(-123)));
        try testing.expectEqual(false, F.from(-123).lt(F.from(-124)));
        try testing.expectEqual(true, F.from(-123).lt(F.from(-12)));
        try testing.expectEqual(false, F.from(-12).lt(F.from(-123)));

        try testing.expectEqual(false, F.inf.lt(F.inf));
        try testing.expectEqual(false, F.inf.lt(F.minus_inf));
        try testing.expectEqual(true, F.minus_inf.lt(F.inf));
        try testing.expectEqual(false, F.nan.lt(F.nan));
    }
}

test "abs" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.from(123).abs().eql(F.from(123)));
        try testing.expect(F.from(-123).abs().eql(F.from(123)));
        try testing.expect(F.from(0).abs().eql(F.from(0)));
        try testing.expect(F.minus_inf.abs().eql(F.inf));
    }
}

test "neg" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.from(123).neg().eql(F.from(-123)));
        try testing.expect(F.from(-123).neg().eql(F.from(123)));
        try testing.expect(F.from(0).neg().eql(F.from(-0.0)));
        try testing.expect(F.minus_inf.neg().eql(F.inf));
    }
}

test "normalize" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        const S = @FieldType(F, "significand");
        const E = @FieldType(F, "exponent");

        try testing.expectEqual(F{
            .significand = 0.5,
            .exponent = 1,
        }, F.normalize(.{ .significand = 1, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = -123.0 / 128.0,
            .exponent = 7,
        }, F.normalize(.{ .significand = -123, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = 0.0043 * 128.0,
            .exponent = -7,
        }, F.normalize(.{ .significand = 0.0043, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = 0,
            .exponent = 0,
        }, F.normalize(.{ .significand = 0, .exponent = 0 }));

        @setEvalBranchQuota(10_000);
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) + 1))
                F{
                    .significand = 0.5,
                    .exponent = math.floatExponentMin(S) + 1,
                }
            else
                F.zero,
            F.normalize(.{ .significand = math.floatMin(S), .exponent = 0 }),
        );
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S) + 1))
                F{
                    .significand = 0.5,
                    .exponent = math.floatExponentMin(S) - math.floatFractionalBits(S) + 1,
                }
            else
                F.zero,
            F.normalize(.{ .significand = math.floatTrueMin(S), .exponent = 0 }),
        );

        try testing.expectEqual(F{
            .significand = 0,
            .exponent = 0,
        }, F.normalize(.{ .significand = 0, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = 0,
            .exponent = 0,
        }, F.normalize(.{ .significand = -0.0, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = math.inf(S),
            .exponent = 0,
        }, F.normalize(.{ .significand = math.inf(S), .exponent = 0 }));
        try testing.expect(math.isNan(
            F.normalize(.{ .significand = math.nan(S), .exponent = 0 }).significand,
        ));
    }
}

test "floatExponent" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{i32})) |F| {
        try testing.expectEqual(0, F.floatExponent(0.5));
        try testing.expectEqual(-1, F.floatExponent(0.3));
        try testing.expectEqual(1, F.floatExponent(-1.0));
        try testing.expectEqual(120, F.floatExponent(1e36));

        try testing.expectEqual(-132, F.floatExponent(1e-40));
        try testing.expectEqual(-132, F.floatExponent(-1e-40));
    }
}

test "add" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{i11})) |F| {
        try testing.expectEqual(F.from(0), F.from(0).add(.from(0)));
        try testing.expectEqual(F.from(1), F.from(1).add(.from(0)));
        try testing.expectEqual(F.from(444), F.from(123).add(.from(321)));
        try testing.expectEqual(F.from(0), F.from(123).add(.from(-123)));
        try testing.expectEqual(F.from(4.75), F.from(1.5).add(.from(3.25)));
        try testing.expectEqual(F.from(1e38), F.from(1e38).add(.from(1e-38)));
        {
            const expected = F.from(1e36);
            const actual = F.from(1e38).add(.from(-0.99e38));
            try testing.expectEqual(expected.exponent, actual.exponent);
            try testing.expect(math.approxEqRel(
                @FieldType(F, "significand"),
                expected.significand,
                actual.significand,
                f64_error_tolerance,
            ));
        }

        try testing.expectEqual(F.inf, F.max_value.add(.max_value));
        try testing.expectEqual(F.minus_inf, F.min_value.add(.min_value));
        try testing.expectEqual(F.zero, F.min_value.add(.max_value));
        try testing.expectEqual(F.zero, F.max_value.add(.min_value));

        // Only valid when exponent is i11
        try testing.expect(!F.inf.eql(.from(0.6e308)));
        try testing.expectEqual(F.inf, F.from(0.6e308).add(.from(0.6e308)));

        try testing.expectEqual(F.minus_inf, F.from(12).add(F.minus_inf));
        try testing.expect(F.inf.add(F.minus_inf).isNan());
        try testing.expect(F.nan.add(.from(2)).isNan());
    }
}

test "sub" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{i11})) |F| {
        try testing.expectEqual(F.from(0), F.from(0).sub(.from(0)));
        try testing.expectEqual(F.from(1), F.from(1).sub(.from(0)));
        try testing.expectEqual(F.from(-198), F.from(123).sub(.from(321)));
        try testing.expectEqual(F.from(246), F.from(123).sub(.from(-123)));
        try testing.expectEqual(F.from(0), F.from(123).sub(.from(123)));
        try testing.expectEqual(F.from(-1.75), F.from(1.5).sub(.from(3.25)));
        try testing.expectEqual(F.from(1e38), F.from(1e38).sub(.from(1e-38)));
        {
            const expected = F.from(1e36);
            const actual = F.from(1e38).sub(.from(0.99e38));
            try testing.expectEqual(expected.exponent, actual.exponent);
            try testing.expect(math.approxEqRel(
                @FieldType(F, "significand"),
                expected.significand,
                actual.significand,
                f64_error_tolerance,
            ));
        }

        try testing.expectEqual(F.zero, F.max_value.sub(.max_value));
        try testing.expectEqual(F.zero, F.min_value.sub(.min_value));
        try testing.expectEqual(F.minus_inf, F.min_value.sub(.max_value));
        try testing.expectEqual(F.inf, F.max_value.sub(.min_value));

        // Only valid when exponent is i11
        try testing.expect(!F.inf.eql(.from(0.6e308)));
        try testing.expectEqual(F.inf, F.from(0.6e308).sub(.from(-0.6e308)));

        try testing.expectEqual(F.inf, F.from(12).sub(F.minus_inf));
        try testing.expect(F.inf.sub(F.inf).isNan());
        try testing.expect(F.nan.sub(.from(2)).isNan());
    }
}

test "mul" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i23, i64 })) |F| {
        try testing.expectEqual(F.from(0), F.from(0).mul(.from(0)));
        try testing.expectEqual(F.from(0), F.from(1).mul(.from(0)));
        try testing.expectEqual(F.from(3.5), F.from(1).mul(.from(3.5)));
        try testing.expectEqual(F.from(39483), F.from(123).mul(.from(321)));
        try testing.expectEqual(F.from(4.875), F.from(1.5).mul(.from(3.25)));
        try testing.expectEqual(F.from(-151782), F.from(123).mul(.from(-1234)));
        try testing.expect(F.from(3.74496).approxEqRel(F.from(-0.83).mul(.from(-4.512)), f64_error_tolerance));
        try testing.expect(F.from(1).approxEqRel(F.from(1e38).mul(.from(1e-38)), f64_error_tolerance));

        try testing.expect(
            (F{
                .significand = 0.89117166164618254333829281056332,
                .exponent = 2045,
            }).approxEqRel(F.from(0.6e308).mul(.from(0.6e308)), f64_error_tolerance),
        );
        try testing.expectEqual(F.minus_inf, F.inf.mul(.minus_inf));
        try testing.expectEqual(F.inf, F.inf.mul(.inf));
        try testing.expectEqual(F.inf, F.inf.mul(.from(1)));
        try testing.expect(F.inf.mul(.from(0)).isNan());
        try testing.expect(F.inf.mul(.nan).isNan());
        try testing.expect(F.nan.mul(.from(2)).isNan());
    }
}
