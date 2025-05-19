const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

const exp2_128 = @import("exp2_128.zig").exp2_128;

/// Represents a floating-point number as `significand * 2^exponent`.
/// `abs(significand)` is in the interval `[0.5, 1)`.
///
/// Special cases:
///  - `+-0   => significand = +-0,   exponent = 0`
///  - `+-inf => significand = +-inf, exponent = 0`
///  - `nan   => significand = nan,   exponent = undefined`
pub fn BigFloat(S: type, E: type) type {
    assert(@typeInfo(S) == .float);
    switch (@typeInfo(E)) {
        .int => |info| assert(info.signedness == .signed),
        else => @compileError("exponent must be an signed int"),
    }

    return struct {
        significand: S,
        exponent: E,

        const Self = @This();
        const max_exponent = math.maxInt(E);
        const min_exponent = math.minInt(E);

        // zig fmt: off
        pub const zero: Self =      .{ .significand = 0,                         .exponent = 0 };
        pub const minusZero: Self = .{ .significand = -0.0,                      .exponent = 0 };
        pub const inf: Self =       .{ .significand = math.inf(S),               .exponent = 0 };
        pub const minusInf: Self =  .{ .significand = -math.inf(S),              .exponent = 0 };
        pub const nan: Self =       .{ .significand = math.nan(S),               .exponent = undefined };
        /// Largest value smaller than `inf`.
        pub const maxValue: Self =  .{ .significand = 1 - math.floatEpsAt(S, 1), .exponent = math.maxInt(E) };
        /// Smallest value larger than `minusInf`.
        pub const minValue: Self =  .{ .significand = math.floatEpsAt(S, 1) - 1, .exponent = math.maxInt(E) };
        /// Smallest value larger than `zero`.
        pub const epsilon: Self =   .{ .significand = 0.5,                       .exponent = math.minInt(E) };
        // zig fmt: on

        pub fn from(x: anytype) Self {
            const T = @TypeOf(x);
            switch (@typeInfo(T)) {
                .int, .comptime_int => {
                    if (x == 0) return zero;

                    const x_msb = 1 + math.log2(@abs(x));
                    const x_bits = switch (T) {
                        comptime_int => x_msb,
                        else => @typeInfo(T).int.bits,
                    };
                    // Zig ints go up to 65,536 bits, so using i32 is always safe
                    const exponent = 1 + @as(i32, math.log2_int(std.meta.Int(.unsigned, x_bits), @abs(x)));
                    if (exponent > max_exponent) return if (x > 0) inf else minusInf;

                    // Bit shift to ensure x fits in the range of S
                    const shift = @max(0, @as(i32, @intCast(x_msb)) - math.floatFractionalBits(S) - 1);
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
                    if (fr.exponent < min_exponent) return if (fr.significand > 0) zero else minusZero;
                    if (fr.exponent > max_exponent) return if (fr.significand > 0) inf else minusInf;
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
    };
}

const testing = std.testing;

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
        try testing.expectEqualDeep(F{
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
        try testing.expectEqual(F{
            .significand = math.inf(@FieldType(F, "significand")),
            .exponent = 0,
        }, F.from(math.inf(@FieldType(F, "significand"))));
        try testing.expect(math.isNan(
            F.from(math.nan(@FieldType(F, "significand"))).significand,
        ));
    }
}

test "format" {
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i53, i64 })) |F| {
        try testing.expectFmt("0e0", "{e}", .{F.zero});
        try testing.expectFmt("-0e0", "{e}", .{F.from(-0.0)});
        try testing.expectFmt("inf", "{e}", .{F.inf});
        try testing.expectFmt("-inf", "{e}", .{F.minusInf});
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
        try testing.expectFmt("-inf", "{d}", .{F.minusInf});
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
        try testing.expectEqual(true, F.minusInf.isInf());
        try testing.expectEqual(false, F.from(0).isInf());
        try testing.expectEqual(false, F.from(123).isInf());
        try testing.expectEqual(false, F.nan.isInf());
    }
}

test "isNan" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(true, F.nan.isNan());
        try testing.expectEqual(false, F.inf.isNan());
        try testing.expectEqual(false, F.minusInf.isNan());
        try testing.expectEqual(false, F.from(0).isNan());
        try testing.expectEqual(false, F.from(123).isNan());
    }
}

test "eql" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(true, F.from(123).eql(F.from(123)));
        try testing.expectEqual(false, F.from(123).eql(F.from(122)));
        try testing.expectEqual(true, F.from(0).eql(F.from(-0.0)));
        try testing.expectEqual(true, F.inf.eql(F.inf));
        try testing.expectEqual(false, F.inf.eql(F.minusInf));
        try testing.expectEqual(false, F.nan.eql(F.nan));
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
        try testing.expectEqual(true, F.inf.gt(F.minusInf));
        try testing.expectEqual(false, F.minusInf.gt(F.inf));
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
        try testing.expectEqual(false, F.inf.lt(F.minusInf));
        try testing.expectEqual(true, F.minusInf.lt(F.inf));
        try testing.expectEqual(false, F.nan.lt(F.nan));
    }
}

test "abs" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.from(123).abs().eql(F.from(123)));
        try testing.expect(F.from(-123).abs().eql(F.from(123)));
        try testing.expect(F.from(0).abs().eql(F.from(0)));
        try testing.expect(F.minusInf.abs().eql(F.inf));
    }
}

test "neg" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.from(123).neg().eql(F.from(-123)));
        try testing.expect(F.from(-123).neg().eql(F.from(123)));
        try testing.expect(F.from(0).neg().eql(F.from(-0.0)));
        try testing.expect(F.minusInf.neg().eql(F.inf));
    }
}
