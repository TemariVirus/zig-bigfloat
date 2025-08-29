const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const testing = std.testing;
const Writer = std.Io.Writer;

pub const Options = struct {
    Significand: type,
    Exponent: type,
    /// This should only be disabled to increase compilation speed.
    /// Binary sizes are smaller when this is enabled as the baked constants
    /// take up less space than the code for generating them at runtime.
    bake_render: bool = @import("builtin").mode != .Debug,
    format_buf_size: ?usize = null,
};

/// Represents a floating-point number as `significand * 2^exponent`.
/// `abs(significand)` is in the interval `[1, 2)`.
///
/// Special cases:
///  - `+-0   => significand = +-0,   exponent = 0`
///  - `+-inf => significand = +-inf, exponent = 0`
///  - `nan   => significand = nan,   exponent = 0`
pub fn BigFloat(comptime float_options: Options) type {
    const S = float_options.Significand;
    const E = float_options.Exponent;
    @setEvalBranchQuota(10000);
    comptime assert(@typeInfo(S) == .float);
    switch (@typeInfo(E)) {
        .int => |info| comptime assert(info.signedness == .signed),
        else => @compileError("exponent must be a signed int"),
    }
    // TODO: document limits of S and E sizes

    const Render = @import("schubfach.zig").Render(S, E, float_options.bake_render);
    const format_buf_size = float_options.format_buf_size orelse
        Render.Decimal.maxDigitCount + Render.Decimal.maxExponentDigitCount + 3 + 20;

    // Using a packed struct increases performance by 45% to 140%;
    return packed struct {
        /// The significand, normalized to the range `[1, 2)`.
        /// `normalize` must be called after modifying this field directly.
        significand: S,
        /// The base-2 exponent. `normalize` must be called after modifying this field directly.
        exponent: E,

        const Self = @This();

        pub const Decimal = Render.Decimal;

        // zig fmt: off
        pub const zero: Self =       .{ .significand = 0,                        .exponent = 0 };
        pub const minus_zero: Self = .{ .significand = -0.0,                     .exponent = 0 };
        pub const inf: Self =        .{ .significand = math.inf(S),              .exponent = 0 };
        pub const minus_inf: Self =  .{ .significand = -math.inf(S),             .exponent = 0 };
        pub const nan: Self =        .{ .significand = math.nan(S),              .exponent = 0 };
        /// Largest value smaller than `inf`.
        pub const max_value: Self =  .{ .significand = math.nextAfter(S, 2, 0),  .exponent = math.maxInt(E) };
        /// Smallest value larger than `minus_inf`.
        pub const min_value: Self =  .{ .significand = math.nextAfter(S, -2, 0), .exponent = math.maxInt(E) };
        /// Smallest value larger than `zero`.
        pub const epsilon: Self =    .{ .significand = 1,                        .exponent = math.minInt(E) };
        // zig fmt: on

        pub fn init(x: anytype) Self {
            const T = @TypeOf(x);
            switch (@typeInfo(T)) {
                .int, .comptime_int => {
                    if (x == 0) return zero;

                    // Zig ints go up to 65,535 bits, so using i32 is always safe
                    const exponent: i32 = @intCast(math.log2(@abs(x)));
                    if (exponent > math.maxInt(E)) return if (x > 0) inf else minus_inf;

                    // Bit shift to ensure x fits in the range of S
                    const shift = @max(0, exponent - math.floatFractionalBits(S));
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
                    const exponent = fr.exponent - 1;
                    if (math.isNan(fr.significand)) return nan;
                    if (math.isInf(fr.significand)) return if (math.signbit(fr.significand)) minus_inf else inf;
                    if (fr.significand == 0 or exponent < math.minInt(E)) return if (math.signbit(fr.significand)) minus_zero else zero;
                    if (exponent > math.maxInt(E)) return if (math.signbit(fr.significand)) minus_inf else inf;
                    return .{
                        .significand = math.lossyCast(S, 2 * fr.significand),
                        .exponent = @intCast(exponent),
                    };
                },
                else => @compileError("x must be an int or float"),
            }
        }

        pub fn toFloat(self: Self, FloatT: type) FloatT {
            comptime assert(@typeInfo(FloatT) == .float);

            const f: FloatT = @floatCast(self.significand);
            return math.ldexp(f, math.clamp(
                self.exponent,
                math.minInt(i32),
                @as(i32, math.maxInt(i32)),
            ));
        }

        pub fn parse(str: []const u8) std.fmt.ParseFloatError!Self {
            _ = str;
            @panic("TODO");
        }

        /// Returns the decimal scientific representation of `w`.
        /// The result is not normalized, i.e., the digits may have trailing zeros.
        pub fn toDecimal(self: Self) Decimal {
            assert(math.isFinite(self.significand));
            assert(1 <= self.significand and self.significand < 2);

            if (self.significand == 0) return .{ .digits = 0, .exponent = 0 };
            return Render.toDecimal(self.significand, self.exponent);
        }

        pub fn format(self: Self, writer: *Writer) Writer.Error!void {
            // TODO: change the numbers to follow std when this PR is merged.
            // https://github.com/ziglang/zig/pull/22971#issuecomment-2676157243
            const decimal_min: Self = .init(1e-6);
            const decimal_max: Self = .init(1e15);
            if (self.abs().lt(decimal_min) or self.abs().gte(decimal_max)) {
                return self.formatNumber(writer, .{ .mode = .scientific });
            }
            return self.formatNumber(writer, .{ .mode = .decimal });
        }

        /// Only formats special cases (nan, inf).
        /// Returns true if a special case was formatted.
        /// Otherwise, returns false and nothing is written to `writer`.
        fn formatSpecial(self: Self, writer: *Writer, case: std.fmt.Case) Writer.Error!bool {
            if (self.isNan()) {
                try writer.writeAll(switch (case) {
                    .lower => "nan",
                    .upper => "NAN",
                });
                return true;
            }
            if (self.isInf()) {
                try writer.writeAll(switch (case) {
                    .lower => "inf",
                    .upper => "INF",
                });
                return true;
            }
            return false;
        }

        pub fn formatNumber(self: Self, writer: *Writer, options: std.fmt.Number) Writer.Error!void {
            if (math.signbit(self.significand)) try writer.writeByte('-');
            if (try formatSpecial(self, writer, options.case)) {
                return;
            }

            const s = switch (options.mode) {
                .decimal => @panic("TODO"),
                .scientific => blk: {
                    var buf: [format_buf_size]u8 = undefined;
                    var w = std.Io.Writer.fixed(&buf);
                    formatScientific(self.abs(), &w, options.precision) catch break :blk "(BigFloat)";
                    break :blk w.buffered();
                },
                .binary, .octal => @panic("TODO"),
                .hex => {
                    var discard_writer: Writer.Discarding = .init(&.{});
                    formatHex(self.abs(), &discard_writer.writer, options.case, options.precision) catch unreachable;
                    const len: usize = @intCast(discard_writer.fullCount());

                    const padding = @max(len, options.width orelse len) - len;
                    if (padding == 0) {
                        return formatHex(self.abs(), writer, options.case, options.precision);
                    }
                    switch (options.alignment) {
                        .left => {},
                        .center => try writer.splatByteAll(options.fill, padding / 2),
                        .right => try writer.splatByteAll(options.fill, padding),
                    }
                    try formatHex(self.abs(), writer, options.case, options.precision);
                    switch (options.alignment) {
                        .left => try writer.splatByteAll(options.fill, padding),
                        .center => try writer.splatByteAll(options.fill, (padding + 1) / 2),
                        .right => {},
                    }
                    return;
                },
            };
            return writer.alignBuffer(s, options.width orelse s.len, options.alignment, options.fill);
        }

        pub fn formatDecimal(self: Self, writer: *Writer, precision: ?usize) Writer.Error![]const u8 {
            _ = self; // autofix
            _ = writer; // autofix
            _ = precision; // autofix
            @panic("TODO");
        }

        pub fn formatScientific(self: Self, writer: *Writer, precision: ?usize) Writer.Error!void {
            if (self.significand == 0) return writer.writeAll("0e0");
            assert(self.significand > 0);
            assert(math.isNormal(self.significand));

            const decimal = if (precision) |p| blk: {
                const d = self.toDecimal();
                break :blk d.roundToEven(d.digitCount() - @min(p + 1, d.digitCount()));
            } else self.toDecimal().removeTrailingZeros();

            const digits_str = blk: {
                var buf: [Decimal.maxDigitCount]u8 = undefined;
                var digit_writer = std.Io.Writer.fixed(&buf);
                digit_writer.print("{d}", .{decimal.digits}) catch unreachable;
                break :blk digit_writer.buffered();
            };

            const p = precision orelse digits_str.len - 1;
            const actual_exponent = decimal.exponent + @as(i32, @intCast(digits_str.len)) - 1;
            if (p == 0) return writer.print("{s}e{d}", .{ digits_str, actual_exponent });
            try writer.print("{s}.{s}", .{ digits_str[0..1], digits_str[1..@min(p + 1, digits_str.len)] });
            try writer.splatByteAll('0', @max(p + 1, digits_str.len) - digits_str.len);
            return writer.print("e{d}", .{actual_exponent});
        }

        pub fn formatHex(self: Self, writer: *Writer, case: std.fmt.Case, precision: ?usize) Writer.Error!void {
            if (self.significand == 0) {
                try writer.writeAll("0x0");
                if (precision) |p| {
                    if (p > 0) {
                        try writer.writeAll(".");
                        try writer.splatByteAll('0', p);
                    }
                } else {
                    try writer.writeAll(".0");
                }
                try writer.writeAll("p0");
                return;
            }

            assert(self.significand > 0);
            assert(math.isNormal(self.significand));

            const C = std.meta.Int(.unsigned, @typeInfo(S).float.bits);

            const mantissa_bits = std.math.floatMantissaBits(S);
            const fractional_bits = std.math.floatFractionalBits(S);
            const mantissa_mask = (1 << mantissa_bits) - 1;

            const as_bits: C = @bitCast(self.significand);
            var mantissa = as_bits & mantissa_mask;
            var exponent: std.meta.Int(.signed, @typeInfo(E).int.bits + 1) = self.exponent;

            if (fractional_bits == mantissa_bits)
                mantissa |= 1 << fractional_bits; // Add the implicit integer bit.

            const mantissa_digits = (fractional_bits + 3) / 4;
            // Fill in zeroes to round the fraction width to a multiple of 4.
            mantissa <<= mantissa_digits * 4 - fractional_bits;

            if (precision) |p| {
                // Round if needed.
                if (p < mantissa_digits) {
                    // We always have at least 4 extra bits.
                    var extra_bits = (mantissa_digits - p) * 4;
                    // The result LSB is the Guard bit, we need two more (Round and
                    // Sticky) to round the value.
                    while (extra_bits > 2) {
                        mantissa = (mantissa >> 1) | (mantissa & 1);
                        extra_bits -= 1;
                    }
                    // Round to nearest, tie to even.
                    mantissa |= @intFromBool(mantissa & 0b100 != 0);
                    mantissa += 1;
                    // Drop the excess bits.
                    mantissa >>= 2;
                    // Restore the alignment.
                    mantissa <<= @as(std.math.Log2Int(C), @intCast((mantissa_digits - p) * 4));

                    const overflow = mantissa & (1 << 1 + mantissa_digits * 4) != 0;
                    // Prefer a normalized result in case of overflow.
                    if (overflow) {
                        mantissa >>= 1;
                        exponent += 1;
                    }
                }
            }

            // +1 for the decimal part.
            var buf: [1 + mantissa_digits]u8 = undefined;
            assert(std.fmt.printInt(&buf, mantissa, 16, case, .{ .fill = '0', .width = 1 + mantissa_digits }) == buf.len);

            try writer.writeAll("0x");
            try writer.writeByte(buf[0]);
            const trimmed = std.mem.trimRight(u8, buf[1..], "0");
            if (precision) |p| {
                if (p > 0) try writer.writeAll(".");
            } else if (trimmed.len > 0) {
                try writer.writeAll(".");
            }
            try writer.writeAll(trimmed);
            // Add trailing zeros if explicitly requested.
            if (precision) |p| if (p > 0) {
                if (p > trimmed.len)
                    try writer.splatByteAll('0', p - trimmed.len);
            };
            try writer.writeAll("p");
            try writer.printInt(exponent, 10, case, .{});
        }

        pub fn sign(self: Self) S {
            return math.sign(self.significand);
        }

        pub fn signBit(self: Self) bool {
            return math.signbit(self.significand);
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
            const abs_max = if (lhs_abs.gt(rhs_abs)) lhs_abs else rhs_abs;
            const abs_diff = lhs.sub(rhs).abs();
            return abs_diff.lte(abs_max.mul(.init(tolerance)));
        }

        pub fn gt(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand > rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent < rhs.exponent else lhs.exponent > rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand > rhs.significand);
        }

        pub fn gte(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand >= rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent < rhs.exponent else lhs.exponent > rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand >= rhs.significand);
        }

        pub fn lt(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand < rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent > rhs.exponent else lhs.exponent < rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand < rhs.significand);
        }

        pub fn lte(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand <= rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent > rhs.exponent else lhs.exponent < rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand <= rhs.significand);
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

        /// Returns e where `x = s * 2^e` and `abs(s)` is in the interval `[1, 2)`.
        ///
        /// Asserts that `x` is finite and non-zero.
        fn floatExponent(x: S) i32 {
            assert(math.isFinite(x));
            assert(x != 0);

            const Int: type = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
            const MantInt: type = std.meta.Int(.unsigned, math.floatMantissaBits(S));
            const ExpInt = std.meta.Int(.unsigned, math.floatExponentBits(S));
            const bias: comptime_int = (1 << (math.floatExponentBits(S) - 1)) - 1;
            const ones_place: comptime_int = math.floatMantissaBits(S) - math.floatFractionalBits(S);

            const v: Int = @bitCast(x);
            const m: MantInt = @truncate(v);
            const e: ExpInt = @truncate(v >> math.floatMantissaBits(S));

            return switch (e) {
                // subnormal
                0 => -math.floatExponentMax(S) - @as(i32, @clz(m)) + ones_place,
                // normal
                else => @as(i32, e) - bias,
            };
        }

        /// Returns x * 2^n.
        /// Asserts that `x` is finite and in the interval `[1, 2)`.
        /// Asserts that `n` is non-positive and greater than `-floatExponentMax(S)`.
        fn ldexpFast(x: S, n: i32) S {
            assert(!math.isNan(x));
            assert(1.0 <= @abs(x) and @abs(x) < 2.0);
            assert(n <= 0);
            assert(n > -math.floatExponentMax(S));

            const SBits = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
            const mantissa_bits = math.floatMantissaBits(S);
            const repr = @as(SBits, @bitCast(x));
            return @as(S, @bitCast(repr - (@as(SBits, @intCast(-n)) << mantissa_bits)));
        }

        /// Normalizes the significand and exponent of `x` so that the significand is in the
        /// interval `[1, 2)`, or returns one of the special cases for zero, infinity, or NaN.
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
            const ExpInt = std.meta.Int(.signed, @max(
                @typeInfo(E).int.bits,
                @typeInfo(@TypeOf(exp_offset)).int.bits,
            ) + 1);
            const new_exponent = @as(ExpInt, x.exponent) + @as(ExpInt, exp_offset);
            if (new_exponent > math.maxInt(E)) {
                return if (x.significand > 0) inf else minus_inf;
            }
            if (new_exponent < math.minInt(E)) return zero;
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
            if (exp_diff > math.floatFractionalBits(S) + 1) return lhs;

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
            if (exp_diff > math.floatFractionalBits(S) + 1) return lhs;

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
            if (exponent > math.maxInt(E)) return if (significand > 0) inf else minus_inf;
            if (exponent < math.minInt(E)) return zero;
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
            types[i * es.len + j] = BigFloat(.{
                .Significand = s,
                .Exponent = e,
            });
        }
    }
    return types;
}

test "init" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
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
            if (comptime fitsInt(E, math.floatExponentMin(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S),
                }
            else
                F.zero,
            F.init(math.floatMin(S)),
        );
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S) - math.floatFractionalBits(S),
                }
            else
                F.zero,
            F.init(math.floatTrueMin(S)),
        );

        try testing.expectEqual(F{
            .significand = math.inf(S),
            .exponent = 0,
        }, F.init(math.inf(S)));
        try testing.expect(math.isNan(
            F.init(math.nan(S)).significand,
        ));
    }
}

test "toFloat" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i16, i19, i32 })) |F| {
        const S = @FieldType(F, "significand");

        try testing.expectEqual(@as(S, 0), F.init(0).toFloat(S));
        try testing.expect(math.isNegativeZero(F.init(-0.0).toFloat(S)));
        try testing.expectEqual(@as(S, 1), F.init(1).toFloat(S));
        try testing.expectEqual(@as(S, -521.122), F.init(-521.122).toFloat(S));
        try testing.expectEqual(@as(S, 1e23), F.init(1e23).toFloat(S));
        try testing.expectEqual(@as(S, 1e-23), F.init(1e-23).toFloat(S));
        try testing.expectEqual(@as(S, -1e-45), F.init(-1e-45).toFloat(S));

        try testing.expectEqual(math.inf(S), F.inf.toFloat(S));
        try testing.expectEqual(-math.inf(S), F.minus_inf.toFloat(S));
        try testing.expect(math.isNan(F.nan.toFloat(S)));

        try testing.expectEqual(math.inf(S), F.max_value.toFloat(S));
        try testing.expectEqual(-math.inf(S), F.min_value.toFloat(S));
        try testing.expectEqual(@as(S, 0), F.epsilon.toFloat(S));
    }
}

test "parse" {
    // TODO
    return error.SkipZigTest;
}

test "formatDecimal" {
    // Crazy large numbers were verified by calculating them in log10 form in wolfram alpha
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i53, i64 })) |F| {
        _ = F; // autofix
        // try testing.expectFmt("0", "{d}", .{F.zero});
        // try testing.expectFmt("-0", "{d}", .{F.init(-0.0)});
        // try testing.expectFmt("inf", "{d}", .{F.inf});
        // try testing.expectFmt("-inf", "{d}", .{F.minus_inf});
        // try testing.expectFmt("nan", "{d}", .{F.nan});
        // try testing.expectFmt("-nan", "{d}", .{F.nan.neg()});
        // try testing.expectFmt("     12345     ", "{d:^15}", .{F.init(12345)});
        // try testing.expectFmt(
        //     "-762981672489762158671378613432987234.12",
        //     "{d:.2}",
        //     .{F.init(-762981672489762158671378613432987234.123)},
        // );
        // try testing.expectFmt(
        //     "0.00000000000000000000006126734632",
        //     "{d:.32}",
        //     .{F.init(6.1267346318123e-23)},
        // );
    }
}

test "formatScientific" {
    // Crazy large numbers were verified by calculating them in log10 form in wolfram alpha
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i53, i64 })) |F| {
        try testing.expectFmt("0e0", "{e}", .{F.zero});
        try testing.expectFmt("-0e0", "{e}", .{F.init(-0.0)});
        try testing.expectFmt("inf", "{e}", .{F.inf});
        try testing.expectFmt("-inf", "{e}", .{F.minus_inf});
        try testing.expectFmt("nan", "{e}", .{F.nan});
        try testing.expectFmt("-nan", "{e}", .{F.nan.neg()});
        try testing.expectFmt("1.2345e4", "{e}", .{F.init(12345)});
        try testing.expectFmt(
            "-7.629816727e35",
            "{e:.9}",
            .{F.init(-762981672689762158671378613432987234.123)},
        );
        try testing.expectFmt(
            "     6.1267e-23     ",
            "{e:^20.4}",
            .{F.init(6.1267346318123e-23)},
        );
        try testing.expectFmt(
            "1.2300000000000000000000000000000000000000e0",
            "{e:.40}",
            .{F.init(1.23)},
        );
        try testing.expectFmt(
            "1e0",
            "{e:.0}",
            .{F.init(1.23)},
        );
        try testing.expectFmt(
            "1e1",
            "{e:.0}",
            .{F.init(9.9)},
        );
        try testing.expectFmt(
            "3.00000000000000000000e0",
            "{e:.20}",
            .{F.init(3)},
        );
        try testing.expectFmt("6.969e69696969696969", "{e}", .{F{
            .significand = 1.1936405809786527348488982195707378,
            .exponent = 231528321764877,
        }});

        try testing.expectFmt("0e0", "{E}", .{F.zero});
        try testing.expectFmt("INF", "{E}", .{F.inf});
        try testing.expectFmt("-NAN", "{E}", .{F.nan.neg()});
        try testing.expectFmt("1.2345e4", "{E}", .{F.init(12345)});
    }

    const Tiny = BigFloat(.{
        .Significand = f16,
        .Exponent = i1,
        .bake_render = true,
    });
    try testing.expectFmt("5.4e-1", "{e}", .{Tiny.init(0.54)});
    try testing.expectFmt("-1.999e0", "{e}", .{Tiny.min_value});
    try testing.expectFmt("1.999e0", "{e}", .{Tiny.max_value});
    try testing.expectFmt("-5e-1", "{e}", .{Tiny.epsilon.neg()});

    const Big = BigFloat(.{
        .Significand = f128,
        .Exponent = i1000,
        .bake_render = true,
    });
    try testing.expectFmt("1.2e0", "{e}", .{Big.init(1.2)});
    try testing.expectFmt(
        "-1.74873540141175310457335403747254e1612781156876002906875571082584823201862449472531419045065794674069106383716117052919457727193362579413227119899182498925712743046831971577883865387701955331933942161297844459829891638312939843401472423805414763286757270514940834671439119961744112646874840130936383580584822955244454679295610137107954",
        "{e}",
        .{Big.min_value},
    );
    try testing.expectFmt(
        "1.74873540141175310457335403747254e1612781156876002906875571082584823201862449472531419045065794674069106383716117052919457727193362579413227119899182498925712743046831971577883865387701955331933942161297844459829891638312939843401472423805414763286757270514940834671439119961744112646874840130936383580584822955244454679295610137107954",
        "{e}",
        .{Big.max_value},
    );
    try testing.expectFmt(
        "-5.718418001904122048746596962547775e-1612781156876002906875571082584823201862449472531419045065794674069106383716117052919457727193362579413227119899182498925712743046831971577883865387701955331933942161297844459829891638312939843401472423805414763286757270514940834671439119961744112646874840130936383580584822955244454679295610137107955",
        "{e}",
        .{Big.epsilon.neg()},
    );
}

test "formatHex" {
    // Crazy large numbers were verified by calculating them in log10 form in wolfram alpha
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i53, i64 })) |F| {
        try testing.expectFmt("0x0.0p0", "{x}", .{F.zero});
        try testing.expectFmt("-0x0.0p0", "{x}", .{F.init(-0.0)});
        try testing.expectFmt("inf", "{x}", .{F.inf});
        try testing.expectFmt("-inf", "{x}", .{F.minus_inf});
        try testing.expectFmt("nan", "{x}", .{F.nan});
        try testing.expectFmt("-nan", "{x}", .{F.nan.neg()});
        try testing.expectFmt("0x1.81c8p13", "{x}", .{F.init(12345)});
        try testing.expectFmt(
            "-0x1.25e3cd373p119",
            "{x:.9}",
            .{F.init(-762981672689762158671378613432987234.123)},
        );
        try testing.expectFmt(
            "    0x1.2845p-74    ",
            "{x:^20.4}",
            .{F.init(6.1267346318123e-23)},
        );
        try testing.expectFmt(
            switch (@FieldType(F, "significand")) {
                f64 => "0x1.3ae147ae147ae000000000000000000000000000p0",
                f128 => "0x1.3ae147ae147ae147ae147ae147ae000000000000p0",
                else => unreachable,
            },
            "{x:.40}",
            .{F.init(1.23)},
        );
        try testing.expectFmt(
            "0x1p0",
            "{x:.0}",
            .{F.init(1.23)},
        );
        try testing.expectFmt(
            "0x1p1",
            "{x:.0}",
            .{F.init(1.9)},
        );
        try testing.expectFmt(
            "0x1.80000000000000000000p1",
            "{x:.20}",
            .{F.init(3)},
        );
        try testing.expectFmt(
            switch (@FieldType(F, "significand")) {
                f64 => "0x1.31926dda7b543p231528321764877",
                f128 => "0x1.31926dda7b542cff4177e4f16523p231528321764877",
                else => unreachable,
            },
            "{x}",
            .{F{
                .significand = 1.1936405809786527348488982195707378,
                .exponent = 231528321764877,
            }},
        );

        try testing.expectFmt("0x0.0p0", "{X}", .{F.zero});
        try testing.expectFmt("INF", "{X}", .{F.inf});
        try testing.expectFmt("-NAN", "{X}", .{F.nan.neg()});
        try testing.expectFmt("0x1.81C8p13", "{X}", .{F.init(12345)});
    }

    const Tiny = BigFloat(.{
        .Significand = f16,
        .Exponent = i1,
        .bake_render = true,
    });
    try testing.expectFmt("0x1.148p-1", "{x}", .{Tiny.init(0.54)});
    try testing.expectFmt("-0x1.FFCp0", "{X}", .{Tiny.min_value});
    try testing.expectFmt("0x1.ffcp0", "{x}", .{Tiny.max_value});
    try testing.expectFmt("-0x1p-1", "{x}", .{Tiny.epsilon.neg()});

    const Big = BigFloat(.{
        .Significand = f128,
        .Exponent = i1000,
        .bake_render = true,
    });
    try testing.expectFmt("0x1.3333333333333333333333333333p0", "{x}", .{Big.init(1.2)});
    try testing.expectFmt(
        "-0x1.ffffffffffffffffffffffffffffp5357543035931336604742125245300009052807024058527668037218751941851755255624680612465991894078479290637973364587765734125935726428461570217992288787349287401967283887412115492710537302531185570938977091076523237491790970633699383779582771973038531457285598238843271083830214915826312193418602834034687",
        "{x}",
        .{Big.min_value},
    );
    try testing.expectFmt(
        "0x1.FFFFFFFFFFFFFFFFFFFFFFFFFFFFp5357543035931336604742125245300009052807024058527668037218751941851755255624680612465991894078479290637973364587765734125935726428461570217992288787349287401967283887412115492710537302531185570938977091076523237491790970633699383779582771973038531457285598238843271083830214915826312193418602834034687",
        "{X}",
        .{Big.max_value},
    );
    try testing.expectFmt(
        "-0x1p-5357543035931336604742125245300009052807024058527668037218751941851755255624680612465991894078479290637973364587765734125935726428461570217992288787349287401967283887412115492710537302531185570938977091076523237491790970633699383779582771973038531457285598238843271083830214915826312193418602834034688",
        "{x}",
        .{Big.epsilon.neg()},
    );
}

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

test "isInf" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.inf.isInf());
        try testing.expect(F.minus_inf.isInf());
        try testing.expect(!F.init(0).isInf());
        try testing.expect(!F.init(123).isInf());
        try testing.expect(!F.nan.isInf());
    }
}

test "isNan" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.nan.isNan());
        try testing.expect(!F.inf.isNan());
        try testing.expect(!F.minus_inf.isNan());
        try testing.expect(!F.init(0).isNan());
        try testing.expect(!F.init(123).isNan());
    }
}

test "eql" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(F.init(123).eql(F.init(123)));
        try testing.expect(!F.init(123).eql(F.init(122)));
        try testing.expect(F.init(0).eql(F.init(-0.0)));
        try testing.expect(F.inf.eql(F.inf));
        try testing.expect(!F.inf.eql(F.max_value));
        try testing.expect(!F.inf.eql(F.minus_inf));
        try testing.expect(!F.nan.eql(F.nan));
    }
}

test "approxEqRel" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        // Exactly equal
        try testing.expect(F.init(123).approxEqRel(F.init(123), 1e-6));
        try testing.expect(!F.init(123).approxEqRel(F.init(122), 1e-6));
        try testing.expect(F.init(0).approxEqRel(F.init(-0.0), 1e-6));
        try testing.expect(F.inf.approxEqRel(.inf, 1e-6));
        try testing.expect(!F.inf.approxEqRel(.max_value, 1e-6));
        try testing.expect(!F.inf.approxEqRel(.minus_inf, 1e-6));
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
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(!F.init(0).gt(F.init(0)));
        try testing.expect(!F.init(-0.0).gt(F.init(0)));
        try testing.expect(!F.init(0).gt(F.init(-0.0)));
        try testing.expect(!F.init(-0.0).gt(F.init(-0.0)));

        try testing.expect(F.init(123).gt(F.init(122)));
        try testing.expect(!F.init(123).gt(F.init(123)));
        try testing.expect(!F.init(123).gt(F.init(124)));
        try testing.expect(F.init(123).gt(F.init(12)));
        try testing.expect(!F.init(12).gt(F.init(123)));

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

        try testing.expect(!F.inf.gt(F.inf));
        try testing.expect(F.inf.gt(F.minus_inf));
        try testing.expect(!F.minus_inf.gt(F.inf));
        try testing.expect(!F.nan.gt(F.nan));
    }
}

test "lt" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expect(!F.init(0).lt(F.init(0)));
        try testing.expect(!F.init(-0.0).lt(F.init(0)));
        try testing.expect(!F.init(0).lt(F.init(-0.0)));
        try testing.expect(!F.init(-0.0).lt(F.init(-0.0)));

        try testing.expect(!F.init(123).lt(F.init(122)));
        try testing.expect(!F.init(123).lt(F.init(123)));
        try testing.expect(F.init(123).lt(F.init(124)));
        try testing.expect(!F.init(123).lt(F.init(12)));
        try testing.expect(F.init(12).lt(F.init(123)));

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

        try testing.expect(!F.inf.lt(F.inf));
        try testing.expect(!F.inf.lt(F.minus_inf));
        try testing.expect(F.minus_inf.lt(F.inf));
        try testing.expect(!F.nan.lt(F.nan));
    }
}

test "abs" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(F.init(123).abs(), F.init(123));
        try testing.expectEqual(F.init(-123).abs(), F.init(123));
        try testing.expectEqual(F.init(0).abs(), F.init(0));
        try testing.expectEqual(F.minus_inf.abs(), F.inf);
    }
}

test "neg" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(F.init(123).neg(), F.init(-123));
        try testing.expectEqual(F.init(-123).neg(), F.init(123));
        try testing.expectEqual(F.init(0).neg(), F.init(-0.0));
        try testing.expectEqual(F.minus_inf.neg(), F.inf);
    }
}

test "normalize" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        const S = @FieldType(F, "significand");
        const E = @FieldType(F, "exponent");

        try testing.expectEqual(F{
            .significand = 1,
            .exponent = 1,
        }, F.normalize(.{ .significand = 2, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = -123.0 / 64.0,
            .exponent = 6,
        }, F.normalize(.{ .significand = -123, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = 0.0043 * 256.0,
            .exponent = -8,
        }, F.normalize(.{ .significand = 0.0043, .exponent = 0 }));
        try testing.expectEqual(F{
            .significand = 0,
            .exponent = 0,
        }, F.normalize(.{ .significand = 0, .exponent = 0 }));

        @setEvalBranchQuota(10_000);
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S),
                }
            else
                F.zero,
            F.normalize(.{ .significand = math.floatMin(S), .exponent = 0 }),
        );
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S) - math.floatFractionalBits(S),
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
        try testing.expectEqual(0, F.floatExponent(1));
        try testing.expectEqual(-1, F.floatExponent(0.6));
        try testing.expectEqual(1, F.floatExponent(-2.0));
        try testing.expectEqual(119, F.floatExponent(1e36));

        try testing.expectEqual(-133, F.floatExponent(1e-40));
        try testing.expectEqual(-133, F.floatExponent(-1e-40));
    }
}

test "add" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{i11})) |F| {
        try testing.expectEqual(F.init(0), F.init(0).add(F.init(0)));
        try testing.expectEqual(F.init(1), F.init(1).add(F.init(0)));
        try testing.expectEqual(F.init(444), F.init(123).add(F.init(321)));
        try testing.expectEqual(F.init(0), F.init(123).add(F.init(-123)));
        try testing.expectEqual(F.init(4.75), F.init(1.5).add(F.init(3.25)));
        try testing.expectEqual(F.init(1e38), F.init(1e38).add(F.init(1e-38)));
        {
            const expected = F.init(1e36);
            const actual = F.init(1e38).add(F.init(-0.99e38));
            try testing.expectEqual(expected.exponent, actual.exponent);
            try testing.expect(math.approxEqRel(
                @FieldType(F, "significand"),
                expected.significand,
                actual.significand,
                f64_error_tolerance,
            ));
        }

        try testing.expectEqual(F.inf, F.max_value.add(F.max_value));
        try testing.expectEqual(F.minus_inf, F.min_value.add(F.min_value));
        try testing.expectEqual(F.zero, F.min_value.add(F.max_value));
        try testing.expectEqual(F.zero, F.max_value.add(F.min_value));

        // Only valid when exponent is i11
        try testing.expect(!F.init(0.9e308).isInf());
        try testing.expectEqual(F.inf, F.init(0.9e308).add(F.init(0.9e308)));

        try testing.expectEqual(F.minus_inf, F.init(12).add(F.minus_inf));
        try testing.expect(F.inf.add(F.minus_inf).isNan());
        try testing.expect(F.nan.add(F.init(2)).isNan());
    }
}

test "sub" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{i11})) |F| {
        try testing.expectEqual(F.init(0), F.init(0).sub(F.init(0)));
        try testing.expectEqual(F.init(1), F.init(1).sub(F.init(0)));
        try testing.expectEqual(F.init(-198), F.init(123).sub(F.init(321)));
        try testing.expectEqual(F.init(246), F.init(123).sub(F.init(-123)));
        try testing.expectEqual(F.init(0), F.init(123).sub(F.init(123)));
        try testing.expectEqual(F.init(-1.75), F.init(1.5).sub(F.init(3.25)));
        try testing.expectEqual(F.init(1e38), F.init(1e38).sub(F.init(1e-38)));
        {
            const expected = F.init(1e36);
            const actual = F.init(1e38).sub(F.init(0.99e38));
            try testing.expectEqual(expected.exponent, actual.exponent);
            try testing.expect(math.approxEqRel(
                @FieldType(F, "significand"),
                expected.significand,
                actual.significand,
                f64_error_tolerance,
            ));
        }

        try testing.expectEqual(F.zero, F.max_value.sub(F.max_value));
        try testing.expectEqual(F.zero, F.min_value.sub(F.min_value));
        try testing.expectEqual(F.minus_inf, F.min_value.sub(F.max_value));
        try testing.expectEqual(F.inf, F.max_value.sub(F.min_value));

        // Only valid when exponent is i11
        try testing.expect(!F.init(0.9e308).isInf());
        try testing.expectEqual(F.inf, F.init(0.9e308).sub(F.init(-0.9e308)));

        try testing.expectEqual(F.inf, F.init(12).sub(F.minus_inf));
        try testing.expect(F.inf.sub(F.inf).isNan());
        try testing.expect(F.nan.sub(F.init(2)).isNan());
    }
}

test "mul" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i23, i64 })) |F| {
        try testing.expectEqual(F.init(0), F.init(0).mul(F.init(0)));
        try testing.expectEqual(F.init(0), F.init(1).mul(F.init(0)));
        try testing.expectEqual(F.init(3.5), F.init(1).mul(F.init(3.5)));
        try testing.expectEqual(F.init(39483), F.init(123).mul(F.init(321)));
        try testing.expectEqual(F.init(4.875), F.init(1.5).mul(F.init(3.25)));
        try testing.expectEqual(F.init(-151782), F.init(123).mul(F.init(-1234)));
        try testing.expect(F.init(3.74496).approxEqRel(
            F.init(-0.83).mul(F.init(-4.512)),
            f64_error_tolerance,
        ));
        try testing.expect(F.init(1).approxEqRel(
            F.init(1e38).mul(F.init(1e-38)),
            f64_error_tolerance,
        ));
        try testing.expect(
            (F{
                .significand = 0.89117166164618254333829281056332,
                .exponent = 2045,
            }).approxEqRel(F.init(0.6e308).mul(F.init(0.6e308)), f64_error_tolerance),
        );

        try testing.expectEqual(F.minus_inf, F.inf.mul(F.minus_inf));
        try testing.expectEqual(F.inf, F.inf.mul(F.inf));
        try testing.expectEqual(F.inf, F.inf.mul(F.init(1)));
        try testing.expect(F.inf.mul(F.init(0)).isNan());
        try testing.expect(F.inf.mul(F.nan).isNan());
        try testing.expect(F.nan.mul(F.init(2)).isNan());
    }
}
