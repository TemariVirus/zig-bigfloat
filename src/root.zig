const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const testing = std.testing;
const Writer = std.Io.Writer;

pub const Options = struct {
    /// The floating-point type used for the significand.
    Significand: type,
    /// The signed integer type used for the exponent.
    Exponent: type,
    /// Whether to bake constants used for rendering decimal representations more quickly.
    ///
    /// This should only be disabled to increase compilation speed.
    /// Binary sizes are smaller when this is enabled as the baked constants
    /// take up less space than the code for generating them at runtime.
    bake_render: bool = @import("builtin").mode != .Debug,
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
        pub const inf: Self =        .{ .significand = math.inf(S),              .exponent = 0 };
        pub const minus_inf: Self =  .{ .significand = -math.inf(S),             .exponent = 0 };
        pub const nan: Self =        .{ .significand = math.nan(S),              .exponent = 0 };
        /// Largest value smaller than `inf`.
        pub const max_value: Self =  .{ .significand = math.nextAfter(S, 2, 0),  .exponent = math.maxInt(E) };
        /// Smallest value larger than `minus_inf`.
        pub const min_value: Self =  .{ .significand = math.nextAfter(S, -2, 0), .exponent = math.maxInt(E) };
        /// Smallest value larger than `0`.
        pub const epsilon: Self =    .{ .significand = 1,                        .exponent = math.minInt(E) };
        // zig fmt: on

        /// Create a new `BigFloat` with the closest representable value to `x`.
        pub fn init(x: anytype) Self {
            const zero: Self = .{ .significand = 0, .exponent = 0 };
            const minus_zero: Self = .{ .significand = -0.0, .exponent = 0 };

            const T = @TypeOf(x);
            switch (@typeInfo(T)) {
                .int => |info| {
                    if (x == 0) return zero;

                    const Unsigned = std.meta.Int(.unsigned, info.bits);
                    // Zig ints go up to 65,535 bits, so using i32 is always safe
                    var exponent: i32 = math.log2_int(Unsigned, @abs(x));

                    const shift = @max(0, exponent - math.floatFractionalBits(S));
                    const frac_mask = (@as(Unsigned, 1) << @intCast(shift)) - 1;
                    const half = frac_mask >> 1;
                    const frac = @as(Unsigned, @bitCast(x)) & frac_mask;
                    // Ties are rounded away from zero
                    const round_up: T = @intFromBool(frac > half);

                    // Bit shift to ensure x fits in the range of S
                    var significand: S = @floatFromInt((x >> @intCast(shift)) + round_up);
                    significand = math.ldexp(significand, shift - exponent);
                    if (significand == 2) {
                        significand = 1;
                        exponent += 1;
                    }
                    if (exponent > math.maxInt(E)) return if (x > 0) inf else minus_inf;

                    return .{
                        .significand = significand,
                        .exponent = @intCast(exponent),
                    };
                },
                .comptime_int => {
                    if (x == 0) return zero;
                    const exponent = math.log2(@abs(x));
                    const Int = std.meta.Int(.signed, exponent + 2);
                    return comptime init(@as(Int, x));
                },
                .float => {
                    const fr = math.frexp(x);
                    const significand, const exponent = blk: {
                        var significand = math.lossyCast(S, 2 * fr.significand);
                        var exponent = fr.exponent - 1;
                        if (significand == 2) {
                            significand = 1;
                            exponent += 1;
                        }
                        break :blk .{ significand, exponent };
                    };

                    if (math.isNan(significand)) return nan;
                    if (math.isInf(significand)) return .{ .significand = significand, .exponent = inf.exponent };
                    if (significand == 0 or exponent < math.minInt(E)) {
                        return if (math.signbit(significand)) minus_zero else zero;
                    }
                    if (exponent > math.maxInt(E)) return .{ .significand = significand, .exponent = inf.exponent };

                    return .{
                        .significand = significand,
                        .exponent = @intCast(exponent),
                    };
                },
                .comptime_float => {
                    // comptime_float internally is a f128; this preserves precision.
                    return comptime init(@as(f128, x));
                },
                else => @compileError("x must be an int or float"),
            }
        }

        /// Converts `self` to the closest representable value of `FloatT`.
        pub fn toFloat(self: Self, FloatT: type) FloatT {
            comptime assert(@typeInfo(FloatT) == .float);

            const f: FloatT = @floatCast(self.significand);
            return math.ldexp(f, @intCast(math.clamp(
                self.exponent,
                math.minInt(i32),
                @as(i32, math.maxInt(i32)),
            )));
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

        /// Returns the maximum buffer size required to format a `BigFloat` with the given options.
        pub fn maxFormatLength(options: std.fmt.Number) usize {
            const e_bits: comptime_int = @typeInfo(E).int.bits;
            const width = switch (options.mode) {
                .decimal =>
                // 2^(e_bits - 3) < log10(2) * 2^(e_bits - 1) < 2^(e_bits - 2)
                if (e_bits - 3 >= @typeInfo(usize).int.bits)
                    math.maxInt(usize)
                else
                    // When p is null, the longest value is `epsilon`.
                    // When p is non-null, the longest value is `max_value`.
                    1 + // Negative sign
                        1 + // Decimal point
                        // Leading zeros when p is null. Otherwise, the integer part.
                        @as(usize, @intFromFloat(@ceil(@log10(2.0) * math.ldexp(@as(f64, 1.0), e_bits - 1)))) +
                        // Non-zero digits when p is null. Otherwise, the fractional part.
                        (if (options.precision) |p| p else Decimal.maxDigitCount),
                .scientific => 1 + // Negative sign
                    1 + // Decimal point
                    (if (options.precision) |p| p + 1 else Decimal.maxDigitCount) + // Significand
                    1 + // 'e'
                    Decimal.maxExponentDigitCount,
                .binary, .octal => @panic("TODO"),
                .hex => 1 + // Negative sign
                    2 + // '0x'
                    1 + // Integer part
                    1 + // Hex point
                    (if (options.precision) |p|
                        p
                    else
                        (math.floatFractionalBits(S) + 3) / 4) + // Fractional part
                    1 + // 'p'
                    Decimal.maxExponentDigitCount,
            };
            // The longest special cases have length 4 (-inf, -nan)
            return @max(width, 4, options.width orelse 0);
        }

        /// The default formatting function. Called when using the `{f}` format specifier.
        pub fn format(self: Self, writer: *Writer) Writer.Error!void {
            // TODO: change the numbers to follow std when this PR is merged.
            // https://github.com/ziglang/zig/pull/22971#issuecomment-2676157243
            const decimal_min: Self = .init(1e-6);
            const decimal_max: Self = .init(1e15);
            if (self.significand != 0 and (self.abs().lt(decimal_min) or self.abs().gte(decimal_max))) {
                return self.formatNumber(writer, .{ .mode = .scientific });
            }
            return self.formatNumber(writer, .{ .mode = .decimal });
        }

        fn calculatePadding(len: usize, width: usize, alignment: std.fmt.Alignment) struct { usize, usize } {
            const padding = @max(len, width) - len;
            const left_padding = switch (alignment) {
                .left => 0,
                .center => padding / 2,
                .right => padding,
            };
            const right_padding = switch (alignment) {
                .left => padding,
                .center => (padding + 1) / 2,
                .right => 0,
            };
            return .{ left_padding, right_padding };
        }

        /// Formats `self` according to `options`. It is recommended to use a format string instead.
        pub fn formatNumber(self: Self, writer: *Writer, options: std.fmt.Number) Writer.Error!void {
            if (options.width == null) return formatNumberNoWidth(self, writer, options);

            // If possible, use writer's buffer to align without printing twice.
            const remaining_capacity = writer.buffer.len - writer.end;
            if (remaining_capacity >= maxFormatLength(options)) {
                const start = writer.end;
                try formatNumberNoWidth(self, writer, options);
                const len = writer.end - start;
                const left_padding, const right_padding = calculatePadding(len, options.width.?, options.alignment);
                if (left_padding != 0) {
                    @memmove(writer.buffer[start + left_padding ..][0..len], writer.buffer[start..writer.end]);
                }
                @memset(writer.buffer[start..][0..left_padding], options.fill);
                @memset(writer.buffer[start + left_padding + len ..][0..right_padding], options.fill);
                writer.end += left_padding + right_padding;
                return;
            }

            var discard_writer: Writer.Discarding = .init(&.{});
            formatNumberNoWidth(self, &discard_writer.writer, options) catch unreachable;
            const len: usize = @intCast(discard_writer.fullCount());

            const left_padding, const right_padding = calculatePadding(len, options.width.?, options.alignment);
            try writer.splatByteAll(options.fill, left_padding);
            try formatNumberNoWidth(self, writer, options);
            try writer.splatByteAll(options.fill, right_padding);
        }

        /// Only formats special cases (nan, inf).
        /// Returns true if a special case was formatted.
        /// Otherwise, returns false and nothing is written to `writer`.
        pub fn formatSpecial(self: Self, writer: *Writer, case: std.fmt.Case) Writer.Error!bool {
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

        fn formatNumberNoWidth(self: Self, writer: *Writer, options: std.fmt.Number) Writer.Error!void {
            if (math.signbit(self.significand)) try writer.writeByte('-');
            if (try formatSpecial(self, writer, options.case)) return;

            switch (options.mode) {
                .decimal => try formatDecimal(self.abs(), writer, options.precision),
                .scientific => try formatScientific(self.abs(), writer, options.precision),
                .binary, .octal => @panic("TODO"),
                .hex => try formatHex(self.abs(), writer, options.case, options.precision),
            }
        }

        /// Formats the decimal expansion of `self`. Called when using the `{d}` format specifier.
        ///
        /// Example: 123.45
        pub fn formatDecimal(self: Self, writer: *Writer, precision: ?usize) Writer.Error!void {
            if (self.significand == 0) {
                try writer.writeByte('0');
                if (precision) |p| {
                    try writer.writeByte('.');
                    try writer.splatByteAll('0', p);
                }
                return;
            }

            assert(self.significand > 0);
            assert(math.isNormal(self.significand));

            const decimal = if (precision) |p| blk: {
                const d = self.toDecimal();
                if (-d.exponent > p +| d.digitCount()) {
                    try writer.writeAll("0.");
                    return writer.splatByteAll('0', p);
                }
                const UsizePlus1 = std.meta.Int(.unsigned, 1 + @typeInfo(usize).int.bits);
                break :blk d.round(@intCast(math.clamp(
                    @as(UsizePlus1, @intCast(@max(0, -d.exponent))) -| p,
                    0,
                    d.digitCount(),
                )));
            } else self.toDecimal().removeTrailingZeros();

            const digits_str = blk: {
                var buf: [Decimal.maxDigitCount]u8 = undefined;
                var digit_writer = std.Io.Writer.fixed(&buf);
                digit_writer.print("{d}", .{decimal.digits}) catch unreachable;
                break :blk digit_writer.buffered();
            };

            const DP = std.meta.Int(.signed, 1 + @max(1 + @typeInfo(usize).int.bits, @typeInfo(@TypeOf(decimal.exponent)).int.bits));
            const decimal_point = @as(DP, digits_str.len) + decimal.exponent;
            const decimal_point_clamped: usize = @intCast(math.clamp(decimal_point, 0, digits_str.len));
            // Integer part
            if (decimal_point <= 0) {
                try writer.writeByte('0');
            } else {
                try writer.print("{s}", .{digits_str[0..decimal_point_clamped]});
                if (decimal_point_clamped == digits_str.len) {
                    const zeros = decimal_point - digits_str.len;
                    // Prevent overflow in writer.splatByteAll
                    // If you ever hit this point you should be using scientific notation.
                    if (zeros > math.maxInt(usize) - writer.end) {
                        @branchHint(.cold);
                        return Writer.Error.WriteFailed;
                    }
                    try writer.splatByteAll('0', @intCast(zeros));
                }
            }

            // No fraction part
            if (precision != null and precision.? == 0) return;
            if (precision == null and decimal_point >= digits_str.len) return;

            // Fraction part
            try writer.writeByte('.');
            var left = precision;
            if (decimal_point < 0) {
                const leading_zeros = math.cast(usize, -decimal_point) orelse {
                    @branchHint(.cold);
                    return Writer.Error.WriteFailed;
                };

                const zeros = @min(left orelse math.maxInt(usize), leading_zeros);
                // Prevent overflow in writer.splatByteAll
                // If you ever hit this point you should be using scientific notation.
                if (zeros > math.maxInt(usize) - writer.end) {
                    @branchHint(.cold);
                    return Writer.Error.WriteFailed;
                }
                try writer.splatByteAll('0', zeros);
                if (left) |l| {
                    left = l - @min(l, leading_zeros);
                }
            }
            if (decimal_point_clamped < digits_str.len) {
                try writer.print("{s}", .{digits_str[decimal_point_clamped..][0..@min(
                    left orelse math.maxInt(usize),
                    digits_str.len - decimal_point_clamped,
                )]});
                if (left) |l| {
                    left = l - @min(l, digits_str.len - decimal_point_clamped);
                }
            }
            if (left) |l| {
                try writer.splatByteAll('0', l);
            }
        }

        /// Formats the scientific decimal expansion of `self`. Called when using the `{e}` and `{E}` format specifiers.
        ///
        /// Example: 1.2345e2 (aka 1.2345 * 10^2 = 123.45)
        pub fn formatScientific(self: Self, writer: *Writer, precision: ?usize) Writer.Error!void {
            if (self.significand == 0) {
                try writer.writeByte('0');
                if (precision) |p| {
                    try writer.writeByte('.');
                    try writer.splatByteAll('0', p);
                }
                return writer.writeAll("e0");
            }

            assert(self.significand > 0);
            assert(math.isNormal(self.significand));

            const decimal = if (precision) |p| blk: {
                const d = self.toDecimal();
                break :blk d.round(d.digitCount() - @min(p + 1, d.digitCount()));
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

        /// Formats the scientific hexadecimal expansion of `self`. Called when using the `{x}` and `{X}` format specifiers.
        ///
        /// Example: 0x1.eddp6 (aka 1.928955078125 * 2^6 ≈ 123.45)
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

        /// Returns -1, 0, or 1.
        pub fn sign(self: Self) S {
            return math.sign(self.significand);
        }

        /// Returns whether `self` is negative or negative 0.
        pub fn signBit(self: Self) bool {
            return math.signbit(self.significand);
        }

        /// Returns a value with the magnitude of `magnitude` and the sign of `_sign`.
        pub fn copysign(magnitude: Self, _sign: S) Self {
            return .{
                .significand = math.copysign(magnitude.significand, _sign),
                .exponent = magnitude.exponent,
            };
        }

        /// Returns whether `self` is an infinity, ignoring sign.
        pub fn isInf(self: Self) bool {
            return math.isInf(self.significand);
        }

        /// Returns whether `self` is NaN.
        pub fn isNan(self: Self) bool {
            return math.isNan(self.significand);
        }

        /// Returns whether `self` is in canonical form.
        ///
        /// - For +0, -0, +inf, -inf, and nan, the exponent must be 0.
        /// - For all other values, @abs(significand) must be in the interval [1, 2).
        pub fn isCanonical(self: Self) bool {
            if (!math.isFinite(self.significand) or self.significand == 0) {
                return self.exponent == 0;
            } else {
                return 1.0 <= @abs(self.significand) and @abs(self.significand) < 2.0;
            }
        }

        /// Returns whether `lhs` and `rhs` have equal value.
        ///
        /// NaN values are never considered equal to any value.
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

        /// Returns whether `lhs` is greater than `rhs`.
        ///
        /// This function always returns `false` if either `lhs` or `rhs` is NaN.
        pub fn gt(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand > rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent < rhs.exponent else lhs.exponent > rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand > rhs.significand);
        }

        /// Returns whether `lhs` is greater than or euqal to `rhs`.
        ///
        /// This function always returns `false` if either `lhs` or `rhs` is NaN.
        pub fn gte(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand >= rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent < rhs.exponent else lhs.exponent > rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand >= rhs.significand);
        }

        /// Returns whether `lhs` is smaller than `rhs`.
        ///
        /// This function always returns `false` if either `lhs` or `rhs` is NaN.
        pub fn lt(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand < rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent > rhs.exponent else lhs.exponent < rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand < rhs.significand);
        }

        /// Returns whether `lhs` is smaller than or equal to `rhs`.
        ///
        /// This function always returns `false` if either `lhs` or `rhs` is NaN.
        pub fn lte(lhs: Self, rhs: Self) bool {
            if (lhs.sign() != rhs.sign()) {
                return lhs.significand <= rhs.significand;
            }
            const exp_cmp = if (lhs.signBit()) lhs.exponent > rhs.exponent else lhs.exponent < rhs.exponent;
            return exp_cmp or (lhs.exponent == rhs.exponent and lhs.significand <= rhs.significand);
        }

        /// Returns the absolute value of `self`.
        pub fn abs(self: Self) Self {
            return .{
                .significand = @abs(self.significand),
                .exponent = self.exponent,
            };
        }

        /// Returns `-self`.
        pub fn neg(self: Self) Self {
            return .{
                .significand = -self.significand,
                .exponent = self.exponent,
            };
        }

        /// Returns `1 / self`.
        pub fn inv(self: Self) Self {
            if (!math.isFinite(self.significand) or self.significand == 0) {
                @branchHint(.unlikely);
                comptime assert(nan.exponent == inf.exponent);
                comptime assert(init(0.0).exponent == inf.exponent);
                return .{
                    .significand = 1 / self.significand,
                    .exponent = inf.exponent,
                };
            }

            if (@abs(self.significand) == 1) {
                return if (self.exponent == math.minInt(E))
                    inf.copysign(self.significand)
                else
                    .{
                        .significand = self.significand,
                        .exponent = -self.exponent,
                    };
            }

            return .{
                .significand = 2 / self.significand,
                .exponent = -1 - self.exponent,
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
        /// Asserts that `x` is normal.
        /// Asserts that the result is normal.
        fn ldexpFast(x: S, n: i32) S {
            assert(math.isNormal(x));
            {
                const result_exponent = floatExponent(x) +| n;
                // The trick below doesn't work if the result is subnormal or infinity.
                assert(result_exponent <= math.floatExponentMax(S));
                assert(result_exponent >= math.floatExponentMin(S));
            }

            const SBits = std.meta.Int(.signed, @typeInfo(S).float.bits);
            const mantissa_bits = math.floatMantissaBits(S);
            const repr: SBits = @bitCast(x);
            const exp_diff = @as(SBits, @intCast(n)) << mantissa_bits;
            return @bitCast(repr + exp_diff);
        }

        /// Normalizes the significand and exponent of `x` so that the significand is in the
        /// interval `[1, 2)`, or returns one of the special cases for zero, infinity, or NaN.
        /// `-0` is normalized to `0`.
        ///
        /// `normalize` must be called after modifying the significand or exponent of `x` directly.
        pub fn normalize(x: Self) Self {
            comptime assert(nan.exponent == inf.exponent);
            if (!math.isFinite(x.significand)) return .{ .significand = x.significand, .exponent = inf.exponent };
            return normalizeFinite(x);
        }

        /// Performs the same function as `normalize`, but asserts that `x.significand` is finite.
        pub fn normalizeFinite(x: Self) Self {
            assert(math.isFinite(x.significand));

            if (x.significand == 0) return init(0);

            const exp_offset = floatExponent(x.significand);
            const ExpInt = std.meta.Int(.signed, @max(
                @typeInfo(E).int.bits,
                @typeInfo(@TypeOf(exp_offset)).int.bits,
            ) + 1);
            const new_exponent = @as(ExpInt, x.exponent) + @as(ExpInt, exp_offset);
            if (new_exponent > math.maxInt(E)) {
                return inf.copysign(x.significand);
            }
            if (new_exponent < math.minInt(E)) return init(0);
            return .{
                .significand = math.ldexp(x.significand, -exp_offset),
                .exponent = @intCast(new_exponent),
            };
        }

        /// Returns the largest value between `lhs` and `rhs`.
        pub fn max(lhs: Self, rhs: Self) Self {
            return if (lhs.gt(rhs)) lhs else rhs;
        }

        /// Returns the smallest value between `lhs` and `rhs`.
        pub fn min(lhs: Self, rhs: Self) Self {
            return if (lhs.lt(rhs)) lhs else rhs;
        }

        /// Returns `lhs + rhs`.
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

            // The exponent difference is too large, we can just return lhs
            const exp_diff = math.sub(E, lhs.exponent, rhs.exponent) catch return lhs;
            if (exp_diff > math.floatFractionalBits(S) + 1) return lhs;

            const normalized_rhs = ldexpFast(rhs.significand, @intCast(-exp_diff));
            return normalizeFinite(.{
                .significand = lhs.significand + normalized_rhs,
                .exponent = lhs.exponent,
            });
        }

        /// Returns `lhs - rhs`.
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

            // The exponent difference is too large, we can just return lhs
            const exp_diff = math.sub(E, lhs.exponent, rhs.exponent) catch return lhs;
            if (exp_diff > math.floatFractionalBits(S) + 1) return lhs;

            const normalized_rhs = ldexpFast(rhs.significand, @intCast(-exp_diff));
            return normalizeFinite(.{
                .significand = lhs.significand - normalized_rhs,
                .exponent = lhs.exponent,
            });
        }

        /// Returns `lhs * rhs`.
        pub fn mul(lhs: Self, rhs: Self) Self {
            const significand = lhs.significand * rhs.significand;
            if (!math.isFinite(significand) or significand == 0) {
                comptime assert(nan.exponent == inf.exponent);
                comptime assert(nan.exponent == init(0).exponent);
                return .{ .significand = significand, .exponent = 0 };
            }

            const ExpInt = std.meta.Int(.signed, @max(32, @typeInfo(E).int.bits) + 2);
            const exp_offset = floatExponent(significand);
            const exponent = @as(ExpInt, lhs.exponent) + @as(ExpInt, rhs.exponent) + exp_offset;
            if (exponent > math.maxInt(E)) return inf.copysign(significand);
            if (exponent < math.minInt(E)) return init(0);
            return .{
                .significand = math.ldexp(significand, -exp_offset),
                .exponent = @intCast(exponent),
            };
        }

        /// Returns `base` raised to the power of `power`.
        /// Relative error grows logarithmically with respect to `|power|`.
        ///
        /// This function is faster than `pow` but less accurate.
        ///
        /// Special Cases ordered by precedence:
        ///  - pow(nan, y)    = nan
        ///  - pow(x, nan)    = nan
        ///  - pow(x, +-0)    = 1
        ///  - pow(1, y)      = 1
        ///  - pow(-1, +-inf) = 1
        ///  - pow(x, 1)      = x
        ///  - pow(+-0, +inf) = +0
        ///  - pow(+-0, -inf) = +inf
        ///  - pow(-0, y)     = nan for finite non-integer y
        ///  - pow(x, y)      = nan for x < 0 and finite non-integer y
        ///  - pow(+0, y)     = +0 when y > 0
        ///  - pow(+0, y)     = +inf when y < 0
        ///  - pow(-0, y)     = pow(+0, y) when y is an even integer
        ///  - pow(-0, y)     = -pow(+0, y) when y is an odd integer
        ///  - pow(x, +inf)   = +inf when |x| > 1
        ///  - pow(x, +inf)   = +0 when |x| < 1
        ///  - pow(x, -inf)   = +0 when |x| > 1
        ///  - pow(x, -inf)   = +inf when |x| < 1
        ///  - pow(+inf, y)   = +inf when y > 0
        ///  - pow(+inf, y)   = +0 when y < 0
        ///  - pow(-inf, y)   = pow(+inf, y) when y is an even integer
        ///  - pow(-inf, y)   = -pow(+inf, y) when y is an odd integer
        pub fn pow(base: Self, power: Self) Self {
            // x^y = 2^(log2(x) * y)

            if (base.isNan()) {
                @branchHint(.unlikely);
                return nan;
            }
            if (power.significand == 0) return init(1);
            // log2 and exp2 are highly unlikely to round-trip
            if (power.eql(init(1))) return base;
            if (!base.signBit()) return exp2(log2(base).mul(power));
            if (base.eql(.init(-1)) and power.isInf()) return init(1);

            const abs_result = exp2(log2(base.neg()).mul(power));

            const Int: type = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
            const power_repr: Int = @bitCast(power.significand);
            const frac_mask = (@as(Int, 1) << math.floatFractionalBits(S)) - 1;
            const power_mantissa = (power_repr & frac_mask) | (1 << math.floatFractionalBits(S));
            // Number is too big to be represented exactly, assume it is an even integer
            if (power.exponent > math.floatFractionalBits(S) or power.isInf()) {
                return abs_result;
            }

            const binary_point: math.Log2Int(Int) = @intCast(@min(
                math.floatFractionalBits(S) - power.exponent,
                math.floatFractionalBits(S) + 1,
            ));
            const ones_bit: u1 = @truncate(power_mantissa >> binary_point);
            const fraction = power_mantissa & ((@as(Int, 1) << binary_point) - 1);
            if (fraction != 0) {
                @branchHint(.unlikely);
                return nan;
            }
            return if (ones_bit == 1) abs_result.neg() else abs_result;
        }

        /// Returns `base` raised to the power of `power`.
        /// Relative error grows logarithmically with respect to `|power|`.
        ///
        /// This function is slower than `pow` but more accurate.
        ///
        /// Special Cases ordered by precedence:
        ///  - powi(nan, y)  = nan
        ///  - powi(x, 0)    = 1
        ///  - powi(1, y)    = 1
        ///  - powi(x, 1)    = x
        ///  - powi(+0, y)   = +0 when y > 0
        ///  - powi(+0, y)   = +inf when y < 0
        ///  - powi(-0, y)   = powi(+0, y) when y is even
        ///  - powi(-0, y)   = -powi(+0, y) when y is odd
        ///  - powi(+inf, y) = +inf when y > 0
        ///  - powi(+inf, y) = +0 when y < 0
        ///  - powi(-inf, y) = powi(+inf, y) when y is even
        ///  - powi(-inf, y) = -powi(+inf, y) when y is odd
        pub fn powi(base: Self, power: E) Self {
            if (math.isNan(base.significand)) {
                @branchHint(.unlikely);
                return nan;
            }
            if (power == 0) return init(1);

            if (power < 0) {
                const inverse: Self = base.inv();
                // -power can overflow
                const powered = powi(inverse, -1 - power);
                return powered.mul(inverse);
            }

            var result = base;
            var bit = math.log2_int(std.meta.Int(.unsigned, @typeInfo(E).int.bits -| 1), @intCast(power));
            while (bit > 0) {
                result = result.mul(result);
                bit -= 1;
                if (((power >> bit) & 1) == 1) {
                    result = result.mul(base);
                }
            }
            return result;
        }

        /// Returns `2` raised to the power of `self`.
        ///
        /// Special cases:
        ///  - `-0, 0 => 1`
        ///  - `+inf  => +inf`
        ///  - `-inf  => 0`
        ///  - `nan   => nan`
        pub fn exp2(self: Self) Self {
            // 2^(s * 2^e) = 2^(i + f) ; 0 <= f < 1
            //             = 2^f * 2^i ; 1 <= s < 2
            // i = floor(s * 2^e)      ; i >= 0

            if (self.isNan()) {
                @branchHint(.unlikely);
                return nan;
            }
            if (self.significand == math.inf(S)) {
                @branchHint(.unlikely);
                return inf;
            }
            if (self.significand < 0) {
                return exp2(self.neg()).inv();
            }
            if (self.exponent < 0 or self.significand == 0) {
                // 0 <= s * 2^e < 1
                // 1 <= 2^(s * 2^e) < 2
                const e = @max(self.exponent, math.minInt(i32));
                const @"2^e" = math.ldexp(@as(S, 1), @intCast(e));
                return .{ .significand = @exp2(self.significand * @"2^e"), .exponent = 0 };
            }
            if (self.exponent >= @typeInfo(E).int.bits) {
                // Result always overflows
                return inf;
            }

            // Enough bits for E and the fractional bits of S
            const SE = std.meta.Int(.unsigned, @typeInfo(E).int.bits + math.floatFractionalBits(S));
            const SMask = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
            const SInt = std.meta.Int(.unsigned, math.floatFractionalBits(S) + 1);
            const FInt = std.meta.Int(.unsigned, math.floatFractionalBits(S));
            const @"2^e" = @as(SE, 1) << @intCast(self.exponent);
            const s = @as(SInt, @truncate(@as(SMask, @bitCast(self.significand)))) | (1 << math.floatFractionalBits(S));
            const exponent = @"2^e" * s;
            if (exponent >> math.floatFractionalBits(S) > math.maxInt(E)) {
                return inf;
            }
            const i: E = @intCast(exponent >> math.floatFractionalBits(S));

            const f_mantissa: FInt = @truncate(exponent);
            const f = math.ldexp(@as(S, @floatFromInt(f_mantissa)), -math.floatFractionalBits(S));
            return .{ .significand = @exp2(f), .exponent = i };
        }

        /// Returns the base-2 logarithm of `self`.
        ///
        /// Special cases:
        ///  - `< 0   => nan`
        ///  - `-0, 0 => -inf`
        ///  - `+inf  => +inf`
        ///  - `nan   => nan`
        pub fn log2(self: Self) Self {
            // log2(s * 2^e) = log2(s) + e

            // Result always fits in the range of S
            if (math.minInt(E) >= -math.floatMax(S)) {
                return init(@log2(self.significand) + @as(S, @floatFromInt(self.exponent)));
            }
            // Result always fits in the range of f64
            if (math.minInt(E) >= -math.floatMax(f64) and @typeInfo(S).float.bits <= 64) {
                const s: f64 = @log2(self.significand);
                const e: f64 = @floatFromInt(self.exponent);
                return init(s + e);
            }
            return init(@log2(self.significand)).add(init(self.exponent));
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

fn expectCanonicalPassthrough(actual: anytype) !@TypeOf(actual) {
    try testing.expect(actual.isCanonical());
    return actual;
}

fn expectApproxEqRel(expected: anytype, actual: anytype, tolerance: comptime_float) !void {
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

fn expectBitwiseEqual(expected: anytype, actual: anytype) !void {
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
    try testing.expectEqual(expected.exponent, actual.exponent);
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
                F.init(0),
            F.init(math.floatMin(S)),
        );
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S)))
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

test "format" {
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i23, i64 })) |F| {
        // Format options are not passed down when using the default format function
        try testing.expectFmt("0", "{f}", .{F.init(0)});
        try testing.expectFmt("-0", "{f}", .{F.init(-0.0)});
        try testing.expectFmt("0", "{f:.5}", .{F.init(0)});
        try testing.expectFmt("inf", "{f}", .{F.inf});
        try testing.expectFmt("-inf", "{f}", .{F.minus_inf});
        try testing.expectFmt("nan", "{f}", .{F.nan});
        try testing.expectFmt("-nan", "{f}", .{F.nan.neg()});
        try testing.expectFmt(
            switch (@FieldType(F, "significand")) {
                f64 => "-7.629816726897621e31",
                f128 => "-7.6298167268976215867137861343298125e31",
                else => unreachable,
            },
            "{f:.9}",
            .{F.init(-76298167268976215867137861343298.123)},
        );
        try testing.expectFmt(
            "-0.0061267346318123",
            "{f:^100.4}",
            .{F.init(-6.1267346318123e-3)},
        );
    }
}

test "formatDecimal" {
    // Crazy large numbers were verified by calculating them in log10 form in wolfram alpha
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i8, i12 })) |F| {
        try testing.expectFmt("0", "{d}", .{F.init(0)});
        try testing.expectFmt("-0", "{d}", .{F.init(-0.0)});
        try testing.expectFmt("0.00000", "{d:.5}", .{F.init(0)});
        try testing.expectFmt("inf", "{d}", .{F.inf});
        try testing.expectFmt("-inf", "{d}", .{F.minus_inf});
        try testing.expectFmt("nan", "{d}", .{F.nan});
        try testing.expectFmt("-nan", "{d}", .{F.nan.neg()});
        try testing.expectFmt("12345", "{d}", .{F.init(12345)});
        try testing.expectFmt(
            switch (@FieldType(F, "significand")) {
                f64 => "-76298167268976210000000000000000.000000000",
                f128 => "-76298167268976215867137861343298.125000000",
                else => unreachable,
            },
            "{d:.9}",
            .{F.init(-76298167268976215867137861343298.123)},
        );
        try testing.expectFmt(
            "      -0.0061       ",
            "{d:^20.4}",
            .{F.init(-6.1267346318123e-3)},
        );
        try testing.expectFmt(
            "1.2300000000000000000000000000000000000000",
            "{d:.40}",
            .{F.init(1.23)},
        );
        try testing.expectFmt(
            "1",
            "{d:.0}",
            .{F.init(1.23)},
        );
        try testing.expectFmt(
            "10",
            "{d:.0}",
            .{F.init(9.9)},
        );
        try testing.expectFmt(
            "0.0000",
            "{d:.4}",
            .{F.init(1e-10)},
        );
        try testing.expectFmt(
            "0.1",
            "{d:.1}",
            .{F.init(0.05)},
        );
        try testing.expectFmt(
            switch (@FieldType(F, "significand")) {
                f64 => "69696969696969700",
                f128 => "69696969696969696.96969696969696969",
                else => unreachable,
            },
            "{d}",
            .{F.init(69696969696969696.96969696969696969)},
        );

        var buf: [1024]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try writer.print("{d:>.1}", .{F.init(256)});
        try testing.expectEqualStrings("256.0", writer.buffered());
    }

    const Tiny = BigFloat(.{
        .Significand = f16,
        .Exponent = i1,
        .bake_render = true,
    });
    try testing.expectFmt("0.54", "{d}", .{Tiny.init(0.54)});
    try testing.expectFmt("-1.999", "{d}", .{Tiny.min_value});
    try testing.expectFmt("1.999", "{d}", .{Tiny.max_value});
    try testing.expectFmt("-0.5", "{d}", .{Tiny.epsilon.neg()});

    const Big = BigFloat(.{
        .Significand = f128,
        .Exponent = i1000,
        .bake_render = true,
    });
    try testing.expectFmt("1.2", "{d}", .{Big.init(1.2)});
    try testing.expectFmt(
        "-10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "{d}",
        .{Big.init(-1e100)},
    );
    try testing.expectFmt(
        "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "{d}",
        .{Big.init(1e100)},
    );
    try testing.expectFmt(
        "-0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
        "{d}",
        .{Big.init(-1e-100)},
    );
}

test "formatScientific" {
    // Crazy large numbers were verified by calculating them in log10 form in wolfram alpha
    inline for (bigFloatTypes(&.{ f64, f128 }, &.{ i53, i64 })) |F| {
        try testing.expectFmt("0e0", "{e}", .{F.init(0)});
        try testing.expectFmt("-0e0", "{e}", .{F.init(-0.0)});
        try testing.expectFmt("0.00000e0", "{e:.5}", .{F.init(0)});
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
            "    -6.1267e-23     ",
            "{e:^20.4}",
            .{F.init(-6.1267346318123e-23)},
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

        try testing.expectFmt("0e0", "{E}", .{F.init(0)});
        try testing.expectFmt("INF", "{E}", .{F.inf});
        try testing.expectFmt("-NAN", "{E}", .{F.nan.neg()});
        try testing.expectFmt("1.2345e4", "{E}", .{F.init(12345)});

        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try writer.print("{e:>0}", .{F.init(256)});
        try testing.expectEqualStrings("2.56e2", writer.buffered());
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
        try testing.expectFmt("0x0.0p0", "{x}", .{F.init(0)});
        try testing.expectFmt("-0x0.0p0", "{x}", .{F.init(-0.0)});
        try testing.expectFmt("0x0.00000p0", "{x:.5}", .{F.init(0)});
        try testing.expectFmt("inf", "{x}", .{F.inf});
        try testing.expectFmt("-inf", "{x}", .{F.minus_inf});
        try testing.expectFmt("nan", "{x}", .{F.nan});
        try testing.expectFmt("-nan", "{x}", .{F.nan.neg()});
        try testing.expectFmt("aaaaaaaa-nan", "{x:a>12}", .{F.nan.neg()});
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

        try testing.expectFmt("0x0.0p0", "{X}", .{F.init(0)});
        try testing.expectFmt("INF", "{X}", .{F.inf});
        try testing.expectFmt("-NAN", "{X}", .{F.nan.neg()});
        try testing.expectFmt("0x1.81C8p13", "{X}", .{F.init(12345)});

        var buf: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        try writer.print("{x:>0}", .{F.init(256)});
        try testing.expectEqualStrings("0x1p8", writer.buffered());
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
        try testing.expect(!F.inf.gt(F.nan));
        try testing.expect(!F.nan.gt(F.inf));
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
        try testing.expect(!F.inf.lt(F.nan));
        try testing.expect(!F.nan.lt(F.inf));
        try testing.expect(!F.nan.lt(F.nan));
    }
}

// TODO: test gte() and lte()

test "abs" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(
            F.init(123),
            try expectCanonicalPassthrough(F.init(123).abs()),
        );
        try testing.expectEqual(
            F.init(123),
            try expectCanonicalPassthrough(F.init(-123).abs()),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(0).abs()),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.abs()),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.minus_inf.abs()),
        );
        try testing.expect(F.nan.abs().isNan());
    }
}

test "neg" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i8, i16, i19, i32 })) |F| {
        try testing.expectEqual(
            F.init(-123),
            try expectCanonicalPassthrough(F.init(123).neg()),
        );
        try testing.expectEqual(
            F.init(123),
            try expectCanonicalPassthrough(F.init(-123).neg()),
        );
        try testing.expectEqual(
            F.init(-0.0),
            try expectCanonicalPassthrough(F.init(0).neg()),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.inf.neg()),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.minus_inf.neg()),
        );
        try testing.expect(F.nan.neg().isNan());
    }
}

test "inv" {
    inline for (bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{ i11, i16, i19, i32 })) |F| {
        try testing.expectEqual(
            F.init(0.5),
            try expectCanonicalPassthrough(F.init(2).inv()),
        );
        try testing.expectEqual(
            F.init(-0.5),
            try expectCanonicalPassthrough(F.init(-2).inv()),
        );
        try testing.expectEqual(
            F.init(4),
            try expectCanonicalPassthrough(F.init(0.25).inv()),
        );
        try testing.expectEqual(
            F.init(-4),
            try expectCanonicalPassthrough(F.init(-0.25).inv()),
        );
        try expectApproxEqRel(
            F.init(4.6853308382384842767325064973825399e-57),
            try expectCanonicalPassthrough(F.init(2.134321e56).inv()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(-3.1362728358050739153222272871181604e32),
            try expectCanonicalPassthrough(F.init(-3.188498107e-33).inv()),
            f64_error_tolerance,
        );

        try testing.expectEqual(
            F{
                .significand = 2 / F.max_value.significand,
                .exponent = -1 - F.max_value.exponent,
            },
            try expectCanonicalPassthrough(F.max_value.inv()),
        );
        try testing.expectEqual(
            F{
                .significand = 2 / F.min_value.significand,
                .exponent = -1 - F.min_value.exponent,
            },
            try expectCanonicalPassthrough(F.min_value.inv()),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.epsilon.inv()),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.epsilon.neg().inv()),
        );

        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.inf.inv()),
        );
        try testing.expectEqual(
            F.init(-0.0),
            try expectCanonicalPassthrough(F.minus_inf.inv()),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.init(0).inv()),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.init(-0.0).inv()),
        );
        try testing.expect(F.nan.inv().isNan());
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
        try testing.expectEqual(F.init(1.545), F.init(1.545).normalize());

        @setEvalBranchQuota(10_000);
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S),
                }
            else
                F.init(0),
            F.normalize(.{ .significand = math.floatMin(S), .exponent = 0 }),
        );
        try testing.expectEqual(
            if (comptime fitsInt(E, math.floatExponentMin(S) - math.floatFractionalBits(S)))
                F{
                    .significand = 1,
                    .exponent = math.floatExponentMin(S) - math.floatFractionalBits(S),
                }
            else
                F.init(0),
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
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(0).add(F.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).add(F.init(0))),
        );
        try testing.expectEqual(
            F.init(444),
            try expectCanonicalPassthrough(F.init(123).add(F.init(321))),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(123).add(F.init(-123))),
        );
        try testing.expectEqual(
            F.init(4.75),
            try expectCanonicalPassthrough(F.init(1.5).add(F.init(3.25))),
        );
        try testing.expectEqual(
            F.init(1e38),
            try expectCanonicalPassthrough(F.init(1e38).add(F.init(1e-38))),
        );
        try expectApproxEqRel(
            F.init(1e36),
            try expectCanonicalPassthrough(F.init(1e38).add(F.init(-0.99e38))),
            f64_error_tolerance,
        );

        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.max_value.add(F.max_value)),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.min_value.add(F.min_value)),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.min_value.add(F.max_value)),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.max_value.add(F.min_value)),
        );

        // Only valid when exponent is i11
        try testing.expect(!F.init(0.9e308).isInf());
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.init(0.9e308).add(F.init(0.9e308))),
        );
        try testing.expectEqual(
            F.init(0.9e308),
            try expectCanonicalPassthrough(F.init(0.9e308).add(F.init(0.9e-308))),
        );

        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.init(12).add(F.minus_inf)),
        );
        try testing.expect(F.inf.add(F.minus_inf).isNan());
        try testing.expect(F.minus_inf.add(F.inf).isNan());
        try testing.expect(F.nan.add(F.init(2)).isNan());
    }
}

test "sub" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{i11})) |F| {
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(0).sub(F.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).sub(F.init(0))),
        );
        try testing.expectEqual(
            F.init(-198),
            try expectCanonicalPassthrough(F.init(123).sub(F.init(321))),
        );
        try testing.expectEqual(
            F.init(246),
            try expectCanonicalPassthrough(F.init(123).sub(F.init(-123))),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(123).sub(F.init(123))),
        );
        try testing.expectEqual(
            F.init(-1.75),
            try expectCanonicalPassthrough(F.init(1.5).sub(F.init(3.25))),
        );
        try testing.expectEqual(
            F.init(1e38),
            try expectCanonicalPassthrough(F.init(1e38).sub(F.init(1e-38))),
        );
        try expectApproxEqRel(
            F.init(1e36),
            try expectCanonicalPassthrough(F.init(1e38).sub(F.init(0.99e38))),
            f64_error_tolerance,
        );

        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.max_value.sub(F.max_value)),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.min_value.sub(F.min_value)),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.min_value.sub(F.max_value)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.max_value.sub(F.min_value)),
        );

        // Only valid when exponent is i11
        try testing.expect(!F.init(0.9e308).isInf());
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.init(0.9e308).sub(F.init(-0.9e308))),
        );
        try testing.expectEqual(
            F.init(0.9e308),
            try expectCanonicalPassthrough(F.init(0.9e308).sub(F.init(0.9e-308))),
        );

        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.init(12).sub(F.minus_inf)),
        );
        try testing.expect(F.inf.sub(F.inf).isNan());
        try testing.expect(F.minus_inf.sub(F.minus_inf).isNan());
        try testing.expect(F.nan.sub(F.init(2)).isNan());
    }
}

test "mul" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i23, i64 })) |F| {
        try expectBitwiseEqual(F.init(0), F.init(0).mul(F.init(0)));
        try expectBitwiseEqual(F.init(-0.0), F.init(-0.0).mul(F.init(0)));
        try expectBitwiseEqual(F.init(-0.0), F.init(0).mul(F.init(-0.0)));
        try expectBitwiseEqual(F.init(0.0), F.init(-0.0).mul(F.init(-0.0)));

        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(1).mul(F.init(0))),
        );
        try testing.expectEqual(
            F.init(3.5),
            try expectCanonicalPassthrough(F.init(1).mul(F.init(3.5))),
        );
        try testing.expectEqual(
            F.init(39483),
            try expectCanonicalPassthrough(F.init(123).mul(F.init(321))),
        );
        try testing.expectEqual(
            F.init(4.875),
            try expectCanonicalPassthrough(F.init(1.5).mul(F.init(3.25))),
        );
        try testing.expectEqual(
            F.init(-151782),
            try expectCanonicalPassthrough(F.init(123).mul(F.init(-1234))),
        );
        try expectApproxEqRel(
            F.init(3.74496),
            try expectCanonicalPassthrough(F.init(-0.83).mul(F.init(-4.512))),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1e38).mul(F.init(1e-38))),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 0.89117166164618254333829281056332, .exponent = 2045 },
            try expectCanonicalPassthrough(F.init(0.6e308).mul(F.init(0.6e308))),
            f64_error_tolerance,
        );

        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.inf.mul(F.minus_inf)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.mul(F.inf)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.mul(F.init(1))),
        );
        try testing.expect(F.inf.mul(F.init(0)).isNan());
        try testing.expect(F.inf.mul(F.nan).isNan());
        try testing.expect(F.nan.mul(F.init(2)).isNan());
    }
}

test "pow" {
    const large_power_tolerance = 1e-7;
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F{ .significand = 1, .exponent = 100_000_000 },
            try expectCanonicalPassthrough(F.init(2).pow(.init(100_000_000))),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = -100_000_000 },
            try expectCanonicalPassthrough(F.init(2).pow(.init(-100_000_000))),
        );
        try expectApproxEqRel(
            F{ .significand = 1.1099157202316952388898221715929617, .exponent = 56 },
            try expectCanonicalPassthrough(F.init(23.4).pow(.init(12.345))),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.8019386188912608618888586275231166, .exponent = -57 },
            try expectCanonicalPassthrough(F.init(23.4).pow(.init(-12.345))),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.5745848124494259913414545428268009, .exponent = 561535380 },
            try expectCanonicalPassthrough(F.init(23.4).pow(.init(123456789))),
            large_power_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.2701761024157203962625228005634512, .exponent = -561535381 },
            try expectCanonicalPassthrough(F.init(23.4).pow(.init(-123456789))),
            large_power_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.0124222013774393757001601268900093, .exponent = 0 },
            try expectCanonicalPassthrough(F.init(1.000_000_000_1).pow(.init(123456789.01234))),
            large_power_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.9754604326919372321867791758592745, .exponent = -1 },
            try expectCanonicalPassthrough(F.init(1.000_000_000_1).pow(.init(-123456789.01234))),
            large_power_tolerance,
        );
        try expectApproxEqRel(
            F.init(1.0 / 1.23),
            try expectCanonicalPassthrough(F.init(1.23).pow(.init(-1))),
            f64_error_tolerance,
        );

        const max_exp = math.maxInt(@FieldType(F, "exponent"));
        const min_exp = math.minInt(@FieldType(F, "exponent"));
        const is_even = @typeInfo(@FieldType(F, "exponent")).int.bits > math.floatMantissaBits(@FieldType(F, "significand"));
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.init(100).pow(.init(max_exp))),
        );
        try testing.expectEqual(
            if (is_even) F.inf else F.minus_inf,
            try expectCanonicalPassthrough(F.init(-100).pow(.init(max_exp))),
        );
        try expectBitwiseEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(100).pow(.init(-max_exp))),
        );
        try expectBitwiseEqual(
            if (is_even) F.init(0.0) else F.init(-0.0),
            try expectCanonicalPassthrough(F.init(-100).pow(.init(-max_exp))),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.min_value.pow(.init(2))),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            try expectCanonicalPassthrough(F.min_value.pow(.init(-212389))),
        );
        if (math.floatFractionalBits(@FieldType(F, "significand")) + 1 >= @typeInfo(@FieldType(F, "exponent")).int.bits) {
            try testing.expectEqual(
                F{ .significand = 1, .exponent = max_exp },
                try expectCanonicalPassthrough(F.init(2).pow(.init(max_exp))),
            );
            try testing.expectEqual(
                F{ .significand = 1, .exponent = min_exp + 1 },
                try expectCanonicalPassthrough(F.init(2).pow(.init(min_exp + 1))),
            );
        }
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.epsilon.pow(.init(2))),
        );
        try testing.expectEqual(
            F.epsilon,
            try expectCanonicalPassthrough(F.epsilon.pow(.init(1))),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.epsilon.pow(.init(-1))),
        );

        // Special cases
        // nan^y = nan
        try testing.expect(F.nan.pow(.init(1.23)).isNan());
        try testing.expect(F.nan.pow(.init(0)).isNan());
        try testing.expect(F.nan.pow(.init(-1)).isNan());

        // x^nan = nan
        try testing.expect(F.init(1.23).pow(.nan).isNan());
        try testing.expect(F.init(0).pow(.nan).isNan());
        try testing.expect(F.init(-0.0).pow(.nan).isNan());
        try testing.expect(F.init(-1).pow(.nan).isNan());

        // x^0 = 1
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1.23).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-1.23).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1.23e123).pow(.init(-0.0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(0).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(0).pow(.init(-0.0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-0.0).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-0.0).pow(.init(-0.0))),
        );

        // 1^y = 1
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).pow(.init(1))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).pow(.init(0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).pow(.init(-0.0))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).pow(.init(100_000_000))),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).pow(.init(-12.3876))),
        );

        // -1^+-inf = 1
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-1).pow(.inf)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-1).pow(.minus_inf)),
        );

        // x^1 = x
        try testing.expectEqual(
            F.init(-1.2),
            try expectCanonicalPassthrough(F.init(-1.2).pow(.init(1))),
        );
        try testing.expectEqual(
            F.init(1.233e-12),
            try expectCanonicalPassthrough(F.init(1.233e-12).pow(.init(1))),
        );
        try testing.expectEqual(
            F.max_value,
            try expectCanonicalPassthrough(F.max_value.pow(.init(1))),
        );
        try testing.expectEqual(
            F.epsilon,
            try expectCanonicalPassthrough(F.epsilon.pow(.init(1))),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.pow(.init(1))),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.minus_inf.pow(.init(1))),
        );

        // +-0^+inf = +0
        try expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.inf),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.inf),
        );

        // +-0^-inf = +inf
        try expectBitwiseEqual(
            F.inf,
            F.init(0).pow(.minus_inf),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(-0.0).pow(.minus_inf),
        );

        // -0^y = nan for finite non-integer y
        try testing.expect(F.init(-0.0).pow(.init(1.5)).isNan());
        try testing.expect(F.init(-0.0).pow(.init(-313.23)).isNan());
        try testing.expect(F.init(-0.0).pow(.init(0.0123)).isNan());

        // x^y = nan for x < 0 and finite non-integer y
        try testing.expect(F.init(-1).pow(.init(1.5)).isNan());
        try testing.expect(F.init(-4.654e12).pow(.init(-313.23)).isNan());
        try testing.expect(F.init(-1.2).pow(.init(0.0123)).isNan());

        // +0^y = +0 when y > 0, +inf when y < 0
        try expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(1.875)),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(0).pow(.init(-1)),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(0).pow(.init(187432)),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(0).pow(.init(-1493874.321)),
        );

        // -0^y = +0^y when y is an even integer
        try expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.init(2)),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(-0.0).pow(.init(-2)),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).pow(.init(187432)),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(-0.0).pow(.init(-1493874)),
        );

        // -0^y = -(+0^y) when y is an odd integer
        try expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).pow(.init(1)),
        );
        try expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).pow(.init(-1)),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).pow(.init(187431)),
        );
        try expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).pow(.init(-1493873)),
        );

        // x^+inf = +inf when |x| > 1
        try expectBitwiseEqual(
            F.inf,
            F.init(1.2).pow(.inf),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(-1.00001).pow(.inf),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(1e30).pow(.inf),
        );

        // x^+inf = +0 when |x| < 1
        try expectBitwiseEqual(
            F.init(0),
            F.init(0.8).pow(.inf),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(-0.99999).pow(.inf),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(1e-30).pow(.inf),
        );

        // x^-inf = +0 when |x| > 1
        try expectBitwiseEqual(
            F.init(0),
            F.init(1.2).pow(.minus_inf),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(-1.00001).pow(.minus_inf),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(1e30).pow(.minus_inf),
        );

        // x^-inf = +inf when |x| < 1
        try expectBitwiseEqual(
            F.inf,
            F.init(0.8).pow(.minus_inf),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(-0.99999).pow(.minus_inf),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(1e-30).pow(.minus_inf),
        );

        // +inf^y = +inf when y > 0, +0 when y < 0
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.pow(.init(1.321))),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.pow(.init(18937210))),
        );
        try expectBitwiseEqual(
            F.init(0.0),
            F.inf.pow(.init(-1)),
        );
        try expectBitwiseEqual(
            F.init(0.0),
            F.inf.pow(.init(-1421987.413)),
        );

        // -inf^y = +inf^y when y is an even integer
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.minus_inf.pow(.init(2))),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.minus_inf.pow(.init(12309874))),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.minus_inf.pow(.init(-2)),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.minus_inf.pow(.init(-123098)),
        );

        // -inf^y = -(+inf^y) when y is an odd integer
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.minus_inf.pow(.init(1))),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.minus_inf.pow(.init(123099))),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.pow(.init(-1)),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.pow(.init(-1230987)),
        );
    }
}

test "powi" {
    const large_power_tolerance = 1e-7;
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F{ .significand = 1, .exponent = 100_000_000 },
            try expectCanonicalPassthrough(F.init(2).powi(100_000_000)),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = -100_000_000 },
            try expectCanonicalPassthrough(F.init(2).powi(-100_000_000)),
        );
        try expectApproxEqRel(
            F{ .significand = 1.4961341053179219085744446147145936, .exponent = 54 },
            try expectCanonicalPassthrough(F.init(23.4).powi(12)),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.3367785634263105096278138105491006, .exponent = -55 },
            try expectCanonicalPassthrough(F.init(23.4).powi(-12)),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.5745848124494259913414545428268009, .exponent = 561535380 },
            try expectCanonicalPassthrough(F.init(23.4).powi(123456789)),
            large_power_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.2701761024157203962625228005634512, .exponent = -561535381 },
            try expectCanonicalPassthrough(F.init(23.4).powi(-123456789)),
            large_power_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.0124222013761900467037236039862069, .exponent = 0 },
            try expectCanonicalPassthrough(F.init(1.000_000_000_1).powi(123456789)),
            large_power_tolerance,
        );
        try expectApproxEqRel(
            F{ .significand = 1.9754604326943749503606006445672171, .exponent = -1 },
            try expectCanonicalPassthrough(F.init(1.000_000_000_1).powi(-123456789)),
            large_power_tolerance,
        );

        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).powi(1)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).powi(100_000_000)),
        );
        try testing.expectEqual(
            F.init(1.23),
            try expectCanonicalPassthrough(F.init(1.23).powi(1)),
        );
        try testing.expectEqual(
            F.init(1.0 / 1.23),
            try expectCanonicalPassthrough(F.init(1.23).powi(-1)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1.23).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-1.23).powi(0)),
        );

        const max_exp = math.maxInt(@FieldType(F, "exponent"));
        const min_exp = math.minInt(@FieldType(F, "exponent"));
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.init(100).powi(max_exp)),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.init(-100).powi(max_exp)),
        );
        try expectBitwiseEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(100).powi(-max_exp)),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            try expectCanonicalPassthrough(F.init(-100).powi(-max_exp)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.min_value.powi(2)),
        );
        try testing.expectEqual(
            F.min_value.inv(),
            try expectCanonicalPassthrough(F.min_value.powi(-1)),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = max_exp },
            try expectCanonicalPassthrough(F.init(2).powi(max_exp)),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = min_exp },
            try expectCanonicalPassthrough(F.init(2).powi(min_exp)),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.epsilon.powi(2)),
        );
        try testing.expectEqual(
            F.epsilon,
            try expectCanonicalPassthrough(F.epsilon.powi(1)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.epsilon.powi(-1)),
        );

        // Special cases
        // nan^y = nan
        try testing.expect(F.nan.powi(123).isNan());
        try testing.expect(F.nan.powi(0).isNan());

        // x^0 = 1
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-1.2).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(0).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.inf.powi(0)),
        );

        // 1^y = 1
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).powi(0)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).powi(1)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).powi(-123876)),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(1).powi(3981)),
        );

        // x^1 = x
        try testing.expectEqual(
            F.init(-1.2),
            try expectCanonicalPassthrough(F.init(-1.2).powi(1)),
        );
        try testing.expectEqual(
            F.init(1.233e-12),
            try expectCanonicalPassthrough(F.init(1.233e-12).powi(1)),
        );
        try testing.expectEqual(
            F.max_value,
            try expectCanonicalPassthrough(F.max_value.powi(1)),
        );
        try testing.expectEqual(
            F.epsilon,
            try expectCanonicalPassthrough(F.epsilon.powi(1)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.powi(1)),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.minus_inf.powi(1)),
        );

        // +0^y = +0 when y > 0, +inf when y < 0
        try expectBitwiseEqual(
            F.init(0),
            F.init(0).powi(1),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(0).powi(-1),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(0).powi(187432),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(0).powi(-1493874),
        );

        // -0^y = +0^y when y is even
        try expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).powi(2),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(-0.0).powi(-2),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.init(-0.0).powi(187432),
        );
        try expectBitwiseEqual(
            F.inf,
            F.init(-0.0).powi(-1493874),
        );

        // -0^y = -(+0^y) when y is odd
        try expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).powi(1),
        );
        try expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).powi(-1),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            F.init(-0.0).powi(187431),
        );
        try expectBitwiseEqual(
            F.minus_inf,
            F.init(-0.0).powi(-1493873),
        );

        // +inf^y = +inf when y > 0, +0 when y < 0
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.powi(1)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.powi(18937210)),
        );
        try expectBitwiseEqual(
            F.init(0.0),
            F.inf.powi(-1),
        );
        try expectBitwiseEqual(
            F.init(0.0),
            F.inf.powi(-1421987),
        );

        // -inf^y = +inf^y when y is even
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.minus_inf.powi(2)),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.minus_inf.powi(12309874)),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.minus_inf.powi(-2),
        );
        try expectBitwiseEqual(
            F.init(0),
            F.minus_inf.powi(-123098),
        );

        // -inf^y = -(+inf^y) when y is odd
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.minus_inf.powi(1)),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.minus_inf.powi(123099)),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.powi(-1),
        );
        try expectBitwiseEqual(
            F.init(-0.0),
            F.minus_inf.powi(-1230987),
        );
    }
}

test "exp2" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F.init(2),
            try expectCanonicalPassthrough(F.init(1).exp2()),
        );
        try testing.expectEqual(
            F.init(1.0 / 2.0),
            try expectCanonicalPassthrough(F.init(-1).exp2()),
        );
        try testing.expectEqual(
            F.init(1024),
            try expectCanonicalPassthrough(F.init(10).exp2()),
        );
        try testing.expectEqual(
            F.init(1.0 / 1024.0),
            try expectCanonicalPassthrough(F.init(-10).exp2()),
        );
        try expectApproxEqRel(
            F.init(2.3456698984637576073197579763422596),
            try expectCanonicalPassthrough(F.init(1.23).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(1.9830380770415906313713607977912150e-4),
            try expectCanonicalPassthrough(F.init(-12.3).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            switch (@FieldType(F, "significand")) {
                f64 => F{ .significand = 1.3195078889668167666275307021103743, .exponent = 907374182 },
                f80 => F{ .significand = 1.3195079107941892571016437098436364, .exponent = 907374182 },
                f128 => F{ .significand = 1.3195079107728942593740019523158827, .exponent = 907374182 },
                else => unreachable,
            },
            try expectCanonicalPassthrough(F.init(9.073741824e8).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            switch (@FieldType(F, "significand")) {
                f64 => F{ .significand = 1.7411010690456652445660984990257473, .exponent = -937374183 },
                f80 => F{ .significand = 1.7411011265781988192919530481853739, .exponent = -937374183 },
                f128 => F{ .significand = 1.7411011265922482782725399850457871, .exponent = -937374183 },
                else => unreachable,
            },
            try expectCanonicalPassthrough(F.init(-9.373741822e8).exp2()),
            f64_error_tolerance,
        );

        try expectApproxEqRel(
            F.init(1.4142135623730950488016887242096981),
            try expectCanonicalPassthrough(F.init(0.5).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(1.0892989912812542821268342891053001),
            try expectCanonicalPassthrough(F.init(0.1234).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(1.0000000000000022250024495974269185),
            try expectCanonicalPassthrough(F.init(3.21e-15).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(0.50034669373129031626878431965192960),
            try expectCanonicalPassthrough(F.init(-0.999).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(0.87055056329612413913627001747974610),
            try expectCanonicalPassthrough(F.init(-0.2).exp2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(0.99999999999999783738079665297297308),
            try expectCanonicalPassthrough(F.init(-3.12e-15).exp2()),
            f64_error_tolerance,
        );

        // Only valid when E is i64 or smaller
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.init(1e19).exp2()),
        );
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(-1e19).exp2()),
        );

        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.min_value.exp2()),
        );
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.max_value.exp2()),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.epsilon.exp2()),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.epsilon.neg().exp2()),
        );

        // -0, 0 => 1
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(-0.0).exp2()),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(0).exp2()),
        );

        // +inf => +inf
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.exp2()),
        );

        // -inf => 0
        try expectBitwiseEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.minus_inf.exp2()),
        );

        // nan => nan
        try testing.expect(F.nan.exp2().isNan());
    }
}

test "log2" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F.init(0),
            try expectCanonicalPassthrough(F.init(1).log2()),
        );
        try testing.expectEqual(
            F.init(1),
            try expectCanonicalPassthrough(F.init(2).log2()),
        );
        try testing.expectEqual(
            F.init(-1),
            try expectCanonicalPassthrough(F.init(1.0 / 2.0).log2()),
        );
        try testing.expectEqual(
            F.init(20),
            try expectCanonicalPassthrough(F.init(1024 * 1024).log2()),
        );
        try testing.expectEqual(
            F.init(-20),
            try expectCanonicalPassthrough(F.init(1.0 / 1024.0 / 1024.0).log2()),
        );

        try expectApproxEqRel(
            F.init(6.9477830262554195105713484746171828),
            try expectCanonicalPassthrough(F.init(123.45).log2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(-3.0180012584066675330396098138509877),
            try expectCanonicalPassthrough(F.init(0.12345).log2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(15171.544267666148357902627905870537),
            try expectCanonicalPassthrough(F.init(1.23e4567).log2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(-15170.946951035019327544869763085553),
            try expectCanonicalPassthrough(F.init(1.23e-4567).log2()),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(123456789.30392683648069481483070962),
            try expectCanonicalPassthrough(
                (F{ .significand = 1.2345, .exponent = 123456789 }).log2(),
            ),
            f64_error_tolerance,
        );
        try expectApproxEqRel(
            F.init(-123456788.69607316351930518516929038),
            try expectCanonicalPassthrough(
                (F{ .significand = 1.2345, .exponent = -123456789 }).log2(),
            ),
            f64_error_tolerance,
        );

        // < 0 => nan
        try testing.expect(F.init(-1).log2().isNan());
        try testing.expect(F.minus_inf.log2().isNan());
        try testing.expect(F.epsilon.neg().log2().isNan());

        // -0, 0 => -inf
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.init(-0.0).log2()),
        );
        try testing.expectEqual(
            F.minus_inf,
            try expectCanonicalPassthrough(F.init(0.0).log2()),
        );

        // +inf => +inf
        try testing.expectEqual(
            F.inf,
            try expectCanonicalPassthrough(F.inf.log2()),
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
    try expectApproxEqRel(
        Small.init(0.2992080183872788182197666346168540),
        try expectCanonicalPassthrough(Small.init(1.23).log2()),
        f16_error_tolerance,
    );
    try testing.expectEqual(
        Small.init(123456),
        try expectCanonicalPassthrough(
            (Small{ .significand = 1, .exponent = 123456 }).log2(),
        ),
    );
    try testing.expectEqual(
        Small.init(-123456),
        try expectCanonicalPassthrough(
            (Small{ .significand = 1, .exponent = -123456 }).log2(),
        ),
    );
    try expectApproxEqRel(
        Small.init(12345678.304006068589101766689691059),
        try expectCanonicalPassthrough(
            (Small{ .significand = 1.2345678, .exponent = 12345678 }).log2(),
        ),
        f16_error_tolerance,
    );
    try expectApproxEqRel(
        Small.init(2147483647.9992953870234106272584284),
        try expectCanonicalPassthrough(Small.max_value.log2()),
        f16_error_tolerance,
    );
    try testing.expectEqual(
        Small.init(-2147483648),
        try expectCanonicalPassthrough(Small.epsilon.log2()),
    );

    // f64 goes up to around 2^1024 before hitting inf
    const Big = BigFloat(.{
        .Significand = f64,
        .Exponent = i1030,
    });
    try expectApproxEqRel(
        Big.init(0.2986583155645151788790713924919448),
        try expectCanonicalPassthrough(Big.init(1.23).log2()),
        f64_error_tolerance,
    );
    try testing.expectEqual(
        Big.init(123456),
        try expectCanonicalPassthrough(
            (Big{ .significand = 1, .exponent = 123456 }).log2(),
        ),
    );
    try testing.expectEqual(
        Big.init(-123456),
        try expectCanonicalPassthrough(
            (Big{ .significand = 1, .exponent = -123456 }).log2(),
        ),
    );
    try expectApproxEqRel(
        Big.init(12345678.304006068589101766689691059),
        try expectCanonicalPassthrough(
            (Big{ .significand = 1.2345678, .exponent = 12345678 }).log2(),
        ),
        f64_error_tolerance,
    );
    try expectApproxEqRel(
        Big.init(5.7526180315594109047337766105248791e309),
        try expectCanonicalPassthrough(Big.max_value.log2()),
        f64_error_tolerance,
    );
    try expectApproxEqRel(
        Big.init(-5.7526180315594109047337766105248791e309),
        try expectCanonicalPassthrough(Big.epsilon.log2()),
        f64_error_tolerance,
    );
}
