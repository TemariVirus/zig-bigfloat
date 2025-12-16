const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const testing = std.testing;
const Writer = std.Io.Writer;

const _exp2 = @import("exp2.zig").exp2;
const _log2 = @import("log2.zig").log2;
const test_utils = @import("test_utils.zig");

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
        pub const inf: Self =       .{ .significand = math.inf(S),             .exponent = 0 };
        pub const nan: Self =       .{ .significand = math.nan(S),             .exponent = 0 };
        /// Largest value smaller than `+inf`.
        /// The smallest value larger than `-inf` is `max_value.neg()`.
        pub const max_value: Self = .{ .significand = math.nextAfter(S, 2, 0), .exponent = math.maxInt(E) };
        /// Smallest value larger than `+0`.
        pub const min_value: Self = .{ .significand = 1,                       .exponent = math.minInt(E) };
        // zig fmt: on

        /// Returns a `BigFloat` with the closest representable value to `x`.
        /// Use `initExact` if you want an exact conversion.
        ///
        /// Special cases:
        ///  - `+-inf => +-inf`
        ///  - `nan   => nan`
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
                    if (exponent > math.maxInt(E)) {
                        return if (x > 0) inf else inf.neg();
                    }

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

        /// Returns a `BigFloat` with the exact same value as `x`.
        /// Use `init` if you want a lossy conversion.
        ///
        /// Special cases:
        ///  - `+-inf => +-inf`
        ///  - `nan   => nan`
        pub fn initExact(x: anytype) ?Self {
            return if (canRepresentExact(x)) .init(x) else null;
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

        // Returns whether `x` can be represented exactly by `BigFloat`.
        pub fn canRepresentExact(x: anytype) bool {
            if (x == 0) return true;

            const T = @TypeOf(x);
            switch (@typeInfo(T)) {
                .int => |info| {
                    const Unsigned = std.meta.Int(.unsigned, info.bits);
                    const log2_x = math.log2_int(Unsigned, @abs(x));
                    const frac_bits_needed = @as(i32, log2_x) - @ctz(x);
                    // Precision check
                    if (frac_bits_needed > math.floatFractionalBits(S)) return false;
                    // Size check
                    return log2_x <= math.maxInt(E);
                },
                .comptime_int => {
                    const exponent = math.log2(@abs(x));
                    const Int = std.meta.Int(.signed, exponent + 2);
                    return comptime canRepresentExact(@as(Int, x));
                },
                .float => {
                    if (math.isInf(x) or math.isNan(x)) return true;
                    const lossy = init(x).toFloat(T);
                    return lossy == x;
                },
                .comptime_float => {
                    // comptime_float internally is a f128
                    return comptime canRepresentExact(@as(f128, x));
                },
                else => @compileError("x must be an int or float"),
            }
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
                    // When p is null, the longest value is `min_value`.
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
        /// - For +-0, +-inf, and nan, the exponent must be 0.
        /// - For all other values, abs(significand) must be in the interval [1, 2).
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
        ///
        /// Special cases:
        ///  - `nan   => nan`
        ///  - `+-0   => +-inf`
        ///  - `+-inf => +-0`
        pub fn inv(self: Self) Self {
            if (!math.isFinite(self.significand) or self.significand == 0) {
                @branchHint(.unlikely);
                comptime assert(nan.exponent == inf.exponent);
                comptime assert(init(0.0).exponent == inf.exponent);
                return .{
                    .significand = 1.0 / self.significand,
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
                .significand = 2.0 / self.significand,
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
        ///
        /// Special cases:
        ///  - `x + nan     => nan`
        ///  - `nan + y     => nan`
        ///  - `+inf + -inf => nan`
        ///  - `-inf + +inf => nan`
        ///  - `x + +-inf   => +-inf` for finite x
        ///  - `+-inf + y   => +-inf` for finite y
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
            if (exp_diff > math.floatFractionalBits(S) + 2) return lhs;

            const normalized_rhs = ldexpFast(rhs.significand, @intCast(-exp_diff));
            return normalizeFinite(.{
                .significand = lhs.significand + normalized_rhs,
                .exponent = lhs.exponent,
            });
        }

        /// Returns `lhs - rhs`.
        ///
        /// Special cases:
        ///  - `x - nan     => nan`
        ///  - `nan - y     => nan`
        ///  - `+inf - +inf => nan`
        ///  - `-inf - -inf => nan`
        ///  - `x - +-inf   => -+inf` for finite x
        ///  - `+-inf - y   => +-inf` for finite y
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
            if (exp_diff > math.floatFractionalBits(S) + 2) return lhs;

            const normalized_rhs = ldexpFast(rhs.significand, @intCast(-exp_diff));
            return normalizeFinite(.{
                .significand = lhs.significand - normalized_rhs,
                .exponent = lhs.exponent,
            });
        }

        /// Returns `lhs * rhs`.
        ///
        /// Special cases:
        ///  - `x * nan     => nan`
        ///  - `nan * y     => nan`
        ///  - `+-inf * +-0 => nan`
        ///  - `+-0 * +-inf => nan`
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
        /// For `|power| >= 1`, relative error is proportional to `log(|power|)`.
        /// For `|power| < 1`, absolute error is approximately `5e-17`.
        ///
        /// This function is faster than `powi` but usually less accurate.
        ///
        /// Special Cases:
        ///  - pow(x, +-0)    = 1
        ///  - pow(+-0, y)    = +-inf  for y an odd integer < 0
        ///  - pow(+-0, −inf) = +inf
        ///  - pow(+-0, +inf) = +0
        ///  - pow(+-0, y)    = +-0    for finite y > 0 an odd integer
        ///  - pow(−1, +-inf) = 1
        ///  - pow(+1, y)     = 1
        ///  - pow(x, +inf)   = +0     for −1 < x < 1
        ///  - pow(x, +inf)   = +inf   for x < −1 or for 1 < x (including +-inf)
        ///  - pow(x, −inf)   = +inf   for −1 < x < 1
        ///  - pow(x, −inf)   = +0     for x < −1 or for 1 < x (including +-inf)
        ///  - pow(+inf, y)   = +0     for a number y < 0
        ///  - pow(+inf, y)   = +inf   for a number y > 0
        ///  - pow(−inf, y)   = −0     for finite y < 0 an odd integer
        ///  - pow(−inf, y)   = −inf   for finite y > 0 an odd integer
        ///  - pow(−inf, y)   = +0     for finite y < 0 and not an odd integer
        ///  - pow(−inf, y)   = +inf   for finite y > 0 and not an odd integer
        ///  - pow(+-0, y)    = +inf   for finite y < 0 and not an odd integer
        ///  - pow(+-0, y)    = +0     for finite y > 0 and not an odd integer
        ///  - pow(x, y)      = nan    for finite x < 0 and finite non-integer y
        ///  - pow(x, 1)      = x
        ///  - pow(nan, y)    = nan    for y != +-0
        ///  - pow(x, nan)    = nan    for x != 1
        pub fn pow(base: Self, power: Self) Self {
            // x^y = 2^(log2(x) * y)

            if (power.significand == 0 or base.eql(init(1))) return init(1);
            // log2 and exp2 are highly unlikely to round-trip
            if (power.eql(init(1))) return base;
            if (base.isNan() or power.isNan()) {
                @branchHint(.unlikely);
                return nan;
            }
            if (power.isInf()) {
                return if (base.eql(init(-1)))
                    init(1)
                else if ((base.significand == 0 or base.exponent < 0) ==
                    math.isPositiveInf(power.significand))
                    init(0)
                else
                    inf;
            }
            if (base.isInf()) {
                return if (math.isNegativeInf(base.significand))
                    pow(init(-0.0), power.neg())
                else if (power.significand < 0)
                    init(0)
                else
                    inf;
            }
            if (!base.signBit()) return exp2(log2(base).mul(power));

            const Int: type = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
            const power_repr: Int = @bitCast(power.significand);
            const frac_mask = (@as(Int, 1) << math.floatFractionalBits(S)) - 1;
            const power_mantissa = (power_repr & frac_mask) | (1 << math.floatFractionalBits(S));
            const binary_point: math.Log2Int(Int) = @intCast(
                math.floatFractionalBits(S) -
                    math.clamp(power.exponent, -1, math.floatFractionalBits(S)),
            );
            const is_odd = ((power_mantissa >> binary_point) & 1 == 1) and
                (power.exponent <= math.floatFractionalBits(S));
            const fraction = power_mantissa & ((@as(Int, 1) << binary_point) - 1);

            if (base.significand != 0 and fraction != 0) {
                return nan;
            }
            const abs_result = exp2(log2(base.neg()).mul(power));
            return if (is_odd and fraction == 0) abs_result.neg() else abs_result;
        }

        /// Returns `base` raised to the power of `power`.
        /// Relative error is proportional to `log(|power|)`.
        ///
        /// This function is slower than `pow` but usually more accurate.
        ///
        /// Special Cases:
        ///  - powi(x, 0)    = 1
        ///  - powi(+-0, n)  = +-inf  for odd n < 0
        ///  - powi(+-0, n)  = +inf   for even n < 0
        ///  - powi(+-0, n)  = +0     for even n > 0
        ///  - powi(+-0, n)  = +-0    for odd n > 0
        ///  - powi(+inf, n) = +inf   for n > 0
        ///  - powi(−inf, n) = −inf   for odd n > 0
        ///  - powi(−inf, n) = +inf   for even n > 0
        ///  - powi(+inf, n) = +0     for n < 0
        ///  - powi(−inf, n) = −0     for odd n < 0
        ///  - powi(−inf, n) = +0     for even n < 0
        ///  - powi(nan, n)  = nan    for n != 0
        pub fn powi(base: Self, power: E) Self {
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
        ///  - `+-0  => 1`
        ///  - `+inf => +inf`
        ///  - `-inf => 0`
        ///  - `nan  => nan`
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
                return .{ .significand = _exp2(self.significand * @"2^e"), .exponent = 0 };
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
            return .{ .significand = _exp2(f), .exponent = i };
        }

        /// Returns the base-2 logarithm of `self`.
        ///
        /// Special cases:
        ///  - `< 0  => nan`
        ///  - `+-0  => -inf`
        ///  - `+inf => +inf`
        ///  - `nan  => nan`
        pub fn log2(self: Self) Self {
            // log2(s * 2^e) = log2(s) + e

            // Use extra bits for accuracy
            const _S = switch (S) {
                f16 => f32,
                f32 => f64,
                else => S,
            };
            // Result always fits in the range of _S
            if (math.minInt(E) >= -math.floatMax(_S)) {
                return init(_log2(@as(_S, self.significand)) + @as(_S, @floatFromInt(self.exponent)));
            }
            // Result always fits in the range of f64
            if (math.minInt(E) >= -math.floatMax(f64) and @typeInfo(S).float.bits <= 64) {
                const s: f64 = @log2(self.significand);
                const e: f64 = @floatFromInt(self.exponent);
                return init(s + e);
            }
            return init(_log2(self.significand)).add(init(self.exponent));
        }
    };
}

test {
    testing.refAllDecls(@This());
    _ = @import("schubfach.zig");

    _ = @import("tests/init.zig");
    _ = @import("tests/to.zig");
    _ = @import("tests/parse.zig");
    _ = @import("tests/format.zig");
    _ = @import("tests/sign.zig");
    _ = @import("tests/isinf.zig");
    _ = @import("tests/isnan.zig");
    _ = @import("tests/normalize.zig");
    _ = @import("tests/unary_ops.zig");
    _ = @import("tests/compare.zig");
    _ = @import("tests/add.zig");
    _ = @import("tests/sub.zig");
    _ = @import("tests/mul.zig");
    _ = @import("tests/exp.zig");
    _ = @import("tests/log.zig");
    _ = @import("tests/pow.zig");

    _ = @import("tests/fuzz.zig");
}

test "floatExponent" {
    inline for (test_utils.bigFloatTypes(&.{ f32, f64, f80, f128 }, &.{i32})) |F| {
        try testing.expectEqual(0, F.floatExponent(1));
        try testing.expectEqual(-1, F.floatExponent(0.6));
        try testing.expectEqual(1, F.floatExponent(-2.0));
        try testing.expectEqual(119, F.floatExponent(1e36));

        try testing.expectEqual(-133, F.floatExponent(1e-40));
        try testing.expectEqual(-133, F.floatExponent(-1e-40));
    }
}
