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

        pub fn init(x: anytype) Self {
            const T = @TypeOf(x);
            switch (@typeInfo(T)) {
                .int, .comptime_int => {
                    if (x == 0) return .{ .significand = 0, .exponent = 0 };

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
                    if (fr.significand == 0 or exponent < math.minInt(E))
                        return if (math.signbit(fr.significand))
                            .{ .significand = -0.0, .exponent = 0 }
                        else
                            .{ .significand = 0.0, .exponent = 0 };
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

        pub fn format(self: Self, writer: *Writer) Writer.Error!void {
            // TODO: change the numbers to follow std when this PR is merged.
            // https://github.com/ziglang/zig/pull/22971#issuecomment-2676157243
            const decimal_min: Self = comptime .init(1e-6);
            const decimal_max: Self = comptime .init(1e15);
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

            if (x.significand == 0) return .init(0);

            const exp_offset = floatExponent(x.significand);
            const ExpInt = std.meta.Int(.signed, @max(
                @typeInfo(E).int.bits,
                @typeInfo(@TypeOf(exp_offset)).int.bits,
            ) + 1);
            const new_exponent = @as(ExpInt, x.exponent) + @as(ExpInt, exp_offset);
            if (new_exponent > math.maxInt(E)) {
                return if (x.significand > 0) inf else minus_inf;
            }
            if (new_exponent < math.minInt(E)) return .init(0);
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
            if (significand == 0) return .init(0);

            const ExpInt = std.meta.Int(.signed, @max(32, @typeInfo(E).int.bits) + 2);
            const exp_offset = floatExponent(significand);
            const exponent = @as(ExpInt, lhs.exponent) + @as(ExpInt, rhs.exponent) + exp_offset;
            if (exponent > math.maxInt(E)) return if (significand > 0) inf else minus_inf;
            if (exponent < math.minInt(E)) return .init(0);
            return .{
                .significand = math.ldexp(significand, -exp_offset),
                .exponent = @intCast(exponent),
            };
        }

        /// Returns `base` raised to the power of `power`.
        pub fn powi(base: Self, power: E) Self {
            if (!math.isFinite(base.significand) or base.significand == 0) return base;
            if (power == 0) return .init(1);
            if (base.exponent == math.minInt(E)) return .init(0);

            if (power < 0) {
                const inverse: Self = if (base.significand == 1)
                    .{
                        .significand = 1,
                        .exponent = -base.exponent,
                    }
                else
                    .{
                        .significand = 2 / base.significand,
                        .exponent = -base.exponent - 1,
                    };
                // -power can overflow
                const powered = powi(inverse, -(power + 1));
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
        try testing.expectEqual(F.init(0), F.min_value.add(F.max_value));
        try testing.expectEqual(F.init(0), F.max_value.add(F.min_value));

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

        try testing.expectEqual(F.init(0), F.max_value.sub(F.max_value));
        try testing.expectEqual(F.init(0), F.min_value.sub(F.min_value));
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

test "powi" {
    inline for (bigFloatTypes(&.{ f64, f80, f128 }, &.{ i31, i64 })) |F| {
        try testing.expectEqual(
            F{ .significand = 1, .exponent = 100_000_000 },
            F.init(2).powi(100_000_000),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = -100_000_000 },
            F.init(2).powi(-100_000_000),
        );
        try testing.expect(
            (F{
                .significand = 1.49613410531792190857444461471459362655878067016602,
                .exponent = 54,
            }).approxEqRel(F.init(23.4).powi(12), f64_error_tolerance),
        );
        try testing.expect(
            (F{
                .significand = 1.33677856342631050962781381054910058599950329555425,
                .exponent = -55,
            }).approxEqRel(F.init(23.4).powi(-12), f64_error_tolerance),
        );
        try testing.expect(
            (F{
                .significand = 1.57458481244942599134145454282680092718933815099252,
                .exponent = 561535380,
            }).approxEqRel(F.init(23.4).powi(123456789), f64_error_tolerance),
        );
        try testing.expect(
            (F{
                .significand = 1.27017610241572039626252280056345120742834904876593,
                .exponent = -561535381,
            }).approxEqRel(F.init(23.4).powi(-123456789), f64_error_tolerance),
        );
        try testing.expect(
            (F{
                .significand = 1.01242220137619004670372360398620690451219085629840,
                .exponent = 0,
            }).approxEqRel(F.init(1.000_000_000_1).powi(123456789), f64_error_tolerance),
        );
        try testing.expect(
            (F{
                .significand = 1.97546043269437495036060064456721714137725455710222,
                .exponent = -1,
            }).approxEqRel(F.init(1.000_000_000_1).powi(-123456789), f64_error_tolerance),
        );

        try testing.expectEqual(
            F.init(1),
            F.init(1).powi(1),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1).powi(0),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1).powi(100_000_000),
        );
        try testing.expectEqual(
            F.init(1.23),
            F.init(1.23).powi(1),
        );
        try testing.expectEqual(
            F.init(1.0 / 1.23),
            F.init(1.23).powi(-1),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(1.23).powi(0),
        );
        try testing.expectEqual(
            F.init(1),
            F.init(-1.23).powi(0),
        );

        try testing.expectEqual(
            F.inf,
            F.init(100).powi(math.maxInt(@FieldType(F, "exponent"))),
        );
        try testing.expectEqual(
            F.inf,
            F.min_value.powi(2),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = math.maxInt(@FieldType(F, "exponent")) },
            F.init(2).powi(math.maxInt(@FieldType(F, "exponent"))),
        );
        try testing.expectEqual(
            F{ .significand = 1, .exponent = math.minInt(@FieldType(F, "exponent")) },
            F.init(2).powi(math.minInt(@FieldType(F, "exponent"))),
        );
        try testing.expectEqual(
            F.init(0),
            F.epsilon.powi(2),
        );
        try testing.expectEqual(
            F.epsilon,
            F.epsilon.powi(1),
        );
        try testing.expectEqual(
            F.epsilon,
            F.epsilon.powi(-1),
        );

        try testing.expectEqual(
            F.inf,
            F.init(100).powi(math.maxInt(@FieldType(F, "exponent"))),
        );
        try testing.expectEqual(
            F.inf,
            F.inf.powi(3),
        );
        try testing.expectEqual(
            F.minus_inf,
            F.minus_inf.powi(3),
        );
        try testing.expectEqual(
            F.inf,
            F.minus_inf.powi(2),
        );
        try testing.expectEqual(
            F.init(1),
            F.inf.powi(0),
        );
        try testing.expectEqual(
            F.init(1),
            F.minus_inf.powi(0),
        );
    }
}
