const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const Writer = std.Io.Writer;

/// Calculates the left and right padding needed to format a number with string length `len`.
pub fn calculatePadding(len: usize, options: std.fmt.Number) struct { usize, usize } {
    const padding = @max(len, options.width.?) - len;
    const left_padding = switch (options.alignment) {
        .left => 0,
        .center => padding / 2,
        .right => padding,
    };
    const right_padding = switch (options.alignment) {
        .left => padding,
        .center => (padding + 1) / 2,
        .right => 0,
    };
    return .{ left_padding, right_padding };
}

/// Formats `self` according to `options`. `options.width` is ignored.
pub fn formatNoWidth(
    comptime maxDigitCount: usize,
    bf: anytype,
    writer: *Writer,
    options: std.fmt.Number,
) Writer.Error!void {
    if (math.signbit(bf.significand)) try writer.writeByte('-');
    if (try formatSpecial(bf.significand, writer, options.case)) return;

    return switch (options.mode) {
        .decimal => formatDecimal(maxDigitCount, bf.abs(), writer, options.precision),
        .scientific => formatScientific(maxDigitCount, bf.abs(), writer, options.precision),
        .binary => formatPowerOf2Base(bf.abs(), writer, 2, "0b", options.case, options.precision),
        .octal => formatPowerOf2Base(bf.abs(), writer, 8, "0o", options.case, options.precision),
        .hex => formatPowerOf2Base(bf.abs(), writer, 16, "0x", options.case, options.precision),
    };
}

/// Only formats special cases (nan, inf).
/// Returns true if a special case was formatted.
/// Otherwise, returns false and nothing is written to `writer`.
fn formatSpecial(significand: anytype, writer: *Writer, case: std.fmt.Case) Writer.Error!bool {
    if (math.isNan(significand)) {
        try writer.writeAll(switch (case) {
            .lower => "nan",
            .upper => "NAN",
        });
        return true;
    }
    if (math.isInf(significand)) {
        try writer.writeAll(switch (case) {
            .lower => "inf",
            .upper => "INF",
        });
        return true;
    }
    return false;
}

/// Formats the decimal expansion of `self`. Called when using the `{d}` format specifier.
///
/// Example: 123.45
fn formatDecimal(
    comptime maxDigitCount: usize,
    bf: anytype,
    writer: *Writer,
    precision: ?usize,
) Writer.Error!void {
    if (bf.significand == 0) {
        try writer.writeByte('0');
        if (precision) |p| {
            if (p > 0) {
                try writer.writeByte('.');
                try writer.splatByteAll('0', p);
            }
        }
        return;
    }

    assert(bf.significand > 0);
    assert(math.isNormal(bf.significand));

    const decimal = if (precision) |p| blk: {
        const d = bf.toDecimal();
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
    } else bf.toDecimal().removeTrailingZeros();

    const digits_str = blk: {
        var buf: [maxDigitCount]u8 = undefined;
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
fn formatScientific(
    comptime maxDigitCount: usize,
    bf: anytype,
    writer: *Writer,
    precision: ?usize,
) Writer.Error!void {
    if (bf.significand == 0) {
        try writer.writeByte('0');
        if (precision) |p| {
            try writer.writeByte('.');
            try writer.splatByteAll('0', p);
        }
        return writer.writeAll("e0");
    }

    assert(bf.significand > 0);
    assert(math.isNormal(bf.significand));

    const decimal = if (precision) |p| blk: {
        const d = bf.toDecimal();
        break :blk d.round(d.digitCount() - @min(p + 1, d.digitCount()));
    } else bf.toDecimal().removeTrailingZeros();

    const digits_str = blk: {
        var buf: [maxDigitCount]u8 = undefined;
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

/// Formats the scientific base-n expansion of `self`. Asserts that `base` is a power of 2.
///
/// Example (base 16): 0x1.eddp6 (aka 1.928955078125 * 2^6 ≈ 123.45)
fn formatPowerOf2Base(
    self: anytype,
    writer: *Writer,
    comptime base: u8,
    comptime prefix: []const u8,
    case: std.fmt.Case,
    precision: ?usize,
) Writer.Error!void {
    comptime assert(base > 1);
    comptime assert(math.isPowerOfTwo(base));
    const S = @FieldType(@TypeOf(self), "significand");
    const E = @FieldType(@TypeOf(self), "exponent");

    if (self.significand == 0) {
        try writer.writeAll(prefix ++ "0");
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
    const bits_per_digit = comptime math.log2(base);

    const mantissa_bits = std.math.floatMantissaBits(S);
    const fractional_bits = std.math.floatFractionalBits(S);
    const mantissa_mask = (1 << mantissa_bits) - 1;

    const as_bits: C = @bitCast(self.significand);
    var mantissa = as_bits & mantissa_mask;
    var exponent: std.meta.Int(.signed, @typeInfo(E).int.bits + 1) = self.exponent;

    if (fractional_bits == mantissa_bits)
        mantissa |= 1 << fractional_bits; // Add the implicit integer bit.

    const mantissa_digits = (fractional_bits + bits_per_digit - 1) / bits_per_digit;
    // Fill in zeroes to round the fraction width to a whole number of digits.
    mantissa <<= mantissa_digits * bits_per_digit - fractional_bits;

    if (precision) |p| {
        // Round if needed.
        if (p < mantissa_digits) {
            var extra_bits = (mantissa_digits - p) * bits_per_digit;
            // The result LSB is the Guard bit, we need two more (Round and
            // Sticky) to round the value.
            while (extra_bits > 2) {
                mantissa = (mantissa >> 1) | (mantissa & 1);
                extra_bits -= 1;
            }
            // Round to nearest, tie to even.
            mantissa |= @intFromBool(mantissa & 0b100 != 0);
            mantissa += 1;
            // Drop the extra bits.
            mantissa >>= @intCast(extra_bits);
            // Restore the alignment.
            mantissa <<= @as(std.math.Log2Int(C), @intCast((mantissa_digits - p) * bits_per_digit));

            const overflow = mantissa & (1 << 1 + mantissa_digits * bits_per_digit) != 0;
            // Prefer a normalized result in case of overflow.
            if (overflow) {
                mantissa >>= 1;
                exponent += 1;
            }
        }
    }

    // +1 for the decimal part.
    var buf: [1 + mantissa_digits]u8 = undefined;
    assert(std.fmt.printInt(&buf, mantissa, base, case, .{ .fill = '0', .width = 1 + mantissa_digits }) == buf.len);

    try writer.writeAll(prefix);
    try writer.writeByte(buf[0]);
    const trimmed = std.mem.trimRight(u8, buf[1..], "0");
    if (precision) |p| {
        if (p > 0) try writer.writeAll(".");
    } else if (trimmed.len > 0) {
        try writer.writeAll(".");
    }
    try writer.writeAll(trimmed);
    // Add trailing zeros if explicitly requested.
    if (precision) |p| {
        if (p > trimmed.len)
            try writer.splatByteAll('0', p - trimmed.len);
    }
    try writer.writeAll("p");
    try writer.printInt(exponent, 10, case, .{});
}
