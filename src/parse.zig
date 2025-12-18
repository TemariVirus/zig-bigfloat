const std = @import("std");
const fmt = std.fmt;
const math = std.math;
const assert = std.debug.assert;
const Reader = std.Io.Reader;

fn MantissaInt(T: type) type {
    return switch (T) {
        f16, f32 => u32,
        f64 => u64,
        f80, f128 => u128,
        else => @compileError("Unsupported type"),
    };
}

/// Returns the number of leading zeros discarded.
fn discardLeadingZeros(reader: *Reader) usize {
    assert(reader.end == reader.buffer.len);

    const leading_zero_count = std.mem.indexOfNone(u8, reader.buffered(), "0") orelse reader.bufferedLen();
    reader.toss(leading_zero_count);
    return leading_zero_count;
}

/// Peeks the next digit from `reader`, skipping a `_` if needed.
fn scanDigit(reader: *Reader, base: u8) fmt.ParseFloatError!?u8 {
    assert(reader.end == reader.buffer.len);

    if (reader.bufferedLen() == 0) {
        return null;
    }

    const i: usize = if (reader.bufferedLen() >= 2 and reader.buffered()[0] == '_') 1 else 0;
    const d = switch (reader.buffered()[i]) {
        '0'...'9' => |c| c - '0',
        'A'...'F' => |c| c - 'A' + 10,
        'a'...'f' => |c| c - 'a' + 10,
        // Cannot have 2 underscores in a row
        '_' => return error.InvalidCharacter,
        else => return null,
    };
    if (d >= base) {
        return error.InvalidCharacter;
    }

    reader.toss(i + 1);
    return d;
}

/// Puts a continuous stream of digits from `reader` into `buf`, skipping `_` characters.
/// Returns the number of digits parsed.
fn takeDigits(reader: *Reader, buf: []u8, base: u8) fmt.ParseFloatError!usize {
    switch (reader.peekByte() catch return 0) {
        '_' => return error.InvalidCharacter,
        else => {},
    }

    var i: usize = 0;
    while (try scanDigit(reader, base)) |d| {
        buf[i] = d;
        i += 1;
        if (i >= buf.len) break;
    }
    while (try scanDigit(reader, base)) |d| {
        // Keep sticky bit for rounding
        buf[buf.len - 1] |= @intFromBool(d != 0);
    }
    return i;
}

fn parseExponent(T: type, reader: *Reader, comptime negative: bool) !T {
    var exponent: T = try scanDigit(reader, 10) orelse return error.InvalidCharacter;
    if (negative) {
        exponent = -exponent;
    }

    const add = if (negative) math.sub else math.add;
    while (try scanDigit(reader, 10)) |d| {
        exponent = try math.mul(T, exponent, 10);
        exponent = try add(T, exponent, d);
    }
    return exponent;
}

/// Parses a floating point number with a `base` that is a power of 2.
/// Returns a tuple containing the normalized significand and exponent,
/// such that `abs(significand)` is in the interval `[1, 2)` and the value
/// of the float is `significand * 2^exponent`.
///
/// `reader` is assumed to come from `std.Io.Reader.fixed`.
pub fn parsePowerOf2Base(
    S: type,
    E: type,
    comptime base: u8,
    reader: *Reader,
) (Reader.Error || fmt.ParseFloatError)!struct { S, E } {
    comptime assert(base > 1);
    comptime assert(math.isPowerOfTwo(base));

    const C = std.meta.Int(.unsigned, @typeInfo(S).float.bits);
    const M = MantissaInt(S);
    const bits_per_digit: comptime_int = comptime math.log2(base);
    // 2 Extra bits needed for rounding
    const bit_width_with_rounding = math.floatFractionalBits(S) + 3;
    const max_digit_count = (bit_width_with_rounding + 2 * bits_per_digit - 2) / bits_per_digit;

    // Discard leading zeros
    var has_leading_zero = discardLeadingZeros(reader) > 0;

    var digits: [max_digit_count]u8 = undefined;
    var digit_count = try takeDigits(reader, &digits, base);
    var binary_point: isize = @intCast(digit_count);
    if (std.mem.startsWith(u8, reader.buffered(), ".")) {
        reader.toss(1);
        if (digit_count == 0) {
            // 0x.000123... case
            binary_point = -@as(isize, @intCast(discardLeadingZeros(reader)));
            has_leading_zero = has_leading_zero or (binary_point < 0);
        }
        const frac_digits = try takeDigits(reader, digits[digit_count..], base);
        // Trim trailing zeros
        digit_count += std.mem.trimEnd(u8, digits[digit_count..][0..frac_digits], &.{0}).len;
    }
    if (!has_leading_zero and digit_count == 0) {
        return error.InvalidCharacter;
    }

    // String was all zeros
    if (digit_count == 0) {
        if (std.ascii.startsWithIgnoreCase(reader.buffered(), "p")) {
            // Discard exponent
            reader.toss(1);
            while (try scanDigit(reader, 10)) |_| {}
        }
        return .{ 0, 0 };
    }

    // Adjust binary point
    binary_point *= bits_per_digit;
    binary_point -= @clz(digits[0]) - (8 - bits_per_digit);
    binary_point -= 1;

    var mantissa: M = 0;
    for (digits[0..digit_count]) |d| {
        mantissa <<= bits_per_digit;
        mantissa |= @as(M, d);
    }
    assert(mantissa != 0);

    // Align mantissa
    const shift = @as(i32, @clz(mantissa)) - (@typeInfo(M).int.bits - bit_width_with_rounding);
    if (shift >= 0) {
        mantissa <<= @intCast(shift);
    } else {
        const dropped_mask = (@as(M, 1) << @intCast(-shift)) - 1;
        const dropped_bits = mantissa & dropped_mask;
        mantissa >>= @intCast(-shift);
        // Keep sticky bit for rounding
        mantissa |= @intFromBool(dropped_bits != 0);
    }
    // Round to nearest, tie to even
    if (mantissa & 0b11 == 0b11) {
        mantissa += 0b100;
    } else if (mantissa & 0b11 == 0b10) {
        mantissa += mantissa & 0b100;
    }
    // Drop rounding bits
    mantissa >>= 2;
    // Handle overflow
    if (mantissa >= @as(M, 1) << (math.floatMantissaBits(S) + 1)) {
        mantissa >>= 1;
        binary_point += 1;
    }
    // Remove implicit one
    assert(mantissa < @as(M, 1) << (math.floatMantissaBits(S) + 1));
    if (math.floatMantissaBits(S) == math.floatFractionalBits(S)) {
        mantissa &= ~(@as(M, 1) << math.floatMantissaBits(S));
    }
    // Insert biased 0 exponent
    const s_repr: C = @as(C, math.floatExponentMax(S) << math.floatMantissaBits(S)) | @as(C, @truncate(mantissa));
    const significand: S = @bitCast(s_repr);

    const EPlus = math.IntFittingRange(
        math.minInt(E) + math.minInt(isize),
        math.maxInt(E) + math.maxInt(isize),
    );
    const exponent: EPlus = if (std.ascii.startsWithIgnoreCase(reader.buffered(), "p")) exponent: {
        reader.toss(1);
        const negative = std.mem.startsWith(u8, reader.buffered(), "-");
        if (negative or std.mem.startsWith(u8, reader.buffered(), "+")) {
            reader.toss(1);
        }
        if (std.mem.startsWith(u8, reader.buffered(), "_")) return error.InvalidCharacter;

        const overflow: struct { S, E } = if (negative) .{ 0, 0 } else .{ math.inf(S), 0 };
        const exp: EPlus = blk: {
            break :blk if (negative)
                parseExponent(EPlus, reader, true)
            else
                parseExponent(EPlus, reader, false);
        } catch |err| switch (err) {
            error.Overflow => return overflow,
            error.InvalidCharacter => return error.InvalidCharacter,
        };
        break :exponent math.add(EPlus, exp, binary_point) catch return overflow;
    } else binary_point;

    if (exponent < math.minInt(E)) {
        return .{ 0, 0 };
    }
    if (exponent > math.maxInt(E)) {
        return .{ math.inf(S), 0 };
    }
    return .{ significand, @intCast(exponent) };
}

pub fn parseBase10(
    S: type,
    E: type,
    reader: *Reader,
) (Reader.Error || fmt.ParseFloatError)!struct { S, E } {
    _ = reader; // autofix
    @panic("TODO: implement base 10 parsing");
}
