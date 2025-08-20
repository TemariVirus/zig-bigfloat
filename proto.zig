const std = @import("std");
const assert = std.debug.assert;
const math = std.math;

pub fn main() void {
    const F = f32;
    var f: F = 0;
    var i: Carrier(F) = 0;
    std.debug.print("{}\n", .{@import("builtin").mode});
    while (f < math.inf(F)) {
        const actual = blk: {
            var buf: [128]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            formatScientific(F, &w, f) catch unreachable;
            break :blk w.buffered();
        };
        const expected = blk: {
            var buf: [128]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            w.print("{e}", .{f}) catch unreachable;
            break :blk w.buffered();
        };

        if (!std.mem.eql(u8, actual, expected)) {
            std.debug.print("{e}\na {s}\ne {s}\n\n", .{ f, actual, expected });
        }

        f = math.nextAfter(F, f, math.inf(F));
        i += 1;
        if (i % (1 << 21) == 0) {
            std.debug.print("current: {e}\n", .{f});
        }
    }
}

// 2^-k = k significant digits in base 10
// f16:   10+1 bits ->  11 s.f.
// f32:   23+1 bits ->  24 s.f.
// f64:   52+1 bits ->  53 s.f.
// f80:   63+1 bits ->  64 s.f.
// f128: 112+1 bits -> 113 s.f.
fn digitCount(T: type) comptime_int {
    return 1 + switch (T) {
        f16 => 11,
        f32 => 24,
        f64 => 53,
        f80 => 64,
        f128 => 113,
        else => @compileError("T must be a float type"),
    };
}

// f16:   11 s.f. -> u37
// f32:   24 s.f. -> u80
// f64:   53 s.f. -> u177
// f80:   64 s.f. -> u213
// f128: 113 s.f. -> u376
fn Digits(T: type) type {
    // Bit sizes rounded up to nearest multiple of 64
    // because zig can't handle 10^60 in a u224 for some reason
    return switch (T) {
        f16 => u64,
        f32 => u128,
        f64 => u192,
        f80 => u256,
        f128 => u384,
        else => @compileError("T must be a float type"),
    };
}

fn Carrier(T: type) type {
    return std.meta.Int(.unsigned, @typeInfo(T).float.bits);
}

fn floatDigits(Float: type, f: Float) struct { Digits(Float), i32 } {
    if (f == 0) return .{ 0, 0 };

    const C = Carrier(Float);
    const D = Digits(Float);
    const mantissa_bits = math.floatMantissaBits(Float);
    const mantissa_mask = (1 << mantissa_bits) - 1;
    const exponent_bits = math.floatExponentBits(Float);
    const exponent_mask = (1 << exponent_bits) - 1;

    const repr: C = @bitCast(f);
    const mantissa = if (Float == f80)
        repr & mantissa_mask
    else if (math.isInf(f))
        1 << mantissa_bits
    else
        repr & mantissa_mask | (@as(D, @intFromBool(math.isNormal(f))) << mantissa_bits);
    // i32 is big enough for even f128
    const biased_exponent: i32 = @intCast((repr >> mantissa_bits) & exponent_mask);
    const exponent = @as(i32, @max(
        math.floatExponentMin(Float),
        biased_exponent - math.floatExponentMax(Float),
    )) - math.floatFractionalBits(Float);

    var digits: D = mantissa;
    var exponent10: i32 = 0;
    const ten: comptime_int = comptime math.pow(D, 10, digitCount(Float));
    assert(digits < ten);
    while (digits < ten / 10) {
        digits *= 10;
        exponent10 -= 1;
    }
    for (0..@abs(exponent)) |_| {
        if (exponent > 0) {
            digits <<= 1;
        } else {
            digits >>= 1;
        }
        if (digits >= ten) {
            digits /= 10;
            exponent10 += 1;
        } else if (digits < ten / 10) {
            digits *= 10;
            exponent10 -= 1;
        }
    }
    return .{ digits, exponent10 };
}

fn truncRound(Float: type, d: Digits(Float), pow10: Digits(Float)) Digits(Float) {
    const d_frac = d % pow10;
    const d_int = d - d_frac;
    return d_int;
}

fn absDiff(Float: type, a: Digits(Float), b: Digits(Float)) Digits(Float) {
    return if (a > b) a - b else b - a;
}

fn validApprox(
    Float: type,
    low: Digits(Float),
    mid: Digits(Float),
    high: Digits(Float),
    trunc_mid: Digits(Float),
    is_mantissa_even: bool,
) bool {
    if (absDiff(Float, low, trunc_mid) < absDiff(Float, mid, trunc_mid) or
        absDiff(Float, high, trunc_mid) < absDiff(Float, mid, trunc_mid))
    {
        return false;
    }
    if (absDiff(Float, low, trunc_mid) > absDiff(Float, mid, trunc_mid) and
        absDiff(Float, high, trunc_mid) > absDiff(Float, mid, trunc_mid))
    {
        return true;
    }
    return is_mantissa_even;
}

fn toDecimal(Float: type, f: Float) struct { Digits(Float), i32 } {
    assert(f > 0);

    // Bounds
    const mid, var exponent10 = floatDigits(Float, f);
    const low = blk: {
        const l, _ = floatDigits(Float, math.nextAfter(Float, f, -math.inf(Float)));
        break :blk if (l < mid) l else l / 10;
    };
    const high = blk: {
        const h, _ = floatDigits(Float, math.nextAfter(Float, f, math.inf(Float)));
        break :blk if (h > mid) h else h * 10;
    };
    const is_mantissa_even = (@as(Carrier(Float), @bitCast(f)) & 1) == 0;

    var pow10: Digits(Float) = 1;
    var digits = while (pow10 < mid) : ({
        pow10 *= 10;
        exponent10 += 1;
    }) {
        const trunc_mid = truncRound(Float, mid, pow10 * 10);
        const trunc_mid2 = trunc_mid + (pow10 * 10);
        // Try dropping another digit if either truncation is valid
        if (validApprox(Float, low, mid, high, trunc_mid, is_mantissa_even) or
            validApprox(Float, low, mid, high, trunc_mid2, is_mantissa_even))
        {
            continue;
        }

        // We dropped too many digits, go back and find the valid truncation closest to mid
        const trunc_mid3 = truncRound(Float, mid, pow10);
        const trunc_mid4 = trunc_mid3 + pow10;
        if (!validApprox(Float, low, mid, high, trunc_mid3, is_mantissa_even)) break trunc_mid4 / pow10;
        if (!validApprox(Float, low, mid, high, trunc_mid4, is_mantissa_even)) break trunc_mid3 / pow10;
        if (absDiff(Float, mid, trunc_mid3) < absDiff(Float, mid, trunc_mid4)) break trunc_mid3 / pow10;
        if (absDiff(Float, mid, trunc_mid3) > absDiff(Float, mid, trunc_mid4)) break trunc_mid4 / pow10;
        const final = trunc_mid3 / pow10;
        // Break ties by rounding to even
        break final + (final % 2);
    } else 1; // If mid is a power of 10, truncate everything

    // Remove trailing zeros
    while (digits % 10 == 0 and digits != 0) {
        digits /= 10;
        exponent10 += 1;
    }
    return .{ digits, exponent10 };
}

fn formatScientific(Float: type, w: *std.Io.Writer, f: Float) !void {
    // Special cases
    if (math.signbit(f)) try w.writeByte('-');
    if (math.isNan(f)) return w.writeAll("nan");
    if (math.isInf(f)) return w.writeAll("inf");
    if (f == 0) return w.writeAll("0e0");
    if (f < 0) return formatScientific(Float, w, -f);

    const digits, const exponent10 = toDecimal(Float, f);
    const digits_str = blk: {
        var buf: [digitCount(Float) + 1]u8 = undefined;
        var digit_writer = std.Io.Writer.fixed(&buf);
        digit_writer.print("{d}", .{digits}) catch unreachable;
        break :blk digit_writer.buffered();
    };

    const digit_count: i32 = @intCast(digits_str.len);
    const actual_exponent = exponent10 + digit_count - 1;
    if (digit_count == 1) return w.print("{s}e{d}", .{ digits_str, actual_exponent });
    return w.print("{s}.{s}e{d}", .{ digits_str[0..1], digits_str[1..], actual_exponent });
}
