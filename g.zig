const std = @import("std");
const assert = std.debug.assert;
const math = std.math;
const meta = std.meta;

// r = floor(log2(10^k)) - 127
// 10^k = g 2^r
// 10^k 2^-r <= ceil(g)
// 10^k 2^(-floor(log2(10^k)) + 127) <= ceil(g)
// 10^k 2^ceil(-log2(10^k) + 127) <= ceil(g)
// fr = log2(10^k) - floor(log2(10^k))
// log2(10^k) - fr = floor(log2(10^k))
// fr = k*log2(10) - floor(k*log2(10))
//
// 10^k 2^(fr - log2(10^k) + 127) <= ceil(g)
// 2^fr 2^127 <= ceil(g)
// 2^r < 10^k <= g 2^r
pub fn main() void {
    for (0..77) |j| {
        const k = @as(i32, @intCast(j)) - 31;
        assert(k >= math.minInt(i32));
        const output_bits = 64;
        const guard: comptime_int = comptime (math.log2_int(u16, output_bits) + 1);
        const bits = output_bits + guard;
        @setEvalBranchQuota(bits * 10);
        const loggya = comptime log2_10(bits);
        const loggy = loggya + @intFromBool(k >= 0);
        const T = meta.Int(.unsigned, bits);
        const T2 = meta.Int(.unsigned, 2 * bits);

        var fr: T = @truncate((@as(T2, loggy) * @abs(k)) << 2);
        if (k < 0) {
            // Negate without casting
            fr = ~fr +% 1;
        }
        var g = pow2(T, fr);

        const one = 1 << guard;
        const frac = g & (one - 1);
        g >>= guard;
        g += @intFromBool(frac > 0);
        std.debug.print("0x{X}, // {}\n", .{ g, k });
    }
}

pub fn main3() void {
    const k: i32 = 324;
    // 0x88B3A28A05EADE3A491AF84CC6ED52EB 10
    // 0x88B3A28A05EADE3A491AF84CC6ED472C 23

    // 0xECC5F45AA573D300217C1828AB5BFC5D
    // 0xED47F011818A54978740D09001540ED9 10
    // 0xED47F011818A54978740D090015C2CDB 20
    // 0xED47F011818A54978740D090015C3278 100
    const extra_bits = 100;
    const output_bits = 128;
    const bits = output_bits + extra_bits;
    const low_bits = bits / 2;
    const low_mask = (1 << low_bits) - 1;
    const T = meta.Int(.unsigned, bits + 1);

    var g: T = 10;
    var r: i32 = k - 1;
    g <<= @intCast(@clz(g) - 1);
    while (r > 0) {
        while (g > math.maxInt(T) / 20) {
            // Round up
            g += 1;
            g >>= 1;
        }
        if (@mod(r, 2) == 1) {
            g *= 10;
            r -= 1;
        } else {
            const low = g & low_mask; // 127
            const high = g >> low_bits; // 128
            const hh = high * high; // << 256
            const hh_shift: math.Log2IntCeil(T) = @intCast(@clz(hh) - 2);
            const lh = low * high; // << 128
            const lh_shift = low_bits - 1 - hh_shift; // - 1 to account for * 2
            const half_bit = @as(T, 1) << (lh_shift - 1);
            const lh_rounded = lh + half_bit;
            g = (hh << hh_shift) + (lh_rounded >> lh_shift);
            r = @divTrunc(r, 2);
        }
    }
    g <<= @intCast(@clz(g) - 1);
    // Round up
    const half_bit = 1 << (extra_bits - 1);
    g += half_bit;
    g >>= extra_bits;

    std.debug.print("0x{X}\n", .{g});
}

pub fn main2() void {
    const k: i32 = -292;

    const extra_bits = 10;
    const output_bits = 128;
    const T = meta.Int(.unsigned, output_bits + extra_bits + 1);
    var g: T = 1 << (output_bits + extra_bits - 1);
    var r: i32 = k;
    while (r < 0) : (r += 1) {
        // Round up
        g += 5;
        g /= 10;
        while (g <= math.maxInt(T) / 4) {
            g <<= 1;
        }
    }
    g <<= @intCast(@clz(g) - 1);
    // Round up
    const half_bit = 1 << (extra_bits - 1);
    g += half_bit;
    g >>= extra_bits;

    std.debug.print("0x{X}\n", .{g});
}

/// Returns the high bits of a * b.
fn mulHigh(T: type, a: T, b: T) T {
    const bits = @typeInfo(T).int.bits;
    assert(@typeInfo(T).int.signedness == .unsigned);

    // Multiplying with a wider type is faster than 3 smaller multiplications.
    if (bits <= 65_535 / 2) {
        const result = math.mulWide(T, a, b);
        return @truncate(result >> bits);
    }

    const shift = bits / 2;
    const mask = (1 << shift) - 1;
    const al = a & mask;
    const ah = a >> shift;
    const bl = b & mask;
    const bh = b >> shift;

    const lh = al * bh;
    const hl = ah * bl;
    const m, const of = @addWithOverflow(lh, hl);
    const hh = ah * bh;
    return hh + (m >> shift) + (@as(T, of) << (bits - shift));
}

/// Returns floor(log2(10) * 2^(bits - 2)).
fn log2_10(comptime bits: u16) meta.Int(.unsigned, bits) {
    assert(2 <= bits and bits <= 32_766);
    const T = meta.Int(.unsigned, 2 * (bits + 1));

    // log2(10) = 3 + fractional part
    var result: meta.Int(.unsigned, bits) = 3;

    // fractional part = log2(10 * 2^-3)
    //                 = log2(1.25)
    // v = 1.25 * 2^bits
    //   = 5 * 2^(bits-2)
    var v: T = 5 << (bits - 2);
    for (0..bits - 2) |_| {
        const v2 = v * v;
        result <<= 1;

        const one = 1 << (2 * bits);
        if (v2 >= 2 * one) {
            result |= 1;
            v = v2 >> (bits + 1);
        } else {
            v = v2 >> bits;
        }
    }

    return result;
}

/// Returns 2^(n * 2^-@typeInfo(T).int.bits)) * 2^(@typeInfo(T).int.bits - 1).
/// The answer may be slightly overestimated.
fn pow2(T: type, n: T) T {
    const bits: comptime_int = @typeInfo(T).int.bits;
    assert(3 <= bits and bits <= 29127);
    assert(@typeInfo(T).int.signedness == .unsigned);
    const guard = bits / 12;
    const p: comptime_int = 2 * (bits + guard);
    const P = meta.Int(.unsigned, p);

    // [2^1, 2^0.5, 2^0.25, 2^0.125, ...]
    const pow2s = comptime pow: {
        @setEvalBranchQuota(bits * p * 2);
        var pow: P = 1 << (p - 1); // Start with 2^1 = 2
        var pow2s: [bits]P = undefined;
        for (0..bits) |i| {
            const bits_lost = p / 2 - 1;
            pow = (sqrt(P, pow) + 1) << bits_lost;
            pow2s[i] = pow;
        }
        break :pow pow2s;
    };

    var result: P = 1 << (p - 1);
    for (0..bits) |i| {
        const mask = @as(T, 1) << @intCast(bits - i - 1);
        if (n & mask == mask) {
            result = mulHigh(P, result, pow2s[i]);
        }
    }
    return @truncate(math.shl(P, result, @as(i32, @clz(result)) - (p - bits)));
}

/// Returns the square root of n, rounded down.
fn sqrt(T: type, n: T) T {
    assert(@typeInfo(T).int.signedness == .unsigned);

    const m = n - 1; // Round down
    // Halfing the number of bits is very close to square rooting
    const log2_n: math.Log2IntCeil(T) = math.log2_int(T, n);
    var x: T = n >> ((log2_n + 1) / 2);
    while (true) {
        // Newton-raphson method
        const next_x = (x + (m / x)) >> 1;
        if (x == next_x) return next_x;
        x = next_x;
    }
    return x;
}
