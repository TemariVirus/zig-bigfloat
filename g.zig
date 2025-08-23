const std = @import("std");

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
    for (0..100) |j| {
        @setEvalBranchQuota(100_000);
        const k = @as(i128, j);
        const output_bits = 512;
        const loggy = comptime log2_10(output_bits) + 1;
        const T = std.meta.Int(.unsigned, output_bits);
        const T2 = std.meta.Int(.unsigned, 2 * output_bits);

        // var fr = blk: {
        //     const shift = (output_bits) / 2;
        //     const mask = (1 << shift) - 1;
        //     const rl = loggy & mask;
        //     const rh = loggy >> shift;
        //     const ul = @as(T, @abs(k)) & mask;
        //     const uh = @as(T, @abs(k)) >> shift;
        //     const ll = (rl * ul) << 2;
        //     const lh = rl * uh;
        //     const hl = rh * ul;
        //     break :blk ll + ((lh + hl) << (output_bits - 254));
        // };
        // Apparently faster
        var fr: T = @truncate((@as(T2, loggy) * @abs(k)) << 2);
        if (k < 0) {
            // Negate without casting
            fr = ~fr +% 1;
        }
        const g = pow2(fr);
        std.debug.print("0x{X} {}\n", .{ g, k });
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
    const T = std.meta.Int(.unsigned, bits + 1);

    var g: T = 10;
    var r: i32 = k - 1;
    g <<= @intCast(@clz(g) - 1);
    while (r > 0) {
        while (g > std.math.maxInt(T) / 20) {
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
            const hh_shift: std.math.Log2IntCeil(T) = @intCast(@clz(hh) - 2);
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
    const T = std.meta.Int(.unsigned, output_bits + extra_bits + 1);
    var g: T = 1 << (output_bits + extra_bits - 1);
    var r: i32 = k;
    while (r < 0) : (r += 1) {
        // Round up
        g += 5;
        g /= 10;
        while (g <= std.math.maxInt(T) / 4) {
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

/// Returns floor(log2(10) * 2^(bits - 2)).
fn log2_10(comptime bits: u16) std.meta.Int(.unsigned, bits) {
    if (bits >= 32_767) @compileError("Too many bits");
    const T = std.meta.Int(.unsigned, 2 * (bits + 1));

    // log2(10) = 3 + fractional part
    var result: std.meta.Int(.unsigned, bits) = 3;

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

/// Returns 2^(n * 2^-@typeInfo(@TypeOf(n)).int.bits)) * 2^(@typeInfo(@TypeOf(n)).int.bits - 1).
/// The answer may be slightly overestimated.
fn pow2(n: anytype) std.meta.Int(.unsigned, @typeInfo(@TypeOf(n)).int.bits) {
    const bits = @typeInfo(@TypeOf(n)).int.bits;
    const guard = bits / 8;
    const p = 2 * (bits + guard);
    const T = std.meta.Int(.unsigned, p);

    var unit: T = 1 << (p - 1);
    var result: T = 1 << (p - 1);
    for (0..bits) |i| {
        unit = (sqrt(unit) + 1) << (p / 2 - 1);
        const mask = @as(@TypeOf(n), 1) << @intCast(bits - i - 1);
        if (n & mask == mask) {
            const shift = p / 2;
            const mask2 = (1 << shift) - 1;
            const rl = result & mask2;
            const rh = result >> shift;
            const ul = unit & mask2;
            const uh = unit >> shift;
            const lh = rl * uh;
            const hl = rh * ul;
            const hh = rh * uh;
            result = hh + ((lh + hl) >> shift);
        }
    }
    return @truncate(std.math.shl(T, result, @as(i32, @clz(result)) - bits - (2 * guard)));
}

/// Returns the square root of n, rounded down.
fn sqrt(n: anytype) @TypeOf(n) {
    const T = @TypeOf(n);
    std.debug.assert(@typeInfo(T).int.signedness == .unsigned);
    const bits = @typeInfo(T).int.bits / 2;

    var result: T = 0;
    for (0..bits) |i| {
        const b = @as(T, 1) << @intCast(bits - i - 1);
        result |= b;
        if (result * result > n) {
            result ^= b;
        }
    }
    return result;
}
