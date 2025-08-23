const std = @import("std");

pub fn main() void {
    // @log is not precise enough
    // log10(2)
    // const b: u128 = @bitCast(@as(f128, 3.3219280948873623478703194294893901758648313930245806120547563958159347766));
    // const b: u128 = @bitCast(@as(f128, 0.30102999566398119521373889472449302676818988146210854131042746112710818927));
    const b: u128 = @bitCast(@as(f128, 1.0893418703486831447820173030146839006797769601931));
    const mask = (1 << std.math.floatMantissaBits(f128)) - 1;
    const m = (b & mask) | (1 << std.math.floatMantissaBits(f128));
    std.debug.print("{b}\n", .{m});
    @setEvalBranchQuota(1_000_000);
    std.debug.print("{b}\n", .{pow2(@as(u113, 42010168373378879565782048137661639979 >> 15))});
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

/// Returns round(log10(2) * 2^(bits + 1)). Ties are rounded to +inf.
fn log10_2(comptime bits: u16) std.meta.Int(.unsigned, bits) {
    if (bits >= 32_767) @compileError("Too many bits");
    const p = 2 * (bits + 1);
    const T = std.meta.Int(.unsigned, p);
    // log10(2) = 1 / log2(10)
    const inv = @as(T, log2_10(bits)) << (p - bits - 1);

    // Trial division
    var v: T = 1 << (p - 1);
    var result: std.meta.Int(.unsigned, bits) = 0;
    for (0..bits) |_| {
        result <<= 1;
        if (v >= inv) {
            result |= 1;
            v -= inv;
        }
        v <<= 1;
    }

    return result;
}

/// Returns 2^(n * 2^-@typeInfo(@TypeOf(n)).int.bits)) * 2^(@typeInfo(@TypeOf(n)).int.bits - 1).
/// The answer may be slightly overestimated.
fn pow2(n: anytype) std.meta.Int(.unsigned, @typeInfo(@TypeOf(n)).int.bits) {
    const bits = @typeInfo(@TypeOf(n)).int.bits;
    const guard = 4;
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
