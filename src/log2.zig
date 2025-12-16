//! Ported from zig, which itself is ported from musl.
//! Both are licensed under the MIT license:
//! https://codeberg.org/ziglang/zig/src/branch/master/LICENSE
//! https://git.musl-libc.org/cgit/musl/tree/COPYRIGHT
//!
//! https://codeberg.org/ziglang/zig/src/branch/master/lib/compiler_rt/log2.zig
//! https://git.musl-libc.org/cgit/musl/tree/src/math/log2f.c
//! https://git.musl-libc.org/cgit/musl/tree/src/math/log2.c

const std = @import("std");
const math = std.math;

fn log2f(x_: f32) f32 {
    const ivln2hi: f32 = 1.4428710938e+00;
    const ivln2lo: f32 = -1.7605285393e-04;
    const Lg1: f32 = 0xaaaaaa.0p-24;
    const Lg2: f32 = 0xccce13.0p-25;
    const Lg3: f32 = 0x91e9ee.0p-25;
    const Lg4: f32 = 0xf89e26.0p-26;

    var x = x_;
    var u: u32 = @bitCast(x);
    var ix = u;
    var k: i32 = 0;

    // x < 2^(-126)
    if (ix < 0x00800000 or ix >> 31 != 0) {
        // log(+-0) = -inf
        if (ix << 1 == 0) {
            return -math.inf(f32);
        }
        // log(-#) = nan
        if (ix >> 31 != 0) {
            return math.nan(f32);
        }

        k -= 25;
        x *= 0x1.0p25;
        ix = @bitCast(x);
    } else if (ix >= 0x7F800000) {
        return x;
    } else if (ix == 0x3F800000) {
        return 0;
    }

    // x into [sqrt(2) / 2, sqrt(2)]
    ix += 0x3F800000 - 0x3F3504F3;
    k += @as(i32, @intCast(ix >> 23)) - 0x7F;
    ix = (ix & 0x007FFFFF) + 0x3F3504F3;
    x = @bitCast(ix);

    const f = x - 1.0;
    const s = f / (2.0 + f);
    const z = s * s;
    const w = z * z;
    const t1 = w * (Lg2 + w * Lg4);
    const t2 = z * (Lg1 + w * Lg3);
    const R = t2 + t1;
    const hfsq = 0.5 * f * f;

    var hi = f - hfsq;
    u = @bitCast(hi);
    u &= 0xFFFFF000;
    hi = @bitCast(u);
    const lo = f - hi - hfsq + s * (hfsq + R);
    return (lo + hi) * ivln2lo + lo * ivln2hi + hi * ivln2hi + @as(f32, @floatFromInt(k));
}

fn log2d(x_: f64) f64 {
    const ivln2hi: f64 = 1.44269504072144627571e+00;
    const ivln2lo: f64 = 1.67517131648865118353e-10;
    const Lg1: f64 = 6.666666666666735130e-01;
    const Lg2: f64 = 3.999999999940941908e-01;
    const Lg3: f64 = 2.857142874366239149e-01;
    const Lg4: f64 = 2.222219843214978396e-01;
    const Lg5: f64 = 1.818357216161805012e-01;
    const Lg6: f64 = 1.531383769920937332e-01;
    const Lg7: f64 = 1.479819860511658591e-01;

    var x = x_;
    var ix: u64 = @bitCast(x);
    var hx: u32 = @intCast(ix >> 32);
    var k: i32 = 0;

    if (hx < 0x00100000 or hx >> 31 != 0) {
        // log(+-0) = -inf
        if (ix << 1 == 0) {
            return -math.inf(f64);
        }
        // log(-#) = nan
        if (hx >> 31 != 0) {
            return math.nan(f64);
        }

        // subnormal, scale x
        k -= 54;
        x *= 0x1.0p54;
        hx = @intCast(@as(u64, @bitCast(x)) >> 32);
    } else if (hx >= 0x7FF00000) {
        return x;
    } else if (hx == 0x3FF00000 and ix << 32 == 0) {
        return 0;
    }

    // x into [sqrt(2) / 2, sqrt(2)]
    hx += 0x3FF00000 - 0x3FE6A09E;
    k += @as(i32, @intCast(hx >> 20)) - 0x3FF;
    hx = (hx & 0x000FFFFF) + 0x3FE6A09E;
    ix = (@as(u64, hx) << 32) | (ix & 0xFFFFFFFF);
    x = @bitCast(ix);

    const f = x - 1.0;
    const hfsq = 0.5 * f * f;
    const s = f / (2.0 + f);
    const z = s * s;
    const w = z * z;
    const t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    const R = t2 + t1;

    // hi + lo = f - hfsq + s * (hfsq + R) ~ log(1 + f)
    var hi = f - hfsq;
    var hii = @as(u64, @bitCast(hi));
    hii &= @as(u64, math.maxInt(u64)) << 32;
    hi = @bitCast(hii);
    const lo = f - hi - hfsq + s * (hfsq + R);

    var val_hi = hi * ivln2hi;
    var val_lo = (lo + hi) * ivln2lo + lo * ivln2hi;

    // spadd(val_hi, val_lo, y)
    const y: f64 = @floatFromInt(k);
    const ww = y + val_hi;
    val_lo += (y - ww) + val_hi;
    val_hi = ww;

    return val_lo + val_hi;
}

pub fn log2(x: anytype) @TypeOf(x) {
    return switch (@TypeOf(x)) {
        f16 => @floatCast(log2f(x)),
        f32 => log2f(x),
        f64 => log2d(x),
        f80, f128, comptime_float => log2d(@floatCast(x)),
        else => @compileError("Unsupported type"),
    };
}
