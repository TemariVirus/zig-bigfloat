//! Crude implementation of schubfach for arbitrary floating point types.
//! https://drive.google.com/file/d/1IEeATSVnEE6TkrHlCYNY2GjaraBjOT4f/view

const std = @import("std");
const assert = std.debug.assert;
const math = std.math;
const meta = std.meta;

fn TryInt(comptime signedness: std.builtin.Signedness, bits: comptime_int) ?type {
    if (bits > math.maxInt(u16)) return null;
    return meta.Int(signedness, bits);
}

/// Returns whether x is divisible by 2^n
fn divisibleByPow2(T: type, x: T, n: math.Log2Int(T)) bool {
    const @"2^n" = @as(T, 1) << n;
    return x & (@"2^n" - 1) == 0;
}

/// Returns the high bits of `a * b`.
fn mulHigh(T: type, a: T, b: T) T {
    const bits = @typeInfo(T).int.bits;
    comptime assert(@typeInfo(T).int.signedness == .unsigned);

    // Multiplying with a wider type is faster than 3 smaller multiplications.
    if (TryInt(.unsigned, bits * 2)) |WideT| {
        const result = @as(WideT, a) * @as(WideT, b);
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

/// Returns the first `bits` bits of `log2(x)` truncated.
fn log2(comptime bits: u16, x: comptime_int) meta.Int(.unsigned, bits) {
    comptime assert(bits >= 2);
    const T = TryInt(.unsigned, 2 * (@as(comptime_int, bits) + 1)) orelse @compileError("Too many bits");
    const X = math.IntFittingRange(0, x);

    // log2(x) = int + fractional part
    const int: comptime_int = comptime math.log2_int(X, x);
    const frac_bits = bits - int + @ctz(@as(X, x));
    var result: meta.Int(.unsigned, bits) = int;

    // fractional part = log2(x * 2^-int)
    // v = x * 2^-int * 2^bits
    //   = x * 2^(bits - int)
    var v: T = x << (bits - int);
    for (0..frac_bits) |_| {
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

/// Returns `round(2^@typeInfo(@TypeOf(x)).int.bits / x)`. Ties are rounded to +inf.
fn inverse(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const bits: comptime_int = @typeInfo(T).int.bits;
    comptime assert(@typeInfo(T).int.signedness == .unsigned);
    const p = 2 * (bits + 1);
    const P = TryInt(.unsigned, p) orelse @compileError("Too many bits");
    const inv = @as(P, x) << (p - bits - 1);

    // Trial division
    var v: P = 1 << (p - 1);
    var result: T = 0;
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

/// Returns the square root of n, rounded down.
fn sqrt(T: type, n: T) T {
    comptime assert(@typeInfo(T).int.signedness == .unsigned);

    const m = n - 1; // Round down
    // Halfing the number of bits is very close to square rooting
    const digit_count = @typeInfo(T).int.bits - @clz(n);
    var x: T = n >> @intCast(digit_count / 2);
    while (true) {
        // Newton-raphson method
        const next_x = (x + (m / x)) >> 1;
        if (x == next_x) return next_x;
        x = next_x;
    }
    return x;
}

/// S: The floating point type.
/// _E: The exponent type.
/// bake_logs: Whether to bake in expensive logarithmic constants.
/// This should only be disabled to increase compilation speed.
/// Binary sizes are smaller when this is enabled as the constants take up
/// less space than the code for generating them at runtime, surprisingly.
pub fn Render(S: type, _E: type, comptime bake_logs: bool) type {
    const C = meta.Int(.unsigned, @max(18, @typeInfo(S).float.bits));
    const Cx2 = meta.Int(.unsigned, 2 * @typeInfo(C).int.bits);
    const E = math.IntFittingRange(
        math.minInt(_E) - math.floatFractionalBits(S),
        math.maxInt(_E),
    );

    // The required precision of intermediate values increases with the bit size
    // of _E, putting a hard cap on it as integers can have at most 65,535 bits.
    if (@typeInfo(_E).int.bits < 1) {
        @compileError("Too few bits in exponent");
    }
    if (@typeInfo(_E).int.bits > 14_556) {
        @compileError("Too many bits in exponent");
    }

    return struct {
        /// The scientific representation of a floating point number. `digits * 10^exponent`.
        pub const Decimal = struct {
            digits: C,
            exponent: E,

            const log10_2f = 0.3010299956639811952137388947244930;
            const log10_log10_2f = -0.5213902276543248032662554139456424;

            /// Maximum number of decimal digits in `digits`.
            pub const maxDigitCount: comptime_int =
                1 + @floor(log10_2f * @as(f128, @typeInfo(C).int.bits));

            /// Maximum number of decimal digits in `exponent`. Includes the negative sign.
            pub const maxExponentDigitCount: comptime_int =
                2 + @floor(log10_2f * @as(f128, @typeInfo(_E).int.bits - 1) + log10_log10_2f);

            /// Removes trailing zeros from `digits` and adjusts `exponent` accordingly.
            pub fn removeTrailingZeros(self: @This()) @This() {
                var copy = self;
                while (copy.digits % 10 == 0 and copy.digits != 0) {
                    copy.digits /= 10;
                    copy.exponent += 1;
                }
                return copy;
            }

            /// Ties are broken by rounding away from zero.
            pub fn round(self: @This(), digits: u16) @This() {
                assert(digits <= self.digitCount());
                if (digits == 0) return self;

                const pow_10 = math.powi(C, 10, digits) catch unreachable;
                const half_pow10 = pow_10 / 2;

                var new_exponent = self.exponent + @as(E, @intCast(digits));

                const new_digit_count = @max(1, self.digitCount() - digits);
                var new_digits = self.digits / pow_10;
                const remainder = self.digits % pow_10;
                const round_up = remainder >= half_pow10;
                new_digits += @intFromBool(round_up);
                // Rounding caused new_digits have an extra digit
                if (digitCount2(new_digits) > new_digit_count) {
                    new_exponent += 1;
                    new_digits /= 10;
                }

                return .{
                    .digits = new_digits,
                    .exponent = new_exponent,
                };
            }

            /// Returns the number of decimal digits in `digits`.
            pub fn digitCount(self: @This()) u16 {
                return digitCount2(self.digits);
            }

            fn digitCount2(d: C) u16 {
                // C is at most u128 which means we only need to go up to 1e38.
                inline for (0..39) |i| {
                    if (d >= comptime math.powi(C, 10, 38 - i) catch continue) {
                        return 39 - i;
                    }
                }
                // 0 has 1 digit
                return 1;
            }
        };

        const log2_3_bits: comptime_int = blk: {
            const bits: comptime_int = @typeInfo(E).int.bits;
            const log_bits: comptime_int = 2 * bits + 6;
            // Used in negLog10_075
            break :blk log_bits + negLog10_075GuardBits(log_bits) + 2;
        };

        fn log2_3_baked(comptime bake: bool) meta.Int(.unsigned, log2_3_bits) {
            if (bake) {
                @setEvalBranchQuota(5 * log2_3_bits);
                const baked = comptime log2_3_baked(false);
                return baked;
            }
            return log2(log2_3_bits, 3);
        }

        fn log2_3(comptime bits: u16) meta.Int(.unsigned, bits) {
            const v = log2_3_baked(bake_logs);
            const v_bits = @typeInfo(@TypeOf(v)).int.bits;
            comptime assert(bits <= v_bits);
            return @truncate(v >> (v_bits - bits));
        }

        const log2_10_bits: comptime_int = blk: {
            const bits: comptime_int = @typeInfo(E).int.bits;
            const log_bits: comptime_int = 2 * bits + 6;
            // Used in negLog10_075
            const a = log_bits + negLog10_075GuardBits(log_bits);
            // Used in pow10
            const b = 2 + @max(
                @typeInfo(Cx2).int.bits + 8,
                math.floatFractionalBits(S) + @typeInfo(E).int.bits + 1,
            );
            break :blk @max(a, b);
        };

        fn log2_10_baked(comptime bake: bool) meta.Int(.unsigned, log2_10_bits) {
            if (bake) {
                @setEvalBranchQuota(5 * log2_10_bits);
                const baked = comptime log2_10_baked(false);
                return baked;
            }
            return log2(log2_10_bits, 10);
        }

        fn log2_10(comptime bits: u16) meta.Int(.unsigned, bits) {
            const v = log2_10_baked(bake_logs);
            const v_bits = @typeInfo(@TypeOf(v)).int.bits;
            comptime assert(bits <= v_bits);
            return @truncate(v >> (v_bits - bits));
        }

        fn log10_2_baked(comptime bake: bool) meta.Int(.unsigned, log2_10_bits) {
            if (bake) {
                @setEvalBranchQuota(5 * log2_10_bits);
                const baked = comptime log10_2_baked(false);
                return baked;
            }
            return inverse(log2_10(log2_10_bits));
        }

        fn log10_2(comptime bits: u16) meta.Int(.unsigned, bits) {
            const v = log10_2_baked(bake_logs);
            const v_bits = @typeInfo(@TypeOf(v)).int.bits;
            comptime assert(bits <= v_bits);
            return @truncate(v >> (v_bits - bits));
        }

        fn negLog10_075GuardBits(comptime bits: u16) comptime_int {
            return bits / 8 + 3;
        }

        /// Returns `-floor(log10(0.75) * 2^bits)`.
        fn negLog10_075(comptime bits: u16, comptime bake: bool) meta.Int(.unsigned, bits) {
            if (bake) {
                @setEvalBranchQuota(10 * @as(comptime_int, bits));
                const baked = comptime negLog10_075(bits, false);
                return baked;
            }

            comptime assert(bits >= 3);
            const guard: comptime_int = negLog10_075GuardBits(bits);
            const p = 2 * (@as(comptime_int, bits) + guard);
            const WideT = TryInt(.unsigned, p) orelse @compileError("Too many bits");

            // log10(0.75)  = log2(3/4) / log2(10)
            //              = (log2(3) - 2) / log2(10)
            // -log10(0.75) = (2 - log2(3)) / log2(10)
            // Add 2 to remove leading zeros
            const numerator: WideT = 0 -% log2_3(bits + guard + 2);
            const denominator: WideT = log10_2(bits + guard) -% 1;
            const result = (numerator * denominator) >> (p - bits + 2);
            return @truncate(result + 1);
        }

        /// Returns `2^(n * 2^-@typeInfo(Cx2).int.bits)) * 2^(@typeInfo(Cx2).int.bits - 1)`.
        /// The answer may be slightly overestimated.
        fn pow2(n: Cx2) Cx2 {
            const bits: comptime_int = @typeInfo(Cx2).int.bits;
            comptime assert(@typeInfo(Cx2).int.signedness == .unsigned);
            comptime assert(bits >= 3);
            const guard: comptime_int = bits / 12;
            const p: comptime_int = 2 * (bits + guard);
            const P = TryInt(.unsigned, p) orelse @compileError("Too many bits");

            // [2^1, 2^0.5, 2^0.25, 2^0.125, ...]
            // The size of this table seems to be generally the same or smaller than the code
            // for generating it. The cost of evaluating this block is also negligible (3-12ms).
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
                const mask = @as(Cx2, 1) << @intCast(bits - i - 1);
                if (n & mask == mask) {
                    result = mulHigh(P, result, pow2s[i]);
                }
            }
            return @truncate(math.shl(P, result, @as(i32, @clz(result)) - (p - bits)));
        }

        /// Returns `k = floor(log_10(2^q))` or `floor(log_10(0.75 * 2^q))`.
        fn computeK(q: E, lower_boundary_is_closer: bool) E {
            const bits: comptime_int = @typeInfo(E).int.bits;
            comptime assert(@typeInfo(E).int.signedness == .signed);
            const p: comptime_int = 3 * bits + 6;
            const P = TryInt(.signed, p) orelse @compileError("Too many bits");
            const log_bits = p - bits;

            const @"log10(2)": P = log10_2(log_bits - 1);
            const @"-log10(0.75)": P = negLog10_075(log_bits, bake_logs);
            return @truncate(
                (@"log10(2)" * q - if (lower_boundary_is_closer) @"-log10(0.75)" else 0) >>
                    log_bits,
            );
        }

        /// Returns floor(log_2(10^e))
        fn floorLog2Pow10(e: E) E {
            const bits: comptime_int = @typeInfo(E).int.bits;
            comptime assert(@typeInfo(E).int.signedness == .signed);
            const p = 3 * bits + 6;
            const P = TryInt(.signed, p) orelse @compileError("Too many bits");

            const @"log2(10)": P = log2_10(p - bits);
            // Sub 2 to remove leading zeros
            return @truncate((@"log2(10)" * e) >> (p - bits - 2));
        }

        fn pow10(k: E) Cx2 {
            // There are unique beta and r such that 10^k = beta 2^r and
            // 2^63 <= beta < 2^64, namely r = floor(log_2 10^k) - 63 and
            // beta = 2^-r 10^k.
            // Let g = ceil(beta), so (g-1) 2^r < 10^k <= g 2^r, with the latter
            // value being a pretty good overestimate for 10^k.

            // r = floor(log2(10^k)) - 127
            // 10^k <= g 2^r
            // 10^k 2^-r <= g
            // 10^k 2^(-floor(log2(10^k)) + 127) <= g
            // 10^k 2^ceil(-log2(10^k) + 127) <= g
            // f = log2(10^k) - floor(log2(10^k))
            // log2(10^k) - f = floor(log2(10^k))
            // f = k*log2(10) - floor(k*log2(10))
            // 10^k 2^(f - log2(10^k) + 127) <= g
            // 2^f 2^127 <= g

            // @abs(k) must not overflow.
            // In practice this never happens because:
            // q >= math.minInt(_E), log10(2) < 0.5
            // k = log10(2) * q > math.minInt(_E) > math.minInt(E)
            assert(k > math.minInt(E));

            const bits: comptime_int = @typeInfo(Cx2).int.bits;
            const p: comptime_int = @max(
                bits + 8,
                math.floatFractionalBits(S) + @typeInfo(E).int.bits + 1,
            );
            const P = meta.Int(.unsigned, p);

            const @"log2(10)": P = @truncate(log2_10(p + 2));
            var f: P = (@"log2(10)" + @intFromBool(k >= 0)) *% @abs(k);
            if (k < 0) {
                f = 0 -% f;
            }
            const g = pow2(@truncate(f >> (p - bits)));
            return g + 1;
        }

        /// Returns the high bits of `a * b`, rounded to the nearest odd number if some bits were truncated.
        fn mulHighRoundToOdd(g: Cx2, b: C) C {
            const bits: comptime_int = @typeInfo(C).int.bits;

            const al: Cx2 = @as(C, @truncate(g));
            const ah: Cx2 = g >> bits;
            const low = (al * b) >> bits;
            const high = ah * b;
            const out = high + low; // Cannot overflow
            const frac: C = @truncate(out);
            const int: C = @truncate(out >> bits);
            return int | @intFromBool(frac > 1);
        }

        /// Returns the decimal scientific representation of `w`.
        /// The result is not normalized, i.e., the digits may have trailing zeros.
        /// `w` is asserted to be in the interval `[1, 2)`.
        pub fn toDecimal(w: S, e: _E) Decimal {
            assert(math.isFinite(w));
            assert(1 <= w and w < 2);

            const mant_bits = math.floatMantissaBits(S);
            const mant_mask = (1 << mant_bits) - 1;
            const fract_bits = math.floatFractionalBits(S);
            const fract_mask = (1 << fract_bits) - 1;

            const br: C = @as(meta.Int(.unsigned, @typeInfo(S).float.bits), @bitCast(w));

            const significand: C = if (mant_bits != fract_bits)
                // Hidden bit is always stored
                br & mant_mask
            else
                // Add hidden bit for normal numbers
                br & mant_mask | (@as(C, 1) << fract_bits);
            // Account for significand being multiplied by 2^fract_bits
            const exponent = @as(E, e) - fract_bits;

            // Fast path
            if (0 <= -exponent and -exponent <= fract_bits and
                divisibleByPow2(C, significand, @intCast(-exponent)))
            {
                return .{ .digits = significand >> @intCast(-exponent), .exponent = 0 };
            }

            const fraction = br & fract_mask;
            const is_even = significand % 2 == 0;
            const lower_boundary_is_closer = fraction == 0;

            const cb_l = 4 * significand - 2 + @intFromBool(lower_boundary_is_closer);
            const cb = 4 * significand;
            const cb_r = 4 * significand + 2;
            const k = computeK(exponent, lower_boundary_is_closer);
            const h: u3 = @intCast(exponent + floorLog2Pow10(-k) + 1);
            assert(1 <= h and h <= 4);

            // Convert cb and friends to decimal
            const scale = pow10(-k);
            const vb_l = mulHighRoundToOdd(scale, cb_l << h);
            const vb = mulHighRoundToOdd(scale, cb << h);
            const vb_r = mulHighRoundToOdd(scale, cb_r << h);

            // Create inclusive bounds according to "round ties to even" rule
            const lower = vb_l + @intFromBool(!is_even);
            const upper = vb_r - @intFromBool(!is_even);
            const s = vb / 4;

            if (s >= 10) {
                const sp = s / 10;
                const up_inside = lower <= 40 * sp;
                const wp_inside = 40 * sp + 40 <= upper;
                // if (up_inside || wp_inside) // NB: At most one of u' and w' is in R_v.
                if (up_inside != wp_inside) {
                    return .{ .digits = sp + @intFromBool(wp_inside), .exponent = k + 1 };
                }
            }

            const u_inside = lower <= 4 * s;
            const w_inside = 4 * s + 4 <= upper;
            if (u_inside != w_inside) {
                return .{ .digits = s + @intFromBool(w_inside), .exponent = k };
            }

            // s & 1 == vb & 0x4
            const mid = 4 * s + 2; // = 2(s + t)
            const round_up = vb > mid or (vb == mid and (s & 1) != 0);

            return .{ .digits = s + @intFromBool(round_up), .exponent = k };
        }
    };
}
