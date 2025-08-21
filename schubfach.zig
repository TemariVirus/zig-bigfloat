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
            // std.debug.print("{b}\n", .{(@as(u32, @bitCast(f)) & ((1 << 23) - 1)) | (1 << 23)});
            std.debug.print("{e}\na {s}\ne {s}\n\n", .{ f, actual, expected });
        }

        f = math.nextAfter(F, f, math.inf(F));
        i += 1;
        if (i % (1 << 24) == 0) {
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
    // TODO: use stricter bounds
    return 1 + switch (T) {
        f16 => 11,
        f32 => 24,
        f64 => 53,
        f80 => 64,
        f128 => 113,
        else => @compileError("T must be a float type"),
    };
}

fn Carrier(T: type) type {
    return std.meta.Int(.unsigned, @typeInfo(T).float.bits);
}

/// Returns whether x is divisible by 2^n
fn divisibleByPow2(x: anytype, n: anytype) bool {
    const pow2 = @as(@TypeOf(x), 1) << @intCast(n);
    return (x & (pow2 - 1)) == 0;
}

/// Returns floor(x / 2^n)
fn floorDivPow2(x: anytype, n: comptime_int) @TypeOf(x) {
    return x >> n;
}

// Returns floor(log_2(10^e))
fn floorLog2Pow10(e: i32) i32 {
    // TODO: what are these magic numbers? Can we extend them?
    assert(e >= -1233);
    assert(e <= 1233);
    return floorDivPow2(e * 1741647, 19);
}

fn roundToOdd(g: u64, cp: u32) u32 {
    const g0 = g & 0xFFFFFFFF;
    const g1 = g >> 32;
    const p0 = (g0 * cp) >> 32;
    const p1 = g1 * cp;
    const p = p1 + p0; // Cannot overflow
    const y0: u32 = @truncate(p);
    const y1: u32 = @truncate(p >> 32);
    return y1 | @intFromBool(y0 > 1);

    // const p: u128 = @as(u128, g) * cp;
    // const y1: u32 = @truncate(p >> 64);
    // const y0: u32 = @truncate(p >> 32);
    // return y1 | @intFromBool(y0 > 1);
}

fn computePow10Single(k: i32) u64 {
    // There are unique beta and r such that 10^k = beta 2^r and
    // 2^63 <= beta < 2^64, namely r = floor(log_2 10^k) - 63 and
    // beta = 2^-r 10^k.
    // Let g = ceil(beta), so (g-1) 2^r < 10^k <= g 2^r, with the latter
    // value being a pretty good overestimate for 10^k.

    // NB: Since for all the required exponents k, we have g < 2^64,
    //     all constants can be stored in 128-bit integers.

    const kMin = -31;
    const kMax = 45;
    const g: [kMax - kMin + 1]u64 = .{
        0x81CEB32C4B43FCF5, // -31
        0xA2425FF75E14FC32, // -30
        0xCAD2F7F5359A3B3F, // -29
        0xFD87B5F28300CA0E, // -28
        0x9E74D1B791E07E49, // -27
        0xC612062576589DDB, // -26
        0xF79687AED3EEC552, // -25
        0x9ABE14CD44753B53, // -24
        0xC16D9A0095928A28, // -23
        0xF1C90080BAF72CB2, // -22
        0x971DA05074DA7BEF, // -21
        0xBCE5086492111AEB, // -20
        0xEC1E4A7DB69561A6, // -19
        0x9392EE8E921D5D08, // -18
        0xB877AA3236A4B44A, // -17
        0xE69594BEC44DE15C, // -16
        0x901D7CF73AB0ACDA, // -15
        0xB424DC35095CD810, // -14
        0xE12E13424BB40E14, // -13
        0x8CBCCC096F5088CC, // -12
        0xAFEBFF0BCB24AAFF, // -11
        0xDBE6FECEBDEDD5BF, // -10
        0x89705F4136B4A598, //  -9
        0xABCC77118461CEFD, //  -8
        0xD6BF94D5E57A42BD, //  -7
        0x8637BD05AF6C69B6, //  -6
        0xA7C5AC471B478424, //  -5
        0xD1B71758E219652C, //  -4
        0x83126E978D4FDF3C, //  -3
        0xA3D70A3D70A3D70B, //  -2
        0xCCCCCCCCCCCCCCCD, //  -1
        0x8000000000000000, //   0
        0xA000000000000000, //   1
        0xC800000000000000, //   2
        0xFA00000000000000, //   3
        0x9C40000000000000, //   4
        0xC350000000000000, //   5
        0xF424000000000000, //   6
        0x9896800000000000, //   7
        0xBEBC200000000000, //   8
        0xEE6B280000000000, //   9
        0x9502F90000000000, //  10
        0xBA43B74000000000, //  11
        0xE8D4A51000000000, //  12
        0x9184E72A00000000, //  13
        0xB5E620F480000000, //  14
        0xE35FA931A0000000, //  15
        0x8E1BC9BF04000000, //  16
        0xB1A2BC2EC5000000, //  17
        0xDE0B6B3A76400000, //  18
        0x8AC7230489E80000, //  19
        0xAD78EBC5AC620000, //  20
        0xD8D726B7177A8000, //  21
        0x878678326EAC9000, //  22
        0xA968163F0A57B400, //  23
        0xD3C21BCECCEDA100, //  24
        0x84595161401484A0, //  25
        0xA56FA5B99019A5C8, //  26
        0xCECB8F27F4200F3A, //  27
        0x813F3978F8940985, //  28
        0xA18F07D736B90BE6, //  29
        0xC9F2C9CD04674EDF, //  30
        0xFC6F7C4045812297, //  31
        0x9DC5ADA82B70B59E, //  32
        0xC5371912364CE306, //  33
        0xF684DF56C3E01BC7, //  34
        0x9A130B963A6C115D, //  35
        0xC097CE7BC90715B4, //  36
        0xF0BDC21ABB48DB21, //  37
        0x96769950B50D88F5, //  38
        0xBC143FA4E250EB32, //  39
        0xEB194F8E1AE525FE, //  40
        0x92EFD1B8D0CF37BF, //  41
        0xB7ABC627050305AE, //  42
        0xE596B7B0C643C71A, //  43
        0x8F7E32CE7BEA5C70, //  44
        0xB35DBF821AE4F38C, //  45
    };

    assert(k >= kMin);
    assert(k <= kMax);
    return g[@intCast(k - kMin)];
}

fn Decimal(T: type) type {
    return struct {
        digits: Carrier(T),
        exponent: i32,

        pub fn removeTrailingZeros(self: @This()) @This() {
            var copy = self;
            while (copy.digits % 10 == 0 and copy.digits != 0) {
                copy.digits /= 10;
                copy.exponent += 1;
            }
            return copy;
        }
    };
}

fn toDecimal(T: type, w: T) Decimal(T) {
    std.debug.assert(math.isFinite(w));
    std.debug.assert(w > 0);
    // const NextT = switch (T) {
    //     f16 => f32,
    //     f32 => f64,
    //     f64 => f80,
    //     f80 => f128,
    //     // f128 => ???,
    //     else => @compileError("Unsupported type"),
    // };

    const mant_bits = math.floatMantissaBits(T);
    const mant_mask = (1 << mant_bits) - 1;
    const fract_bits = math.floatFractionalBits(T);
    const fract_mask = (1 << fract_bits) - 1;
    const exp_bits = math.floatExponentBits(T);
    const exp_bias = math.floatExponentMax(T);

    const C = Carrier(T);
    const ExpInt = std.meta.Int(.signed, exp_bits);
    const ExpMask = std.meta.Int(.unsigned, exp_bits);
    // const E = math.IntFittingRange(math.floatExponentMin(T) - mant_bits, math.floatExponentMax(T));

    const br: C = @bitCast(w);
    const exp_masked: ExpMask = @truncate(br >> mant_bits);
    const is_normal = exp_masked != 0;
    // TODO: fix when w is infinity or true min
    // const w_minus: NextT = @floatCast(math.nextAfter(T, w, -math.inf(T)));
    // const w_plus: NextT = @floatCast(math.nextAfter(T, w, math.inf(T)));

    // F_w = math.ldexp(@as(T, @floatFromInt(f_c)), -mant_bits)
    const c: C = if (T == f80)
        br & mant_mask
    else
        br & mant_mask | (@as(C, @intFromBool(is_normal)) << fract_bits);
    const e_w: i32 = if (is_normal)
        @as(ExpInt, @bitCast(exp_masked -% exp_bias))
    else
        math.floatExponentMin(T);
    const q: i32 = e_w - fract_bits;
    if (is_normal and -(fract_bits + 1) < q and q <= 0 and divisibleByPow2(c, -q)) {
        // std.debug.print("{} * 2^{} = {}\n", .{ c, q, c >> @intCast(-q) });
        return .{ .digits = c >> @intCast(-q), .exponent = 0 };
    }

    const ieee_fraction = br & fract_mask;
    const is_even = c % 2 == 0;
    const lower_boundary_is_closer = ieee_fraction == 0 and exp_masked > 1;

    const cb_l = 4 * c - 2 + @intFromBool(lower_boundary_is_closer);
    const cb = 4 * c;
    const cb_r = 4 * c + 2;
    // TODO: what are these magic numbers? Can we extend them?
    assert(q >= -1500);
    assert(q <= 1500);
    // 10^k == 2^q
    // k = floor(q * log_10(2))
    // 1262611 = floor(2^22 * log_10(2))
    // -524032 = floor(2^22 * log_10(0.75))
    //
    // 94.354182398921953954 * 2^(3Q/8)
    // min_k = -149
    // log2(149) + 1.73202084564 = 8.9...
    // round up and add 3 just in case = u12
    // 1233 = floor(2^12 * log_10(2))
    // -512 = floor(2^12 * log_10(0.75))
    // const k = floorDivPow2(q * 1262611 - @as(i32, if (lower_boundary_is_closer) 524031 else 0), 22);
    const k = floorDivPow2(q * 1233 - @as(i32, if (lower_boundary_is_closer) 512 else 0), 12);

    const h: u3 = @intCast(q + floorLog2Pow10(-k) + 1);
    assert(1 <= h and h <= 4);

    const pow10: u64 = computePow10Single(-k);
    const vb_l = roundToOdd(pow10, cb_l << h);
    const vb = roundToOdd(pow10, cb << h);
    // std.debug.print("{} {} {}\n", .{ cb_l, cb, cb_r });
    const vb_r = roundToOdd(pow10, cb_r << h);
    // std.debug.print("{} {} {}\n", .{ vb_l, vb, vb_r });
    // std.debug.print("{} {} {}\n", .{ vb_l / 4, vb / 4, vb_r / 4 });

    // Create inclusive bounds according to "round ties to even" rule
    const lower = vb_l + @intFromBool(!is_even);
    const upper = vb_r - @intFromBool(!is_even);
    const s = vb / 4;

    // // const delta: NextT = (w_plus - w_minus) * 0.5;
    // const delta: NextT = math.ldexp(@as(
    //     NextT,
    //     if (c == (1 << mant_bits) and is_normal)
    //         0.75
    //     else
    //         1,
    // ), q);

    // // Step 1
    // const k_0 = -@floor(@log10(delta));

    // // Step 2
    // const k_1_w0: D = @intFromFloat(@floor(math.pow(NextT, 10, k_0 - 1) * @as(NextT, @floatCast(w))));
    // const k_1_w1 = k_1_w0 + 1;

    // // Step 3
    // const k_w0: D = @intFromFloat(@floor(math.pow(NextT, 10, k_0) * @as(NextT, @floatCast(w))));
    // const k_w1 = k_w0 + 1;

    // std.debug.print("{d} {d}\n", .{ k_1_w0, k_1_w1 });
    // std.debug.print("{d} {d}\n", .{ k_w0, k_w1 });

    // std.debug.print("{} {} {}\n", .{ cb_l, cb, cb_r });
    // std.debug.print("{} {}\n", .{ k, h });
    // std.debug.print("{} {} {}\n", .{ lower, s, upper });

    // Continue schubfach
    if (s >= 10) // vb >= 40
    {
        const sp = s / 10; // = vb / 40
        const up_inside = lower <= 40 * sp;
        // std.debug.print("{} {}\n", .{ lower, 40 * sp });
        const wp_inside = 40 * sp + 40 <= upper;
        // std.debug.print("{} {}\n", .{ lower / 4, upper / 4 });
        // std.debug.print("{} {}\n", .{ up_inside, wp_inside });
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

    // NB: s & 1 == vb & 0x4
    const mid = 4 * s + 2; // = 2(s + t)
    const round_up = vb > mid or (vb == mid and (s & 1) != 0);

    return .{ .digits = s + @intFromBool(round_up), .exponent = k };
}

fn formatScientific(Float: type, w: *std.Io.Writer, f: Float) !void {
    // Special cases
    if (math.signbit(f)) try w.writeByte('-');
    if (math.isNan(f)) return w.writeAll("nan");
    if (math.isInf(f)) return w.writeAll("inf");
    if (f == 0) return w.writeAll("0e0");
    if (f < 0) return formatScientific(Float, w, -f);

    const decimal = toDecimal(Float, f).removeTrailingZeros();
    const digits_str = blk: {
        var buf: [digitCount(Float) + 1]u8 = undefined;
        var digit_writer = std.Io.Writer.fixed(&buf);
        digit_writer.print("{d}", .{decimal.digits}) catch unreachable;
        break :blk digit_writer.buffered();
    };

    const digit_count: i32 = @intCast(digits_str.len);
    const actual_exponent = decimal.exponent + digit_count - 1;
    if (digit_count == 1) return w.print("{s}e{d}", .{ digits_str, actual_exponent });
    return w.print("{s}.{s}e{d}", .{ digits_str[0..1], digits_str[1..], actual_exponent });
}
