const std = @import("std");
const BigFloat = @import("bigfloat").BigFloat;
const BigFloat2 = @import("bigfloat").BigFloat2;
const ExpFloat = @import("bigfloat").ExpFloat;

// CPU: i7-8700
pub fn main() void {
    // ==============================
    // Addition: not stored variables
    // ==============================
    // NativeInt(i64)        1.383GFLOP/s over 0.776s
    // NativeFloat(f32)      1.392GFLOP/s over 0.771s
    // NativeFloat(f64)      1.401GFLOP/s over 0.767s
    // NativeFloat(f128)   112.173MFLOP/s over 1.197s | 12.485x
    // BigFloat(f32,i32)     0.149GFLOP/s over 0.902s |  9.414x
    // BigFloat(f32,i96)     0.136GFLOP/s over 0.984s | 10.269x
    // BigFloat(f64,i64)     0.140GFLOP/s over 0.959s | 10.011x
    // BigFloat(f128,i128)  18.598MFLOP/s over 0.902s | 75.306x
    // BigFloat(i32,i32)   129.902MFLOP/s over 0.517s | 10.781x
    // BigFloat(i32,i96)   115.499MFLOP/s over 0.581s | 12.126x
    // BigFloat(i64,i64)   123.853MFLOP/s over 0.542s | 11.308x
    // BigFloat(i128,i128)  95.253MFLOP/s over 0.705s | 14.703x
    // ExpFloat              1.403GFLOP/s over 0.766s |  0.999x
    // ==========================
    // Addition: stored variables
    // ==========================
    // NativeInt(i64)        1.405GFLOP/s over 0.764s
    // NativeFloat(f32)      1.392GFLOP/s over 0.772s
    // NativeFloat(f64)      1.405GFLOP/s over 0.764s
    // NativeFloat(f128)   112.555MFLOP/s over 1.192s | 12.483x
    // BigFloat(f32,i32)     0.514GFLOP/s over 1.045s |  2.734x
    // BigFloat(f32,i96)     0.411GFLOP/s over 1.307s |  3.419x
    // BigFloat(f64,i64)     0.417GFLOP/s over 1.288s |  3.370x
    // BigFloat(f128,i128)  20.484MFLOP/s over 0.819s | 68.592x
    // BigFloat(i32,i32)     0.500GFLOP/s over 0.269s |  2.812x
    // BigFloat(i32,i96)     0.345GFLOP/s over 0.389s |  4.067x
    // BigFloat(i64,i64)     0.456GFLOP/s over 0.294s |  3.083x
    // BigFloat(i128,i128)   0.210GFLOP/s over 0.639s |  6.685x
    // ExpFloat              1.394GFLOP/s over 0.770s |  1.008x
    benchAdd();

    // ====================================
    // Multiplication: not stored variables
    // ====================================
    // NativeInt(i64)        1.331GFLOP/s over 0.807s
    // NativeFloat(f32)      1.311GFLOP/s over 0.819s
    // NativeFloat(f64)      1.349GFLOP/s over 0.796s
    // NativeFloat(f128)    86.173MFLOP/s over 0.779s | 15.652x
    // BigFloat(f32,i32)     1.365GFLOP/s over 0.787s |  0.988x
    // BigFloat(f32,i96)     1.092GFLOP/s over 0.984s |  1.236x
    // BigFloat(f64,i64)     1.083GFLOP/s over 0.991s |  1.245x
    // BigFloat(f128,i128)  21.497MFLOP/s over 0.780s | 62.741x
    // BigFloat(i32,i32)     1.345GFLOP/s over 0.798s |  1.003x
    // BigFloat(i32,i96)     1.079GFLOP/s over 0.995s |  1.250x
    // BigFloat(i64,i64)     1.085GFLOP/s over 0.989s |  1.243x
    // BigFloat(i128,i128)   1.365GFLOP/s over 0.393s |  0.988x
    // ExpFloat              1.345GFLOP/s over 0.798s |  1.003x
    // ================================
    // Multiplication: stored variables
    // ================================
    // NativeInt(i64)        1.365GFLOP/s over 0.787s
    // NativeFloat(f32)      1.341GFLOP/s over 0.801s
    // NativeFloat(f64)      1.363GFLOP/s over 0.788s
    // NativeFloat(f128)    87.979MFLOP/s over 0.763s | 15.490x
    // BigFloat(f32,i32)     0.840GFLOP/s over 0.639s |  1.623x
    // BigFloat(f32,i96)     0.769GFLOP/s over 0.698s |  1.771x
    // BigFloat(f64,i64)     0.660GFLOP/s over 0.813s |  2.064x
    // BigFloat(f128,i128)  21.257MFLOP/s over 0.789s | 64.110x
    // BigFloat(i32,i32)     0.486GFLOP/s over 0.552s |  2.804x
    // BigFloat(i32,i96)     0.424GFLOP/s over 0.632s |  3.211x
    // BigFloat(i64,i64)     0.397GFLOP/s over 0.677s |  3.436x
    // BigFloat(i128,i128) 130.940MFLOP/s over 1.025s | 10.408x
    // ExpFloat              1.302GFLOP/s over 0.824s |  1.046x
    benchMul();
}

fn NativeInt(T: type) type {
    return struct {
        i: T,

        const Self = @This();

        pub const inf: Self = .{ .i = std.math.maxInt(T) };
        pub const minusInf: Self = .{ .i = std.math.minInt(T) + 1 };
        pub const nan: Self = .{ .i = std.math.minInt(T) };

        pub fn from(value: anytype) Self {
            return .{ .i = std.math.lossyCast(T, value) };
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            return .{ .i = lhs.i +| rhs.i };
        }

        pub fn mul(lhs: Self, rhs: Self) Self {
            return .{ .i = lhs.i *| rhs.i };
        }
    };
}

fn NativeFloat(T: type) type {
    return struct {
        f: T,

        const Self = @This();

        pub const inf: Self = .{ .f = std.math.inf(T) };
        pub const minusInf: Self = .{ .f = -std.math.inf(T) };
        pub const nan: Self = .{ .f = std.math.nan(T) };

        pub fn from(value: T) Self {
            return .{ .f = value };
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f + rhs.f };
        }

        pub fn mul(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f * rhs.f };
        }
    };
}

fn printResult(T: type, flops: u64, time_taken: u64, base_flops: ?f64) void {
    const name = blk: {
        const dot = if (std.mem.lastIndexOfScalar(u8, @typeName(T), '.')) |dot| dot + 1 else 0;
        break :blk @typeName(T)[dot..];
    };
    var buf: [9]u8 = undefined;
    const flops_str = std.fmt.bufPrint(&buf, "{:.3}", .{std.fmt.fmtIntSizeDec(flops)}) catch unreachable;
    std.debug.print("{s:<19} {s:>8}FLOP/s over {d:>5.3}s", .{
        name,
        flops_str[0 .. flops_str.len - 1],
        @as(f64, @floatFromInt(time_taken)) / std.time.ns_per_s,
    });
    if (base_flops) |base| {
        std.debug.print(" | {d:>6.3}x", .{base / @as(f64, @floatFromInt(flops))});
    }
    std.debug.print("\n", .{});
}

fn timeAdd1(F: type, comptime iters: u64, base_flops: ?f64) f64 {
    const start = std.time.nanoTimestamp();

    for (0..iters) |_| {
        std.mem.doNotOptimizeAway((comptime F.from(0)).add(comptime .from(0)));
        std.mem.doNotOptimizeAway((comptime F.from(1)).add(comptime .from(0)));
        std.mem.doNotOptimizeAway((comptime F.from(123)).add(comptime .from(321)));
        std.mem.doNotOptimizeAway((comptime F.from(1.5)).add(comptime .from(3.25)));
        std.mem.doNotOptimizeAway((comptime F.from(1e38)).add(comptime .from(1e-38)));
        std.mem.doNotOptimizeAway((comptime F.from(1e38)).add(comptime .from(1e30)));
        // std.mem.doNotOptimizeAway((comptime F.from(1e38)).add(comptime .from(-0.99e38)));
        std.mem.doNotOptimizeAway((comptime F.from(12)).add(.inf));
        // std.mem.doNotOptimizeAway(F.inf.add(.minusInf));
        std.mem.doNotOptimizeAway(F.nan.add(comptime .from(12)));
    }
    const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
    const flops = (8 * iters) * std.time.ns_per_s / time_taken;

    printResult(F, flops, time_taken, base_flops);
    return @floatFromInt(flops);
}

fn timeAdd2(F: type, comptime iters: u64, base_flops: ?f64) f64 {
    const zero: F = .from(0);
    const one: F = .from(1);
    const @"123": F = .from(123);
    const @"321": F = .from(321);
    const @"1.5": F = .from(1.5);
    const @"3.25": F = .from(3.25);
    const @"1e38": F = .from(1e38);
    const @"1e-38": F = .from(1e-38);
    const @"1e30": F = .from(1e30);
    const @"12": F = .from(12);

    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        std.mem.doNotOptimizeAway(zero.add(zero));
        std.mem.doNotOptimizeAway(one.add(zero));
        std.mem.doNotOptimizeAway(@"123".add(@"321"));
        std.mem.doNotOptimizeAway(@"1.5".add(@"3.25"));
        std.mem.doNotOptimizeAway(@"1e38".add(@"1e-38"));
        std.mem.doNotOptimizeAway(@"1e38".add(@"1e30"));
        // std.mem.doNotOptimizeAway(F.from(1e38).add(.from(-0.99e38)));
        std.mem.doNotOptimizeAway(@"12".add(.inf));
        // std.mem.doNotOptimizeAway(F.inf.add(.minusInf));
        std.mem.doNotOptimizeAway(F.nan.add(@"12"));
    }
    const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
    const flops = (8 * iters) * std.time.ns_per_s / time_taken;

    printResult(F, flops, time_taken, base_flops);
    return @floatFromInt(flops);
}

fn benchAdd() void {
    std.debug.print(
        \\==============================
        \\Addition: not stored variables
        \\==============================
        \\
    , .{});
    {
        _ = timeAdd1(NativeInt(i64), 1 << 27, null);

        const base1 = timeAdd1(NativeFloat(f32), 1 << 27, null);
        const base2 = timeAdd1(NativeFloat(f64), 1 << 27, null);
        const base = @max(base1, base2);

        _ = timeAdd1(NativeFloat(f128), 1 << 24, base);
        _ = timeAdd1(BigFloat(f32, i32), 1 << 27, base);
        _ = timeAdd1(BigFloat(f32, i96), 1 << 27, base);
        _ = timeAdd1(BigFloat(f64, i64), 1 << 27, base);
        _ = timeAdd1(BigFloat(f128, i128), 1 << 21, base);
        _ = timeAdd1(BigFloat2(i32, i32), 1 << 27, base);
        _ = timeAdd1(BigFloat2(i32, i96), 1 << 26, base);
        _ = timeAdd1(BigFloat2(i64, i64), 1 << 26, base);
        _ = timeAdd1(BigFloat2(i128, i128), 1 << 26, base);
        _ = timeAdd1(ExpFloat, 1 << 27, base);
    }

    std.debug.print(
        \\==========================
        \\Addition: stored variables
        \\==========================
        \\
    , .{});
    {
        _ = timeAdd2(NativeInt(i64), 1 << 27, null);

        const base1 = timeAdd2(NativeFloat(f32), 1 << 27, null);
        const base2 = timeAdd2(NativeFloat(f64), 1 << 27, null);
        const base = @max(base1, base2);

        _ = timeAdd2(NativeFloat(f128), 1 << 24, base);
        _ = timeAdd2(BigFloat(f32, i32), 1 << 26, base);
        _ = timeAdd2(BigFloat(f32, i96), 1 << 26, base);
        _ = timeAdd2(BigFloat(f64, i64), 1 << 26, base);
        _ = timeAdd2(BigFloat(f128, i128), 1 << 21, base);
        _ = timeAdd2(BigFloat2(i32, i32), 1 << 26, base);
        _ = timeAdd2(BigFloat2(i32, i96), 1 << 25, base);
        _ = timeAdd2(BigFloat2(i64, i64), 1 << 26, base);
        _ = timeAdd2(BigFloat2(i128, i128), 1 << 24, base);
        _ = timeAdd2(ExpFloat, 1 << 27, base);
    }
}

fn timeMul1(F: type, comptime iters: u64, base_flops: ?f64) f64 {
    const start = std.time.nanoTimestamp();

    for (0..iters) |_| {
        std.mem.doNotOptimizeAway((comptime F.from(0)).mul(comptime .from(0)));
        std.mem.doNotOptimizeAway((comptime F.from(1)).mul(comptime .from(0)));
        std.mem.doNotOptimizeAway((comptime F.from(123)).mul(comptime .from(321)));
        std.mem.doNotOptimizeAway((comptime F.from(1.5)).mul(comptime .from(3.25)));
        std.mem.doNotOptimizeAway((comptime F.from(1e38)).mul(comptime .from(1e-38)));
        std.mem.doNotOptimizeAway((comptime F.from(1e38)).mul(comptime .from(1e30)));
        // std.mem.doNotOptimizeAway((comptime F.from(1e38)).mul(comptime .from(-0.99e38)));
        std.mem.doNotOptimizeAway((comptime F.from(12)).mul(.inf));
        // std.mem.doNotOptimizeAway(F.inf.mul(.minusInf));
        std.mem.doNotOptimizeAway(F.nan.mul(comptime .from(12)));
    }
    const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
    const flops = (8 * iters) * std.time.ns_per_s / time_taken;

    printResult(F, flops, time_taken, base_flops);
    return @floatFromInt(flops);
}

fn timeMul2(F: type, comptime iters: u64, base_flops: ?f64) f64 {
    const zero: F = .from(0);
    const one: F = .from(1);
    const @"123": F = .from(123);
    const @"321": F = .from(321);
    const @"1.5": F = .from(1.5);
    const @"3.25": F = .from(3.25);
    const @"1e38": F = .from(1e38);
    const @"1e-38": F = .from(1e-38);
    const @"1e30": F = .from(1e30);
    const @"12": F = .from(12);

    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        std.mem.doNotOptimizeAway(zero.mul(zero));
        std.mem.doNotOptimizeAway(one.mul(zero));
        std.mem.doNotOptimizeAway(@"123".mul(@"321"));
        std.mem.doNotOptimizeAway(@"1.5".mul(@"3.25"));
        std.mem.doNotOptimizeAway(@"1e38".mul(@"1e-38"));
        std.mem.doNotOptimizeAway(@"1e38".mul(@"1e30"));
        // std.mem.doNotOptimizeAway(F.from(1e38).mul(.from(-0.99e38)));
        std.mem.doNotOptimizeAway(@"12".mul(.inf));
        // std.mem.doNotOptimizeAway(F.inf.mul(.minusInf));
        std.mem.doNotOptimizeAway(F.nan.mul(@"12"));
    }
    const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
    const flops = (8 * iters) * std.time.ns_per_s / time_taken;

    printResult(F, flops, time_taken, base_flops);
    return @floatFromInt(flops);
}

fn benchMul() void {
    std.debug.print(
        \\====================================
        \\Multiplication: not stored variables
        \\====================================
        \\
    , .{});
    {
        _ = timeMul1(NativeInt(i64), 1 << 27, null);

        const base1 = timeMul1(NativeFloat(f32), 1 << 27, null);
        const base2 = timeMul1(NativeFloat(f64), 1 << 27, null);
        const base = @max(base1, base2);

        _ = timeMul1(NativeFloat(f128), 1 << 23, base);
        _ = timeMul1(BigFloat(f32, i32), 1 << 27, base);
        _ = timeMul1(BigFloat(f32, i96), 1 << 27, base);
        _ = timeMul1(BigFloat(f64, i64), 1 << 27, base);
        _ = timeMul1(BigFloat(f128, i128), 1 << 21, base);
        _ = timeMul1(BigFloat2(i32, i32), 1 << 27, base);
        _ = timeMul1(BigFloat2(i32, i96), 1 << 27, base);
        _ = timeMul1(BigFloat2(i64, i64), 1 << 27, base);
        _ = timeMul1(BigFloat2(i128, i128), 1 << 26, base);
        _ = timeMul1(ExpFloat, 1 << 27, base);
    }

    std.debug.print(
        \\================================
        \\Multiplication: stored variables
        \\================================
        \\
    , .{});
    {
        _ = timeMul2(NativeInt(i64), 1 << 27, null);

        const base1 = timeMul2(NativeFloat(f32), 1 << 27, null);
        const base2 = timeMul2(NativeFloat(f64), 1 << 27, null);
        const base = @max(base1, base2);

        _ = timeMul2(NativeFloat(f128), 1 << 23, base);
        _ = timeMul2(BigFloat(f32, i32), 1 << 26, base);
        _ = timeMul2(BigFloat(f32, i96), 1 << 26, base);
        _ = timeMul2(BigFloat(f64, i64), 1 << 26, base);
        _ = timeMul2(BigFloat(f128, i128), 1 << 21, base);
        _ = timeMul2(BigFloat2(i32, i32), 1 << 25, base);
        _ = timeMul2(BigFloat2(i32, i96), 1 << 25, base);
        _ = timeMul2(BigFloat2(i64, i64), 1 << 25, base);
        _ = timeMul2(BigFloat2(i128, i128), 1 << 24, base);
        _ = timeMul2(ExpFloat, 1 << 27, base);
    }
}
