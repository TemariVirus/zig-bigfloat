const std = @import("std");

const BigFloat = @import("bigfloat").BigFloat;

// CPU: i7-1165G7
pub fn main() void {
    // ==========
    //  Addition
    // ==========
    // NativeFloat(f32)      1.406GFLOP/s over 0.954s
    // NativeFloat(f64)      1.407GFLOP/s over 0.954s
    // NativeFloat(f128)     0.148GFLOP/s over 0.568s |  9.521x
    // BigFloat(f32,i32)     0.465GFLOP/s over 0.722s |  3.027x
    // BigFloat(f32,i96)     0.417GFLOP/s over 0.402s |  3.371x
    // BigFloat(f64,i64)     0.375GFLOP/s over 0.448s |  3.754x
    // BigFloat(f128,i128)  20.468MFLOP/s over 1.025s | 68.737x
    benchAdd();

    // ================
    //  Multiplication
    // ================
    // NativeFloat(f32)      1.432GFLOP/s over 0.937s
    // NativeFloat(f64)      1.423GFLOP/s over 0.943s
    // NativeFloat(f128)   114.939MFLOP/s over 0.730s | 12.463x
    // BigFloat(f32,i32)     0.532GFLOP/s over 0.631s |  2.692x
    // BigFloat(f32,i96)     0.498GFLOP/s over 0.674s |  2.879x
    // BigFloat(f64,i64)     0.527GFLOP/s over 0.637s |  2.717x
    // BigFloat(f128,i128)  23.111MFLOP/s over 0.907s | 61.981x
    benchMul();
}

fn NativeFloat(T: type) type {
    return struct {
        f: T,

        const Self = @This();

        pub const inf: Self = .{ .f = std.math.inf(T) };
        pub const minus_inf: Self = .{ .f = -std.math.inf(T) };
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

fn timeAdd(F: type, comptime iters: u64, base_flops: ?f64) f64 {
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
    const @"-0.99e38": F = .from(-0.99e38);

    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        std.mem.doNotOptimizeAway(zero.add(zero));
        std.mem.doNotOptimizeAway(one.add(zero));
        std.mem.doNotOptimizeAway(@"123".add(@"321"));
        std.mem.doNotOptimizeAway(@"1.5".add(@"3.25"));
        std.mem.doNotOptimizeAway(@"1e38".add(@"1e-38"));
        std.mem.doNotOptimizeAway(@"1e38".add(@"1e30"));
        std.mem.doNotOptimizeAway(F.from(1e38).add(@"-0.99e38"));
        std.mem.doNotOptimizeAway(@"12".add(.inf));
        std.mem.doNotOptimizeAway(F.inf.add(.minus_inf));
        std.mem.doNotOptimizeAway(F.nan.add(@"12"));
    }
    const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
    const flops = (10 * iters) * std.time.ns_per_s / time_taken;

    printResult(F, flops, time_taken, base_flops);
    return @floatFromInt(flops);
}

fn benchAdd() void {
    std.debug.print(
        \\==========
        \\ Addition
        \\==========
        \\
    , .{});
    {
        const base1 = timeAdd(NativeFloat(f32), 1 << 27, null);
        const base2 = timeAdd(NativeFloat(f64), 1 << 27, null);
        const base = @max(base1, base2);

        _ = timeAdd(NativeFloat(f128), 1 << 23, base);
        _ = timeAdd(BigFloat(f32, i32), 1 << 25, base);
        _ = timeAdd(BigFloat(f32, i96), 1 << 24, base);
        _ = timeAdd(BigFloat(f64, i64), 1 << 24, base);
        _ = timeAdd(BigFloat(f128, i128), 1 << 21, base);
    }
}

fn timeMul(F: type, comptime iters: u64, base_flops: ?f64) f64 {
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
    const @"-0.99e38": F = .from(-0.99e38);

    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        std.mem.doNotOptimizeAway(zero.mul(zero));
        std.mem.doNotOptimizeAway(one.mul(zero));
        std.mem.doNotOptimizeAway(@"123".mul(@"321"));
        std.mem.doNotOptimizeAway(@"1.5".mul(@"3.25"));
        std.mem.doNotOptimizeAway(@"1e38".mul(@"1e-38"));
        std.mem.doNotOptimizeAway(@"1e38".mul(@"1e30"));
        std.mem.doNotOptimizeAway(F.from(1e38).mul(@"-0.99e38"));
        std.mem.doNotOptimizeAway(@"12".mul(.inf));
        std.mem.doNotOptimizeAway(F.inf.mul(.minus_inf));
        std.mem.doNotOptimizeAway(F.nan.mul(@"12"));
    }
    const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
    const flops = (10 * iters) * std.time.ns_per_s / time_taken;

    printResult(F, flops, time_taken, base_flops);
    return @floatFromInt(flops);
}

fn benchMul() void {
    std.debug.print(
        \\================
        \\ Multiplication
        \\================
        \\
    , .{});
    {
        const base1 = timeMul(NativeFloat(f32), 1 << 27, null);
        const base2 = timeMul(NativeFloat(f64), 1 << 27, null);
        const base = @max(base1, base2);

        _ = timeMul(NativeFloat(f128), 1 << 23, base);
        _ = timeMul(BigFloat(f32, i32), 1 << 25, base);
        _ = timeMul(BigFloat(f32, i96), 1 << 25, base);
        _ = timeMul(BigFloat(f64, i64), 1 << 25, base);
        _ = timeMul(BigFloat(f128, i128), 1 << 21, base);
    }
}
