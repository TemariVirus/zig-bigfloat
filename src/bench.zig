const std = @import("std");

const BigFloat = @import("bigfloat").BigFloat;

const INLINE_ITERS = 16;

// CPU: i7-1165G7
pub fn main() void {
    // ==========
    //  Addition
    // ==========
    // NativeFloat(f32)      2.189GFLOP/s over 0.990s
    // NativeFloat(f64)      2.055GFLOP/s over 1.076s
    // NativeFloat(f128)     0.149GFLOP/s over 1.061s | 14.676x
    // BigFloat(f32,i32)     0.715GFLOP/s over 0.964s |  3.063x
    // BigFloat(f32,i96)     0.622GFLOP/s over 0.844s |  3.517x
    // BigFloat(f64,i64)     0.620GFLOP/s over 0.975s |  3.529x
    // BigFloat(f128,i128)  24.081MFLOP/s over 1.006s | 90.910x
    std.debug.print(
        \\==========
        \\ Addition
        \\==========
        \\
    , .{});
    bench(runAdd, runAdd_flops, 5);

    // NativeFloat(f32)      1.839GFLOP/s over 0.821s
    // NativeFloat(f64)      1.810GFLOP/s over 0.832s
    // NativeFloat(f128)   111.263MFLOP/s over 1.023s | 16.524x
    // BigFloat(f32,i32)     0.665GFLOP/s over 1.288s |  2.763x
    // BigFloat(f32,i96)     0.666GFLOP/s over 1.211s |  2.762x
    // BigFloat(f64,i64)     0.653GFLOP/s over 0.933s |  2.817x
    // BigFloat(f128,i128)  22.936MFLOP/s over 1.550s | 80.161x
    std.debug.print(
        \\================
        \\ Multiplication
        \\================
        \\
    , .{});
    bench(runMul, runMul_flops, 5);
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

        pub fn sub(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f - rhs.f };
        }

        pub fn mul(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f * rhs.f };
        }

        pub fn div(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f / rhs.f };
        }

        pub fn pow(base: Self, exponent: Self) Self {
            return .{ .f = std.math.pow(T, base.f, exponent.f) };
        }
    };
}

fn iterCount(run: *const fn () void, target_ns: u64) u64 {
    var iters: u64 = 1;

    // Find rough number of iterations needed to take at least 50ms
    const ns_taken = while (true) : (iters *= 2) {
        const start = std.time.nanoTimestamp();
        for (0..iters) |_| {
            inline for (0..INLINE_ITERS) |_| {
                run();
            }
        }
        const ns_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
        if (ns_taken >= 50 * std.time.ns_per_ms) {
            break ns_taken;
        }
    };

    // Extrapolate to target_ns
    return (INLINE_ITERS * iters * target_ns) / ns_taken;
}

fn timeIters(run: *const fn () void, iters: u64) u64 {
    const start = std.time.nanoTimestamp();
    for (0..iters / INLINE_ITERS) |_| {
        inline for (0..INLINE_ITERS) |_| {
            run();
        }
    }
    return @intCast(std.time.nanoTimestamp() - start);
}

fn printResult(T: type, iters: u64, ns_taken: u64, base_flops: ?f64) void {
    const flops = iters * std.time.ns_per_s / ns_taken;
    const name = blk: {
        const dot = if (std.mem.lastIndexOfScalar(u8, @typeName(T), '.')) |dot| dot + 1 else 0;
        break :blk @typeName(T)[dot..];
    };

    var buf: [9]u8 = undefined;
    const flops_str = std.fmt.bufPrint(&buf, "{:.3}", .{std.fmt.fmtIntSizeDec(flops)}) catch unreachable;
    std.debug.print("{s:<19} {s:>8}FLOP/s over {d:>5.3}s", .{
        name,
        flops_str[0 .. flops_str.len - 1],
        @as(f64, @floatFromInt(ns_taken)) / std.time.ns_per_s,
    });

    if (base_flops) |base| {
        std.debug.print(" | {d:>6.3}x", .{base / @as(f64, @floatFromInt(flops))});
    }

    std.debug.print("\n", .{});
}

fn closure(func: fn (anytype) void, comptime value: anytype) *const fn () void {
    return &(struct {
        fn t() void {
            func(value);
        }
    }).t;
}

fn bench(
    run: fn (type) void,
    run_flops: u64,
    run_count: usize,
) void {
    const types = [_]type{
        NativeFloat(f32),
        NativeFloat(f64),
        NativeFloat(f128),
        BigFloat(f32, i32),
        BigFloat(f32, i96),
        BigFloat(f64, i64),
        BigFloat(f128, i128),
    };
    var iters: [types.len]u64 = undefined;
    var ns_takens: [types.len]u64 = @splat(std.math.maxInt(u64));

    inline for (types, &iters) |F, *its| {
        its.* = iterCount(closure(run, F), std.time.ns_per_s);
    }
    for (0..run_count) |_| {
        // Interleave types to minimise effects of random CPU spikes
        inline for (types, iters, &ns_takens) |F, its, *ns| {
            // `run` is deterministic, so we use the minimum to get the most accurate timing
            ns.* = @min(ns.*, timeIters(closure(run, F), its));
        }
    }

    const base1: f64 = @floatFromInt((run_flops * iters[0] * std.time.ns_per_s) / ns_takens[0]);
    const base2: f64 = @floatFromInt((run_flops * iters[1] * std.time.ns_per_s) / ns_takens[1]);
    const base_flops = @max(base1, base2);

    inline for (types[0..2], iters[0..2], ns_takens[0..2]) |F, it, ns| {
        printResult(F, run_flops * it, ns, null);
    }
    inline for (types[2..], iters[2..], ns_takens[2..]) |F, it, ns| {
        printResult(F, run_flops * it, ns, base_flops);
    }
}

const runAdd_flops = 10;
fn runAdd(F: type) void {
    const zero: F = comptime .from(0);
    const one: F = comptime .from(1);
    const @"123": F = comptime .from(123);
    const @"321": F = comptime .from(321);
    const @"1.5": F = comptime .from(1.5);
    const @"3.25": F = comptime .from(3.25);
    const @"1e38": F = comptime .from(1e38);
    const @"1e-38": F = comptime .from(1e-38);
    const @"1e30": F = comptime .from(1e30);
    const @"12": F = comptime .from(12);
    const @"-0.99e38": F = comptime .from(-0.99e38);

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

const runMul_flops = 10;
fn runMul(F: type) void {
    const zero: F = comptime .from(0);
    const one: F = comptime .from(1);
    const @"123": F = comptime .from(123);
    const @"321": F = comptime .from(321);
    const @"1.5": F = comptime .from(1.5);
    const @"3.25": F = comptime .from(3.25);
    const @"1e38": F = comptime .from(1e38);
    const @"1e-38": F = comptime .from(1e-38);
    const @"1e30": F = comptime .from(1e30);
    const @"12": F = comptime .from(12);
    const @"-0.99e38": F = comptime .from(-0.99e38);

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
