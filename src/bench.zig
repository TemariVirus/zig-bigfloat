const std = @import("std");
const assert = std.debug.assert;

// TODO: timings are too fickle due to laptop heat throttling
// CPU: i7-1165G7
pub fn main() void {
    std.debug.print(
        \\==========
        \\ Addition
        \\==========
        \\
    , .{});
    // NativeFloat(f32)      1.233GFLOP/s over 0.895s
    // NativeFloat(f64)      1.300GFLOP/s over 0.894s
    // NativeFloat(f80)      0.466GFLOP/s over 0.967s |  5.585x
    // BigFloat(f32,i32)     0.574GFLOP/s over 1.142s |  4.533x
    // BigFloat(f32,i96)     0.427GFLOP/s over 1.020s |  6.083x
    // BigFloat(f64,i64)     0.545GFLOP/s over 0.864s |  4.771x
    // BigFloat(f64,i128)    0.350GFLOP/s over 1.056s |  7.437x
    //OR
    // NativeFloat(f32)      2.108GFLOP/s over 1.020s
    // NativeFloat(f64)      2.040GFLOP/s over 1.013s
    // NativeFloat(f80)     13.864MFLOP/s over 1.014s | 294.270x
    // BigFloat(f32,i32)     0.275GFLOP/s over 1.025s | 14.828x
    // BigFloat(f32,i96)     0.210GFLOP/s over 1.021s | 19.450x
    // BigFloat(f64,i64)     0.320GFLOP/s over 1.009s | 12.756x
    // BigFloat(f64,i128)    0.238GFLOP/s over 1.009s | 17.112x
    bench(runAdd, 3);

    std.debug.print(
        \\================
        \\ Multiplication
        \\================
        \\
    , .{});
    // NativeFloat(f32)      1.247GFLOP/s over 0.870s
    // NativeFloat(f64)      1.294GFLOP/s over 0.910s
    // NativeFloat(f80)      0.473GFLOP/s over 0.980s |  5.465x
    // BigFloat(f32,i32)     0.490GFLOP/s over 1.043s |  5.277x
    // BigFloat(f32,i96)     0.932GFLOP/s over 1.022s |  2.777x
    // BigFloat(f64,i64)     0.935GFLOP/s over 1.016s |  2.766x
    // BigFloat(f64,i128)    0.772GFLOP/s over 0.937s |  3.350x
    //OR
    // NativeFloat(f32)      0.907GFLOP/s over 0.986s
    // NativeFloat(f64)      1.918GFLOP/s over 1.047s
    // NativeFloat(f80)     12.541MFLOP/s over 0.992s | 305.841x
    // BigFloat(f32,i32)     0.228GFLOP/s over 1.056s | 16.788x
    // BigFloat(f32,i96)     0.187GFLOP/s over 1.048s | 20.519x
    // BigFloat(f64,i64)     0.187GFLOP/s over 1.014s | 20.480x
    // BigFloat(f64,i128)    0.180GFLOP/s over 1.006s | 21.350x
    bench(runMul, 3);
}

const RunFn = fn (type, anytype) void;

fn NativeFloat(T: type) type {
    return struct {
        f: T,

        const Self = @This();

        pub const inf: Self = .{ .f = std.math.inf(T) };
        pub const minus_inf: Self = .{ .f = -std.math.inf(T) };
        pub const nan: Self = .{ .f = std.math.nan(T) };

        pub inline fn add(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f + rhs.f };
        }

        pub inline fn sub(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f - rhs.f };
        }

        pub inline fn mul(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f * rhs.f };
        }

        pub inline fn div(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f / rhs.f };
        }

        pub inline fn pow(base: Self, exponent: Self) Self {
            return .{ .f = std.math.pow(T, base.f, exponent.f) };
        }

        pub fn randomArray(comptime bytes: usize, random_bytes: [bytes]u8) [bytes / @sizeOf(Self)]Self {
            assert(bytes % @sizeOf(Self) == 0);
            return @bitCast(random_bytes);
        }
    };
}

fn BigFloat(S: type, E: type) type {
    const T = @import("bigfloat").BigFloat(.{
        .Significand = S,
        .Exponent = E,
        .bake_render = true,
    });
    return struct {
        f: T,

        const Self = @This();

        pub const inf: Self = .{ .f = T.inf };
        pub const minus_inf: Self = .{ .f = T.minus_inf };
        pub const nan: Self = .{ .f = T.nan };

        pub inline fn add(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f.add(rhs.f) };
        }

        pub inline fn sub(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f.sub(rhs.f) };
        }

        pub inline fn mul(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f.mul(rhs.f) };
        }

        pub fn randomArray(comptime bytes: usize, random_bytes: [bytes]u8) [bytes / @sizeOf(Self)]Self {
            assert(bytes % @sizeOf(Self) == 0);
            var randoms: [bytes / @sizeOf(Self)]Self = @bitCast(random_bytes);
            for (&randoms) |*r| {
                // Ensure the random value is in cannon form
                if (std.math.isNan(r.f.significand)) {
                    r.f = .nan;
                } else if (r.f.significand == std.math.inf(S)) {
                    r.f = .inf;
                } else if (r.f.significand == -std.math.inf(S)) {
                    r.f = .minus_inf;
                } else if (r.f.significand == 0) {
                    r.f = .init(0);
                } else {
                    r.f.significand = std.math.ldexp(r.f.significand, -std.math.ilogb(r.f.significand));
                }
            }
            return randoms;
        }
    };
}

fn iterCount(run: *const RunFn, data: anytype, target_ns: u64) u64 {
    var iters: u64 = 1;

    // Find rough number of iterations needed to take at least 50ms
    const ns_taken = while (true) : (iters *= 2) {
        const start = std.time.nanoTimestamp();
        for (0..iters) |_| {
            run(@TypeOf(data), &data);
        }
        const ns_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
        const time_limit = 50 * std.time.ns_per_ms;
        if (ns_taken >= time_limit) {
            break ns_taken;
        }
    };

    // Extrapolate to target_ns
    return (iters * target_ns) / ns_taken;
}

fn timeIters(run: *const RunFn, data: anytype, iters: u64) u64 {
    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        run(@TypeOf(data), &data);
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
    const flops_str = std.fmt.bufPrint(&buf, "{B:.3}", .{flops}) catch unreachable;
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

fn arrayFlops(T: type, bytes: usize, args_per_flop: usize) usize {
    return bytes / args_per_flop / @sizeOf(T);
}

fn bench(run: *const RunFn, run_count: usize) void {
    const biggest_bytes = 32;
    const args_per_flop = 2;
    // Random data should be the same size for all types to be fair in terms of caching
    const random_len = 16 * args_per_flop * biggest_bytes;
    var random_buf: [random_len]u8 = undefined;
    var random: std.Random.Xoshiro256 = .init(123456789_850_907);
    random.fill(&random_buf);

    const types = [_]type{
        NativeFloat(f32),
        NativeFloat(f64),
        NativeFloat(f80),
        BigFloat(f32, i32),
        BigFloat(f32, i96),
        BigFloat(f64, i64),
        BigFloat(f64, i128),
    };
    var iters: [types.len]u64 = undefined;
    var ns_takens: [types.len]u64 = @splat(std.math.maxInt(u64));

    inline for (types, &iters) |F, *its| {
        its.* = iterCount(run, F.randomArray(random_len, random_buf), std.time.ns_per_s);
    }
    for (0..run_count) |_| {
        // Interleave types to minimise effects of random CPU spikes
        inline for (types, iters, &ns_takens) |F, its, *ns| {
            // `run` is deterministic, so we use the minimum to get the most accurate timing
            ns.* = @min(ns.*, timeIters(run, F.randomArray(random_len, random_buf), its));
        }
    }

    const base1: f64 = @floatFromInt((arrayFlops(types[0], random_len, args_per_flop) * iters[0] * std.time.ns_per_s) / ns_takens[0]);
    const base2: f64 = @floatFromInt((arrayFlops(types[0], random_len, args_per_flop) * iters[1] * std.time.ns_per_s) / ns_takens[1]);
    const base_flops = @max(base1, base2);

    inline for (types[0..2], iters[0..2], ns_takens[0..2]) |F, it, ns| {
        printResult(F, arrayFlops(F, random_len, args_per_flop) * it, ns, null);
    }
    inline for (types[2..], iters[2..], ns_takens[2..]) |F, it, ns| {
        printResult(F, arrayFlops(F, random_len, args_per_flop) * it, ns, base_flops);
    }
}

fn runAdd(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    inline for (0..array_info.len / 2) |i| {
        const args = data[i * 2 ..][0..2];
        std.mem.doNotOptimizeAway(args[0].add(args[1]));
    }
}

fn runMul(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    inline for (0..array_info.len / 2) |i| {
        const args = data[i * 2 ..][0..2];
        std.mem.doNotOptimizeAway(args[0].mul(args[1]));
    }
}
