const std = @import("std");
const assert = std.debug.assert;

// CPU: i7-8700
pub fn main() void {
    std.debug.print(
        \\==========
        \\ Addition
        \\==========
        \\
    , .{});
    // NativeFloat(f32)      1.419GFLOP/s over 0.984s
    // NativeFloat(f64)      1.406GFLOP/s over 0.988s
    // NativeFloat(f128)    82.457MFLOP/s over 0.966s | 17.204x
    // BigFloat(f32,i32)     0.143GFLOP/s over 0.984s |  9.952x
    // BigFloat(f32,i96)   110.214MFLOP/s over 0.972s | 12.871x
    // BigFloat(f64,i64)   125.964MFLOP/s over 0.976s | 11.262x
    // BigFloat(f64,i128)  122.179MFLOP/s over 0.984s | 11.611x
    bench(runAdd, 2, 3);

    std.debug.print(
        \\================
        \\ Multiplication
        \\================
        \\
    , .{});
    // NativeFloat(f32)      1.406GFLOP/s over 0.990s
    // NativeFloat(f64)      1.391GFLOP/s over 0.997s
    // NativeFloat(f128)    73.024MFLOP/s over 0.978s | 19.260x
    // BigFloat(f32,i32)     0.209GFLOP/s over 0.979s |  6.733x
    // BigFloat(f32,i96)     0.162GFLOP/s over 0.962s |  8.665x
    // BigFloat(f64,i64)     0.167GFLOP/s over 0.975s |  8.419x
    // BigFloat(f64,i128)    0.170GFLOP/s over 0.982s |  8.280x
    bench(runMul, 2, 3);

    // NativeFloat(f32)     23.768MFLOP/s over 1.000s
    // NativeFloat(f64)     21.411MFLOP/s over 0.999s
    // NativeFloat(f128)     1.579MFLOP/s over 0.975s | 15.049x
    // BigFloat(f32,i32)     4.028MFLOP/s over 1.008s |  5.900x
    // BigFloat(f32,i96)     3.804MFLOP/s over 0.866s |  6.248x
    // BigFloat(f64,i64)     0.936MFLOP/s over 0.994s | 25.386x
    // BigFloat(f64,i128)    0.921MFLOP/s over 1.003s | 25.806x
    std.debug.print(
        \\==================
        \\ FormatScientific
        \\==================
        \\
    , .{});
    bench(runFmt, 1, 3);
}

const RunFn = fn (type, anytype) void;

fn NativeFloat(T: type) type {
    return struct {
        f: T,

        const Self = @This();

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
            comptime assert(bytes % @sizeOf(Self) == 0);
            var randoms: [bytes / @sizeOf(Self)]Self = @bitCast(random_bytes);
            for (&randoms) |*r| {
                if (std.math.isFinite(r.f)) {
                    const frexp = std.math.frexp(r.f);
                    // Keep floats in a reasonable range
                    r.f = std.math.ldexp(frexp.significand * 2, @rem(frexp.exponent - 1, 16));
                }
            }
            return randoms;
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
            comptime assert(bytes % @sizeOf(Self) == 0);
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
                    // Keep floats in a reasonable range
                    r.f.exponent = @rem(r.f.exponent, 16);
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

fn bench(run: *const RunFn, comptime args_per_flop: usize, run_count: usize) void {
    const biggest_bytes = 32;
    // Random data should be the same size for all types to be fair in terms of caching
    const random_len = 16 * args_per_flop * biggest_bytes;
    var random_buf: [random_len]u8 = undefined;
    var random: std.Random.Xoshiro256 = .init(123456789_850_907);
    random.fill(&random_buf);

    const types = [_]type{
        NativeFloat(f32),
        NativeFloat(f64),
        NativeFloat(f128), // f128 seems to perform better than f80 on my machine
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
    const base2: f64 = @floatFromInt((arrayFlops(types[1], random_len, args_per_flop) * iters[1] * std.time.ns_per_s) / ns_takens[1]);
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

fn runFmt(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    var discard: std.Io.Writer.Discarding = .init(&.{});
    inline for (0..array_info.len) |i| {
        const arg = data[i];
        discard.writer.print("{e}", .{arg.f}) catch unreachable;
    }
    std.mem.doNotOptimizeAway(discard.count);
}
