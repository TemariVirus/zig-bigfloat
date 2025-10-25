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
    // NativeFloat(f32)      1.425GFLOP/s over 0.994s
    // NativeFloat(f64)      1.435GFLOP/s over 0.986s
    // NativeFloat(f128)    86.920MFLOP/s over 0.992s | 16.512x
    // BigFloat(f32,i32)     0.269GFLOP/s over 0.991s |  5.339x
    // BigFloat(f32,i96)     0.191GFLOP/s over 0.993s |  7.513x
    // BigFloat(f64,i64)     0.168GFLOP/s over 0.996s |  8.554x
    // BigFloat(f64,i128)    0.150GFLOP/s over 0.991s |  9.551x
    bench(runAdd, 2, 3);

    std.debug.print(
        \\================
        \\ Multiplication
        \\================
        \\
    , .{});
    // NativeFloat(f32)      0.591GFLOP/s over 0.869s
    // NativeFloat(f64)      1.432GFLOP/s over 0.930s
    // NativeFloat(f128)    75.146MFLOP/s over 0.988s | 19.053x
    // BigFloat(f32,i32)     0.192GFLOP/s over 0.704s |  7.453x
    // BigFloat(f32,i96)     0.152GFLOP/s over 0.984s |  9.434x
    // BigFloat(f64,i64)     0.166GFLOP/s over 0.589s |  8.608x
    // BigFloat(f64,i128)    0.154GFLOP/s over 0.978s |  9.325x
    bench(runMul, 2, 3);

    std.debug.print(
        \\==================
        \\ FormatScientific
        \\==================
        \\
    , .{});
    // NativeFloat(f32)     25.033MFLOP/s over 0.989s
    // NativeFloat(f64)     21.962MFLOP/s over 0.990s
    // NativeFloat(f128)     2.433MFLOP/s over 0.988s | 10.290x
    // BigFloat(f32,i32)     3.817MFLOP/s over 0.991s |  6.558x
    // BigFloat(f32,i96)     3.825MFLOP/s over 0.993s |  6.544x
    // BigFloat(f64,i64)     1.083MFLOP/s over 0.996s | 23.106x
    // BigFloat(f64,i128)    1.029MFLOP/s over 0.979s | 24.321x
    bench(runFmt, 1, 3);

    std.debug.print(
        \\===============
        \\ Integer Power
        \\===============
        \\
    , .{});
    // NativeFloat(f32)     65.414MFLOP/s over 0.570s
    // NativeFloat(f64)     39.799MFLOP/s over 0.966s
    // NativeFloat(f128)    32.865MFLOP/s over 0.598s |  1.990x
    // BigFloat(f32,i32)    10.241MFLOP/s over 0.827s |  6.388x
    // BigFloat(f32,i96)     9.597MFLOP/s over 0.999s |  6.816x
    // BigFloat(f64,i64)    11.234MFLOP/s over 0.996s |  5.823x
    // BigFloat(f64,i128)   10.387MFLOP/s over 0.990s |  6.298x
    bench(runPowi, 1, 3);

    std.debug.print(
        \\======
        \\ Exp2
        \\======
        \\
    , .{});
    // NativeFloat(f32)      0.251GFLOP/s over 1.002s
    // NativeFloat(f64)      0.292GFLOP/s over 1.000s
    // NativeFloat(f128)    99.830MFLOP/s over 1.002s |  2.922x
    // BigFloat(f32,i32)   103.688MFLOP/s over 0.997s |  2.813x
    // BigFloat(f32,i96)    82.764MFLOP/s over 0.999s |  3.524x
    // BigFloat(f64,i64)    86.573MFLOP/s over 0.994s |  3.369x
    // BigFloat(f64,i128)   72.266MFLOP/s over 0.961s |  4.036x
    bench(runExp2, 1, 3);

    std.debug.print(
        \\======
        \\ Log2
        \\======
        \\
    , .{});
    // NativeFloat(f32)      0.252GFLOP/s over 0.999s
    // NativeFloat(f64)      0.166GFLOP/s over 0.994s
    // NativeFloat(f128)    55.854MFLOP/s over 0.992s |  4.503x
    // BigFloat(f32,i32)   103.572MFLOP/s over 0.995s |  2.428x
    // BigFloat(f32,i96)    81.821MFLOP/s over 0.961s |  3.074x
    // BigFloat(f64,i64)    86.399MFLOP/s over 0.990s |  2.911x
    // BigFloat(f64,i128)   71.893MFLOP/s over 0.973s |  3.498x
    bench(runLog2, 1, 3);
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

        pub inline fn powi(base: Self, exponent: i32) Self {
            // TODO: change this out when std.math.pow is implemented for f16/f80/f128
            const F = switch (T) {
                f16 => f32,
                f32, f64 => T,
                f80, f128 => f64,
                else => unreachable,
            };
            const b: F = @floatCast(base.f);
            return .{ .f = std.math.pow(F, b, @floatFromInt(exponent)) };
        }

        pub inline fn pow(base: Self, exponent: Self) Self {
            return .{ .f = std.math.pow(T, base.f, exponent.f) };
        }

        pub inline fn exp2(self: Self) Self {
            return .{ .f = @exp2(self.f) };
        }

        pub inline fn log2(self: Self) Self {
            return .{ .f = @log2(self.f) };
        }

        pub fn randomArray(comptime bytes: usize, random_bytes: [bytes]u8) [bytes / @sizeOf(Self)]Self {
            comptime assert(bytes % @sizeOf(Self) == 0);
            if (T == f32) {
                return @bitCast(random_bytes);
            }

            const original = NativeFloat(f32).randomArray(bytes, random_bytes);
            var resized: [bytes / @sizeOf(Self)]Self = undefined;
            for (0..resized.len) |i| {
                resized[i] = .{ .f = @floatCast(original[i].f) };
            }
            return resized;
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

        pub inline fn powi(base: Self, power: E) Self {
            return .{ .f = base.f.powi(power) };
        }

        pub inline fn exp2(self: Self) Self {
            return .{ .f = self.f.log2() };
        }

        pub inline fn log2(self: Self) Self {
            return .{ .f = self.f.log2() };
        }

        pub fn randomArray(comptime bytes: usize, random_bytes: [bytes]u8) [bytes / @sizeOf(Self)]Self {
            comptime assert(bytes % @sizeOf(Self) == 0);
            const original = NativeFloat(f32).randomArray(bytes, random_bytes);
            var resized: [bytes / @sizeOf(Self)]Self = undefined;
            for (0..resized.len) |i| {
                resized[i] = .{ .f = .init(original[i].f) };
            }
            return resized;
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
    const random_len = 32 * args_per_flop * biggest_bytes;
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

fn runPowi(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    const powers = comptime blk: {
        @setEvalBranchQuota(100 * array_info.len);
        var ps: [array_info.len]i32 = undefined;
        var rng: std.Random.DefaultPrng = .init(0);
        for (&ps) |*p| {
            const power = rng.random().floatNorm(f64) * 1_000 + 0.5;
            p.* = @intFromFloat(power);
        }
        break :blk ps;
    };

    inline for (0..array_info.len) |i| {
        const arg = data[i];
        std.mem.doNotOptimizeAway(arg.powi(powers[i]));
    }
}

fn runExp2(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    inline for (0..array_info.len) |i| {
        const arg = data[i];
        std.mem.doNotOptimizeAway(arg.exp2());
    }
}

fn runLog2(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    inline for (0..array_info.len) |i| {
        const arg = data[i];
        std.mem.doNotOptimizeAway(arg.log2());
    }
}
