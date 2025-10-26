const std = @import("std");
const assert = std.debug.assert;

// CPU: Intel(R) Core(TM) i7-8700 CPU @ 4.60GHz
// ==========
//  Addition
// ==========
// NativeFloat(f32)      1.426GFLOP/s over 1.000s |  1.000x
// NativeFloat(f64)      1.422GFLOP/s over 0.999s |  1.003x
// NativeFloat(f128)    86.836MFLOP/s over 1.000s | 16.421x
// BigFloat(f32,i32)     0.238GFLOP/s over 0.998s |  6.000x
// BigFloat(f32,i96)     0.171GFLOP/s over 0.982s |  8.348x
// BigFloat(f64,i64)     0.158GFLOP/s over 0.999s |  9.047x
// BigFloat(f64,i128)    0.147GFLOP/s over 1.000s |  9.670x
// ================
//  Multiplication
// ================
// NativeFloat(f32)      0.587GFLOP/s over 1.000s |  2.421x
// NativeFloat(f64)      1.422GFLOP/s over 1.000s |  1.000x
// NativeFloat(f128)    74.562MFLOP/s over 1.000s | 19.074x
// BigFloat(f32,i32)     0.202GFLOP/s over 1.000s |  7.027x
// BigFloat(f32,i96)     0.164GFLOP/s over 1.000s |  8.693x
// BigFloat(f64,i64)     0.163GFLOP/s over 0.999s |  8.735x
// BigFloat(f64,i128)    0.150GFLOP/s over 1.000s |  9.470x
// =========
//  Inverse
// =========
// NativeFloat(f32)      1.428GFLOP/s over 0.998s |  1.000x
// NativeFloat(f64)      1.073GFLOP/s over 0.999s |  1.331x
// NativeFloat(f128)    25.854MFLOP/s over 0.997s | 55.229x
// BigFloat(f32,i32)     0.514GFLOP/s over 0.999s |  2.779x
// BigFloat(f32,i96)     0.484GFLOP/s over 0.998s |  2.948x
// BigFloat(f64,i64)     0.594GFLOP/s over 0.993s |  2.403x
// BigFloat(f64,i128)    0.644GFLOP/s over 0.992s |  2.217x
// =======
//  Power
// =======
// NativeFloat(f32)     72.669MFLOP/s over 1.005s |  1.000x
// NativeFloat(f64)     49.436MFLOP/s over 1.001s |  1.470x
// NativeFloat(f128)    30.448MFLOP/s over 1.001s |  2.387x
// BigFloat(f32,i32)    26.888MFLOP/s over 1.002s |  2.703x
// BigFloat(f32,i96)    24.107MFLOP/s over 1.000s |  3.014x
// BigFloat(f64,i64)    21.711MFLOP/s over 0.998s |  3.347x
// BigFloat(f64,i128)   19.545MFLOP/s over 0.995s |  3.718x
// ===============
//  Integer Power
// ===============
// NativeFloat(f32)     68.439MFLOP/s over 0.998s |  1.000x
// NativeFloat(f64)     40.011MFLOP/s over 1.008s |  1.711x
// NativeFloat(f128)    33.169MFLOP/s over 0.987s |  2.063x
// BigFloat(f32,i32)     9.799MFLOP/s over 0.993s |  6.985x
// BigFloat(f32,i96)    11.569MFLOP/s over 1.003s |  5.916x
// BigFloat(f64,i64)    11.547MFLOP/s over 1.016s |  5.927x
// BigFloat(f64,i128)   10.337MFLOP/s over 1.017s |  6.621x
// ======
//  Exp2
// ======
// NativeFloat(f32)      0.255GFLOP/s over 0.996s |  1.158x
// NativeFloat(f64)      0.296GFLOP/s over 0.997s |  1.000x
// NativeFloat(f128)   100.419MFLOP/s over 1.000s |  2.949x
// BigFloat(f32,i32)   104.030MFLOP/s over 0.997s |  2.847x
// BigFloat(f32,i96)    82.907MFLOP/s over 0.991s |  3.572x
// BigFloat(f64,i64)    87.098MFLOP/s over 0.987s |  3.400x
// BigFloat(f64,i128)   71.875MFLOP/s over 1.001s |  4.121x
// ======
//  Log2
// ======
// NativeFloat(f32)      0.254GFLOP/s over 0.994s |  1.000x
// NativeFloat(f64)      0.166GFLOP/s over 1.000s |  1.530x
// NativeFloat(f128)    55.859MFLOP/s over 0.986s |  4.547x
// BigFloat(f32,i32)   104.199MFLOP/s over 0.995s |  2.438x
// BigFloat(f32,i96)    83.089MFLOP/s over 0.996s |  3.057x
// BigFloat(f64,i64)    86.519MFLOP/s over 0.995s |  2.936x
// BigFloat(f64,i128)   72.214MFLOP/s over 1.017s |  3.517x
// ==================
//  FormatScientific
// ==================
// NativeFloat(f32)     24.568MFLOP/s over 0.629s |  1.000x
// NativeFloat(f64)     21.715MFLOP/s over 0.993s |  1.131x
// NativeFloat(f128)     2.478MFLOP/s over 0.991s |  9.914x
// BigFloat(f32,i32)     4.016MFLOP/s over 1.000s |  6.117x
// BigFloat(f32,i96)     3.961MFLOP/s over 0.996s |  6.202x
// BigFloat(f64,i64)     1.081MFLOP/s over 0.983s | 22.730x
// BigFloat(f64,i128)    1.141MFLOP/s over 0.997s | 21.531x

pub fn main() void {
    printCpuInfo() catch std.debug.print("CPU: unknown\n", .{});
    bench("Addition", runAdd, 2, 5);
    bench("Multiplication", runMul, 2, 5);
    bench("Inverse", runInv, 1, 5);
    bench("Power", runPow, 2, 5);
    bench("Integer Power", runPowi, 1, 5);
    bench("Exp2", runExp2, 1, 5);
    bench("Log2", runLog2, 1, 5);
    bench("FormatScientific", runFmt, 1, 5);
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

        pub inline fn inv(lhs: Self) Self {
            return .{ .f = 1.0 / lhs.f };
        }

        pub inline fn pow(base: Self, power: Self) Self {
            // TODO: change this out when std.math.pow is implemented for f16/f80/f128
            const F = switch (T) {
                f16 => f32,
                f32, f64 => T,
                f80, f128 => f64,
                else => unreachable,
            };
            return .{ .f = std.math.pow(F, @floatCast(base.f), @floatCast(power.f)) };
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

        pub inline fn inv(lhs: Self) Self {
            return .{ .f = lhs.f.inv() };
        }

        pub inline fn pow(base: Self, power: Self) Self {
            return .{ .f = base.f.pow(power.f) };
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

fn printCpuInfo() !void {
    switch (@import("builtin").os.tag) {
        .linux => {},
        else => return error.UnsupportedOS,
    }

    const f = try std.fs.openFileAbsolute("/proc/cpuinfo", .{});
    defer f.close();
    var line_buf: [4096]u8 = undefined;
    var reader = f.reader(&line_buf);

    const full_name = while (try reader.interface.takeDelimiter(':')) |key_full| {
        const key = std.mem.trim(u8, key_full, " \t\n");
        if (' ' != try reader.interface.takeByte()) { // Skip leading space
            return error.InvalidFormat;
        }

        if (std.mem.eql(u8, key, "model name")) {
            const value = try reader.interface.takeDelimiter('\n');
            break value orelse return error.InvalidFormat;
        } else {
            _ = try reader.interface.discardDelimiterInclusive('\n');
        }
    } else return error.InvalidFormat;

    const f2 = try std.fs.openFileAbsolute(
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
        .{},
    );
    defer f2.close();
    var max_khz_buff: [32]u8 = undefined;
    const max_khz_str = max_khz_buff[0..try f2.readAll(&max_khz_buff)];
    const max_khz = std.fmt.parseInt(
        u64,
        std.mem.trimRight(u8, max_khz_str, "\n"),
        10,
    ) catch return error.InvalidFormat;

    // Get rid of '@ X.XXGHz' suffix from CPU name
    const at_pos = std.mem.lastIndexOfScalar(u8, full_name, '@') orelse full_name.len;
    const name = std.mem.trim(u8, full_name[0..at_pos], " ");
    const hz_str = try std.fmt.bufPrint(&max_khz_buff, "{B:.2}", .{max_khz * 1000});
    std.debug.print("CPU: {s} @ {s}Hz\n", .{ name, std.mem.trimEnd(u8, hz_str, "B") });
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

fn bench(
    comptime name: []const u8,
    run: *const RunFn,
    comptime args_per_flop: usize,
    run_count: usize,
) void {
    std.debug.print(
        \\{1s}
        \\ {0s}
        \\{1s}
        \\
    , .{ name, "=" ** (name.len + 2) });

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
        printResult(F, arrayFlops(F, random_len, args_per_flop) * it, ns, base_flops);
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

fn runInv(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    inline for (0..array_info.len) |i| {
        const arg = data[i];
        std.mem.doNotOptimizeAway(arg.inv());
    }
}

fn runPow(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    inline for (0..array_info.len / 2) |i| {
        const args = data[i * 2 ..][0..2];
        std.mem.doNotOptimizeAway(args[0].pow(args[1]));
    }
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

fn runFmt(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    var discard: std.Io.Writer.Discarding = .init(&.{});
    inline for (0..array_info.len) |i| {
        const arg = data[i];
        discard.writer.print("{e}", .{arg.f}) catch unreachable;
    }
    std.mem.doNotOptimizeAway(discard.count);
}
