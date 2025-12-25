const std = @import("std");
const assert = std.debug.assert;

// CPU: Intel(R) Core(TM) i7-8700 CPU @ 4.60GHz
// ==========
//  Addition
// ==========
// NativeFloat(f32)      1.509GFLOP/s over 0.961s |  1.000x
// NativeFloat(f64)      1.505GFLOP/s over 0.973s |  1.002x
// NativeFloat(f128)    92.042MFLOP/s over 1.002s | 16.391x
// BigFloat(f32,i32)     0.297GFLOP/s over 1.010s |  5.077x
// BigFloat(f32,i96)     0.208GFLOP/s over 0.999s |  7.247x
// BigFloat(f64,i64)     0.193GFLOP/s over 1.000s |  7.813x
// BigFloat(f64,i128)    0.152GFLOP/s over 0.984s |  9.949x
// ================
//  Multiplication
// ================
// NativeFloat(f32)      0.625GFLOP/s over 0.993s |  2.423x
// NativeFloat(f64)      1.515GFLOP/s over 0.985s |  1.000x
// NativeFloat(f128)    79.391MFLOP/s over 0.997s | 19.083x
// BigFloat(f32,i32)     0.307GFLOP/s over 0.996s |  4.931x
// BigFloat(f32,i96)     0.259GFLOP/s over 0.998s |  5.855x
// BigFloat(f64,i64)     0.296GFLOP/s over 0.986s |  5.125x
// BigFloat(f64,i128)    0.237GFLOP/s over 0.986s |  6.390x
// ==========
//  Division
// ==========
// NativeFloat(f32)      0.437GFLOP/s over 1.013s |  2.562x
// NativeFloat(f64)      1.119GFLOP/s over 1.018s |  1.000x
// NativeFloat(f128)    27.058MFLOP/s over 0.978s | 41.350x
// BigFloat(f32,i32)     0.298GFLOP/s over 0.926s |  3.760x
// BigFloat(f32,i96)     0.258GFLOP/s over 0.979s |  4.332x
// BigFloat(f64,i64)     0.296GFLOP/s over 0.994s |  3.786x
// BigFloat(f64,i128)    0.237GFLOP/s over 0.985s |  4.712x
// =========
//  Inverse
// =========
// NativeFloat(f32)      1.515GFLOP/s over 0.998s |  1.000x
// NativeFloat(f64)      1.137GFLOP/s over 0.982s |  1.332x
// NativeFloat(f128)    27.457MFLOP/s over 1.000s | 55.183x
// BigFloat(f32,i32)     0.545GFLOP/s over 1.001s |  2.781x
// BigFloat(f32,i96)     0.515GFLOP/s over 1.002s |  2.944x
// BigFloat(f64,i64)     0.622GFLOP/s over 0.995s |  2.435x
// BigFloat(f64,i128)    0.663GFLOP/s over 0.995s |  2.287x
// =======
//  Power
// =======
// NativeFloat(f32)     76.645MFLOP/s over 1.002s |  1.000x
// NativeFloat(f64)     52.530MFLOP/s over 1.005s |  1.459x
// NativeFloat(f128)    32.396MFLOP/s over 1.002s |  2.366x
// BigFloat(f32,i32)    29.836MFLOP/s over 0.996s |  2.569x
// BigFloat(f32,i96)    26.667MFLOP/s over 1.001s |  2.874x
// BigFloat(f64,i64)    29.202MFLOP/s over 1.007s |  2.625x
// BigFloat(f64,i128)   26.952MFLOP/s over 1.010s |  2.844x
// ===============
//  Integer Power
// ===============
// NativeFloat(f32)     59.236MFLOP/s over 0.977s |  1.000x
// NativeFloat(f64)     41.716MFLOP/s over 0.986s |  1.420x
// NativeFloat(f128)    35.053MFLOP/s over 1.002s |  1.690x
// BigFloat(f32,i32)    14.123MFLOP/s over 0.960s |  4.194x
// BigFloat(f32,i96)    13.464MFLOP/s over 1.002s |  4.400x
// BigFloat(f64,i64)    14.056MFLOP/s over 0.998s |  4.214x
// BigFloat(f64,i128)   13.472MFLOP/s over 0.993s |  4.397x
// ======
//  Exp2
// ======
// NativeFloat(f32)      0.271GFLOP/s over 0.994s |  1.145x
// NativeFloat(f64)      0.310GFLOP/s over 1.008s |  1.000x
// NativeFloat(f128)   106.260MFLOP/s over 0.990s |  2.921x
// BigFloat(f32,i32)    75.491MFLOP/s over 0.992s |  4.111x
// BigFloat(f32,i96)    64.021MFLOP/s over 0.996s |  4.848x
// BigFloat(f64,i64)    90.506MFLOP/s over 0.997s |  3.429x
// BigFloat(f64,i128)   76.258MFLOP/s over 1.002s |  4.070x
// ======
//  Log2
// ======
// NativeFloat(f32)      0.269GFLOP/s over 0.994s |  1.000x
// NativeFloat(f64)      0.177GFLOP/s over 0.997s |  1.523x
// NativeFloat(f128)    59.129MFLOP/s over 1.003s |  4.553x
// BigFloat(f32,i32)    75.721MFLOP/s over 1.004s |  3.556x
// BigFloat(f32,i96)    64.048MFLOP/s over 1.012s |  4.204x
// BigFloat(f64,i64)    90.710MFLOP/s over 0.996s |  2.968x
// BigFloat(f64,i128)   76.464MFLOP/s over 1.006s |  3.521x
// ==================
//  FormatScientific
// ==================
// NativeFloat(f32)     26.264MFLOP/s over 0.997s |  1.000x
// NativeFloat(f64)     22.210MFLOP/s over 0.997s |  1.182x
// NativeFloat(f128)     2.576MFLOP/s over 1.001s | 10.197x
// BigFloat(f32,i32)     4.109MFLOP/s over 1.004s |  6.392x
// BigFloat(f32,i96)     3.943MFLOP/s over 0.999s |  6.662x
// BigFloat(f64,i64)     1.142MFLOP/s over 0.998s | 23.004x
// BigFloat(f64,i128)    1.126MFLOP/s over 0.980s | 23.328x

pub fn main() void {
    printCpuInfo() catch std.debug.print("CPU: unknown\n", .{});
    bench("Addition", runAdd, 2, 5);
    bench("Multiplication", runMul, 2, 5);
    bench("Division", runDiv, 2, 5);
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

        pub inline fn div(lhs: Self, rhs: Self) Self {
            return .{ .f = lhs.f.div(rhs.f) };
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

fn runDiv(Array: type, data: *const Array) void {
    const array_info = @typeInfo(Array).array;
    inline for (0..array_info.len / 2) |i| {
        const args = data[i * 2 ..][0..2];
        std.mem.doNotOptimizeAway(args[0].div(args[1]));
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
