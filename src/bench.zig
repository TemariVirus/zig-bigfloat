const std = @import("std");
const math = std.math;
const linux = std.os.linux;
const PERF = linux.PERF;

const BigFloat = @import("bigfloat").BigFloat;

// Note: The instruction counts for extremely lightweight benchmarks are inflated
// from having to load the data. (E.g., f64 addition is only 1 instruction, but was recorded as 3)
//
// CPU: Intel(R) Core(TM) i7-8700 CPU @ 4.60GHz
//
// ==========
//  Addition
// ==========
// f32                   1.381GFLOP/s |  1.000x
// f64                   0.850GFLOP/s |  1.624x
// f128                 78.473MFLOP/s | 17.601x
// BigFloat(f32, i32)  117.014MFLOP/s | 11.804x
// BigFloat(f32, i96)   85.523MFLOP/s | 16.150x
// BigFloat(f64, i64)  125.399MFLOP/s | 11.015x
// BigFloat(f64, i128) 100.329MFLOP/s | 13.767x

// f64
//   Wall time:    1.27ns
//   Cycles:       5.4   | 1.18ns
//   Instructions: 3.0
//   Branches:     0.01  | 0.00% miss
// BigFloat(f64, i64)
//   Wall time:    8.58ns
//   Cycles:       36.7  | 7.97ns
//   Instructions: 50.9
//   Branches:     9.69  | 5.76% miss

// ================
//  Multiplication
// ================
// f32                   0.398GFLOP/s |  1.769x
// f64                   0.704GFLOP/s |  1.000x
// f128                 52.784MFLOP/s | 13.334x
// BigFloat(f32, i32)    0.295GFLOP/s |  2.385x
// BigFloat(f32, i96)    0.224GFLOP/s |  3.144x
// BigFloat(f64, i64)    0.247GFLOP/s |  2.844x
// BigFloat(f64, i128)   0.138GFLOP/s |  5.083x

// f64
//   Wall time:    1.53ns
//   Cycles:       6.5   | 1.42ns
//   Instructions: 3.0
//   Branches:     0.01  | 0.00% miss
// BigFloat(f64, i64)
//   Wall time:    4.35ns
//   Cycles:       18.6  | 4.04ns
//   Instructions: 41.0
//   Branches:     5.01  | 0.00% miss

// ==========
//  Division
// ==========
// f32                   0.340GFLOP/s |  1.585x
// f64                   0.538GFLOP/s |  1.000x
// f128                 23.208MFLOP/s | 23.198x
// BigFloat(f32, i32)    0.312GFLOP/s |  1.726x
// BigFloat(f32, i96)    0.215GFLOP/s |  2.509x
// BigFloat(f64, i64)    0.226GFLOP/s |  2.379x
// BigFloat(f64, i128) 131.278MFLOP/s |  4.101x

// f64
//   Wall time:    2.00ns
//   Cycles:       8.5   | 1.86ns
//   Instructions: 3.0
//   Branches:     0.01  | 0.02% miss
// BigFloat(f64, i64)
//   Wall time:    4.75ns
//   Cycles:       20.3  | 4.42ns
//   Instructions: 41.0
//   Branches:     5.01  | 0.00% miss

// =========
//  Inverse
// =========
// f32                   0.849GFLOP/s |  1.080x
// f64                   0.917GFLOP/s |  1.000x
// f128                 26.462MFLOP/s | 34.664x
// BigFloat(f32, i32)    0.501GFLOP/s |  1.831x
// BigFloat(f32, i96)    0.315GFLOP/s |  2.915x
// BigFloat(f64, i64)    0.425GFLOP/s |  2.161x
// BigFloat(f64, i128)   0.365GFLOP/s |  2.514x

// f64
//   Wall time:    1.17ns
//   Cycles:       5.0   | 1.09ns
//   Instructions: 2.0
//   Branches:     0.00  | 0.03% miss
// BigFloat(f64, i64)
//   Wall time:    2.54ns
//   Cycles:       10.8  | 2.36ns
//   Instructions: 19.0
//   Branches:     3.01  | 0.00% miss

// =======
//  Power
// =======
// f32                  39.958MFLOP/s |  1.000x
// f64                  39.064MFLOP/s |  1.023x
// f128                 44.475MFLOP/s |  0.898x
// BigFloat(f32, i32)   23.158MFLOP/s |  1.725x
// BigFloat(f32, i96)   21.342MFLOP/s |  1.872x
// BigFloat(f64, i64)   24.909MFLOP/s |  1.604x
// BigFloat(f64, i128)  21.904MFLOP/s |  1.824x

// f64
//   Wall time:    27.6ns
//   Cycles:       117.8 | 25.6ns
//   Instructions: 130.4
//   Branches:     26.84 | 4.84% miss
// BigFloat(f64, i64)
//   Wall time:    43.2ns
//   Cycles:       184.7 | 40.1ns
//   Instructions: 218.0
//   Branches:     32.56 | 4.31% miss

// ===============
//  Integer Power
// ===============
// f32                  21.818MFLOP/s |  1.000x
// f64                  20.281MFLOP/s |  1.076x
// f128                 42.284MFLOP/s |  0.516x
// BigFloat(f32, i32)   14.058MFLOP/s |  1.552x
// BigFloat(f32, i96)   12.325MFLOP/s |  1.770x
// BigFloat(f64, i64)   14.039MFLOP/s |  1.554x
// BigFloat(f64, i128)  13.061MFLOP/s |  1.670x

// f64
//   Wall time:    53.0ns
//   Cycles:       226.8 | 49.3ns
//   Instructions: 227.9
//   Branches:     47.18 | 11.73% miss
// BigFloat(f64, i64)
//   Wall time:    76.7ns
//   Cycles:       327.7 | 71.2ns
//   Instructions: 548.8
//   Branches:     91.41 | 1.35% miss

// ======
//  Exp2
// ======
// f32                 112.720MFLOP/s |  1.073x
// f64                 121.000MFLOP/s |  1.000x
// f128                 74.622MFLOP/s |  1.622x
// BigFloat(f32, i32)   75.672MFLOP/s |  1.599x
// BigFloat(f32, i96)   63.351MFLOP/s |  1.910x
// BigFloat(f64, i64)   81.324MFLOP/s |  1.488x
// BigFloat(f64, i128)  66.163MFLOP/s |  1.829x

// f64
//   Wall time:    8.89ns
//   Cycles:       38.0  | 8.26ns
//   Instructions: 25.9
//   Branches:     6.21  | 13.18% miss
// BigFloat(f64, i64)
//   Wall time:    13.3ns
//   Cycles:       56.6  | 12.3ns
//   Instructions: 67.0
//   Branches:     15.11 | 7.43% miss

// ======
//  Log2
// ======
// f32                 114.351MFLOP/s |  1.063x
// f64                 121.587MFLOP/s |  1.000x
// f128                 74.913MFLOP/s |  1.623x
// BigFloat(f32, i32)   74.468MFLOP/s |  1.633x
// BigFloat(f32, i96)   62.489MFLOP/s |  1.946x
// BigFloat(f64, i64)   80.155MFLOP/s |  1.517x
// BigFloat(f64, i128)  65.794MFLOP/s |  1.848x

// f64
//   Wall time:    9.08ns
//   Cycles:       37.8  | 8.22ns
//   Instructions: 25.9
//   Branches:     6.21  | 13.17% miss
// BigFloat(f64, i64)
//   Wall time:    13.4ns
//   Cycles:       57.4  | 12.5ns
//   Instructions: 67.0
//   Branches:     15.11 | 7.41% miss

// ==================
//  FormatScientific
// ==================
// f32                  17.270MFLOP/s |  1.032x
// f64                  17.828MFLOP/s |  1.000x
// f128                  2.011MFLOP/s |  8.863x
// BigFloat(f32, i32)    3.616MFLOP/s |  4.930x
// BigFloat(f32, i96)    2.654MFLOP/s |  6.718x
// BigFloat(f64, i64)    0.716MFLOP/s | 24.898x
// BigFloat(f64, i128)   0.630MFLOP/s | 28.291x

// f64
//   Wall time:    60.9ns
//   Cycles:       258.0 | 56.1ns
//   Instructions: 622.6
//   Branches:     67.02 | 3.32% miss
// BigFloat(f64, i64)
//   Wall time:    1.507us
//   Cycles:       6424.2 | 1.396us
//   Instructions: 14105.0
//   Branches:     424.43 | 15.12% miss

// =================
//  ParseScientific
// =================
// f32                  26.246MFLOP/s |  1.000x
// f64                  24.875MFLOP/s |  1.055x
// f128                  0.887kFLOP/s | 29589.280x
// BigFloat(f32, i32)    3.472MFLOP/s |  7.559x
// BigFloat(f32, i96)    3.165MFLOP/s |  8.293x
// BigFloat(f64, i64)    0.686MFLOP/s | 38.277x
// BigFloat(f64, i128)   0.646MFLOP/s | 40.627x

// f64
//   Wall time:    43.2ns
//   Cycles:       184.9 | 40.2ns
//   Instructions: 409.9
//   Branches:     62.64 | 3.19% miss
// BigFloat(f64, i64)
//   Wall time:    1.578us
//   Cycles:       6708.7 | 1.458us
//   Instructions: 14698.5
//   Branches:     504.19 | 12.70% miss

pub fn main() !void {
    const cpu_info = CpuInfo.init() catch |err| {
        std.debug.panic("unable to get CPU info: {t}\n", .{err});
    };
    cpu_info.prettyPrint();

    bench("Addition", runAdd, 2, cpu_info);
    bench("Multiplication", runMul, 2, cpu_info);
    bench("Division", runDiv, 2, cpu_info);
    bench("Inverse", runInv, 1, cpu_info);
    bench("Power", runPow, 2, cpu_info);
    bench("Integer Power", runPowi, 1, cpu_info);
    bench("Exp2", runExp2, 1, cpu_info);
    bench("Log2", runExp2, 1, cpu_info);
    bench("FormatScientific", runFmt, 1, cpu_info);
    bench("ParseScientific", runParse, 1, cpu_info);
}

fn AllocFn(T: type) type {
    return fn (type, usize, std.mem.Allocator) []T;
}
fn FreeFn(T: type) type {
    return fn (type, std.mem.Allocator, []T) void;
}
fn RunFn(T: type) type {
    return fn (type, []const T) void;
}

const CpuInfo = struct {
    name: []const u8,
    max_hz: u64,

    pub fn init() !@This() {
        switch (@import("builtin").os.tag) {
            .linux => {},
            else => @compileError("Unsupported OS"),
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
        var max_khz_buf: [32]u8 = undefined;
        const max_khz_str = max_khz_buf[0..try f2.readAll(&max_khz_buf)];
        const max_khz = std.fmt.parseInt(
            u64,
            std.mem.trimRight(u8, max_khz_str, "\n"),
            10,
        ) catch return error.InvalidFormat;

        // Get rid of '@ X.XXGHz' suffix from CPU name
        const at_pos = std.mem.lastIndexOfScalar(u8, full_name, '@') orelse full_name.len;
        const name = std.mem.trim(u8, full_name[0..at_pos], " ");
        return CpuInfo{ .name = name, .max_hz = max_khz * 1000 };
    }

    pub fn prettyPrint(self: @This()) void {
        var max_hz_buf: [64]u8 = undefined;
        const hz_str = std.fmt.bufPrint(&max_hz_buf, "{B:.2}", .{self.max_hz}) catch unreachable;
        std.debug.print("CPU: {s} @ {s}Hz\n", .{ self.name, std.mem.trimEnd(u8, hz_str, "B") });
    }
};

const Bench = struct {
    perf_fds: [perf_measurements.len]linux.fd_t,
    timer: std.time.Timer,

    const perf_measurements = [_]PERF.COUNT.HW{
        .CPU_CYCLES,
        .INSTRUCTIONS,
        .BRANCH_INSTRUCTIONS,
        .BRANCH_MISSES,
    };

    pub const Result = struct {
        wall_nanos: u64,
        cycles: usize,
        instructions: usize,
        branches: usize,
        branch_misses: usize,

        fn perOp(total: usize, op_count: u64) f64 {
            return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(op_count));
        }

        pub fn flops(result: Result, op_count: u64, cpu_hz: u64) u64 {
            return op_count * cpu_hz / @as(u64, @intCast(result.cycles));
        }

        pub fn prettyPrint(result: Result, op_count: u64, cpu_hz: u64) void {
            std.debug.print("  Wall time:    {f}\n", .{
                smallDuration(perOp(result.wall_nanos, op_count)),
            });

            const cycles_as_nanos = perOp(result.cycles, op_count) * std.time.ns_per_s / @as(f64, @floatFromInt(cpu_hz));
            std.debug.print("  Cycles:       {d:<5.1} | {f}\n", .{
                perOp(result.cycles, op_count),
                smallDuration(cycles_as_nanos),
            });

            std.debug.print("  Instructions: {d:.1}\n", .{perOp(result.instructions, op_count)});

            const branch_miss_rate = if (result.branches == 0)
                0.0
            else
                @as(f64, @floatFromInt(result.branch_misses)) / @as(f64, @floatFromInt(result.branches));
            std.debug.print("  Branches:     {d:<5.2} | {d:.2}% miss\n", .{
                perOp(result.branches, op_count),
                branch_miss_rate * 100.0,
            });
        }
    };

    pub fn init() @This() {
        var self = Bench{
            .perf_fds = @splat(-1),
            .timer = std.time.Timer.start() catch |err| {
                std.debug.panic("unable to start timer: {t}\n", .{err});
            },
        };
        for (perf_measurements, &self.perf_fds) |measurement, *perf_fd| {
            var attr: linux.perf_event_attr = .{
                .type = PERF.TYPE.HARDWARE,
                .config = @intFromEnum(measurement),
                .flags = .{
                    .disabled = true,
                    .exclude_kernel = true,
                    .exclude_hv = true,
                },
            };
            perf_fd.* = std.posix.perf_event_open(&attr, 0, -1, self.perf_fds[0], PERF.FLAG.FD_CLOEXEC) catch |err| {
                std.debug.panic("unable to open perf event: {t}\n", .{err});
            };
        }
        return self;
    }

    pub fn start(self: *@This()) void {
        self.timer.reset();
        _ = linux.ioctl(self.perf_fds[0], PERF.EVENT_IOC.RESET, PERF.IOC_FLAG_GROUP);
        _ = linux.ioctl(self.perf_fds[0], PERF.EVENT_IOC.ENABLE, PERF.IOC_FLAG_GROUP);
    }

    pub fn stop(self: *@This()) Result {
        _ = linux.ioctl(self.perf_fds[0], PERF.EVENT_IOC.DISABLE, PERF.IOC_FLAG_GROUP);
        const res = Result{
            .wall_nanos = self.timer.read(),
            .cycles = readPerfFd(self.perf_fds[0]),
            .instructions = readPerfFd(self.perf_fds[1]),
            .branches = readPerfFd(self.perf_fds[2]),
            .branch_misses = readPerfFd(self.perf_fds[3]),
        };

        for (&self.perf_fds) |*perf_fd| {
            std.posix.close(perf_fd.*);
            perf_fd.* = -1;
        }
        return res;
    }

    fn readPerfFd(fd: linux.fd_t) usize {
        var result: usize = 0;
        const n = std.posix.read(fd, std.mem.asBytes(&result)) catch |err| {
            std.debug.panic("unable to read perf fd: {t}\n", .{err});
        };
        std.debug.assert(n == @sizeOf(usize));
        return result;
    }
};

fn typeName(T: type) []const u8 {
    if (comptime isFloat(T)) {
        return std.fmt.comptimePrint("{}", .{T});
    } else {
        return std.fmt.comptimePrint(
            "BigFloat({}, {})",
            .{ @FieldType(T, "significand"), @FieldType(T, "exponent") },
        );
    }
}

fn getAllocFree(T: type) struct { AllocFn(T), FreeFn(T) } {
    return switch (T) {
        []const u8 => .{ allocStrings, freeStrings },
        else => .{ allocFloats, freeFloats },
    };
}

fn formatSmallDuration(nanos: f64, w: *std.Io.Writer) !void {
    if (nanos >= 100) {
        try w.print("{D}", .{@as(u64, @intFromFloat(nanos))});
    } else if (nanos >= 10) {
        try w.print("{d:.1}ns", .{nanos});
    } else {
        try w.print("{d:.2}ns", .{nanos});
    }
}

fn smallDuration(nanos: f64) std.fmt.Alt(f64, formatSmallDuration) {
    return .{ .data = nanos };
}

fn iterCount(
    T: type,
    InputT: type,
    comptime run: RunFn(InputT),
    comptime args_per_run: usize,
    target_ns: u64,
) u64 {
    var iters: u64 = 1;

    const allocator = std.heap.page_allocator;
    const alloc, const free = getAllocFree(InputT);
    const args = alloc(T, args_per_run, allocator);
    defer free(T, allocator, args);

    // Find rough number of iterations needed to take at least 10ms
    const ns_taken = while (true) : (iters *= 2) {
        const start = std.time.nanoTimestamp();
        for (0..iters) |_| {
            run(T, args);
        }
        const ns_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
        const time_limit = 10 * std.time.ns_per_ms;
        // Prevent overflow
        if (ns_taken >= time_limit or iters *% 2 < iters) {
            break ns_taken;
        }
    };

    // Extrapolate to target_ns
    return (iters * target_ns) / ns_taken;
}

fn bench(
    comptime name: []const u8,
    comptime run: anytype,
    comptime args_per_run: usize,
    cpu_info: CpuInfo,
) void {
    std.debug.print(
        \\
        \\{1s}
        \\ {0s}
        \\{1s}
        \\
    , .{ name, "=" ** (name.len + 2) });

    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    const types = [_]type{
        f32,
        f64,
        f128,
        BigFloat(.{ .Significand = f32, .Exponent = i32, .bake_render = true }),
        BigFloat(.{ .Significand = f32, .Exponent = i96, .bake_render = true }),
        BigFloat(.{ .Significand = f64, .Exponent = i64, .bake_render = true }),
        BigFloat(.{ .Significand = f64, .Exponent = i128, .bake_render = true }),
    };
    var iter_counts: [types.len]u64 = undefined;
    var results: [types.len]Bench.Result = undefined;

    const InputSlice = @typeInfo(@TypeOf(run)).@"fn".params[1].type;
    inline for (types, 0..) |T, i| {
        const InputT = if (InputSlice) |IS| @typeInfo(IS).pointer.child else T;
        const alloc, const free = getAllocFree(InputT);

        iter_counts[i] = iterCount(T, InputT, run, args_per_run, 100 * std.time.ns_per_ms);
        const data = alloc(T, iter_counts[i] * args_per_run, allocator);
        defer free(T, allocator, data);
        var b: Bench = .init();

        run(T, data); // Warmup
        b.start();
        run(T, data); // Actual run
        results[i] = b.stop();
    }

    const base_flops = @max(
        results[0].flops(iter_counts[0], cpu_info.max_hz),
        results[1].flops(iter_counts[1], cpu_info.max_hz),
    );
    inline for (types, 0..) |T, i| {
        const flops = results[i].flops(iter_counts[i], cpu_info.max_hz);
        var flops_buf: [9]u8 = undefined;
        const flops_str = std.fmt.bufPrint(&flops_buf, "{B:.3}", .{flops}) catch unreachable;

        std.debug.print("{s:<19} {s:>8}FLOP/s | {d:>6.3}x\n", .{
            typeName(T),
            flops_str[0 .. flops_str.len - 1],
            @as(f64, @floatFromInt(base_flops)) / @as(f64, @floatFromInt(flops)),
        });
    }

    std.debug.print("\n{s}\n", .{typeName(types[1])});
    results[1].prettyPrint(iter_counts[1], cpu_info.max_hz);
    std.debug.print("{s}\n", .{typeName(types[5])});
    results[5].prettyPrint(iter_counts[5], cpu_info.max_hz);
}

fn isFloat(T: type) bool {
    return @typeInfo(T) == .float;
}

/// Returns a random float evenly distributed in the range [1, 2) or (-2, -1].
fn randomSignificand(T: type, rng: std.Random) T {
    const C = std.meta.Int(.unsigned, @typeInfo(T).float.bits);

    // Mantissa
    var repr: C = rng.int(std.meta.Int(.unsigned, math.floatMantissaBits(T)));
    // Explicit bit is always 1
    if (math.floatMantissaBits(T) != math.floatFractionalBits(T)) {
        repr |= @as(C, 1) << math.floatFractionalBits(T);
    }
    // Exponent is always 0
    repr |= math.floatExponentMax(T) << math.floatMantissaBits(T);
    // Sign
    repr |= @as(C, rng.int(u1)) << (@typeInfo(T).float.bits - 1);

    return @bitCast(repr);
}

fn randomFloat(T: type, rng: std.Random) T {
    if (comptime isFloat(T)) {
        var f: T = undefined;
        while (true) {
            rng.bytes(std.mem.asBytes(&f));
            if (math.isFinite(f)) return f;
        }
    } else {
        return .init(randomFloat(@FieldType(T, "significand"), rng));
    }
}

fn allocFloats(T: type, n: usize, allocator: std.mem.Allocator) []T {
    const floats = allocator.alloc(T, n) catch @panic("OOM");
    var rng: std.Random.Xoshiro256 = .init(123456789_850_907);
    for (floats) |*f| {
        f.* = randomFloat(T, rng.random());
    }
    return floats;
}

fn allocStrings(T: type, n: usize, allocator: std.mem.Allocator) [][]const u8 {
    const chars_per_float = @bitSizeOf(T);
    const strings = allocator.alloc([]const u8, n) catch @panic("OOM");
    var buf = std.ArrayList(u8).initCapacity(allocator, n * chars_per_float) catch @panic("OOM");

    var rng: std.Random.Xoshiro256 = .init(123456789_850_907);
    for (strings) |*s| {
        const f = randomFloat(T, rng.random());

        const start = buf.unusedCapacitySlice().ptr;
        buf.print(allocator, "{e}", .{f}) catch @panic("failed to print float");
        const len = buf.unusedCapacitySlice().ptr - start;
        s.* = start[0..len];
    }

    return strings;
}

fn freeFloats(T: type, allocator: std.mem.Allocator, arr: []T) void {
    allocator.free(arr);
}

fn freeStrings(T: type, allocator: std.mem.Allocator, arr: [][]const u8) void {
    var buf = arr[0];
    buf.len = arr.len * @bitSizeOf(T);
    allocator.free(buf);
    allocator.free(arr);
}

inline fn batchRun(
    comptime runOne: anytype,
    comptime args_per_run: usize,
    comptime batch_size: usize,
    data: anytype,
) void {
    comptime std.debug.assert(batch_size % args_per_run == 0);
    var args = data;

    while (args.len >= batch_size) {
        inline for (0..batch_size / args_per_run) |i| {
            const f = runOne(args[i * args_per_run ..][0..args_per_run]);
            std.mem.doNotOptimizeAway(f);
        }
        args = args[batch_size..];
    }

    // Remainder
    for (0..args.len / args_per_run) |i| {
        const f = runOne(args[i * args_per_run ..][0..args_per_run]);
        std.mem.doNotOptimizeAway(f);
    }
}

fn runAdd(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [2]T) T {
            return if (comptime isFloat(T))
                args[0] + args[1]
            else
                args[0].add(args[1]);
        }
    }).runOne;
    batchRun(runOne, 2, 256, data);
}

fn runMul(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [2]T) T {
            return if (comptime isFloat(T))
                args[0] * args[1]
            else
                args[0].mul(args[1]);
        }
    }).runOne;
    batchRun(runOne, 2, 256, data);
}

fn runDiv(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [2]T) T {
            return if (comptime isFloat(T))
                args[0] / args[1]
            else
                args[0].div(args[1]);
        }
    }).runOne;
    batchRun(runOne, 2, 256, data);
}

fn runInv(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [1]T) T {
            return if (comptime isFloat(T))
                1.0 / args[0]
            else
                args[0].inv();
        }
    }).runOne;
    batchRun(runOne, 1, 256, data);
}

fn runPow(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [2]T) T {
            if (comptime isFloat(T)) {
                // TODO: change this out when std.math.pow is implemented for f16/f80/f128
                const F = switch (T) {
                    f16, f32 => f32,
                    f64, f80, f128 => f64,
                    else => unreachable,
                };
                return math.pow(F, @floatCast(args[0]), @floatCast(args[1]));
            } else {
                return args[0].pow(args[1]);
            }
        }
    }).runOne;
    batchRun(runOne, 2, 2, data);
}

fn runPowi(T: type, data: []const T) void {
    const powi = (struct {
        inline fn powi(base: T, power: i32) T {
            if (comptime isFloat(T)) {
                // TODO: change this out when std.math.pow is implemented for f16/f80/f128
                const F = switch (T) {
                    f16, f32 => f32,
                    f64, f80, f128 => f64,
                    else => unreachable,
                };
                return math.pow(F, @floatCast(base), @floatFromInt(power));
            } else {
                return base.powi(power);
            }
        }
    }).powi;

    const powers = comptime blk: {
        var ps: [256]i32 = undefined;
        @setEvalBranchQuota(100 * ps.len);
        var rng: std.Random.Xoshiro256 = .init(123);
        for (&ps) |*p| {
            const power = rng.random().floatNorm(f64) * 1_000 + 0.5;
            p.* = @intFromFloat(power);
        }
        break :blk ps;
    };

    var args = data;
    while (args.len >= powers.len) {
        inline for (0..powers.len) |i| {
            const f = powi(args[i], powers[i]);
            std.mem.doNotOptimizeAway(f);
        }
        args = args[powers.len..];
    }

    // Remainder
    for (0..args.len) |i| {
        const f = powi(args[i], powers[i]);
        std.mem.doNotOptimizeAway(f);
    }
}

fn runExp2(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [1]T) T {
            return if (comptime isFloat(T))
                @exp2(args[0])
            else
                args[0].exp2();
        }
    }).runOne;
    batchRun(runOne, 1, 256, data);
}

fn runLog2(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [1]T) T {
            return if (comptime isFloat(T))
                @log2(args[0])
            else
                args[0].log2();
        }
    }).runOne;
    batchRun(runOne, 1, 256, data);
}

fn runFmt(T: type, data: []const T) void {
    const runOne = (struct {
        inline fn runOne(args: *const [1]T) u64 {
            var w: std.Io.Writer.Discarding = .init(&.{});
            w.writer.print("{e}", .{args[0]}) catch unreachable;
            return w.count;
        }
    }).runOne;
    batchRun(runOne, 1, 1, data);
}

fn runParse(T: type, data: []const []const u8) void {
    const runOne = (struct {
        inline fn runOne(args: *const [1][]const u8) T {
            const f = if (comptime isFloat(T))
                std.fmt.parseFloat(T, args[0])
            else
                T.parse(args[0]);
            return f catch {
                // Using the unreachable keyword would tell the compiler that parsing never fails,
                // but we don't want any optimizations based on that assumption.
                @branchHint(.cold);
                @panic("unreachable");
            };
        }
    }).runOne;
    batchRun(runOne, 1, 1, data);
}
