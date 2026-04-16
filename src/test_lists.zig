const std = @import("std");
const math = std.math;

fn isEquivalent(a: anytype, b: @TypeOf(a)) bool {
    if (a.isNan()) {
        return b.isNan();
    } else {
        return a.eql(b);
    }
}

fn expect(comptime op: []const u8, args: anytype) !void {
    const BF = std.meta.Child(@TypeOf(args));
    const expected = args[args.len - 1];
    const actual: BF = if (args.len == 2)
        @field(BF, op)(args[0])
    else if (args.len == 3)
        @field(BF, op)(args[0], args[1])
    else
        @compileError("Bad argument count");
    if (isEquivalent(expected, actual)) return;

    std.debug.print("BigFloat type: {}\nexpected {s}(", .{
        @TypeOf(actual),
        op,
    });
    std.debug.print("{e}", .{args[0]});
    for (args[1 .. args.len - 1]) |arg| {
        std.debug.print(", {e}", .{arg});
    }
    std.debug.print(") = {e}, found {e}\n", .{
        expected,
        actual,
    });
    return error.UnexpectedTestResult;
}

fn testOp(mod: anytype, comptime op: []const u8) !void {
    for (mod.values) |v| try expect(op, v);
}

test "consistent add" {
    inline for (.{
        @import("test-lists/add-f32.zig"),
        @import("test-lists/add-f64.zig"),
        @import("test-lists/add-f128.zig"),
    }) |mod| {
        try testOp(mod, "add");
    }
}

test "consistent div" {
    inline for (.{
        @import("test-lists/div-f32.zig"),
        @import("test-lists/div-f64.zig"),
        @import("test-lists/div-f128.zig"),
    }) |mod| {
        try testOp(mod, "div");
    }
}

test "consistent exp2" {
    inline for (.{
        @import("test-lists/exp2-f32.zig"),
        @import("test-lists/exp2-f64.zig"),
        @import("test-lists/exp2-f128.zig"),
    }) |mod| {
        try testOp(mod, "exp2");
    }
}

test "consistent inv" {
    inline for (.{
        @import("test-lists/inv-f32.zig"),
        @import("test-lists/inv-f64.zig"),
        @import("test-lists/inv-f128.zig"),
    }) |mod| {
        try testOp(mod, "inv");
    }
}

test "consistent log2" {
    inline for (.{
        @import("test-lists/log2-f32.zig"),
        @import("test-lists/log2-f64.zig"),
        @import("test-lists/log2-f128.zig"),
    }) |mod| {
        try testOp(mod, "log2");
    }
}

test "consistent mul" {
    inline for (.{
        @import("test-lists/mul-f32.zig"),
        @import("test-lists/mul-f64.zig"),
        @import("test-lists/mul-f128.zig"),
    }) |mod| {
        try testOp(mod, "mul");
    }
}

test "consistent pow" {
    inline for (.{
        @import("test-lists/pow-f32.zig"),
        @import("test-lists/pow-f64.zig"),
        @import("test-lists/pow-f128.zig"),
    }) |mod| {
        try testOp(mod, "pow");
    }
}

test "consistent sub" {
    inline for (.{
        @import("test-lists/sub-f32.zig"),
        @import("test-lists/sub-f64.zig"),
        @import("test-lists/sub-f128.zig"),
    }) |mod| {
        try testOp(mod, "sub");
    }
}
