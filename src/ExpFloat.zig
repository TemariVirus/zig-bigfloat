//! Represents 2^exponent. Assumed to be always positive.

const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

/// Base-2 exponent of the number.
exponent: f64,

const Self = @This();

// zig fmt: off
pub const zero: Self =     .{ .exponent = -math.inf(f64) };
pub const inf: Self =      .{ .exponent = math.inf(f64) };
pub const nan: Self =      .{ .exponent = math.nan(f64) };
/// Largest value smaller than `inf`.
pub const maxValue: Self = .{ .exponent = math.floatMax(f64) };
/// Smallest value larger than `zero`.
pub const minValue: Self = .{ .exponent = math.floatTrueMin(f64) };
// zig fmt: on

pub fn from(value: f64) Self {
    return .{ .exponent = @log2(value) };
}

pub fn isInf(self: Self) bool {
    return self.exponent == inf.exponent;
}

pub fn isNan(self: Self) bool {
    return math.isNan(self.exponent);
}

pub fn eql(a: Self, b: Self) bool {
    return a.exponent == b.exponent;
}

pub fn gt(a: Self, b: Self) bool {
    return a.exponent > b.exponent;
}

pub fn lt(a: Self, b: Self) bool {
    return a.exponent < b.exponent;
}

pub fn add(a: Self, b: Self) Self {
    if (math.isNan(a.exponent) or math.isNan(b.exponent)) return nan;
    if (a.exponent < b.exponent) {
        return @call(.always_inline, add2, .{ b, a });
    }
    return @call(.always_inline, add2, .{ a, b });
}

fn add2(a: Self, b: Self) Self {
    assert(!a.isNan() and !b.isNan());
    assert(a.exponent >= b.exponent);
    // If either arg is inf or zero, the answer is a
    if (math.isInf(a.exponent) or math.isInf(b.exponent)) {
        return a;
    }
    // 2^c = 2^a + 2^b
    // c   = log_2(2^a + 2^b)
    // c   = log_2(2^a * (1 + 2^(b-a)))
    // c   = a + log_2(1 + 2^(b-a))
    // These 2 lines appear to give the same results
    // return .{ .exponent = a.exponent + math.log1p(@exp2(b.exponent - a.exponent)) * math.log2e };
    return .{ .exponent = a.exponent + @log2(1 + @exp2(b.exponent - a.exponent)) };
}

pub fn sub(a: Self, b: Self) Self {
    if (math.isNan(a.exponent) or math.isNan(b.exponent)) return nan;
    if (a.exponent < b.exponent) {
        return @call(.always_inline, sub2, .{ b, a });
    }
    return @call(.always_inline, sub2, .{ a, b });
}

fn sub2(a: Self, b: Self) Self {
    assert(!a.isNan() and !b.isNan());
    assert(a.exponent >= b.exponent);
    // 2^c = 2^a - 2^b
    // c   = log_2(2^a - 2^b)
    // c   = log_2(2^a * (1 - 2^(b-a)))
    // c   = a + log_2(1 - 2^(b-a))
    // These 2 lines appear to give the same results
    // return .{ .exponent = a.exponent + math.log1p(-@exp2(b.exponent - a.exponent)) * math.log2e };
    return .{ .exponent = a.exponent + @log2(1 - @exp2(b.exponent - a.exponent)) };
}

pub fn mul(a: Self, b: Self) Self {
    return .{ .exponent = a.exponent + b.exponent };
}

pub fn pow(a: Self, b: Self) Self {
    return .{ .exponent = a.exponent * @exp2(b.exponent) };
}

pub fn powF64(a: Self, b: f64) Self {
    return .{ .exponent = a.exponent * b };
}

const testing = std.testing;

fn testAlmostEql(expected: Self, actual: Self) !void {
    const result = math.approxEqRel(
        f64,
        expected.exponent,
        actual.exponent,
        1.1102230246251565404236316680908e-13, // 10 ^ (-log_10(2^53) + 3)
    );
    if (!result) {
        std.debug.print("expected {}, found {}\n", .{ expected.exponent, actual.exponent });
        return error.TestExpectedEqual;
    }
}

test "from" {
    try testing.expectEqualDeep(Self{ .exponent = 0 }, Self.from(1));
    try testing.expect(Self.from(@as(i32, -123)).isNan());
    try testing.expectEqual(Self{ .exponent = -7.8614476248473514525971360774895 }, Self.from(0.0043));
    try testing.expectEqual(Self.zero, Self.from(0));
    try testing.expect(Self.from(math.nan(f64)).isNan());
    try testing.expectEqual(
        Self.inf,
        Self.from(math.inf(f64)),
    );
}

test "add" {
    try testing.expectEqualDeep(Self.from(0), Self.from(0).add(.from(0)));
    try testing.expectEqualDeep(Self.from(1), Self.from(1).add(.from(0)));
    try testAlmostEql(Self.from(444), Self.from(123).add(.from(321)));
    try testAlmostEql(Self.from(4.75), Self.from(1.5).add(.from(3.25)));
    try testing.expectEqualDeep(Self.from(1e38), Self.from(1e38).add(.from(1e-38)));

    try testing.expect(!Self.inf.eql(.from(0.6e308)));
    try testAlmostEql(
        Self{ .exponent = 1023.4168876311413969776418037972 },
        Self.from(0.6e308).add(.from(0.6e308)),
    );
    try testing.expect(Self.inf.add(Self.nan).isNan());
    try testing.expect(Self.nan.add(.from(2)).isNan());
}

test "sub" {
    try testAlmostEql(Self.from(1e35), Self.from(1e38).sub(.from(0.999e38)));
    try testAlmostEql(Self.from(0), Self.from(1e38).sub(.from(1e38)));
}
