const std = @import("std");
const Writer = std.Io.Writer;
const allocator = std.heap.wasm_allocator;

const BigFloat = @import("bigfloat").BigFloat;

const BFs = [_]type{
    BigFloat(.{ .Significand = f32, .Exponent = i8 }),
    BigFloat(.{ .Significand = f64, .Exponent = i128 }),
};

const JS = struct {
    extern fn _consoleLog([*]const u8, usize) void;
    extern fn _consoleError([*]const u8, usize) void;

    var console_buffer: ?[]u8 = null;

    pub fn consolePrint(comptime fmt: []const u8, args: anytype) void {
        const buf = if (console_buffer) |b| b else blk: {
            @branchHint(.cold);
            console_buffer = allocator.alloc(u8, 64 * 1024) catch {
                consoleError("Failed to allocate console buffer, using stack");
                var _buf: [1024]u8 = undefined;
                break :blk &_buf;
            };
            break :blk console_buffer.?;
        };

        var w: Writer = .fixed(buf);
        w.print(fmt, args) catch {
            consoleError("consolePrint: string too long, truncating.");
        };
        _consoleLog(w.buffered().ptr, w.buffered().len);
    }

    pub fn consoleError(str: []const u8) void {
        _consoleError(str.ptr, str.len);
    }
};

fn SpecialisedFn(generic: anytype) type {
    return @typeInfo(@TypeOf(generic)).@"fn".return_type.?;
}
fn specialise(generic: anytype) [BFs.len]*const SpecialisedFn(generic) {
    var functions: [BFs.len]*const SpecialisedFn(generic) = undefined;
    for (BFs, 0..) |BF, i| {
        functions[i] = generic(BF);
    }
    return functions;
}

/// Packs `bf_type` into the high bits of the pointer.
fn packBfPtr(bf_type: u8, ptr: anytype) @TypeOf(ptr) {
    const bf_bits = comptime std.math.log2_int_ceil(usize, BFs.len);
    const shift = @bitSizeOf(usize) - bf_bits;
    const bf_mask = (1 << shift) - 1;

    var p = @intFromPtr(ptr);
    p &= bf_mask; // Clear bf type
    p |= (@as(usize, bf_type) << shift); // Set bf type
    return @ptrFromInt(p);
}

/// Undoes `packBfPtr`.
fn unpackBfPtr(ptr: anytype) struct { u8, @TypeOf(ptr) } {
    const bf_bits = comptime std.math.log2_int_ceil(usize, BFs.len);
    const shift = @bitSizeOf(usize) - bf_bits;
    const bf_mask = (1 << shift) - 1;

    const p = @intFromPtr(ptr);
    const bf_type: u8 = @intCast(p >> shift);
    const bf_ptr: @TypeOf(ptr) = @ptrFromInt(p & bf_mask);
    return .{ bf_type, bf_ptr };
}

fn _fromFloat(T: type) fn (f64) *anyopaque {
    return (struct {
        fn impl(x: f64) *anyopaque {
            const bf = allocator.create(T) catch unreachable;
            bf.* = .init(x);
            return bf;
        }
    }).impl;
}
export fn fromFloat(bf_type: u8, x: f64) *anyopaque {
    const funcs = comptime specialise(_fromFloat);
    const ptr = funcs[bf_type](x);
    return packBfPtr(bf_type, ptr);
}

fn _freeBF(T: type) fn (?*const anyopaque) void {
    return (struct {
        fn impl(ptr: ?*const anyopaque) void {
            const bf: ?*const T = @ptrCast(@alignCast(ptr));
            if (bf) |p| allocator.destroy(p);
        }
    }).impl;
}
export fn freeBF(ptr: ?*const anyopaque) void {
    const funcs = comptime specialise(_freeBF);
    const bf_type, const p = unpackBfPtr(ptr);
    funcs[bf_type](p);
}

fn _parse(T: type) fn ([]const u8) ?*anyopaque {
    return (struct {
        fn impl(str: []const u8) ?*anyopaque {
            const bf = allocator.create(T) catch return null;
            bf.* = T.parse(str) catch {
                allocator.destroy(bf);
                return null;
            };
            return bf;
        }
    }).impl;
}
export fn parse(bf_type: u8, ptr: [*]const u8, len: usize) ?*anyopaque {
    const funcs = comptime specialise(_parse);
    if (funcs[bf_type](ptr[0..len])) |bf_ptr| {
        return packBfPtr(bf_type, bf_ptr);
    } else {
        return null;
    }
}

fn _format(T: type) fn (*const anyopaque, ?usize, [*]u8) usize {
    return (struct {
        fn impl(ptr: *const anyopaque, precision: ?usize, buf: [*]u8) usize {
            const bf: *const T = @ptrCast(@alignCast(ptr));
            const options: std.fmt.Number = .{
                .mode = .scientific,
                .precision = precision,
            };
            const max_len = T.maxFormatLength(options);
            var w: std.Io.Writer = .fixed(buf[0..max_len]);
            bf.formatNumber(&w, options) catch unreachable;
            return w.buffered().len;
        }
    }).impl;
}
export fn format(ptr: *const anyopaque, precision: usize, buf: [*]u8) usize {
    const funcs = comptime specialise(_format);
    const bf_type, const p = unpackBfPtr(ptr);
    return funcs[bf_type](
        p,
        if (precision == 0) null else precision,
        buf,
    );
}

fn unary_op(comptime op: []const u8) fn (*const anyopaque) *anyopaque {
    const op_fn = (struct {
        fn impl(T: type) fn (*const anyopaque) *anyopaque {
            return (struct {
                fn impl2(ptr: *const anyopaque) *anyopaque {
                    const x: *const T = @ptrCast(@alignCast(ptr));
                    const result = allocator.create(T) catch unreachable;
                    result.* = @field(T, op)(x.*);
                    return result;
                }
            }).impl2;
        }
    }).impl;

    const funcs = comptime specialise(op_fn);
    return (struct {
        fn impl(ptr: *const anyopaque) *anyopaque {
            const bf_type, const x = unpackBfPtr(ptr);
            const result = funcs[bf_type](x);
            return packBfPtr(bf_type, result);
        }
    }).impl;
}

// Needs to be renamed to avoid name clash with @exp2 and @log2 (?)
export fn exp2Bf(ptr: *const anyopaque) *anyopaque {
    return unary_op("exp2")(ptr);
}
export fn log2Bf(ptr: *const anyopaque) *anyopaque {
    return unary_op("log2")(ptr);
}

fn binary_op(comptime op: []const u8) fn (*const anyopaque, *const anyopaque) *anyopaque {
    const op_fn = (struct {
        fn impl(T: type) fn (*const anyopaque, *const anyopaque) *anyopaque {
            return (struct {
                fn impl2(lhs: *const anyopaque, rhs: *const anyopaque) *anyopaque {
                    const l: *const T = @ptrCast(@alignCast(lhs));
                    const r: *const T = @ptrCast(@alignCast(rhs));
                    const result = allocator.create(T) catch unreachable;
                    result.* = @field(T, op)(l.*, r.*);
                    return result;
                }
            }).impl2;
        }
    }).impl;

    const funcs = comptime specialise(op_fn);
    return (struct {
        fn impl(lhs: *const anyopaque, rhs: *const anyopaque) *anyopaque {
            const bf_type, const l = unpackBfPtr(lhs);
            _, const r = unpackBfPtr(rhs);
            const result = funcs[bf_type](l, r);
            return packBfPtr(bf_type, result);
        }
    }).impl;
}

export fn add(lhs: *const anyopaque, rhs: *const anyopaque) *anyopaque {
    return binary_op("add")(lhs, rhs);
}
export fn sub(lhs: *const anyopaque, rhs: *const anyopaque) *anyopaque {
    return binary_op("sub")(lhs, rhs);
}
export fn mul(lhs: *const anyopaque, rhs: *const anyopaque) *anyopaque {
    return binary_op("mul")(lhs, rhs);
}
export fn div(lhs: *const anyopaque, rhs: *const anyopaque) *anyopaque {
    return binary_op("div")(lhs, rhs);
}
export fn pow(lhs: *const anyopaque, rhs: *const anyopaque) *anyopaque {
    return binary_op("pow")(lhs, rhs);
}
