const std = @import("std");
const Writer = std.Io.Writer;
const allocator = std.heap.wasm_allocator;

const BigFloat = @import("bigfloat").BigFloat;

const BigBF = BigFloat(.{ .Significand = f64, .Exponent = i128, .bake_render = true });
const BFs = [_]type{
    BigFloat(.{ .Significand = f32, .Exponent = i8, .bake_render = true }),
    BigBF,
};

const JS = struct {
    extern fn _consoleLog([*]const u8, usize) void;
    extern fn _consoleError([*]const u8, usize) void;
    extern fn _setInnerHtml([*]const u8, usize, [*]const u8, usize) void;
    extern fn _removeHtmlClass([*]const u8, usize, [*]const u8, usize) void;
    extern fn _addHtmlClass([*]const u8, usize, [*]const u8, usize) void;

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

    pub fn setInnerHtml(id: []const u8, text: []const u8) void {
        _setInnerHtml(id.ptr, id.len, text.ptr, text.len);
    }

    pub fn removeHtmlClass(id: []const u8, class: []const u8) void {
        _removeHtmlClass(id.ptr, id.len, class.ptr, class.len);
    }

    pub fn addHtmlClass(id: []const u8, class: []const u8) void {
        _addHtmlClass(id.ptr, id.len, class.ptr, class.len);
    }
};

const Game = struct {
    const tick_rate: BigBF = .init(100); // fps

    pub var paperclip_count: BigBF = .init(1);
    pub var dim_counts: [8]BigBF = @splat(.init(0));
    pub var dim_purchases: [8]BigBF = @splat(.init(0));

    pub fn dimProduction(dim: u8) BigBF {
        const bought_multipliers: [dim_counts.len]BigBF = comptime .{
            .init(3),
            .init(1e1),
            .init(1e2),
            .init(3e3),
            .init(1e13),
            .init(1e82),
            .init(1e400),
            BigBF.parse("1e1000") catch unreachable,
        };

        const multiplier: BigBF = bought_multipliers[dim].pow(dim_purchases[dim]);
        var prod = dim_counts[dim].mul(multiplier).div(tick_rate);
        if (dim > 0) {
            prod = prod.mul(.pow(dim_counts[dim - 1], .init(0.75)));
        }
        return prod;
    }

    pub fn dimCost(dim: u8) BigBF {
        const base_costs: [dim_counts.len]BigBF = comptime .{
            .init(1),
            .init(1e2),
            .init(1e8),
            .init(1e69),
            BigBF.parse("1e7216") catch unreachable,
            BigBF.parse("1e123456") catch unreachable,
            BigBF.parse("1e42424242") catch unreachable,
            BigBF.parse("1e696969696969") catch unreachable,
        };
        const cost_multipliers: [dim_counts.len]BigBF = comptime .{
            .init(1e1),
            .init(1e8),
            .init(1e82),
            BigBF.parse("1e500") catch unreachable,
            BigBF.parse("1e8000") catch unreachable,
            BigBF.parse("1e100000") catch unreachable,
            BigBF.parse("1e20000000") catch unreachable,
            BigBF.parse("1e1000000000000") catch unreachable,
        };
        return base_costs[dim].mul(cost_multipliers[dim].pow(dim_purchases[dim]));
    }

    pub fn canBuyDim(dim: u8) bool {
        return paperclip_count.gte(dimCost(dim));
    }

    pub fn buyDim(dim: u8) void {
        paperclip_count = paperclip_count.sub(dimCost(dim));
        dim_counts[dim] = dim_counts[dim].add(.init(1));
        dim_purchases[dim] = dim_purchases[dim].add(.init(1));
    }

    pub fn doProduction() void {
        var i: u8 = dim_counts.len - 1;
        while (i > 0) : (i -= 1) {
            const prod = dimProduction(i);
            dim_counts[i - 1] = dim_counts[i - 1].add(prod);
        }
        paperclip_count = .add(paperclip_count, dimProduction(0));
    }

    pub fn render() void {
        printDimToHtml("paperclips", paperclip_count);
        inline for (0..dim_counts.len) |i| {
            const dim_id = std.fmt.comptimePrint("dim{d}", .{i + 1});
            printDimToHtml(dim_id, dim_counts[i]);
            printCostToHtml(dim_id ++ "-cost", dimCost(i));

            const row_id = dim_id ++ "-row";
            if (i > 0 and dim_counts[i - 1].lte(.init(0))) {
                JS.addHtmlClass(row_id, "hidden");
            } else {
                JS.removeHtmlClass(row_id, "hidden");
            }

            const button_id = dim_id ++ "-buy";
            if (canBuyDim(i)) {
                JS.removeHtmlClass(button_id, "not-buyable");
                JS.addHtmlClass(button_id, "buyable");
            } else {
                JS.removeHtmlClass(button_id, "buyable");
                JS.addHtmlClass(button_id, "not-buyable");
            }

            const button_text_id = dim_id ++ "-buy-text";
            const dim_text = std.fmt.comptimePrint(" Dim {d}", .{i + 1});
            if (dim_purchases[i].eql(.init(0))) {
                JS.setInnerHtml(button_text_id, "Buy" ++ dim_text);
            } else {
                JS.setInnerHtml(button_text_id, "Upgrade" ++ dim_text);
            }
        }
    }

    fn printDimToHtml(id: []const u8, value: BigBF) void {
        var buf: [
            @max(
                7,
                BigBF.maxFormatLength(.{ .mode = .scientific, .precision = 3 }),
            )
        ]u8 = undefined;
        var w: Writer = .fixed(&buf);

        if (value.gte(.init(1e6))) {
            w.print("{e:.3}", .{value}) catch unreachable;
        } else {
            w.print("{d:.2}", .{value}) catch unreachable;
        }
        JS.setInnerHtml(id, w.buffered());
    }

    fn printCostToHtml(id: []const u8, value: BigBF) void {
        var buf: [
            @max(
                7,
                BigBF.maxFormatLength(.{ .mode = .scientific, .precision = 3 }),
            )
        ]u8 = undefined;
        var w: Writer = .fixed(&buf);

        const formatted = if (value.gte(.init(1e3))) blk: {
            w.print("{e:.3}", .{value}) catch unreachable;
            const e_index = std.mem.indexOfScalar(u8, w.buffered(), 'e').?;
            const no_trailing_zero = std.mem.trimEnd(u8, w.buffered()[0..e_index], "0");
            const significand = std.mem.trimEnd(u8, no_trailing_zero, ".");
            const exponent = w.buffered()[e_index..];
            @memmove(w.buffered()[significand.len..], exponent);
            break :blk w.buffered()[0..(significand.len + exponent.len)];
        } else blk: {
            w.print("{d:.2}", .{value}) catch unreachable;
            const no_trailing_zero = std.mem.trimEnd(u8, w.buffered(), "0");
            break :blk std.mem.trimEnd(u8, no_trailing_zero, ".");
        };
        JS.setInnerHtml(id, formatted);
    }
};

export fn gameLoop() void {
    Game.doProduction();
    Game.render();
}

export fn buyDim(dim: u8) bool {
    if (Game.canBuyDim(dim - 1)) {
        Game.buyDim(dim - 1);
        return true;
    }
    return false;
}

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

fn growLen(current: usize, target: usize) usize {
    var size = @max(8, current);
    while (size < target) {
        if (size > std.math.maxInt(usize) / 2) return target;
        size *= 2;
    }
    return size;
}

var input_buf: ?[]u8 = null;
export fn ensureBufferSize(len: usize) ?[*]u8 {
    if (input_buf) |buf| {
        if (len <= buf.len) return buf.ptr;
    }

    input_buf = blk: {
        break :blk if (input_buf) |buf|
            allocator.realloc(buf, growLen(buf.len, len))
        else
            allocator.alloc(u8, len);
    } catch {
        JS.consoleError("Wasm: OOM");
        return null;
    };
    return input_buf.?.ptr;
}

/// Packs `bf_type` into the high bits of the pointer.
/// We can safely do this because wasm uses a linear memory model that starts
/// from address 0.
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
