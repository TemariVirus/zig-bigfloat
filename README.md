# zig-bigfloat

zig-bigfloat represents a floating point value as `s * 2^e`,
where `1 >= |s| > 2` is a regular floating point number and `e` is a signed integer.
This allows for extremely large and small numbers to be represented with a fixed number of bits,
without excessive precision by selecting a suitable floating point type.

zig-bigfloat is primarily optimized for speed over precision.

## Usage

In your project folder, run this to add zig-bigfloat to your `build.zig.zon`:

```bash
zig fetch git+https://github.com/TemariVirus/zig-bigfloat#<COMMIT-HASH>
```

Then, add the following to your `build.zig`:

```zig
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ...other build code

    // Import zig-bigfloat's module into your own
    const bigfloat = b.dependency("zig-bigfloat", .{
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("bigfloat", bigfloat.module("bigfloat"));

    // ...other build code
}
```

Now you can use zig-bigfloat in your code:

```zig
const std = @import("std");
const BigFloat = @import("bigfloat").BigFloat;

pub fn main() void {
    const F = BigFloat(.{ .Significand = f64, .Exponent = i32 });

    const pie: F = .init(3.14);
    // pie ^ BOOBIES = 5.097e3979479
    std.debug.print("pie ^ BOOBIES = {e:.3}\n", .{pie.powi(8008135)});
}
```

## Use cases

- Incremental games that require numbers larger than f128 can represent (~10^4932)
- not sure, I just wanted to make an incremental game with big ass numbers
