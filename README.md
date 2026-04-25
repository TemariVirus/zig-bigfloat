# The BFP

BFP (big friendly... point?) represents a floating point value as `s * 2^e`,
where `1 >= |s| > 2` is a regular floating point number and `e` is a signed integer.
This allows for extremely large and small numbers to be represented with a fixed number of bits,
without excessive precision by selecting a suitable floating point type.

BFP is primarily optimized for speed over precision. Benchmark results are in [src/bench.zig](src/bench.zig).

## Usage

In your project folder, run this to add BFP to your `build.zig.zon`:

```bash
zig fetch --save git+https://codeberg.org/TemariVirus/BFP#<COMMIT-HASH>
```

Then, add the following to your `build.zig`:

```zig
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ...other build code

    // Import BFP's module into your own
    const bfp = b.dependency("BFP", .{
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("BFP", bfp.module("BFP"));

    // ...other build code
}
```

Now you can use BFP in your code:

```zig
const std = @import("std");
const F = @import("BFP").BigFloat(.{ .Significand = f64, .Exponent = i64 });

pub fn main() void {
    const pie: F = .init(3.14);
    const ans = pie.powi(8008135); // Or, if you prefer: F.powi(pie, 8008135)
    // pie ** BOOBIES = 5.097e3979479
    std.debug.print("pie ** BOOBIES = {e:.3}\n", .{ans});
}
```

## Features

- 0 memory allocation
- Fast with decent accuracy
- Identical behaviour on any platform (assuming floating point `+`, `-`, `*`, `/` and `@abs` are consistent)

Note: The behaviour of some operations may vary from version to version until the library hits 1.0.

## This library is NOT

- Arbitrary precision (you decide the precision at compile time)
- IEEE-754 compilant (especially with respect to rounding when formatting/parsing)

## Use cases

- Incremental games that require numbers larger than f128 can represent (~10^4932)
- not sure, I just wanted to make an incremental game with big ass numbers
