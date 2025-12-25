# zig-bigfloat

zig-bigfloat represents a floating point value as `s * 2^e`,
where `1 >= |s| > 2` is a regular floating point number and `e` is a signed integer.
This allows for extremely large and small numbers to be represented with a fixed number of bits,
without excessive precision by selecting a suitable floating point type.

zig-bigfloat is primarily optimized for speed over precision. Benchmark results are in [src/bench.zig](src/bench.zig).

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
const F = @import("bigfloat").BigFloat(.{ .Significand = f64, .Exponent = i64 });

pub fn main() void {
    const pie: F = .init(3.14);
    // pie ^ BOOBIES = 5.097e3979479
    std.debug.print("pie ^ BOOBIES = {e:.3}\n", .{pie.powi(8008135)});
    // Or, if you prefer:
    // std.debug.print("pie ^ BOOBIES = {e:.3}\n", .{F.powi(pie, 8008135)});
}
```

## Use cases

- Incremental games that require numbers larger than f128 can represent (~10^4932)
- not sure, I just wanted to make an incremental game with big ass numbers

## TODO

- add decimal parser?
  - https://github.com/tiehuis/parse-number-fxx-test-data
- add exhaustive decimal formatting/parsing roundtrip tests over f16's range
- fuzz test decimal formatting/parsing roundtripping for f32, f64, f80, f128

## A note on correctness

I'm 99% sure that the functions provided are correct, except for base-10 formatting (used by the `{f}`, `{d}` and `{e}` format specifiers).

The base-10 formatting uses Schubfach. I do not fully understand how to determine the precision needed for it to always be correct.
I instead found the minimum precision needed for various bit-width floats to be formatted correctly by comparing it to Zig's float formating, then fit a line above the recorded points, and added an extra bit of precision just in case.
I have been unable to find a failing example, but also do not have a proof of correctness.

If you need formatting and parsing to always roundtrip, use the `{x}`, `{o}` or `{b}` format specifiers, which are always exact.
