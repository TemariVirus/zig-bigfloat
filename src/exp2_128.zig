// Adapted from: https://github.com/LewisGaul/zig-f128math/blob/04eddc93e09c7018129d43675ddf2022099e9b66/src/exp2.zig#L436-L774

// MIT License
//
// Copyright (c) 2021 Lewis Gaul
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const std = @import("std");
const math = struct {
    pub const isNan = std.math.isNan;
    pub const inf = std.math.inf;
    pub const raiseOverflow = std.math.raiseOverflow;
    pub const scalbn = std.math.scalbn;
    pub const doNotOptimizeAway = std.mem.doNotOptimizeAway;

    pub const qnan_u16: u16 = 0x7E00;
    pub const qnan_f16: f16 = @bitCast(qnan_u16);
    pub const qnan_u32: u32 = 0x7FC00000;
    pub const qnan_f32: f32 = @bitCast(qnan_u32);
    pub const qnan_u64: u64 = 0x7FF8 << 48;
    pub const qnan_f64: f64 = @bitCast(qnan_u64);
    pub const qnan_u128: u128 = 0x7FFF8 << 108;
    pub const qnan_f128: f128 = @bitCast(qnan_u128);

    /// Returns the quiet nan representation for type T.
    pub fn nan(comptime T: type) T {
        return switch (T) {
            f16 => qnan_f16,
            f32 => qnan_f32,
            f64 => qnan_f64,
            f128 => qnan_f128,
            else => @compileError("nan not implemented for " ++ @typeName(T)),
        };
    }
};

const exp2_128_table = [_]f128{
    0x1.6a09e667f3bcc908b2fb1366dfeap-1, 0x1.6c012750bdabeed76a99800f4edep-1,
    0x1.6dfb23c651a2ef220e2cbe1bc0d4p-1, 0x1.6ff7df9519483cf87e1b4f3e1e98p-1,
    0x1.71f75e8ec5f73dd2370f2ef0b148p-1, 0x1.73f9a48a58173bd5c9a4e68ab074p-1,
    0x1.75feb564267c8bf6e9aa33a489a8p-1, 0x1.780694fde5d3f619ae02808592a4p-1,
    0x1.7a11473eb0186d7d51023f6ccb1ap-1, 0x1.7c1ed0130c1327c49334459378dep-1,
    0x1.7e2f336cf4e62105d02ba1579756p-1, 0x1.80427543e1a11b60de67649a3842p-1,
    0x1.82589994cce128acf88afab34928p-1, 0x1.8471a4623c7acce52f6b97c6444cp-1,
    0x1.868d99b4492ec80e41d90ac2556ap-1, 0x1.88ac7d98a669966530bcdf2d4cc0p-1,
    0x1.8ace5422aa0db5ba7c55a192c648p-1, 0x1.8cf3216b5448bef2aa1cd161c57ap-1,
    0x1.8f1ae991577362b982745c72eddap-1, 0x1.9145b0b91ffc588a61b469f6b6a0p-1,
    0x1.93737b0cdc5e4f4501c3f2540ae8p-1, 0x1.95a44cbc8520ee9b483695a0e7fep-1,
    0x1.97d829fde4e4f8b9e920f91e8eb6p-1, 0x1.9a0f170ca07b9ba3109b8c467844p-1,
    0x1.9c49182a3f0901c7c46b071f28dep-1, 0x1.9e86319e323231824ca78e64c462p-1,
    0x1.a0c667b5de564b29ada8b8cabbacp-1, 0x1.a309bec4a2d3358c171f770db1f4p-1,
    0x1.a5503b23e255c8b424491caf88ccp-1, 0x1.a799e1330b3586f2dfb2b158f31ep-1,
    0x1.a9e6b5579fdbf43eb243bdff53a2p-1, 0x1.ac36bbfd3f379c0db966a3126988p-1,
    0x1.ae89f995ad3ad5e8734d17731c80p-1, 0x1.b0e07298db66590842acdfc6fb4ep-1,
    0x1.b33a2b84f15faf6bfd0e7bd941b0p-1, 0x1.b59728de559398e3881111648738p-1,
    0x1.b7f76f2fb5e46eaa7b081ab53ff6p-1, 0x1.ba5b030a10649840cb3c6af5b74cp-1,
    0x1.bcc1e904bc1d2247ba0f45b3d06cp-1, 0x1.bf2c25bd71e088408d7025190cd0p-1,
    0x1.c199bdd85529c2220cb12a0916bap-1, 0x1.c40ab5fffd07a6d14df820f17deap-1,
    0x1.c67f12e57d14b4a2137fd20f2a26p-1, 0x1.c8f6d9406e7b511acbc48805c3f6p-1,
    0x1.cb720dcef90691503cbd1e949d0ap-1, 0x1.cdf0b555dc3f9c44f8958fac4f12p-1,
    0x1.d072d4a07897b8d0f22f21a13792p-1, 0x1.d2f87080d89f18ade123989ea50ep-1,
    0x1.d5818dcfba48725da05aeb66dff8p-1, 0x1.d80e316c98397bb84f9d048807a0p-1,
    0x1.da9e603db3285708c01a5b6d480cp-1, 0x1.dd321f301b4604b695de3c0630c0p-1,
    0x1.dfc97337b9b5eb968cac39ed284cp-1, 0x1.e264614f5a128a12761fa17adc74p-1,
    0x1.e502ee78b3ff6273d130153992d0p-1, 0x1.e7a51fbc74c834b548b2832378a4p-1,
    0x1.ea4afa2a490d9858f73a18f5dab4p-1, 0x1.ecf482d8e67f08db0312fb949d50p-1,
    0x1.efa1bee615a27771fd21a92dabb6p-1, 0x1.f252b376bba974e8696fc3638f24p-1,
    0x1.f50765b6e4540674f84b762861a6p-1, 0x1.f7bfdad9cbe138913b4bfe72bd78p-1,
    0x1.fa7c1819e90d82e90a7e74b26360p-1, 0x1.fd3c22b8f71f10975ba4b32bd006p-1,
    0x1.0000000000000000000000000000p+0, 0x1.0163da9fb33356d84a66ae336e98p+0,
    0x1.02c9a3e778060ee6f7caca4f7a18p+0, 0x1.04315e86e7f84bd738f9a20da442p+0,
    0x1.059b0d31585743ae7c548eb68c6ap+0, 0x1.0706b29ddf6ddc6dc403a9d87b1ep+0,
    0x1.0874518759bc808c35f25d942856p+0, 0x1.09e3ecac6f3834521e060c584d5cp+0,
    0x1.0b5586cf9890f6298b92b7184200p+0, 0x1.0cc922b7247f7407b705b893dbdep+0,
    0x1.0e3ec32d3d1a2020742e4f8af794p+0, 0x1.0fb66affed31af232091dd8a169ep+0,
    0x1.11301d0125b50a4ebbf1aed9321cp+0, 0x1.12abdc06c31cbfb92bad324d6f84p+0,
    0x1.1429aaea92ddfb34101943b2588ep+0, 0x1.15a98c8a58e512480d573dd562aep+0,
    0x1.172b83c7d517adcdf7c8c50eb162p+0, 0x1.18af9388c8de9bbbf70b9a3c269cp+0,
    0x1.1a35beb6fcb753cb698f692d2038p+0, 0x1.1bbe084045cd39ab1e72b442810ep+0,
    0x1.1d4873168b9aa7805b8028990be8p+0, 0x1.1ed5022fcd91cb8819ff61121fbep+0,
    0x1.2063b88628cd63b8eeb0295093f6p+0, 0x1.21f49917ddc962552fd29294bc20p+0,
    0x1.2387a6e75623866c1fadb1c159c0p+0, 0x1.251ce4fb2a63f3582ab7de9e9562p+0,
    0x1.26b4565e27cdd257a673281d3068p+0, 0x1.284dfe1f5638096cf15cf03c9fa0p+0,
    0x1.29e9df51fdee12c25d15f5a25022p+0, 0x1.2b87fd0dad98ffddea46538fca24p+0,
    0x1.2d285a6e4030b40091d536d0733ep+0, 0x1.2ecafa93e2f5611ca0f45d5239a4p+0,
    0x1.306fe0a31b7152de8d5a463063bep+0, 0x1.32170fc4cd8313539cf1c3009330p+0,
    0x1.33c08b26416ff4c9c8610d96680ep+0, 0x1.356c55f929ff0c94623476373be4p+0,
    0x1.371a7373aa9caa7145502f45452ap+0, 0x1.38cae6d05d86585a9cb0d9bed530p+0,
    0x1.3a7db34e59ff6ea1bc9299e0a1fep+0, 0x1.3c32dc313a8e484001f228b58cf0p+0,
    0x1.3dea64c12342235b41223e13d7eep+0, 0x1.3fa4504ac801ba0bf701aa417b9cp+0,
    0x1.4160a21f72e29f84325b8f3dbacap+0, 0x1.431f5d950a896dc704439410b628p+0,
    0x1.44e086061892d03136f409df0724p+0, 0x1.46a41ed1d005772512f459229f0ap+0,
    0x1.486a2b5c13cd013c1a3b69062f26p+0, 0x1.4a32af0d7d3de672d8bcf46f99b4p+0,
    0x1.4bfdad5362a271d4397afec42e36p+0, 0x1.4dcb299fddd0d63b36ef1a9e19dep+0,
    0x1.4f9b2769d2ca6ad33d8b69aa0b8cp+0, 0x1.516daa2cf6641c112f52c84d6066p+0,
    0x1.5342b569d4f81df0a83c49d86bf4p+0, 0x1.551a4ca5d920ec52ec620243540cp+0,
    0x1.56f4736b527da66ecb004764e61ep+0, 0x1.58d12d497c7fd252bc2b7343d554p+0,
    0x1.5ab07dd48542958c93015191e9a8p+0, 0x1.5c9268a5946b701c4b1b81697ed4p+0,
    0x1.5e76f15ad21486e9be4c20399d12p+0, 0x1.605e1b976dc08b076f592a487066p+0,
    0x1.6247eb03a5584b1f0fa06fd2d9eap+0, 0x1.6434634ccc31fc76f8714c4ee122p+0,
    0x1.66238825522249127d9e29b92ea2p+0, 0x1.68155d44ca973081c57227b9f69ep+0,
};

// Use a separate table since these values are only 32-bit floats.
const exp2_128_eps_table = [exp2_128_table.len]f32{
    -0x1.5c50p-101, -0x1.5d00p-106, 0x1.8e90p-102,  -0x1.5340p-103,
    0x1.1bd0p-102,  -0x1.4600p-105, -0x1.7a40p-104, 0x1.d590p-102,
    -0x1.d590p-101, 0x1.b100p-103,  -0x1.0d80p-105, 0x1.6b00p-103,
    -0x1.9f00p-105, 0x1.c400p-103,  0x1.e120p-103,  -0x1.c100p-104,
    -0x1.9d20p-103, 0x1.a800p-108,  0x1.4c00p-106,  -0x1.9500p-106,
    0x1.6900p-105,  -0x1.29d0p-100, 0x1.4c60p-103,  0x1.13a0p-102,
    -0x1.5b60p-103, -0x1.1c40p-103, 0x1.db80p-102,  0x1.91a0p-102,
    0x1.dc00p-105,  0x1.44c0p-104,  0x1.9710p-102,  0x1.8760p-103,
    -0x1.a720p-103, 0x1.ed20p-103,  -0x1.49c0p-102, -0x1.e000p-111,
    0x1.86a0p-103,  0x1.2b40p-103,  -0x1.b400p-108, 0x1.1280p-99,
    -0x1.02d8p-102, -0x1.e3d0p-103, -0x1.b080p-105, -0x1.f100p-107,
    -0x1.16c0p-105, -0x1.1190p-103, -0x1.a7d2p-100, 0x1.3450p-103,
    -0x1.67c0p-105, 0x1.4b80p-104,  -0x1.c4e0p-103, 0x1.6000p-108,
    -0x1.3f60p-105, 0x1.93f0p-104,  0x1.5fe0p-105,  0x1.6f80p-107,
    -0x1.7600p-106, 0x1.21e0p-106,  -0x1.3a40p-106, -0x1.40c0p-104,
    -0x1.9860p-105, -0x1.5d40p-108, -0x1.1d70p-106, 0x1.2760p-105,
    0x0.0000p+0,    0x1.21e2p-104,  -0x1.9520p-108, -0x1.5720p-106,
    -0x1.4810p-106, -0x1.be00p-109, 0x1.0080p-105,  -0x1.5780p-108,
    -0x1.d460p-105, -0x1.6140p-105, 0x1.4630p-104,  0x1.ad50p-103,
    0x1.82e0p-105,  0x1.1d3cp-101,  0x1.6100p-107,  0x1.ec30p-104,
    0x1.f200p-108,  0x1.0b40p-103,  0x1.3660p-102,  0x1.d9d0p-103,
    -0x1.02d0p-102, 0x1.b070p-103,  0x1.b9c0p-104,  -0x1.01c0p-103,
    -0x1.dfe0p-103, 0x1.1b60p-104,  -0x1.ae94p-101, -0x1.3340p-104,
    0x1.b3d8p-102,  -0x1.6e40p-105, -0x1.3670p-103, 0x1.c140p-104,
    0x1.1840p-101,  0x1.1ab0p-102,  -0x1.a400p-104, 0x1.1f00p-104,
    -0x1.7180p-103, 0x1.4ce0p-102,  0x1.9200p-107,  -0x1.54c0p-103,
    0x1.1b80p-105,  -0x1.1828p-101, 0x1.5720p-102,  -0x1.a060p-100,
    0x1.9160p-102,  0x1.a280p-104,  0x1.3400p-107,  0x1.2b20p-102,
    0x1.7800p-108,  0x1.cfd0p-101,  0x1.2ef0p-102,  -0x1.2760p-99,
    0x1.b380p-104,  0x1.0048p-101,  -0x1.60b0p-102, 0x1.a1ccp-100,
    -0x1.a640p-104, -0x1.08a0p-101, 0x1.7e60p-102,  0x1.22c0p-103,
    -0x1.7200p-106, 0x1.f0f0p-102,  0x1.eb4ep-99,   0x1.c6e0p-103,
};

pub fn exp2_128(x: f128) f128 {
    const tblsiz: u32 = @intCast(exp2_128_table.len);
    const redux: f128 = 0x1.8p112 / @as(f128, @floatFromInt(tblsiz));

    const P1: f128 = 0x1.62e42fefa39ef35793c7673007e6p-1;
    const P2: f128 = 0x1.ebfbdff82c58ea86f16b06ec9736p-3;
    const P3: f128 = 0x1.c6b08d704a0bf8b33a762bad3459p-5;
    const P4: f128 = 0x1.3b2ab6fba4e7729ccbbe0b4f3fc2p-7;
    const P5: f128 = 0x1.5d87fe78a67311071dee13fd11d9p-10;
    const P6: f128 = 0x1.430912f86c7876f4b663b23c5fe5p-13;
    const P7: f64 = 0x1.ffcbfc588b041p-17;
    const P8: f64 = 0x1.62c0223a5c7c7p-20;
    const P9: f64 = 0x1.b52541ff59713p-24;
    const P10: f64 = 0x1.e4cf56a391e22p-28;

    // Return canonical NaN for any NaN input.
    if (math.isNan(x)) {
        return math.nan(f128);
    }

    const ux: u128 = @bitCast(x);
    const e: u16 = @as(u16, @intCast(ux >> 112)) & 0x7FFF; // exponent

    // |x| >= 16384 or nan
    if (e >= 0x3FFF + 14) {
        // x >= 16384
        if (e >= 0x3FFF + 15 and ux >> 127 == 0) {
            math.raiseOverflow();
            return math.inf(f128);
        }
        // -inf or nan
        if (e == 0x7FFF) {
            return -1 / x;
        }
        // x <= -1022
        if (x < -16382) {
            // underflow
            if (x <= -16495 or x - 0x1p112 + 0x1p112 != x) {
                math.doNotOptimizeAway(@as(f32, @floatCast(-0x1.0p-149 / x)));
            }
            if (x <= -16495) {
                return 0;
            }
        }
    }
    // |x| < 0x1p-114
    else if (e < 0x3FFF - 114) {
        return 1.0 + x;
    }

    // NOTE: musl relies on unsafe behaviours which are replicated below
    // (addition overflow, division truncation, casting). Appears that this
    // produces the intended result but should confirm how GCC/Clang handle this
    // to ensure.

    // reduce x
    var u_f: f128 = x + redux;
    var i_0: u32 = @truncate(@as(u128, @bitCast(u_f)));
    i_0 +%= tblsiz / 2;

    const k_u: u32 = i_0 / tblsiz * tblsiz;
    const k_i: i32 = @divTrunc(@as(i32, @bitCast(k_u)), tblsiz);
    i_0 %= tblsiz;
    u_f -= redux;
    var z: f128 = x - u_f;

    // r = exp2(y) = exp2t[i_0] * p(z - eps[i])
    const t: f128 = exp2_128_table[@intCast(i_0)];
    z -= exp2_128_eps_table[@intCast(i_0)];
    // zig fmt: off
    const r: f128 = t + t * z * (P1 + z * (P2 + z * (P3 + z * (P4 + z * (P5
        + z * (P6 + z * (P7 + z * (P8 + z * (P9 + z * P10)))))))));
    // zig fmt: on

    return math.scalbn(r, k_i);
}
