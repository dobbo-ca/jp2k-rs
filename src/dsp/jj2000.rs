//! JJ2000 rounding reproduces the 9/7 decode of JJ2000 (jj2000.j2k in jai-imageio, ome-jai
//! 0.1.5), which Bio-Formats decodes JPEG 2000 with: SynWTFilterFloatLift9x7's synthesis,
//! InvWTFull's row-then-column order and ImgDataConverter's rounding. Java float has no FMA, so
//! every product is a separate f32 operation. The subband gain is in the dequantization step
//! (the decoder multiplies it in), so the high band is divided by K here, not multiplied by
//! OpenJPEG's two_invK.

// The constants keep the reference's decimal literals; they round to the same f32.
#![allow(clippy::excessive_precision)]

use super::dwt::{DWT97_ALPHA, DWT97_BETA, DWT97_DELTA, DWT97_GAMMA};

/// SynWTFilterFloatLift9x7's KL and KH (1/K and K).
const JJ_KL: f32 = 0.8128930655;
const JJ_KH: f32 = 1.230174106;

/// Writes the synthesis of one Mallat-ordered line (low band `l`, high band `h`) to the
/// interleaved `o`: synthetize_lpf when the line starts on a low sample (`cas` 0),
/// synthetize_hpf otherwise. hpf scales `l` and `h` in place, as JJ2000 does.
pub(super) fn synth_97_jj(o: &mut [f32], l: &mut [f32], h: &mut [f32], cas: usize) {
    let (a, b, g, d) = (DWT97_ALPHA, DWT97_BETA, DWT97_GAMMA, DWT97_DELTA);
    let n = l.len() + h.len();
    if cas == 0 {
        if n > 1 {
            o[0] = l[0] / JJ_KL - (2.0 * d * h[0]) / JJ_KH;
        } else {
            o[0] = l[0];
        }
        let mut i = 2;
        while i + 1 < n {
            let k = i / 2;
            o[i] = l[k] / JJ_KL - (d * (h[k - 1] + h[k])) / JJ_KH;
            i += 2;
        }
        if n % 2 == 1 && n > 2 {
            let m = (n - 1) / 2;
            o[n - 1] = l[m] / JJ_KL - (2.0 * d * h[m - 1]) / JJ_KH;
        }
        let mut i = 1;
        while i + 1 < n {
            o[i] = h[(i - 1) / 2] / JJ_KH - (g * (o[i - 1] + o[i + 1]));
            i += 2;
        }
        if n.is_multiple_of(2) {
            o[n - 1] = h[n / 2 - 1] / JJ_KH - (2.0 * g * o[n - 2]);
        }
        if n > 1 {
            o[0] -= 2.0 * b * o[1];
        }
        let mut i = 2;
        while i + 1 < n {
            o[i] -= b * (o[i - 1] + o[i + 1]);
            i += 2;
        }
        if n % 2 == 1 && n > 2 {
            o[n - 1] -= 2.0 * b * o[n - 2];
        }
        let mut i = 1;
        while i + 1 < n {
            o[i] -= a * (o[i - 1] + o[i + 1]);
            i += 2;
        }
        if n.is_multiple_of(2) {
            o[n - 1] -= 2.0 * a * o[n - 2];
        }
        return;
    }
    if n != 1 {
        for v in l.iter_mut() {
            *v /= JJ_KL;
        }
        for v in h.iter_mut() {
            *v /= JJ_KH;
        }
    } else {
        h[0] /= 2.0;
    }
    let mut i = 1;
    while i + 1 < n {
        let k = (i - 1) / 2;
        o[i] = l[k] - (d * (h[k] + h[k + 1]));
        i += 2;
    }
    if n.is_multiple_of(2) && n > 1 {
        o[n - 1] = l[n / 2 - 1] - (2.0 * d * h[n / 2 - 1]);
    }
    if n > 1 {
        o[0] = h[0] - (2.0 * g * o[1]);
    } else {
        o[0] = h[0];
    }
    let mut i = 2;
    while i + 1 < n {
        o[i] = h[i / 2] - (g * (o[i - 1] + o[i + 1]));
        i += 2;
    }
    if n % 2 == 1 && n > 1 {
        o[n - 1] = h[(n - 1) / 2] - (2.0 * g * o[n - 2]);
    }
    let mut i = 1;
    while i + 1 < n {
        o[i] -= b * (o[i - 1] + o[i + 1]);
        i += 2;
    }
    if n.is_multiple_of(2) && n > 1 {
        o[n - 1] -= 2.0 * b * o[n - 2];
    }
    if n > 1 {
        o[0] -= 2.0 * a * o[1];
    }
    let mut i = 2;
    while i + 1 < n {
        o[i] -= a * (o[i - 1] + o[i + 1]);
        i += 2;
    }
    if n % 2 == 1 && n > 1 {
        o[n - 1] -= 2.0 * a * o[n - 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codestream::Rect;
    use crate::dsp::Rounding;
    use crate::dsp::dwt::{inverse_dwt_97, sn_dn};
    use crate::limits::{Budget, Limits};
    use std::sync::atomic::AtomicBool;

    // The vectors come from a float32 Python model written from the ome-jai 0.1.5 Java
    // (SynWTFilterFloatLift9x7, InvWTFull), as in Go's internal/dsp/jj2000_test.go, so a fused or
    // reordered step fails on both architectures.

    /// `(cas, Mallat-ordered line, JJ2000's synthesis of it)` as f32 bits.
    type Vec1D = (usize, &'static [u32], &'static [u32]);
    /// `(width, height, resolution rects, Mallat plane, JJ2000's InvWTFull result)`.
    type Vec2D = (
        usize,
        usize,
        &'static [[i64; 4]],
        &'static [u32],
        &'static [u32],
    );

    const JJ_1D: &[Vec1D] = &[
        (0, &[0x3fc00000, 0x447a2000], &[0xc3f96000, 0x43fae000]),
        (0, &[0x40f00000, 0x40600000], &[0x40b80000, 0x41140000]),
        (0, &[0x47a765b1], &[0x47a765b1]),
        (0, &[0xbf7f4875, 0xc0360a0c], &[0x3ed9974a, 0xc01ad722]),
        (
            0,
            &[0x45bbe65b, 0x3e27f88c, 0xc42723c5],
            &[0x45dbf865, 0x45270330, 0xc3b2c424],
        ),
        (
            0,
            &[0xc289d54a, 0xc3f3cae0, 0x42a315ef, 0xc79e3aaa],
            &[0xc52ecb13, 0x457870e0, 0x46a4c1bc, 0xc7410b4e],
        ),
        (
            0,
            &[0xc5cb0aca, 0x3f66e77e, 0x3f30b36e, 0xbedd4753, 0x4288332b],
            &[0xc5e2557c, 0xc570338a, 0x43b2f6f8, 0x441d6832, 0xc20ec843],
        ),
        (
            0,
            &[
                0xc57808bb, 0xc4464a9c, 0xc5506916, 0xc363d92e, 0x45f8b2c0, 0x4250c0f2,
            ],
            &[
                0xc5767f5e, 0xc5396888, 0xc51e2280, 0x454acf40, 0xc5ab8152, 0xc59d0216,
            ],
        ),
        (
            0,
            &[
                0x4373ad67, 0xc37d66c5, 0x43cfb52b, 0x41184a7e, 0xbf1c34b7, 0x41fa3ea1, 0xbe90171f,
            ],
            &[
                0x439721f8, 0xc1b3cc4a, 0xc3a45874, 0x42b72eac, 0x43ea7c3d, 0x436ab094, 0xc20ff5d7,
            ],
        ),
        (
            0,
            &[
                0xbf495f34, 0x44593f8e, 0x45881eed, 0xc1065305, 0xc4e95080, 0x3f1bb5bc, 0xc449307a,
                0x47b9f339,
            ],
            &[
                0x445fd620, 0xc4712028, 0x44926952, 0x45b5219a, 0x45ce22bd, 0xc5ac40d0, 0xc6c6f309,
                0x475d9582,
            ],
        ),
        (
            0,
            &[
                0x3edc1035, 0x45d3cb8c, 0x3f204e15, 0x440f75f8, 0xc74bdf05, 0x419e7a1e, 0xc2a41199,
                0x47bc6c9c, 0xc599b8f3,
            ],
            &[
                0xc44636d1, 0x45bb0099, 0x460fc504, 0xc56bd400, 0xc6cce898, 0x4775c13c, 0xc6a3137a,
                0xc714e982, 0xc74ca03a,
            ],
        ),
        (1, &[0xc33f98f5], &[0xc2bf98f5]),
        (1, &[0x44700be2, 0xbe676cd8], &[0x447004a8, 0x4470131e]),
        (
            1,
            &[0x3dc53544, 0x43ea1488, 0xc7261cf3],
            &[0x45d98268, 0x4624492e, 0xc6daa902],
        ),
        (
            1,
            &[0x439bed39, 0x3f5763eb, 0x45e9957c, 0xc7c30357],
            &[0x466cca68, 0x46a8d9be, 0xc75817ac, 0x47510428],
        ),
        (
            1,
            &[0xc76ef1bb, 0xc217d945, 0x42c50492, 0x407cd55f, 0x3ffbb786],
            &[0xc78d2704, 0xc77ccab4, 0xc6ef2230, 0x45597ec9, 0x462ddc55],
        ),
        (
            1,
            &[
                0x45c92793, 0x3e090606, 0x43daac72, 0x3ec2c410, 0x3e0a8367, 0xc5b803d7,
            ],
            &[
                0x45e40912, 0x45d19f3b, 0x45599cbc, 0x44869566, 0xc555aafa, 0x4562e848,
            ],
        ),
        (
            1,
            &[
                0x431028d6, 0xc4368b41, 0x3e897d34, 0x47879277, 0xc42bbcee, 0xc7aa34c0, 0x4489a6be,
            ],
            &[
                0x4712e1a0, 0xc699567d, 0x441c1390, 0x46b9fbf1, 0xc750f7ca, 0x46a82670, 0x4660e9c0,
            ],
        ),
        (
            1,
            &[
                0xc509ddde, 0x4490045e, 0x46d12619, 0xc59e7d5f, 0xc7b3fc60, 0x40c3cb2f, 0x474d9691,
                0x3f6fb27a,
            ],
            &[
                0xc7590f9c, 0x46b44e6a, 0x43664980, 0xc675c6f2, 0x47356a7a, 0x467a8102, 0x45eca937,
                0xc5d98dc2,
            ],
        ),
        (
            1,
            &[
                0x46983b07, 0x43e0af52, 0x42a25a00, 0xc5cf19b9, 0xc2334533, 0xc420ce9d, 0xbeec8597,
                0xc0d55355, 0xc5428244,
            ],
            &[
                0x46b3f40a, 0x46a2235c, 0x4615fd7e, 0xc3e2f85c, 0xc45f7283, 0x43c04ace, 0xc540c8cc,
                0xc5c1287e, 0xc617fcee,
            ],
        ),
    ];

    const JJ_2D: &[Vec2D] = &[
        (
            9,
            7,
            &[[0, 0, 3, 2], [0, 0, 5, 4], [0, 0, 9, 7]],
            &[
                0x42cf5edb, 0xc1999f4f, 0xc4a550b2, 0xc5a16b57, 0xc11891fd, 0x43dff76a, 0xc0fcb944,
                0xc081ec13, 0x43badd01, 0xc2dc9c6d, 0x3f128b55, 0x47952b91, 0xc5d49fbb, 0xc42dac50,
                0xbf4153ca, 0xc245b8c1, 0x43b95bdc, 0x3e54cc59, 0x478da146, 0xbd4b62db, 0xbdcf76fd,
                0xc0dd9542, 0xc793c123, 0x44f87f1e, 0xbf6de54b, 0xc48b901a, 0x429fb2a0, 0xbe3c42dd,
                0x41ae5158, 0xc2bda10a, 0xc60ffe97, 0x45232416, 0x422cdd2f, 0xc0360869, 0xc07c1925,
                0x445e0de2, 0xc41acb25, 0x411cb71a, 0xc5c0cd97, 0xc414e90d, 0xc438aea8, 0xbee4161d,
                0x47a40c9f, 0xbbc1fbfa, 0xc549ebb1, 0xc1b0f0c3, 0x3f1744f3, 0x41d620df, 0xc299c578,
                0xc2981bd8, 0xc0f8aa7e, 0x478eab39, 0x44757933, 0xbd53b492, 0xc24362b6, 0xc4223fab,
                0xc442a968, 0xc4464ab2, 0xbf3e4f0c, 0x42a13114, 0x40051bb6, 0x409cde47, 0xc392bc49,
            ],
            &[
                0xc74aff29, 0xc7209153, 0xc69d6cce, 0xc72b8596, 0x45dbac2b, 0x46a0691e, 0x46cd0882,
                0xc65b59d0, 0xc74a7c5a, 0x4541dfc0, 0xc53ae0f0, 0xc6695170, 0x469eff60, 0xc64fbd3c,
                0xc53c0800, 0x453a0fc4, 0x44d81740, 0x44a6ba10, 0x475e55b8, 0x4730fa7a, 0x4703d9a4,
                0xc5eead8e, 0x46aea034, 0xc5abc1ec, 0xc696a822, 0x46d6ccac, 0x47851d66, 0x46bd3fe8,
                0x4647fc52, 0xc6017972, 0x46b45ab2, 0xc61b1cb0, 0x458841d0, 0x46a76cee, 0x475eb002,
                0x479f8e72, 0xc6ac495e, 0xc6a070b1, 0xc69494f6, 0xc6fc7d69, 0xc5a9104d, 0x46d4e418,
                0x477d45a2, 0x479bc28e, 0x47a48130, 0xc643cf4c, 0xc68fd9b8, 0xc6c5198e, 0xc6b8a564,
                0xc6116ac3, 0x46d614cc, 0x47823cb2, 0x47aab8a0, 0x47ba2164, 0x4528d3ae, 0xc6131345,
                0xc6aaac51, 0xc6853865, 0xc5c568d5, 0x46cc29dd, 0x476d0070, 0x47acac78, 0x47c600f1,
            ],
        ),
        (
            9,
            7,
            &[[1, 1, 3, 2], [1, 1, 5, 4], [1, 1, 10, 8]],
            &[
                0x47995ed3, 0x4703a078, 0x47a724a4, 0x444e03eb, 0xc28f21f9, 0xc420bfa6, 0xc457232b,
                0x440d10e8, 0x3f212c0b, 0x428ee4c6, 0xc4854e21, 0xc5254da0, 0x47a3ec58, 0xbe60f4b9,
                0x3f28259b, 0xbf6f5341, 0xc3dac933, 0x3f12d137, 0xc586a486, 0x440e806f, 0x410bbb41,
                0x3bb3bbd0, 0x460a37d3, 0x3ef5ef28, 0xc4ac061d, 0xc68b6f73, 0xc70af634, 0x428b1d6f,
                0xc5688a41, 0xc4009511, 0x40751209, 0xc5a32789, 0xc5fa57cb, 0x410630ec, 0x42a44dbf,
                0xc51887ee, 0xbfecf186, 0xc26ebbf3, 0xc2b3fb69, 0x40ea5b07, 0x46b4b293, 0x41524341,
                0xbbfda59e, 0xc5c479a8, 0x3ffe2b69, 0x40a69d30, 0xc583ee1c, 0x42af8f2b, 0xc21fc791,
                0x414ce156, 0xc1159d52, 0x3fe0c069, 0x42820a8b, 0x44458230, 0x467f49ce, 0xc19a74ae,
                0xc0088e8d, 0x45ba7e10, 0xc444bebb, 0xbf410de4, 0xc39faa6e, 0x445d1160, 0xc4470fd3,
            ],
            &[
                0x480c8aae, 0x4807edc8, 0x479a1566, 0x4707e7de, 0x4775f798, 0x47b61268, 0x472dff74,
                0xc642c4e9, 0xc6c49805, 0x480d2f4e, 0x480670b3, 0x47aac516, 0x4719b6fe, 0x47741a24,
                0x47aa9f35, 0x472c4b6c, 0xc5e575b7, 0xc68456b8, 0x481dcc4e, 0x4807cd96, 0x47bbf0ae,
                0x474a2a4a, 0x474f3b36, 0x476b4b48, 0x4714938e, 0x4698391c, 0x468a94b4, 0x481586e3,
                0x4811e9e8, 0x47d34634, 0x4778a964, 0x4724bb5d, 0x46f52615, 0x4714545c, 0x472dd6bc,
                0x4739d6f8, 0x48171cac, 0x480b567e, 0x47c56c62, 0x475dc58c, 0x4718ff66, 0x47089b72,
                0x4700840c, 0x47480c90, 0x470e7986, 0x48136529, 0x4802151e, 0x47baefdc, 0x4759edb3,
                0x4727cd6d, 0x472c282e, 0x46d9f0c1, 0x474a6157, 0x4683e5d4, 0x4823ec54, 0x480fc232,
                0x47c68432, 0x47523b00, 0x471e8928, 0x472ed274, 0x46ea8294, 0x4764df3c, 0x46996f46,
            ],
        ),
        (
            6,
            5,
            &[[1, 0, 2, 2], [1, 0, 4, 3], [1, 0, 7, 5]],
            &[
                0xc70f751d, 0x4513a62d, 0x46727337, 0x3f195bd2, 0xc0e2043e, 0x4273d539, 0xc1091c09,
                0x47364029, 0x429673c1, 0xc22e8ccc, 0x425522ac, 0xc3f6600b, 0x474d6ec6, 0xbf6914bd,
                0xbde4b0c2, 0xc0d19648, 0x445b0040, 0x40b0a7c4, 0xc259f4db, 0x45e07a54, 0x45f28cd5,
                0x403ec7fa, 0x477ba8e5, 0x3dd77a24, 0x4234ef60, 0xc3234eb9, 0x448c9b3b, 0xc11320ea,
                0xc3b9b0aa, 0xc0563d69,
            ],
            &[
                0xc796b1b8, 0xc7907cce, 0xc7cebf3a, 0xc791c8eb, 0xc78c0081, 0xc781b846, 0xc7104aa9,
                0xc71df21b, 0xc6547400, 0xc72fab12, 0xc6e2c8c4, 0xc69029a8, 0x470bdfab, 0x4705e601,
                0x45b05f34, 0x45cf074c, 0x45f0300c, 0x4657f53c, 0x46fe2a80, 0x46d8cbce, 0x4580aa2a,
                0xc6813250, 0xc662c314, 0xc5ef9460, 0x469747ac, 0x4649a0ea, 0xc637534c, 0xc72458d4,
                0xc722e9a4, 0xc70c97a2,
            ],
        ),
        (
            5,
            6,
            &[[0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 3, 4], [0, 1, 5, 7]],
            &[
                0xc66da642, 0xc09f5dfb, 0xbff599fc, 0xc400073a, 0x42bc930a, 0x44ba8d2a, 0x3eacf75b,
                0x40931c4e, 0x47bac452, 0x45da866d, 0x3f7da5df, 0x461a83b6, 0x473dade7, 0xc76539ba,
                0x44dd89ea, 0x42b6e31e, 0x4772ff6d, 0x426f812f, 0xc119778f, 0xbf6c76a6, 0xbe13b721,
                0xc296b7e0, 0xc21c0d28, 0x45af3c2b, 0x423cb85b, 0x3dd3df45, 0x46809c92, 0x3e834cf8,
                0x40c92b2d, 0xc0171973,
            ],
            &[
                0x4578e740, 0x452ab830, 0x470ca180, 0x4620e82c, 0xc61770fa, 0x4584d74d, 0xc692fdd5,
                0xc6df3b50, 0xc6850aeb, 0xc54cd0fb, 0xc6fb707a, 0x46b8c70a, 0xc720b612, 0xc6445be4,
                0xc4e1aee0, 0xc759a143, 0x47377966, 0xc7473268, 0xc625ce37, 0xc4fbc814, 0xc6c1acdc,
                0x457c26e0, 0x460c5d54, 0x45033018, 0xc675ba90, 0x4618adb0, 0xc749b401, 0x46d8b552,
                0xc5224e94, 0xc6b80423,
            ],
        ),
        (
            1,
            3,
            &[[1, 0, 1, 2], [1, 0, 2, 3]],
            &[0xc43d94a0, 0x4279cdea, 0xbebd6931],
            &[0xc3d52641, 0xc32e0f70, 0x429d18b0],
        ),
    ];

    fn f32s(b: &[u32]) -> Vec<f32> {
        b.iter().map(|&v| f32::from_bits(v)).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Equal bits, or both NaN (a NaN payload never reaches the output).
    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
    }

    fn rect(r: [i64; 4]) -> Rect {
        Rect {
            x0: r[0],
            y0: r[1],
            x1: r[2],
            y1: r[3],
        }
    }

    /// One resolution level of `rw` x `rh` at origin parity (`cas_x`, `cas_y`), as Go's levelRects.
    fn level_rects(rw: i64, rh: i64, cas_x: i64, cas_y: i64) -> Vec<Rect> {
        vec![
            Rect::default(),
            Rect {
                x0: cas_x,
                y0: cas_y,
                x1: cas_x + rw,
                y1: cas_y + rh,
            },
        ]
    }

    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// Go's randFloats: uniform in (-1, 1) times a power of ten up to 1e5.
        fn float(&mut self) -> f32 {
            let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            (u * 2.0 - 1.0) * 10f32.powi((self.next_u64() % 6) as i32)
        }

        /// Go's jjVals: `float` with overflow, NaN and subnormal values mixed in.
        fn vals(&mut self, n: usize) -> Vec<f32> {
            let sp = [
                f32::NAN,
                0.0,
                -0.0,
                f32::from_bits(1),
                2e38,
                -2e38,
                1.5e38,
                -1.5e38,
                3e38,
                f32::INFINITY,
            ];
            (0..n)
                .map(|_| {
                    let v = self.float();
                    if self.next_u64().is_multiple_of(10) {
                        sp[(self.next_u64() % sp.len() as u64) as usize]
                    } else {
                        v
                    }
                })
                .collect()
        }
    }

    fn run(p: &mut [f32], stride: usize, rects: &[Rect], high: Option<&[bool]>) {
        let mut b = Budget::new(&Limits::default());
        inverse_dwt_97(
            p,
            stride,
            rects,
            high,
            Rounding::JJ2000,
            &mut b,
            &AtomicBool::new(false),
            1,
        )
        .unwrap();
    }

    /// Go's TestSynth97JJ.
    #[test]
    fn synth_97_jj_matches_the_java_model() {
        for (cas, input, out) in JJ_1D {
            let mut line = f32s(input);
            let sn = sn_dn(line.len(), *cas).0;
            let mut got = vec![0f32; line.len()];
            let (l, h) = line.split_at_mut(sn);
            synth_97_jj(&mut got, l, h, *cas);
            assert_eq!(bits(&got), out.to_vec(), "cas {cas} n {}", input.len());
        }
    }

    /// Go's TestInverseDWT97JJ.
    #[test]
    fn inverse_dwt_97_jj_matches_the_java_model() {
        for (w, h, rects, input, out) in JJ_2D {
            let rects: Vec<Rect> = rects.iter().map(|&r| rect(r)).collect();
            let mut p = f32s(input);
            run(&mut p, *w, &rects, None);
            assert_eq!(bits(&p), out.to_vec(), "{w}x{h} {rects:?}");
        }
    }

    /// Go's TestInverseDWT97JJZeroHigh: `high` false (zero high bands, rows skipped) gives the
    /// `None` result.
    #[test]
    fn inverse_dwt_97_jj_zero_high() {
        let mut r = Rng(0x5eed_5eed_1234_5679);
        for (rw, rh) in [(1, 1), (1, 5), (5, 1), (2, 2), (5, 7), (70, 33)] {
            for cas in 0..4i64 {
                let (cas_x, cas_y) = (cas & 1, cas >> 1);
                let sn_x = sn_dn(rw, cas_x as usize).0;
                let sn_y = sn_dn(rh, cas_y as usize).0;
                let mut want = vec![0f32; rw * rh];
                let v: Vec<f32> = (0..sn_x * sn_y).map(|_| r.float()).collect();
                for y in 0..sn_y {
                    want[y * rw..y * rw + sn_x].copy_from_slice(&v[y * sn_x..(y + 1) * sn_x]);
                }
                let mut got = want.clone();
                let rects = level_rects(rw as i64, rh as i64, cas_x, cas_y);
                run(&mut want, rw, &rects, None);
                run(&mut got, rw, &rects, Some(&[false, false]));
                assert_eq!(bits(&got), bits(&want), "{rw}x{rh} cas {cas_x},{cas_y}");
            }
        }
    }

    /// Go's TestInverseDWT97JJBlockMatchesRef. Go's block and split kernels (and their tests,
    /// TestSynth97JJBlock and TestSynth97JJSplit) have no Rust twin: this crate has the one line
    /// kernel only. The property they guard is the one tested here: the whole transform, rows then
    /// columns gathered into blocks, equals `synth_97_jj` applied line by line, NaN and infinity
    /// included, at every origin parity.
    #[test]
    fn inverse_dwt_97_jj_matches_line_by_line() {
        let mut r = Rng(0x0b0b_0b0b_9e37_79b9);
        for (rw, rh) in [(1, 1), (7, 5), (65, 63), (200, 70), (300, 130), (256, 129)] {
            for cas in 0..4i64 {
                let (cas_x, cas_y) = (cas & 1, cas >> 1);
                let (sn_x, sn_y) = (sn_dn(rw, cas_x as usize).0, sn_dn(rh, cas_y as usize).0);
                let mut want = r.vals(rw * rh);
                let mut got = want.clone();
                for y in 0..rh {
                    let mut line = want[y * rw..(y + 1) * rw].to_vec();
                    let (l, h) = line.split_at_mut(sn_x);
                    synth_97_jj(&mut want[y * rw..(y + 1) * rw], l, h, cas_x as usize);
                }
                for x in 0..rw {
                    let mut col: Vec<f32> = (0..rh).map(|y| want[y * rw + x]).collect();
                    let mut o = vec![0f32; rh];
                    let (l, h) = col.split_at_mut(sn_y);
                    synth_97_jj(&mut o, l, h, cas_y as usize);
                    for (y, v) in o.iter().enumerate() {
                        want[y * rw + x] = *v;
                    }
                }
                let rects = level_rects(rw as i64, rh as i64, cas_x, cas_y);
                run(&mut got, rw, &rects, None);
                assert!(same(&got, &want), "{rw}x{rh} cas {cas_x},{cas_y}");
            }
        }
    }
}
