//! The sample-level stages of a decode: dequantization step sizes, the inverse wavelet
//! transforms, the inverse colour transforms and output sample conversion. The lifting steps,
//! the ICT and RCT, the row dequantizations and the output packs have SIMD kernels in [`simd`];
//! the rest is scalar.

mod dwt;
mod jj2000;
pub(crate) mod simd;
#[doc(hidden)]
pub use simd::fuzz;

pub(crate) use dwt::DwtScratch;
pub use dwt::{inverse_dwt_53, inverse_dwt_97, sn_dn};

use crate::codestream::Quantization;
use crate::error::Result;

/// How the 9/7 lift and the ICT round each `c*x + d`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rounding {
    /// Rounds once, as opj_decompress does where its compiler contracts the multiply-add
    /// (arm64, or x86-64 built with FMA). `f32::mul_add` is correctly rounded on every target.
    #[default]
    Fused,
    /// Rounds the product and then the sum, as OpenJPEG built for x86-64 without contraction
    /// does (Ubuntu's libopenjp2, and so libvips).
    Separate,
    /// `Separate` plus JJ2000's own 9/7, dequantization gain and sample rounding. Its lifts use
    /// the JJ2000 kernels and its ICT the `Separate` one.
    JJ2000,
}

impl Rounding {
    /// `a*b + c` rounded as `self` says.
    #[inline]
    pub fn mul_add(self, a: f32, b: f32, c: f32) -> f32 {
        if self != Rounding::Fused {
            // Rust never contracts a separate multiply and add, so this rounds twice.
            a * b + c
        } else {
            a.mul_add(b, c)
        }
    }
}

// T.800 Annex G: multiple component transform, DC level shift, and the final clamp to the
// component's declared precision. These run, in this order, after the inverse DWT (Annex F) for
// each component.

/// T.800 G.2's reversible (5/3) inverse colour transform on components 0, 1, 2, in place.
pub fn inverse_rct(c0: &mut [i32], c1: &mut [i32], c2: &mut [i32]) {
    // Over the shortest of the three, so mismatched lengths cannot panic.
    simd::rct(c0, c1, c2);
}

// ICT coefficients: OpenJPEG's and pdf.js's five-significant-figure literals, not T.800 G.3's
// longer constants, so results stay within a float32 ULP of opj_decompress. Operand order
// matters in float32 and is kept as both reference implementations have it.
const ICT_CR_R: f32 = 1.402;
const ICT_CB_G: f32 = 0.34413;
const ICT_CR_G: f32 = 0.71414;
const ICT_CB_B: f32 = 1.772;

/// T.800 G.3's irreversible (9/7) inverse colour transform on components 0, 1, 2, in place,
/// rounding as `r` says.
pub fn inverse_ict(c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], r: Rounding) {
    simd::ict(c0, c1, c2, r);
}

/// The range a component of the given precision and signedness can hold.
fn precision_range(precision: i32, signed: bool) -> (i32, i32) {
    if signed {
        (-(1i32 << (precision - 1)), (1i32 << (precision - 1)) - 1)
    } else {
        (0, ((1i64 << precision) - 1) as i32)
    }
}

/// Clamps `v` to the range a component of the given precision and signedness can hold.
pub fn clamp_to_precision(v: i32, precision: i32, signed: bool) -> i32 {
    let (lo, hi) = precision_range(precision, signed);
    v.clamp(lo, hi)
}

/// T.800's decoder-side DC shift: 2^(precision-1) for unsigned components and 0 for signed ones.
///
/// `precision` outside [1, 31] is clamped into it, so no value can overflow the shift.
pub fn dc_level_shift(precision: i32, signed: bool) -> i32 {
    if signed {
        0
    } else {
        1i32 << (precision.clamp(1, 31) - 1)
    }
}

/// DC-shifts and clamps a row of reversible (5/3) samples into `dst`. A signed component's true
/// range is then biased up by 2^(precision-1) so every component lands in [0, 2^precision).
pub fn samples_r(dst: &mut [i32], src: &[i32], precision: i32, signed: bool) {
    let shift = dc_level_shift(precision, signed);
    let bias = if signed { 1i32 << (precision - 1) } else { 0 };
    let (lo, hi) = precision_range(precision, signed);
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = v.wrapping_add(shift).clamp(lo, hi).wrapping_add(bias);
    }
}

/// `samples_r`'s irreversible (9/7) counterpart: rounds each f32 sample to the nearest integer
/// (ties to even, as lrintf in opj_decompress) before shifting and clamping.
///
/// The same values as `sample_f`, computed in f32 so that the loop vectorizes: clamping to the
/// range first changes nothing (its ends are integers, so rounding commutes with it; NaN falls
/// to the low end, as `sample_f` maps it to i32::MIN), and then |v| is at most 2^15, where adding
/// and subtracting 1.5 * 2^23 rounds to the nearest integer, ties to even, exactly.
pub fn samples_f(dst: &mut [i32], src: &[f32], precision: i32) {
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23
    let shift = 1i32 << (precision - 1);
    let lo = -(shift as f32);
    let hi = ((1i32 << precision) - 1 - shift) as f32;
    for (d, &v) in dst.iter_mut().zip(src) {
        // Compare-and-select, which is one max and one min instruction (f32::max also
        // spends instructions on NaN); a NaN fails the first comparison and gets `lo`.
        let c = if v > lo { v } else { lo };
        let c = if c < hi { c } else { hi };
        *d = ((c + MAGIC) - MAGIC) as i32 + shift;
    }
}

/// `samples_f` stored as u16 (`as u16` of each sample), in one pass: `src` is at least as long as
/// `dst`.
pub(crate) fn pack_u16(dst: &mut [u16], src: &[f32], precision: i32) {
    simd::pack_u16(dst, src, precision)
}

/// `samples_f` stored as u8 (`as u8` of each sample) into byte `ch` (below 4) of each 4-byte pixel
/// of `dst`, in one pass, the other bytes untouched: `dst` holds at least `4 * src.len()` bytes.
pub(crate) fn pack_u8x4(dst: &mut [u8], src: &[f32], precision: i32, ch: usize) {
    simd::pack_u8x4(dst, src, precision, ch)
}

pub fn sample_f(v: f32, precision: i32) -> i32 {
    let shift = 1i64 << (precision - 1);
    let hi = (1i64 << precision) - 1;
    let f = v as f64;
    let mut iv: i64 = if f > i32::MAX as f64 {
        i32::MAX as i64
    } else if f < i32::MIN as f64 || f.is_nan() {
        // NaN as amd64 converts it.
        i32::MIN as i64
    } else {
        f.round_ties_even() as i64
    };
    iv += shift;
    iv = iv.clamp(0, hi);
    iv as i32
}

/// `samples_f` with JJ2000's rounding: ImgDataConverter adds 0.5 away from zero in f32 and
/// truncates as Java's int cast does, then J2KReadState adds the level shift (wrapping) and
/// clamps.
pub fn samples_jj(dst: &mut [i32], src: &[f32], precision: i32) {
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = sample_jj(v, precision);
    }
}

pub fn sample_jj(mut v: f32, precision: i32) -> i32 {
    if v > 0.0 {
        v += 0.5;
    } else {
        v -= 0.5;
    }
    let mut i: i32 = if v.is_nan() {
        0
    } else if v >= 2147483648.0 {
        i32::MAX
    } else if v <= -2147483648.0 {
        i32::MIN
    } else {
        v as i32
    };
    i = i.wrapping_add(1i32 << (precision - 1));
    i.clamp(0, ((1i64 << precision) - 1) as i32)
}

/// Maps a precision-bit sample to the full 16-bit range by bit replication
/// (`v << shift | v >> (2*precision - 16)`) rather than left-shift alone, so the component's
/// maximum value lands on 0xffff. `precision` can be anywhere in [1, 16]; for precision < 8 the
/// low bits are replicated a full `v << precision` at a time until the register is filled.
/// A `precision` outside [1, 16] is clamped into it (0 would never fill the register).
pub fn expand16(v: u32, precision: u32) -> u16 {
    let precision = precision.clamp(1, 16);
    if precision == 16 {
        return v as u16;
    }
    let mut out = v;
    let mut filled = precision;
    while filled < 16 {
        out = (out << precision) | v;
        filled += precision;
    }
    (out >> (filled - 16)) as u16
}

// -- T.800 Annex E: dequantization --

/// T.800 Annex E's flat band index: 0 for the LL-only resolution (r = 0), otherwise
/// resolution-major HL, LH, HH per level above it.
fn band_index(r: usize, orientation: usize) -> Result<usize> {
    if r == 0 {
        if orientation != 0 {
            return Err(invalid!(
                "resolution 0 has no orientation {orientation} band (LL only)"
            ));
        }
        return Ok(0);
    }
    if orientation == 0 {
        return Err(invalid!(
            "resolution {r} has no LL band (LL only exists at resolution 0)"
        ));
    }
    Ok(1 + 3 * (r - 1) + (orientation - 1))
}

/// Annex E's dequantization step size delta_b (E.1.1.1) and Mb, the total bit-plane count
/// (E.2), for the subband at resolution `r` (0 = the LL-only resolution) and `orientation`.
/// `precision` is the component's bit depth. Like OpenJPEG (t1.c:1798), the "no step size"
/// decision keys on the transform (`reversible`), not on the quantization style: a malformed
/// stream pairing the 9/7 transform with style 0 still needs a real delta. For a reversible
/// transform delta carries no meaning and is returned as 0.
pub fn step_size(
    q: &Quantization,
    r: usize,
    orientation: usize,
    precision: i32,
    reversible: bool,
) -> Result<(f64, i32)> {
    let b = band_index(r, orientation)?;
    let expn: u8;
    let mut mant: u16 = 0;
    match q.quantization_style {
        1 => {
            // scalar derived, T.800 E-5: every band's exponent derives from band 0's
            let Some(&e0) = q.exponents.first() else {
                return Err(invalid!(
                    "scalar-derived quantization has no exponent for band 0"
                ));
            };
            expn = if b == 0 {
                e0
            } else {
                (e0 as i64 - ((b - 1) / 3) as i64).max(0) as u8
            };
            let Some(&m0) = q.mantissas.first() else {
                return Err(invalid!(
                    "scalar-derived quantization has no mantissa for band 0"
                ));
            };
            mant = m0;
        }
        _ => {
            // 0 (reversible) and 2 (expounded): one exponent per band already
            if b >= q.exponents.len() {
                return Err(invalid!(
                    "quantization segment has no band {b} (resolution {r}, orientation {orientation}) among its {} bands",
                    q.exponents.len()
                ));
            }
            expn = q.exponents[b];
            if q.quantization_style == 2 {
                mant = q.mantissas[b];
            }
        }
    }
    let mb = expn as i32 + q.guard_bits as i32 - 1;
    if reversible {
        return Ok((0.0, mb));
    }
    // Rb = precision, with no per-subband gain term: this deliberately diverges from T.800 E-3's
    // Rb = precision + gain(band) to match OpenJPEG's decode-only convention
    // (BUG_WEIRD_TWO_INVK), which drops the gain here and folds it into the inverse DWT's
    // high-band scale instead.
    let rb = precision;
    // exp2 rather than powf(2, x): rb - expn can be negative and this stays exact.
    let delta = (1.0 + mant as f64 / 2048.0) * ((rb - expn as i32) as f64).exp2();
    Ok((delta, mb))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    fn rand_floats(next: &mut impl FnMut() -> u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| ((next() % 2_000_001) as f32 - 1_000_000.0) / 1024.0)
            .collect()
    }

    /// Maps a precision-bit sample to the full 16-bit range, so the component's maximum lands on
    /// 0xffff (12-bit would otherwise land on 0xfff0). The precision < 8 cases guard the
    /// negative shift: classification routes every non-8-bit precision <= 16 here.
    #[test]
    fn expand16_full_range() {
        for (prec, v, want) in [
            (12, 4095, 0xffff),
            (12, 0, 0),
            (16, 65535, 0xffff),
            (9, 511, 0xffff),
            (1, 1, 0xffff),
            (1, 0, 0),
            (2, 3, 0xffff),
            (4, 15, 0xffff),
            (4, 0, 0),
            (7, 127, 0xffff),
        ] {
            assert_eq!(expand16(v, prec), want, "expand16({v}, precision={prec})");
        }
    }

    fn quant(style: u8, guard: u8, exps: &[u8], mants: &[u16]) -> Quantization {
        Quantization {
            quantization_style: style,
            guard_bits: guard,
            exponents: exps.to_vec(),
            mantissas: mants.to_vec(),
        }
    }

    /// `step_size` must not index mantissas[0] for a hand-built style-1 quantization without it.
    #[test]
    fn step_size_style1_missing_mantissa() {
        assert!(step_size(&quant(1, 2, &[8], &[]), 0, 0, 8, false).is_err());
    }

    /// The scalar-derived (style 1) path, T.800 E-5's per-level exponent derivation: expected
    /// values from expn(b) = max(expn(0) - floor((b-1)/3), 0).
    #[test]
    fn step_size_scalar_derived() {
        let q = quant(1, 2, &[10], &[1024]);
        for (r, orient, want_mb, want_delta) in [
            (0, 0, 11, 0.375),
            (1, 1, 11, 0.375), // band 1: (1-1)/3=0, expn=10
            (2, 1, 10, 0.75),  // band 4: (4-1)/3=1, expn=9
            (2, 3, 10, 0.75),  // band 6: (6-1)/3=1, expn=9
        ] {
            let (delta, mb) = step_size(&q, r, orient, 8, false).unwrap();
            assert_eq!((mb, delta), (want_mb, want_delta), "r={r} orient={orient}");
        }
    }

    /// The "no step size" decision keys on the transform, not on the quantization style: a
    /// malformed stream can pair style 0 with the 9/7 transform, which still needs a real delta.
    #[test]
    fn step_size_style_zero_irreversible() {
        let (delta, _) = step_size(&quant(0, 2, &[8], &[0]), 0, 0, 8, false).unwrap();
        assert_ne!(delta, 0.0);
        let (delta, mb) = step_size(&quant(0, 2, &[8], &[0]), 0, 0, 8, true).unwrap();
        assert_eq!((delta, mb), (0.0, 9));
    }

    /// An (r, orientation) pair naming a band that does not exist is an error.
    #[test]
    fn step_size_invalid_band() {
        let q = quant(0, 2, &[8, 7, 7, 6, 6, 6, 5], &[0; 7]);
        assert!(step_size(&q, 2, 0, 8, true).is_err(), "LL at resolution 2");
        assert!(step_size(&q, 0, 1, 8, true).is_err(), "HL at resolution 0");
        assert!(step_size(&q, 9, 1, 8, true).is_err(), "past the last band");
    }

    /// `sample_f` must not depend on the arch's float-to-int conversion.
    #[test]
    fn samples_f_equals_sample_f() {
        let mut next = rng(0x5eed_1234_abcd_ef01);
        let mut v: Vec<f32> = vec![
            0.0,
            -0.0,
            0.5,
            1.5,
            2.5,
            -0.5,
            -1.5,
            -2.5,
            127.5,
            128.5,
            255.5,
            256.5,
            65535.5,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            8388608.0,
            -8388608.0,
            8388609.0,
            16777216.0,
            2147483648.0,
            -2147483904.0,
        ];
        for _ in 0..20000 {
            let r = next();
            // Any bit pattern, and values near the integers where ties and ranges matter.
            v.push(f32::from_bits(r as u32));
            v.push(((r >> 32) % 140000) as f32 / 2.0 - 2000.0);
        }
        for prec in 1..=16 {
            let mut got = vec![0; v.len()];
            samples_f(&mut got, &v, prec);
            for (g, &x) in got.iter().zip(&v) {
                assert_eq!(*g, sample_f(x, prec), "samples_f({x}, {prec})");
            }
        }
    }

    #[test]
    fn sample_f_specials() {
        for prec in 1..=16 {
            let hi = (1i32 << prec) - 1;
            for (v, want) in [
                (f32::NAN, 0),
                (f32::from_bits(0xffc00001), 0),
                (f32::INFINITY, hi),
                (f32::NEG_INFINITY, 0),
            ] {
                assert_eq!(sample_f(v, prec), want, "sample_f({v}, {prec})");
            }
        }
        // Ties go to even, as lrintf does.
        assert_eq!(sample_f(0.5, 8), 128);
        assert_eq!(sample_f(1.5, 8), 130);
        assert_eq!(sample_f(-0.5, 8), 128);
    }

    #[test]
    fn samples_jj_rounding() {
        for (v, prec, want) in [
            (f32::from_bits(0x3effffff), 16, 32769), // 0.5-2^-25: +0.5f rounds to 1
            (-f32::from_bits(0x3effffff), 16, 32767),
            (0.5 - 1.0 / (1 << 24) as f32, 16, 32768),
            (0.5, 16, 32769),
            (-0.5, 16, 32767),
            (2.5, 16, 32771),
            (-2.5, 16, 32765),
            (-0.0, 16, 32768),
            (f32::NAN, 16, 32768),
            (32767.5, 16, 65535),
            (3e9, 16, 0), // Java's int add wraps
            (-3e9, 16, 0),
            (-1.3, 12, 2047),
            (2047.6, 12, 4095),
            (-2048.5, 12, 0),
        ] {
            let mut got = [0i32];
            samples_jj(&mut got, &[v], prec);
            assert_eq!(got[0], want, "samples_jj({v}, {prec})");
        }
    }

    /// Separate rounds the product and then the sum; Fused rounds once. They differ somewhere.
    #[test]
    fn separate_vs_fused_mul_add() {
        let mut next = rng(7);
        let (a, b, c) = (
            rand_floats(&mut next, 4096),
            rand_floats(&mut next, 4096),
            rand_floats(&mut next, 4096),
        );
        let mut differ = 0;
        for i in 0..a.len() {
            let p = a[i] * b[i];
            assert_eq!(Rounding::Separate.mul_add(a[i], b[i], c[i]), p + c[i]);
            assert_eq!(Rounding::JJ2000.mul_add(a[i], b[i], c[i]), p + c[i]);
            if Rounding::Fused.mul_add(a[i], b[i], c[i]) != p + c[i] {
                differ += 1;
            }
        }
        assert!(differ > 0, "Fused equals Separate on every sample");
    }

    /// The fused multiply-add rounds once: (1+2^-23)^2 - (1+2^-22) is exactly 2^-46, which the
    /// two-rounding form loses.
    #[test]
    fn fused_mul_add_is_single_rounding() {
        let a = 1.0f32 + f32::EPSILON;
        let c = -(1.0f32 + 2.0 * f32::EPSILON);
        assert_eq!(
            Rounding::Fused.mul_add(a, a, c),
            f32::EPSILON * f32::EPSILON
        );
        assert_eq!(Rounding::Separate.mul_add(a, a, c), 0.0);
    }

    /// The ICT with Separate rounding follows OpenJPEG's unfused opj_mct_decode_real order.
    #[test]
    fn separate_ict() {
        let mut next = rng(3);
        let n = 4096;
        let (y, u, v) = (
            rand_floats(&mut next, n),
            rand_floats(&mut next, n),
            rand_floats(&mut next, n),
        );
        let (mut c0, mut c1, mut c2) = (y.clone(), u.clone(), v.clone());
        let (mut f0, mut f1, mut f2) = (y.clone(), u.clone(), v.clone());
        inverse_ict(&mut c0, &mut c1, &mut c2, Rounding::Separate);
        inverse_ict(&mut f0, &mut f1, &mut f2, Rounding::Fused);
        let mut differ = 0;
        for i in 0..n {
            let g = y[i] - u[i] * ICT_CB_G;
            let want = [
                v[i] * ICT_CR_R + y[i],
                g - v[i] * ICT_CR_G,
                u[i] * ICT_CB_B + y[i],
            ];
            assert_eq!([c0[i], c1[i], c2[i]], want, "sample {i}");
            if want != [f0[i], f1[i], f2[i]] {
                differ += 1;
            }
        }
        assert!(differ > 0, "Separate ICT equals Fused on every sample");
    }

    #[test]
    fn rct_round_trip() {
        // The forward RCT: Y = (R + 2G + B) >> 2, U = B - G, V = R - G.
        let mut next = rng(5);
        let n = 1000;
        let (r, g, b): (Vec<i32>, Vec<i32>, Vec<i32>) = (
            (0..n).map(|_| (next() % 256) as i32 - 128).collect(),
            (0..n).map(|_| (next() % 256) as i32 - 128).collect(),
            (0..n).map(|_| (next() % 256) as i32 - 128).collect(),
        );
        let mut y: Vec<i32> = (0..n).map(|i| (r[i] + 2 * g[i] + b[i]) >> 2).collect();
        let mut u: Vec<i32> = (0..n).map(|i| b[i] - g[i]).collect();
        let mut v: Vec<i32> = (0..n).map(|i| r[i] - g[i]).collect();
        inverse_rct(&mut y, &mut u, &mut v);
        assert_eq!((y, u, v), (r, g, b));
    }

    #[test]
    fn rct_mismatched_lengths_do_not_panic() {
        inverse_rct(&mut [0; 2], &mut [0; 1], &mut [0; 1]);
    }

    #[test]
    fn clamp_and_shift() {
        assert_eq!(clamp_to_precision(300, 8, false), 255);
        assert_eq!(clamp_to_precision(-5, 8, false), 0);
        assert_eq!(clamp_to_precision(200, 8, true), 127);
        assert_eq!(clamp_to_precision(-200, 8, true), -128);
        assert_eq!(dc_level_shift(8, false), 128);
        assert_eq!(dc_level_shift(8, true), 0);
        let mut out = [0i32; 3];
        samples_r(&mut out, &[-200, 0, 200], 8, true);
        assert_eq!(out, [0, 128, 255]); // clamped to [-128, 127], biased by 128
    }

    /// A precision outside the range the formulas hold for neither hangs nor overflows.
    #[test]
    fn out_of_range_precision_is_clamped() {
        assert_eq!(expand16(1, 0), expand16(1, 1));
        assert_eq!(expand16(0xFFFF, 40), expand16(0xFFFF, 16));
        assert_eq!(dc_level_shift(0, false), 1);
        assert_eq!(dc_level_shift(-5, false), 1);
        assert_eq!(dc_level_shift(99, false), 1 << 30);
    }
}
