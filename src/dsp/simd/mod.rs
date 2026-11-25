//! The f32 kernels of the 9/7 path: one lifting step, a scale, the inverse ICT, the irreversible
//! dequantization of a row and the output packs. `scalar` is the reference; AVX2+FMA and AVX-512
//! (x86_64) and NEON (aarch64) sit beside it and are bit-exact with it: the same IEEE operations in
//! the same order per lane, a fused multiply-add only where `Rounding::Fused` has one. The tier is
//! picked once; `JP2K_SIMD=scalar` forces the scalar one, `JP2K_SIMD=avx2` caps it at AVX2.
//! The i32 kernels of the 5/3 path (the inverse RCT, the two lifting steps and the reversible
//! dequantization of a row) wrap as `wrapping_*` does; they have AVX2 versions (and AVX-512 for
//! the dequantization) and run scalar on aarch64.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

use super::{ICT_CB_B, ICT_CB_G, ICT_CR_G, ICT_CR_R, Rounding};
use crate::tier1::{T1_CHI, T1_PI, T1_SIG};
use std::sync::OnceLock;

/// Proof that `available` detected a SIMD tier's CPU features: its field is private to this
/// module, so only code here can make one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Proof(());

/// An implementation of the kernels. A SIMD variant exists only where `available` found its
/// CPU features, which is what makes the `unsafe` calls to its kernels sound: no other module
/// can build one, as it cannot build its `Proof`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2(Proof),
    /// AVX-512 F and BW, with AVX2 and FMA (which its kernels' tails use).
    #[cfg(target_arch = "x86_64")]
    Avx512(Proof),
    #[cfg(target_arch = "aarch64")]
    Neon(Proof),
}

/// Every tier this CPU can run, scalar first. Detected once.
pub(crate) fn available() -> &'static [Tier] {
    static ALL: OnceLock<Vec<Tier>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut v = vec![Tier::Scalar];
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            v.push(Tier::Avx2(Proof(())));
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                v.push(Tier::Avx512(Proof(())));
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            v.push(Tier::Neon(Proof(())));
        }
        v
    })
}

/// The tier the kernels run: the best available; scalar under `JP2K_SIMD=scalar`, and the best
/// below AVX-512 under `JP2K_SIMD=avx2`. Resolved once.
pub(crate) fn tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(|| {
        let env = std::env::var_os("JP2K_SIMD");
        if env.as_ref().is_some_and(|v| v == "scalar") {
            return Tier::Scalar;
        }
        let all = available();
        #[cfg(target_arch = "x86_64")]
        if env.is_some_and(|v| v == "avx2") {
            let below = all.iter().copied().rfind(|t| !matches!(t, Tier::Avx512(_)));
            return below.unwrap_or(Tier::Scalar);
        }
        all.last().copied().unwrap_or(Tier::Scalar)
    })
}

/// `d[i] = c * (u[i] + v[i]) + d[i]`, rounded as `r` says (`Fused`: once). `u` and `v` are at
/// least as long as `d`.
pub(crate) fn lift(d: &mut [f32], u: &[f32], v: &[f32], c: f32, r: Rounding) {
    lift_with(tier(), d, u, v, c, r)
}

/// `dst[i] = c * (u[i] + v[i]) + d[i]`, rounded as `r` says: `lift` with a destination of its own.
/// `d`, `u` and `v` are at least as long as `dst`.
pub(crate) fn lift_to(dst: &mut [f32], d: &[f32], u: &[f32], v: &[f32], c: f32, r: Rounding) {
    lift_to_with(tier(), dst, d, u, v, c, r)
}

/// `dst[i] = src[i] * k`; `src` is at least as long as `dst`.
pub(crate) fn scale_to(dst: &mut [f32], src: &[f32], k: f32) {
    scale_to_with(tier(), dst, src, k)
}

/// `x[i] *= k`.
pub(crate) fn scale(x: &mut [f32], k: f32) {
    scale_with(tier(), x, k)
}

/// The inverse ICT of three equal-length planes, in place.
pub(crate) fn ict(c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], r: Rounding) {
    ict_with(tier(), c0, c1, c2, r)
}

/// The irreversible dequantization of a row: `mag` as f32 plus the bias `fb[PI bit]`, times `d`,
/// negated where the sign bit is set, and 0 where the sample is not significant. `flags` are the
/// tier-1 flag words, shifted right by `sh` (< 32) to bring the row's own bits to `T1_SIG`,
/// `T1_CHI` and `T1_PI`.
pub(crate) fn dequant(dst: &mut [f32], mag: &[i32], flags: &[u32], sh: u32, fb: [f32; 2], d: f32) {
    dequant_with(tier(), dst, mag, flags, sh, fb, d)
}

/// `dst[i] = samples_f(src[i], precision) as u16`, fused: `src` is at least as long as `dst`.
pub(crate) fn pack_u16(dst: &mut [u16], src: &[f32], precision: i32) {
    pack_u16_with(tier(), dst, src, precision)
}

/// `dst[4 * i + ch] = samples_f(src[i], precision) as u8`, fused, the other bytes untouched: `dst`
/// holds at least `4 * src.len()` bytes and `ch` is below 4.
pub(crate) fn pack_u8x4(dst: &mut [u8], src: &[f32], precision: i32, ch: usize) {
    pack_u8x4_with(tier(), dst, src, precision, ch)
}

/// T.800 G.2's inverse RCT of three planes, in place, over the shortest of the three (so
/// mismatched lengths cannot panic), wrapping.
pub(crate) fn rct(c0: &mut [i32], c1: &mut [i32], c2: &mut [i32]) {
    rct_with(tier(), c0, c1, c2)
}

/// The 5/3 low-band lifting step: `d[i] -= (u[i] + v[i] + 2) >> 2`, wrapping. `u` and `v` are at
/// least as long as `d`.
pub(crate) fn lift53_sub(d: &mut [i32], u: &[i32], v: &[i32]) {
    lift53_sub_with(tier(), d, u, v)
}

/// The 5/3 high-band lifting step: `d[i] += (u[i] + v[i]) >> 1`, wrapping. `u` and `v` are at
/// least as long as `d`.
pub(crate) fn lift53_add(d: &mut [i32], u: &[i32], v: &[i32]) {
    lift53_add_with(tier(), d, u, v)
}

/// The reversible dequantization of a row, in place: `o` plus the bias `rb[PI bit]`, negated
/// where the sign bit is set, and 0 where the sample is not significant, wrapping. `flags` (at
/// least as long as `o`) are the tier-1 flag words, shifted right by `sh` (< 32) as in `dequant`.
pub(crate) fn dequant_rev(o: &mut [i32], flags: &[u32], sh: u32, rb: [i32; 2]) {
    dequant_rev_with(tier(), o, flags, sh, rb)
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn lift_with(t: Tier, d: &mut [f32], u: &[f32], v: &[f32], c: f32, r: Rounding) {
    let n = d.len();
    let (u, v) = (&u[..n], &v[..n]);
    match t {
        Tier::Scalar => scalar::lift(d, u, v, c, r),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::lift(d, u, v, c, r == Rounding::Fused) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => {
            // SAFETY: `Tier::Neon` holds a `Proof`, built only in `available()` after
            // `is_aarch64_feature_detected!("neon")`.
            unsafe { neon::lift(d, u, v, c, r == Rounding::Fused) }
        }
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn scale_with(t: Tier, x: &mut [f32], k: f32) {
    match t {
        Tier::Scalar => scalar::scale(x, k),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::scale(x, k) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => {
            // SAFETY: `Tier::Neon` holds a `Proof`, built only in `available()` after
            // `is_aarch64_feature_detected!("neon")`.
            unsafe { neon::scale(x, k) }
        }
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn ict_with(t: Tier, c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], r: Rounding) {
    let n = c0.len();
    let (c1, c2) = (&mut c1[..n], &mut c2[..n]);
    match t {
        Tier::Scalar => scalar::ict(c0, c1, c2, r),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::ict(c0, c1, c2, r == Rounding::Fused) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => {
            // SAFETY: `Tier::Neon` holds a `Proof`, built only in `available()` after
            // `is_aarch64_feature_detected!("neon")`.
            unsafe { neon::ict(c0, c1, c2, r == Rounding::Fused) }
        }
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn dequant_with(
    t: Tier,
    dst: &mut [f32],
    mag: &[i32],
    flags: &[u32],
    sh: u32,
    fb: [f32; 2],
    d: f32,
) {
    let n = dst.len();
    let (mag, flags) = (&mag[..n], &flags[..n]);
    match t {
        Tier::Scalar => scalar::dequant(dst, mag, flags, sh, fb, d),
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features.
            unsafe { avx2::dequant(dst, mag, flags, sh, fb, d) }
        }
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512(_) => {
            // SAFETY: `Tier::Avx512` holds a `Proof`, built only in `available()` after detecting
            // avx2, fma, avx512f and avx512bw.
            unsafe { avx512::dequant(dst, mag, flags, sh, fb, d) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => {
            // SAFETY: `Tier::Neon` holds a `Proof`, built only in `available()` after
            // `is_aarch64_feature_detected!("neon")`.
            unsafe { neon::dequant(dst, mag, flags, sh, fb, d) }
        }
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn lift_to_with(
    t: Tier,
    dst: &mut [f32],
    d: &[f32],
    u: &[f32],
    v: &[f32],
    c: f32,
    r: Rounding,
) {
    let n = dst.len();
    let (d, u, v) = (&d[..n], &u[..n], &v[..n]);
    match t {
        Tier::Scalar => scalar::lift_to(dst, d, u, v, c, r),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::lift_to(dst, d, u, v, c, r == Rounding::Fused) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => {
            // SAFETY: `Tier::Neon` holds a `Proof`, built only in `available()` after
            // `is_aarch64_feature_detected!("neon")`.
            unsafe { neon::lift_to(dst, d, u, v, c, r == Rounding::Fused) }
        }
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn scale_to_with(t: Tier, dst: &mut [f32], src: &[f32], k: f32) {
    let src = &src[..dst.len()];
    match t {
        Tier::Scalar => scalar::scale_to(dst, src, k),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::scale_to(dst, src, k) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => {
            // SAFETY: `Tier::Neon` holds a `Proof`, built only in `available()` after
            // `is_aarch64_feature_detected!("neon")`.
            unsafe { neon::scale_to(dst, src, k) }
        }
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn pack_u16_with(t: Tier, dst: &mut [u16], src: &[f32], precision: i32) {
    let src = &src[..dst.len()];
    // The SIMD kernels pack with saturation, which is the same as `as u16` only up to 16 bits.
    if !(1..=16).contains(&precision) {
        return scalar::pack_u16(dst, src, precision);
    }
    match t {
        Tier::Scalar => scalar::pack_u16(dst, src, precision),
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features.
            unsafe { avx2::pack_u16(dst, src, precision) }
        }
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512(_) => {
            // SAFETY: `Tier::Avx512` holds a `Proof`, built only in `available()` after detecting
            // avx2, fma, avx512f and avx512bw.
            unsafe { avx512::pack_u16(dst, src, precision) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => {
            // SAFETY: `Tier::Neon` holds a `Proof`, built only in `available()` after
            // `is_aarch64_feature_detected!("neon")`.
            unsafe { neon::pack_u16(dst, src, precision) }
        }
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn pack_u8x4_with(t: Tier, dst: &mut [u8], src: &[f32], precision: i32, ch: usize) {
    let dst = &mut dst[..4 * src.len()];
    // As `pack_u16_with`: the SIMD kernels are `as u8` of the samples only up to 16 bits.
    if !(1..=16).contains(&precision) {
        return scalar::pack_u8x4(dst, src, precision, ch);
    }
    match t {
        Tier::Scalar => scalar::pack_u8x4(dst, src, precision, ch),
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features.
            unsafe { avx2::pack_u8x4(dst, src, precision, ch) }
        }
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512(_) => {
            // SAFETY: `Tier::Avx512` holds a `Proof`, built only in `available()` after detecting
            // avx2, fma, avx512f and avx512bw.
            unsafe { avx512::pack_u8x4(dst, src, precision, ch) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => scalar::pack_u8x4(dst, src, precision, ch),
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn rct_with(t: Tier, c0: &mut [i32], c1: &mut [i32], c2: &mut [i32]) {
    let n = c0.len().min(c1.len()).min(c2.len());
    let (c0, c1, c2) = (&mut c0[..n], &mut c1[..n], &mut c2[..n]);
    match t {
        Tier::Scalar => scalar::rct(c0, c1, c2),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::rct(c0, c1, c2) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => scalar::rct(c0, c1, c2),
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn lift53_sub_with(t: Tier, d: &mut [i32], u: &[i32], v: &[i32]) {
    let n = d.len();
    let (u, v) = (&u[..n], &v[..n]);
    match t {
        Tier::Scalar => scalar::lift53_sub(d, u, v),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::lift53_sub(d, u, v) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => scalar::lift53_sub(d, u, v),
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn lift53_add_with(t: Tier, d: &mut [i32], u: &[i32], v: &[i32]) {
    let n = d.len();
    let (u, v) = (&u[..n], &v[..n]);
    match t {
        Tier::Scalar => scalar::lift53_add(d, u, v),
        // AVX-512 runs the AVX2 kernel: its own lost the measurement (docs/safety.md).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) | Tier::Avx512(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features, and both
            // of these include avx2 and fma.
            unsafe { avx2::lift53_add(d, u, v) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => scalar::lift53_add(d, u, v),
    }
}

#[allow(unsafe_code)] // calls the `#[target_feature]` kernels
pub(crate) fn dequant_rev_with(t: Tier, o: &mut [i32], flags: &[u32], sh: u32, rb: [i32; 2]) {
    let flags = &flags[..o.len()];
    match t {
        Tier::Scalar => scalar::dequant_rev(o, flags, sh, rb),
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) => {
            // SAFETY: a SIMD `Tier` exists only if `available` detected its features.
            unsafe { avx2::dequant_rev(o, flags, sh, rb) }
        }
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512(_) => {
            // SAFETY: `Tier::Avx512` holds a `Proof`, built only in `available()` after detecting
            // avx2, fma, avx512f and avx512bw.
            unsafe { avx512::dequant_rev(o, flags, sh, rb) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => scalar::dequant_rev(o, flags, sh, rb),
    }
}

/// Hooks for the differential fuzz target (`tests/fuzz/`), not part of the API: every tier this CPU can
/// run, by index into [`available`] (0 is scalar; an index past the end runs scalar), so a caller
/// cannot ask for a tier whose CPU features were not detected.
#[doc(hidden)]
pub mod fuzz {
    use super::*;

    /// The names of the available tiers, scalar first.
    pub fn tiers() -> Vec<&'static str> {
        available()
            .iter()
            .map(|t| match t {
                Tier::Scalar => "scalar",
                #[cfg(target_arch = "x86_64")]
                Tier::Avx2(_) => "avx2",
                #[cfg(target_arch = "x86_64")]
                Tier::Avx512(_) => "avx512",
                #[cfg(target_arch = "aarch64")]
                Tier::Neon(_) => "neon",
            })
            .collect()
    }

    fn nth(i: usize) -> Tier {
        available().get(i).copied().unwrap_or(Tier::Scalar)
    }

    /// `d[i] = c * (u[i] + v[i]) + d[i]`; `u` and `v` are at least as long as `d`.
    pub fn lift(tier: usize, d: &mut [f32], u: &[f32], v: &[f32], c: f32, r: Rounding) {
        if u.len() >= d.len() && v.len() >= d.len() {
            lift_with(nth(tier), d, u, v, c, r)
        }
    }

    pub fn scale(tier: usize, x: &mut [f32], k: f32) {
        scale_with(nth(tier), x, k)
    }

    /// The three planes are equal in length.
    pub fn ict(tier: usize, c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], r: Rounding) {
        if c1.len() == c0.len() && c2.len() == c0.len() {
            ict_with(nth(tier), c0, c1, c2, r)
        }
    }

    /// `d`, `u` and `v` are at least as long as `dst`.
    pub fn lift_to(
        tier: usize,
        dst: &mut [f32],
        d: &[f32],
        u: &[f32],
        v: &[f32],
        c: f32,
        r: Rounding,
    ) {
        if d.len() >= dst.len() && u.len() >= dst.len() && v.len() >= dst.len() {
            lift_to_with(nth(tier), dst, d, u, v, c, r)
        }
    }

    /// `src` is at least as long as `dst`.
    pub fn scale_to(tier: usize, dst: &mut [f32], src: &[f32], k: f32) {
        if src.len() >= dst.len() {
            scale_to_with(nth(tier), dst, src, k)
        }
    }

    /// `src` is at least as long as `dst`; `precision` is in [1, 30].
    pub fn pack_u16(tier: usize, dst: &mut [u16], src: &[f32], precision: i32) {
        if src.len() >= dst.len() && (1..=30).contains(&precision) {
            pack_u16_with(nth(tier), dst, src, precision)
        }
    }

    /// `dst` holds at least `4 * src.len()` bytes; `precision` is in [1, 30]; `ch` is below 4.
    pub fn pack_u8x4(tier: usize, dst: &mut [u8], src: &[f32], precision: i32, ch: usize) {
        if dst.len() / 4 >= src.len() && (1..=30).contains(&precision) && ch < 4 {
            pack_u8x4_with(nth(tier), dst, src, precision, ch)
        }
    }

    /// Over the shortest of the three planes.
    pub fn rct(tier: usize, c0: &mut [i32], c1: &mut [i32], c2: &mut [i32]) {
        rct_with(nth(tier), c0, c1, c2)
    }

    /// `u` and `v` are at least as long as `d`.
    pub fn lift53_sub(tier: usize, d: &mut [i32], u: &[i32], v: &[i32]) {
        if u.len() >= d.len() && v.len() >= d.len() {
            lift53_sub_with(nth(tier), d, u, v)
        }
    }

    /// `u` and `v` are at least as long as `d`.
    pub fn lift53_add(tier: usize, d: &mut [i32], u: &[i32], v: &[i32]) {
        if u.len() >= d.len() && v.len() >= d.len() {
            lift53_add_with(nth(tier), d, u, v)
        }
    }

    /// `flags` is at least as long as `o`; `sh` is below 32.
    pub fn dequant_rev(tier: usize, o: &mut [i32], flags: &[u32], sh: u32, rb: [i32; 2]) {
        if flags.len() >= o.len() && sh < 32 {
            dequant_rev_with(nth(tier), o, flags, sh, rb)
        }
    }

    /// `mag` and `flags` are at least as long as `dst`; `sh` is below 32.
    pub fn dequant(
        tier: usize,
        dst: &mut [f32],
        mag: &[i32],
        flags: &[u32],
        sh: u32,
        fb: [f32; 2],
        d: f32,
    ) {
        if mag.len() >= dst.len() && flags.len() >= dst.len() && sh < 32 {
            dequant_with(nth(tier), dst, mag, flags, sh, fb, d)
        }
    }
}

// Where the flag bits sit, as left shifts that bring each to bit 31.
const SIG_UP: i32 = 31 - T1_SIG.trailing_zeros() as i32;
const PI_UP: i32 = 31 - T1_PI.trailing_zeros() as i32;
const CHI_UP: i32 = 31 - T1_CHI as i32;

/// The reference kernels, and the tail loops of the SIMD ones. Lengths are equal.
pub(super) mod scalar {
    use super::*;

    pub fn lift(d: &mut [f32], u: &[f32], v: &[f32], c: f32, r: Rounding) {
        for ((d, &u), &v) in d.iter_mut().zip(u).zip(v) {
            *d = r.mul_add(c, u + v, *d);
        }
    }

    pub fn scale(x: &mut [f32], k: f32) {
        for x in x {
            *x *= k;
        }
    }

    pub fn lift_to(dst: &mut [f32], d: &[f32], u: &[f32], v: &[f32], c: f32, r: Rounding) {
        for (((o, &d), &u), &v) in dst.iter_mut().zip(d).zip(u).zip(v) {
            *o = r.mul_add(c, u + v, d);
        }
    }

    pub fn scale_to(dst: &mut [f32], src: &[f32], k: f32) {
        for (o, &x) in dst.iter_mut().zip(src) {
            *o = x * k;
        }
    }

    pub fn ict(c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], r: Rounding) {
        for ((c0, c1), c2) in c0.iter_mut().zip(c1).zip(c2) {
            let (y, u, v) = (*c0, *c1, *c2);
            *c0 = r.mul_add(v, ICT_CR_R, y);
            *c1 = r.mul_add(-v, ICT_CR_G, r.mul_add(-u, ICT_CB_G, y));
            *c2 = r.mul_add(u, ICT_CB_B, y);
        }
    }

    /// `dsp::samples_f`'s arithmetic, stored as u16.
    pub fn pack_u16(dst: &mut [u16], src: &[f32], precision: i32) {
        const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23
        let shift = 1i32 << (precision - 1);
        let lo = -(shift as f32);
        let hi = ((1i32 << precision) - 1 - shift) as f32;
        for (d, &v) in dst.iter_mut().zip(src) {
            let c = if v > lo { v } else { lo };
            let c = if c < hi { c } else { hi };
            *d = (((c + MAGIC) - MAGIC) as i32 + shift) as u16;
        }
    }

    /// `pack_u16`'s arithmetic, stored `as u8` into byte `ch` of each 4-byte pixel: what
    /// `dsp::samples_f` and then the decoder's 8-bit `put_chan` write.
    pub fn pack_u8x4(dst: &mut [u8], src: &[f32], precision: i32, ch: usize) {
        const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23
        let shift = 1i32 << (precision - 1);
        let lo = -(shift as f32);
        let hi = ((1i32 << precision) - 1 - shift) as f32;
        for (px, &v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
            let c = if v > lo { v } else { lo };
            let c = if c < hi { c } else { hi };
            px[ch] = (((c + MAGIC) - MAGIC) as i32 + shift) as u8;
        }
    }

    /// `dsp::inverse_rct`'s loop, as it was.
    pub fn rct(c0: &mut [i32], c1: &mut [i32], c2: &mut [i32]) {
        for ((c0, c1), c2) in c0.iter_mut().zip(c1.iter_mut()).zip(c2.iter_mut()) {
            let (y, u, v) = (*c0, *c1, *c2);
            let g = y.wrapping_sub(u.wrapping_add(v) >> 2);
            *c0 = v.wrapping_add(g);
            *c1 = g;
            *c2 = u.wrapping_add(g);
        }
    }

    /// The first step of `dwt::inverse_1d_53`, on the split bands.
    pub fn lift53_sub(d: &mut [i32], u: &[i32], v: &[i32]) {
        for ((d, &u), &v) in d.iter_mut().zip(u).zip(v) {
            *d = d.wrapping_sub(u.wrapping_add(v).wrapping_add(2) >> 2);
        }
    }

    /// The second step of `dwt::inverse_1d_53`, on the split bands.
    pub fn lift53_add(d: &mut [i32], u: &[i32], v: &[i32]) {
        for ((d, &u), &v) in d.iter_mut().zip(u).zip(v) {
            *d = d.wrapping_add(u.wrapping_add(v) >> 1);
        }
    }

    /// The reversible write-out of `tier1::State`'s row loop, as it was: bit 4 of the shifted
    /// flags is `T1_SIG`, bit 21 `T1_PI`.
    pub fn dequant_rev(o: &mut [i32], flags: &[u32], sh: u32, rb: [i32; 2]) {
        for (o, &f) in o.iter_mut().zip(flags) {
            let f = f >> sh;
            let m = (f >> T1_CHI & 1).wrapping_neg() as i32;
            let v = (o.wrapping_add(rb[(f >> 21 & 1) as usize]) ^ m).wrapping_sub(m);
            *o = v & ((f >> 4 & 1) as i32).wrapping_neg();
        }
    }

    pub fn dequant(dst: &mut [f32], mag: &[i32], flags: &[u32], sh: u32, fb: [f32; 2], d: f32) {
        for ((o, &m), &f) in dst.iter_mut().zip(mag).zip(flags) {
            let f = f >> sh;
            let v = (m as f32 + fb[usize::from(f & T1_PI != 0)]) * d;
            let sig = if f & T1_SIG != 0 { !0 } else { 0 };
            *o = f32::from_bits((v.to_bits() ^ (f >> T1_CHI & 1) << 31) & sig);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Finite f32 over many magnitudes, with some zeros of both signs.
        fn f32(&mut self) -> f32 {
            let r = self.next();
            match r % 16 {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    ((r >> 8) as u32 as f32 / u32::MAX as f32 - 0.5)
                        * 10f32.powi((r >> 4 & 7) as i32 - 2)
                }
            }
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f32()).collect()
        }
    }

    // 24, 25 and 47 to 49: AVX-512 groups of 16 with an AVX2 group of 8 and a scalar tail after.
    const LENS: [usize; 22] = [
        0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 100,
    ];
    const ROUNDINGS: [Rounding; 2] = [Rounding::Fused, Rounding::Separate];

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn tier_is_detected() {
        #[cfg(target_arch = "x86_64")]
        assert!(
            available().contains(&Tier::Avx2(Proof(()))),
            "AVX2 and FMA expected on x86_64"
        );
        #[cfg(target_arch = "aarch64")]
        assert!(
            available().contains(&Tier::Neon(Proof(()))),
            "NEON expected on aarch64"
        );
        let env = std::env::var_os("JP2K_SIMD");
        let capped = cfg!(target_arch = "x86_64") && env.as_ref().is_some_and(|v| v == "avx2");
        if env.is_some_and(|v| v == "scalar") {
            assert_eq!(tier(), Tier::Scalar);
        } else if capped {
            // The best tier below AVX-512: AVX2, second after scalar (asserted above).
            assert_eq!(Some(&tier()), available().get(1));
        } else {
            assert_eq!(Some(&tier()), available().last());
        }
    }

    /// Whether this CPU has AVX-512 is logged (`--nocapture`), and with `JP2K_EXPECT_AVX512=1` it
    /// is required: the AVX-512 kernels are otherwise left out of every test without a word.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_where_expected() {
        let have = available().contains(&Tier::Avx512(Proof(())));
        eprintln!(
            "AVX-512 F and BW detected: {have}; tiers {:?}",
            fuzz::tiers()
        );
        if std::env::var("JP2K_EXPECT_AVX512").as_deref() == Ok("1") {
            assert!(have, "JP2K_EXPECT_AVX512=1, but this CPU has no AVX-512");
        }
    }

    #[test]
    fn lift_matches_scalar() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for &t in available() {
            for r in ROUNDINGS {
                for rep in 0..40 {
                    for &n in &LENS {
                        // Unaligned starts: slices at offset 0..3 of larger buffers.
                        let off = rep % 4;
                        let (d, u, v) = (rng.vec(n + off), rng.vec(n + off), rng.vec(n + off));
                        let c = rng.f32();
                        let mut want = d.clone();
                        scalar::lift(&mut want[off..], &u[off..], &v[off..], c, r);
                        let mut got = d.clone();
                        lift_with(t, &mut got[off..], &u[off..], &v[off..], c, r);
                        assert_eq!(bits(&got), bits(&want), "{t:?} {r:?} n {n} off {off}");
                    }
                }
            }
        }
    }

    #[test]
    fn scale_matches_scalar() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for &t in available() {
            for rep in 0..40 {
                for &n in &LENS {
                    let off = rep % 4;
                    let x = rng.vec(n + off);
                    let k = rng.f32();
                    let mut want = x.clone();
                    scalar::scale(&mut want[off..], k);
                    let mut got = x.clone();
                    scale_with(t, &mut got[off..], k);
                    assert_eq!(bits(&got), bits(&want), "{t:?} n {n} off {off}");
                }
            }
        }
    }

    #[test]
    fn ict_matches_scalar() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for &t in available() {
            for r in ROUNDINGS {
                for rep in 0..40 {
                    for &n in &LENS {
                        let off = rep % 4;
                        let p = [rng.vec(n + off), rng.vec(n + off), rng.vec(n + off)];
                        let mut want = p.clone();
                        let [a, b, c] = &mut want;
                        scalar::ict(&mut a[off..], &mut b[off..], &mut c[off..], r);
                        let mut got = p.clone();
                        let [a, b, c] = &mut got;
                        ict_with(t, &mut a[off..], &mut b[off..], &mut c[off..], r);
                        for k in 0..3 {
                            assert_eq!(
                                bits(&got[k]),
                                bits(&want[k]),
                                "{t:?} {r:?} n {n} off {off} plane {k}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dequant_matches_scalar() {
        let mut rng = Rng(0x0bad_5eed_0bad_5eed);
        for &t in available() {
            for rep in 0..40 {
                for &n in &LENS {
                    let off = rep % 4;
                    let mag: Vec<i32> = (0..n + off)
                        .map(|_| rng.next() as i32 >> (rng.next() % 24))
                        .collect();
                    let flags: Vec<u32> = (0..n + off).map(|_| rng.next() as u32).collect();
                    let fb = [0.5, (1u64 << (rng.next() % 20)) as f32 * 0.5];
                    let d = rng.f32().abs() + 0.001;
                    let sh = 3 * (rng.next() % 4) as u32;
                    let mut want = vec![0f32; n];
                    scalar::dequant(&mut want, &mag[off..], &flags[off..], sh, fb, d);
                    // The destination starts at an offset too: the store must not need alignment.
                    let mut got = vec![1f32; n + off];
                    dequant_with(t, &mut got[off..], &mag[off..], &flags[off..], sh, fb, d);
                    assert_eq!(bits(&got[off..]), bits(&want), "{t:?} n {n} off {off}");
                }
            }
        }
    }

    #[test]
    fn pack_u16_matches_scalar() {
        let mut rng = Rng(0x5eed_f00d_1234_abcd);
        for &t in available() {
            for prec in [1, 8, 9, 12, 15, 16, 17, 24] {
                for rep in 0..20 {
                    for &n in &LENS {
                        let off = rep % 4;
                        // Spread over, below and above the sample range, with ties and NaN.
                        let top = (1u32 << prec.min(20)) as f32;
                        let src: Vec<f32> = (0..n + off)
                            .map(|_| match rng.next() % 8 {
                                0 => f32::NAN,
                                1 => (rng.next() % 64) as f32 * 0.5 - top / 2.0,
                                _ => (rng.f32() * top).clamp(-4.0 * top, 4.0 * top),
                            })
                            .collect();
                        pack_case(t, prec, &src[off..], off);
                    }
                    // Infinities and exact ties, at the same unaligned starts.
                    let mut src = rng.vec(rep % 4);
                    src.extend([f32::INFINITY, f32::NEG_INFINITY, 0.5, 1.5, -0.5, -2.5]);
                    pack_case(t, prec, &src[rep % 4..], rep % 4);
                }
            }
        }
    }

    fn pack_case(t: Tier, prec: i32, src: &[f32], off: usize) {
        let n = src.len();
        let mut want = vec![0u16; n];
        scalar::pack_u16(&mut want, src, prec);
        // The destination starts at an offset too.
        let mut got = vec![7u16; n + off];
        pack_u16_with(t, &mut got[off..], src, prec);
        assert_eq!(&got[off..], &want[..], "{t:?} prec {prec} n {n} off {off}");
        // And it is `samples_f`, narrowed.
        let mut wide = vec![0i32; n];
        crate::dsp::samples_f(&mut wide, src, prec);
        let narrow: Vec<u16> = wide.iter().map(|&v| v as u16).collect();
        assert_eq!(want, narrow, "prec {prec} n {n}");
    }

    /// `pack_u8x4` equals scalar for each channel and leaves the other three bytes of every pixel
    /// (and the bytes after the pixels) as they were; scalar is `samples_f` then `as u8`.
    #[test]
    fn pack_u8x4_matches_scalar() {
        let mut rng = Rng(0x0dd5_eed5_8888_4444);
        for &t in available() {
            for prec in [1, 7, 8, 9, 12, 16, 17, 24] {
                for rep in 0..10 {
                    for &n in &LENS {
                        let off = rep % 4;
                        let top = (1u32 << prec.min(20)) as f32;
                        let src: Vec<f32> = (0..n + off)
                            .map(|_| match rng.next() % 8 {
                                0 => f32::NAN,
                                1 => (rng.next() % 64) as f32 * 0.5 - top / 2.0,
                                2 => [f32::INFINITY, f32::NEG_INFINITY, 0.5, -2.5][rep % 4],
                                _ => (rng.f32() * top).clamp(-4.0 * top, 4.0 * top),
                            })
                            .collect();
                        let src = &src[off..];
                        let mut wide = vec![0i32; n];
                        crate::dsp::samples_f(&mut wide, src, prec);
                        // The destination starts at an offset and has bytes after the pixels.
                        let before: Vec<u8> =
                            (0..4 * n + off + 5).map(|_| rng.next() as u8).collect();
                        for ch in 0..4 {
                            let mut want = before.clone();
                            scalar::pack_u8x4(&mut want[off..off + 4 * n], src, prec, ch);
                            for (i, (w, b)) in want.iter().zip(&before).enumerate() {
                                let k = i.wrapping_sub(off);
                                if k < 4 * n && k % 4 == ch {
                                    assert_eq!(*w, wide[k / 4] as u8, "prec {prec} n {n} ch {ch}");
                                } else {
                                    assert_eq!(w, b, "prec {prec} n {n} ch {ch} byte {i}");
                                }
                            }
                            let mut got = before.clone();
                            pack_u8x4_with(t, &mut got[off..], src, prec, ch);
                            assert_eq!(got, want, "{t:?} prec {prec} n {n} off {off} ch {ch}");
                        }
                    }
                }
            }
        }
    }

    /// i32 samples over every magnitude, with the extremes, so the wrapping is exercised.
    fn ints(rng: &mut Rng, n: usize) -> Vec<i32> {
        (0..n)
            .map(|_| match rng.next() % 8 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => -1,
                _ => rng.next() as i32 >> (rng.next() % 32),
            })
            .collect()
    }

    #[test]
    fn integer_kernels_match_scalar() {
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        for &t in available() {
            for rep in 0..40 {
                for &n in &LENS {
                    let off = rep % 4;
                    let p = [
                        ints(&mut rng, n + off),
                        ints(&mut rng, n + off),
                        ints(&mut rng, n + off),
                    ];
                    // The RCT runs over the shortest plane; the others keep their tails.
                    let cut = [n + off, n + off - (rep % 2).min(n), n + off];
                    let mut want = p.clone();
                    let [a, b, c] = &mut want;
                    scalar::rct(
                        &mut a[off..cut[0]],
                        &mut b[off..cut[1]],
                        &mut c[off..cut[2]],
                    );
                    let mut got = p.clone();
                    let [a, b, c] = &mut got;
                    rct_with(
                        t,
                        &mut a[off..cut[0]],
                        &mut b[off..cut[1]],
                        &mut c[off..cut[2]],
                    );
                    assert_eq!(got, want, "rct {t:?} n {n} off {off}");
                    let [d, u, v] = &p;
                    let mut want = d.clone();
                    scalar::lift53_sub(&mut want[off..], &u[off..], &v[off..]);
                    let mut got = d.clone();
                    lift53_sub_with(t, &mut got[off..], &u[off..], &v[off..]);
                    assert_eq!(got, want, "lift53_sub {t:?} n {n} off {off}");
                    let mut want = d.clone();
                    scalar::lift53_add(&mut want[off..], &u[off..], &v[off..]);
                    let mut got = d.clone();
                    lift53_add_with(t, &mut got[off..], &u[off..], &v[off..]);
                    assert_eq!(got, want, "lift53_add {t:?} n {n} off {off}");
                    let flags: Vec<u32> = (0..n + off).map(|_| rng.next() as u32).collect();
                    let rb = [ints(&mut rng, 1)[0], ints(&mut rng, 1)[0]];
                    let sh = 3 * (rng.next() % 4) as u32;
                    let mut want = d.clone();
                    scalar::dequant_rev(&mut want[off..], &flags[off..], sh, rb);
                    let mut got = d.clone();
                    dequant_rev_with(t, &mut got[off..], &flags[off..], sh, rb);
                    assert_eq!(got, want, "dequant_rev {t:?} n {n} off {off}");
                }
            }
        }
    }

    #[test]
    fn lift_to_and_scale_to_match_scalar() {
        let mut rng = Rng(0x7a11_0ca7_e5ca_1e55);
        for &t in available() {
            for rep in 0..40 {
                for &n in &LENS {
                    let off = rep % 4;
                    let (d, u, v) = (rng.vec(n + off), rng.vec(n + off), rng.vec(n + off));
                    let c = rng.f32();
                    for r in ROUNDINGS {
                        // `lift_to` is `lift` into a destination of its own.
                        let mut want = d.clone();
                        scalar::lift(&mut want[off..], &u[off..], &v[off..], c, r);
                        let mut got = vec![9f32; n + off];
                        lift_to_with(t, &mut got[off..], &d[off..], &u[off..], &v[off..], c, r);
                        assert_eq!(bits(&got[off..]), bits(&want[off..]), "{t:?} {r:?} n {n}");
                    }
                    let mut want = d.clone();
                    scalar::scale(&mut want[off..], c);
                    let mut got = vec![9f32; n + off];
                    scale_to_with(t, &mut got[off..], &d[off..], c);
                    assert_eq!(
                        bits(&got[off..]),
                        bits(&want[off..]),
                        "{t:?} scale_to n {n}"
                    );
                }
            }
        }
    }
}
