//! AVX-512 (F and BW) kernels, 16 lanes: the ones that beat AVX2 in `examples/kernels.rs`
//! (docs/safety.md lists the ones dropped, for which the AVX-512 tier runs the AVX2 kernel). Each
//! runs whole groups of 16 and leaves the tail (under 16) to the AVX2 kernel, which runs at most
//! one group of 8 and then its scalar tail: no masked load or store. Callers (`super`) run them
//! only for `Tier::Avx512`, which exists only if `avx2`, `fma`, `avx512f` and `avx512bw` were
//! detected; AVX2 and FMA are enabled here too, so the tail calls are safe. Every lane does the
//! same IEEE operations in the same order as the scalar kernel, with the default MXCSR rounding,
//! and no AVX512DQ instruction is used. Slice lengths are equal.

use super::{T1_CHI, T1_PI, T1_SIG, avx2};
use std::arch::x86_64::*;

// The only `unsafe` in this tier: unaligned loads and stores of whole arrays.
#[allow(unsafe_code)]
mod mem {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub(super) fn ld(p: &[f32; 16]) -> __m512 {
        // SAFETY: `p` is 64 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm512_loadu_ps(p.as_ptr()) }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) fn st(p: &mut [f32; 16], v: __m512) {
        // SAFETY: `p` is 64 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm512_storeu_ps(p.as_mut_ptr(), v) }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) fn ldi(p: &[u32; 16]) -> __m512i {
        // SAFETY: `p` is 64 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm512_loadu_si512(p.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) fn ldm(p: &[i32; 16]) -> __m512i {
        // SAFETY: `p` is 64 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm512_loadu_si512(p.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) fn stm(p: &mut [i32; 16], v: __m512i) {
        // SAFETY: `p` is 64 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm512_storeu_si512(p.as_mut_ptr().cast(), v) }
    }

    #[target_feature(enable = "avx")]
    pub(super) fn st16(p: &mut [u16; 16], v: __m256i) {
        // SAFETY: `p` is 32 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), v) }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) fn ldb(p: &[u8; 64]) -> __m512i {
        // SAFETY: `p` is 64 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm512_loadu_si512(p.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) fn stb(p: &mut [u8; 64], v: __m512i) {
        // SAFETY: `p` is 64 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm512_storeu_si512(p.as_mut_ptr().cast(), v) }
    }
}
use mem::{ld, ldb, ldi, ldm, st, st16, stb, stm};

/// `scalar::dequant`: the PI, SIG and CHI bits are tested as masks, so the bias is a masked blend,
/// the sign a masked integer xor and a non-significant sample a zeroing move (+0.0, as `& sig`).
#[target_feature(enable = "avx2,fma,avx512f,avx512bw")]
pub(super) fn dequant(dst: &mut [f32], mag: &[i32], flags: &[u32], sh: u32, fb: [f32; 2], d: f32) {
    let (b0, b1, dv) = (
        _mm512_set1_ps(fb[0]),
        _mm512_set1_ps(fb[1]),
        _mm512_set1_ps(d),
    );
    let count = _mm_cvtsi32_si128(sh as i32);
    let (pi, sig) = (
        _mm512_set1_epi32(T1_PI as i32),
        _mm512_set1_epi32(T1_SIG as i32),
    );
    let (chi, sign) = (_mm512_set1_epi32(1 << T1_CHI), _mm512_set1_epi32(i32::MIN));
    let (oh, ot) = dst.as_chunks_mut::<16>();
    let (mh, mt) = mag.as_chunks::<16>();
    let (fh, ft) = flags.as_chunks::<16>();
    for ((o, m), f) in oh.iter_mut().zip(mh).zip(fh) {
        let f = _mm512_srl_epi32(ldi(f), count);
        let bias = _mm512_mask_blend_ps(_mm512_test_epi32_mask(f, pi), b0, b1);
        let v = _mm512_mul_ps(_mm512_add_ps(_mm512_cvtepi32_ps(ldm(m)), bias), dv);
        let v = _mm512_castps_si512(v);
        let v = _mm512_mask_xor_epi32(v, _mm512_test_epi32_mask(f, chi), v, sign);
        st(
            o,
            _mm512_maskz_mov_ps(_mm512_test_epi32_mask(f, sig), _mm512_castsi512_ps(v)),
        );
    }
    avx2::dequant(ot, mt, ft, sh, fb, d);
}

/// `scalar::pack_u16`'s clamp, round to even by the 1.5 * 2^23 trick and level shift on 16 lanes,
/// as i32. The max/min operand order sends a NaN to the low end, as the scalar compare-and-select
/// does.
#[inline]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw")]
fn samples(x: __m512, lo: __m512, hi: __m512, magic: __m512, sh: __m512i) -> __m512i {
    let c = _mm512_min_ps(_mm512_max_ps(x, lo), hi);
    let r = _mm512_sub_ps(_mm512_add_ps(c, magic), magic);
    _mm512_add_epi32(_mm512_cvttps_epi32(r), sh)
}

/// `scalar::pack_u16` for `precision` in [1, 16]: the samples lie in [0, 65535], so the
/// truncating narrow (VPMOVDW) is `as u16`.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw")]
pub(super) fn pack_u16(dst: &mut [u16], src: &[f32], precision: i32) {
    let shift = 1i32 << (precision - 1);
    let lo = _mm512_set1_ps(-(shift as f32));
    let hi = _mm512_set1_ps(((1i32 << precision) - 1 - shift) as f32);
    let magic = _mm512_set1_ps(12_582_912.0);
    let sh = _mm512_set1_epi32(shift);
    let (dh, dt) = dst.as_chunks_mut::<16>();
    let (sh16, st16t) = src.as_chunks::<16>();
    for (d, s) in dh.iter_mut().zip(sh16) {
        let i = samples(ld(s), lo, hi, magic, sh);
        st16(d, _mm512_cvtepi32_epi16(i));
    }
    avx2::pack_u16(dt, st16t, precision);
}

/// `scalar::pack_u8x4` for `precision` in [1, 16] and `ch` < 4: 16 pixels a step, the low byte
/// of each sample (`as u8`) moved to byte `ch` of its pixel and merged into the pixels read back,
/// so the other three bytes are written unchanged.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw")]
pub(super) fn pack_u8x4(dst: &mut [u8], src: &[f32], precision: i32, ch: usize) {
    let shift = 1i32 << (precision - 1);
    let lo = _mm512_set1_ps(-(shift as f32));
    let hi = _mm512_set1_ps(((1i32 << precision) - 1 - shift) as f32);
    let magic = _mm512_set1_ps(12_582_912.0);
    let sh = _mm512_set1_epi32(shift);
    let at = _mm_cvtsi32_si128(8 * ch as i32);
    let keep = _mm512_sll_epi32(_mm512_set1_epi32(0xff), at);
    let (dh, dt) = dst.as_chunks_mut::<64>();
    let (sh16, st16t) = src.as_chunks::<16>();
    for (d, s) in dh.iter_mut().zip(sh16) {
        let i = samples(ld(s), lo, hi, magic, sh);
        let b = _mm512_sll_epi32(_mm512_and_si512(i, _mm512_set1_epi32(0xff)), at);
        stb(d, _mm512_or_si512(_mm512_andnot_si512(keep, ldb(d)), b));
    }
    avx2::pack_u8x4(dt, st16t, precision, ch);
}

/// `scalar::dequant_rev`: the PI bit picks the bias by a masked blend, the CHI bit negates by a
/// masked `0 - v` (which is `(v ^ -1) - -1`), and a non-significant sample is zeroed.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw")]
pub(super) fn dequant_rev(o: &mut [i32], flags: &[u32], sh: u32, rb: [i32; 2]) {
    let (b0, b1) = (_mm512_set1_epi32(rb[0]), _mm512_set1_epi32(rb[1]));
    let count = _mm_cvtsi32_si128(sh as i32);
    let (pi, sig, chi) = (
        _mm512_set1_epi32(T1_PI as i32),
        _mm512_set1_epi32(T1_SIG as i32),
        _mm512_set1_epi32(1 << T1_CHI),
    );
    let zero = _mm512_setzero_si512();
    let (oh, ot) = o.as_chunks_mut::<16>();
    let (fh, ft) = flags.as_chunks::<16>();
    for (o, f) in oh.iter_mut().zip(fh) {
        let f = _mm512_srl_epi32(ldi(f), count);
        let bias = _mm512_mask_blend_epi32(_mm512_test_epi32_mask(f, pi), b0, b1);
        let v = _mm512_add_epi32(ldm(o), bias);
        let v = _mm512_mask_sub_epi32(v, _mm512_test_epi32_mask(f, chi), zero, v);
        stm(o, _mm512_maskz_mov_epi32(_mm512_test_epi32_mask(f, sig), v));
    }
    avx2::dequant_rev(ot, ft, sh, rb);
}
