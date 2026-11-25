//! AVX2 + FMA kernels, 8 lanes. Each runs whole groups of 8 and leaves the tail to the scalar
//! kernel. Callers (`super`, and the AVX-512 kernels for their tails) run them only for
//! `Tier::Avx2` or `Tier::Avx512`, which exist only if `avx2` and `fma` were detected. Slice
//! lengths are equal.

use super::{CHI_UP, PI_UP, Rounding, SIG_UP, scalar};
use crate::dsp::{ICT_CB_B, ICT_CB_G, ICT_CR_G, ICT_CR_R};
use std::arch::x86_64::*;

// The only `unsafe` in this tier: unaligned loads and stores of whole arrays.
#[allow(unsafe_code)]
mod mem {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) fn ld(p: &[f32; 8]) -> __m256 {
        // SAFETY: `p` is 32 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm256_loadu_ps(p.as_ptr()) }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn st(p: &mut [f32; 8], v: __m256) {
        // SAFETY: `p` is 32 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm256_storeu_ps(p.as_mut_ptr(), v) }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn ldi(p: &[u32; 8]) -> __m256i {
        // SAFETY: `p` is 32 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn ldm(p: &[i32; 8]) -> __m256i {
        // SAFETY: `p` is 32 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn stm(p: &mut [i32; 8], v: __m256i) {
        // SAFETY: `p` is 32 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), v) }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn st16(p: &mut [u16; 8], v: __m128i) {
        // SAFETY: `p` is 16 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm_storeu_si128(p.as_mut_ptr().cast(), v) }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn ldb(p: &[u8; 32]) -> __m256i {
        // SAFETY: `p` is 32 valid bytes and `loadu` has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn stb(p: &mut [u8; 32], v: __m256i) {
        // SAFETY: `p` is 32 valid writable bytes and `storeu` has no alignment requirement.
        unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), v) }
    }
}
use mem::{ld, ldb, ldi, ldm, st, st16, stb, stm};

fn rounding(fused: bool) -> Rounding {
    if fused {
        Rounding::Fused
    } else {
        Rounding::Separate
    }
}

/// `d = c * (u + v) + d`.
#[target_feature(enable = "avx2,fma")]
pub(super) fn lift(d: &mut [f32], u: &[f32], v: &[f32], c: f32, fused: bool) {
    let cv = _mm256_set1_ps(c);
    let (dh, dt) = d.as_chunks_mut::<8>();
    let (uh, ut) = u.as_chunks::<8>();
    let (vh, vt) = v.as_chunks::<8>();
    for ((d, u), v) in dh.iter_mut().zip(uh).zip(vh) {
        let s = _mm256_add_ps(ld(u), ld(v));
        let x = ld(d);
        st(
            d,
            if fused {
                _mm256_fmadd_ps(cv, s, x)
            } else {
                _mm256_add_ps(_mm256_mul_ps(cv, s), x)
            },
        );
    }
    scalar::lift(dt, ut, vt, c, rounding(fused));
}

#[target_feature(enable = "avx2")]
pub(super) fn scale(x: &mut [f32], k: f32) {
    let kv = _mm256_set1_ps(k);
    let (xh, xt) = x.as_chunks_mut::<8>();
    for x in xh {
        st(x, _mm256_mul_ps(ld(x), kv));
    }
    scalar::scale(xt, k);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn ict(c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], fused: bool) {
    let (crr, cbg) = (_mm256_set1_ps(ICT_CR_R), _mm256_set1_ps(ICT_CB_G));
    let (crg, cbb) = (_mm256_set1_ps(ICT_CR_G), _mm256_set1_ps(ICT_CB_B));
    let (h0, t0) = c0.as_chunks_mut::<8>();
    let (h1, t1) = c1.as_chunks_mut::<8>();
    let (h2, t2) = c2.as_chunks_mut::<8>();
    for ((a, b), c) in h0.iter_mut().zip(h1).zip(h2) {
        let (y, u, v) = (ld(a), ld(b), ld(c));
        let (r, g, bl);
        if fused {
            r = _mm256_fmadd_ps(v, crr, y);
            g = _mm256_fnmadd_ps(v, crg, _mm256_fnmadd_ps(u, cbg, y));
            bl = _mm256_fmadd_ps(u, cbb, y);
        } else {
            r = _mm256_add_ps(_mm256_mul_ps(v, crr), y);
            g = _mm256_sub_ps(
                _mm256_sub_ps(y, _mm256_mul_ps(u, cbg)),
                _mm256_mul_ps(v, crg),
            );
            bl = _mm256_add_ps(_mm256_mul_ps(u, cbb), y);
        }
        st(a, r);
        st(b, g);
        st(c, bl);
    }
    scalar::ict(t0, t1, t2, rounding(fused));
}

#[target_feature(enable = "avx2")]
pub(super) fn dequant(dst: &mut [f32], mag: &[i32], flags: &[u32], sh: u32, fb: [f32; 2], d: f32) {
    let (b0, b1, dv) = (
        _mm256_set1_ps(fb[0]),
        _mm256_set1_ps(fb[1]),
        _mm256_set1_ps(d),
    );
    let count = _mm_cvtsi32_si128(sh as i32);
    let sign = _mm256_set1_epi32(i32::MIN);
    let (oh, ot) = dst.as_chunks_mut::<8>();
    let (mh, mt) = mag.as_chunks::<8>();
    let (fh, ft) = flags.as_chunks::<8>();
    for ((o, m), f) in oh.iter_mut().zip(mh).zip(fh) {
        let f = _mm256_srl_epi32(ldi(f), count);
        let pi = _mm256_castsi256_ps(_mm256_srai_epi32::<31>(_mm256_slli_epi32::<PI_UP>(f)));
        let bias = _mm256_blendv_ps(b0, b1, pi);
        let v = _mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(ldm(m)), bias), dv);
        let neg = _mm256_and_si256(_mm256_slli_epi32::<CHI_UP>(f), sign);
        let v = _mm256_xor_ps(v, _mm256_castsi256_ps(neg));
        let sig = _mm256_castsi256_ps(_mm256_srai_epi32::<31>(_mm256_slli_epi32::<SIG_UP>(f)));
        st(o, _mm256_and_ps(v, sig));
    }
    scalar::dequant(ot, mt, ft, sh, fb, d);
}

/// `scalar::pack_u16` for `precision` in [1, 16]: clamp, round to even by the 1.5 * 2^23 trick,
/// add the level shift, narrow. The max/min operand order sends a NaN to the low end, as the
/// scalar compare-and-select does.
#[target_feature(enable = "avx2")]
pub(super) fn pack_u16(dst: &mut [u16], src: &[f32], precision: i32) {
    let shift = 1i32 << (precision - 1);
    let lo = _mm256_set1_ps(-(shift as f32));
    let hi = _mm256_set1_ps(((1i32 << precision) - 1 - shift) as f32);
    let magic = _mm256_set1_ps(12_582_912.0);
    let sh = _mm256_set1_epi32(shift);
    let (dh, dt) = dst.as_chunks_mut::<8>();
    let (sh8, st8) = src.as_chunks::<8>();
    for (d, s) in dh.iter_mut().zip(sh8) {
        let c = _mm256_min_ps(_mm256_max_ps(ld(s), lo), hi);
        let r = _mm256_sub_ps(_mm256_add_ps(c, magic), magic);
        let i = _mm256_add_epi32(_mm256_cvttps_epi32(r), sh);
        // Pack within each 128-bit half, then gather the two low quadwords.
        let p = _mm256_permute4x64_epi64::<0x08>(_mm256_packus_epi32(i, i));
        st16(d, _mm256_castsi256_si128(p));
    }
    scalar::pack_u16(dt, st8, precision);
}

/// `scalar::pack_u8x4` for `precision` in [1, 16] and `ch` < 4: `pack_u16`'s samples, 8 pixels a
/// step, the low byte of each (`as u8`) moved to byte `ch` of its pixel and merged into the pixels
/// read back, so the other three bytes are written unchanged.
#[target_feature(enable = "avx2")]
pub(super) fn pack_u8x4(dst: &mut [u8], src: &[f32], precision: i32, ch: usize) {
    let shift = 1i32 << (precision - 1);
    let lo = _mm256_set1_ps(-(shift as f32));
    let hi = _mm256_set1_ps(((1i32 << precision) - 1 - shift) as f32);
    let magic = _mm256_set1_ps(12_582_912.0);
    let sh = _mm256_set1_epi32(shift);
    let at = _mm_cvtsi32_si128(8 * ch as i32);
    let keep = _mm256_sll_epi32(_mm256_set1_epi32(0xff), at);
    let (dh, dt) = dst.as_chunks_mut::<32>();
    let (sh8, st8) = src.as_chunks::<8>();
    for (d, s) in dh.iter_mut().zip(sh8) {
        let c = _mm256_min_ps(_mm256_max_ps(ld(s), lo), hi);
        let r = _mm256_sub_ps(_mm256_add_ps(c, magic), magic);
        let i = _mm256_add_epi32(_mm256_cvttps_epi32(r), sh);
        let b = _mm256_sll_epi32(_mm256_and_si256(i, _mm256_set1_epi32(0xff)), at);
        stb(d, _mm256_or_si256(_mm256_andnot_si256(keep, ldb(d)), b));
    }
    scalar::pack_u8x4(dt, st8, precision, ch);
}

/// `dst = c * (u + v) + d`: `lift` with a destination of its own.
#[target_feature(enable = "avx2,fma")]
pub(super) fn lift_to(dst: &mut [f32], d: &[f32], u: &[f32], v: &[f32], c: f32, fused: bool) {
    let cv = _mm256_set1_ps(c);
    let (oh, ot) = dst.as_chunks_mut::<8>();
    let (dh, dt) = d.as_chunks::<8>();
    let (uh, ut) = u.as_chunks::<8>();
    let (vh, vt) = v.as_chunks::<8>();
    for (((o, d), u), v) in oh.iter_mut().zip(dh).zip(uh).zip(vh) {
        let s = _mm256_add_ps(ld(u), ld(v));
        let x = ld(d);
        st(
            o,
            if fused {
                _mm256_fmadd_ps(cv, s, x)
            } else {
                _mm256_add_ps(_mm256_mul_ps(cv, s), x)
            },
        );
    }
    scalar::lift_to(ot, dt, ut, vt, c, rounding(fused));
}

#[target_feature(enable = "avx2")]
pub(super) fn scale_to(dst: &mut [f32], src: &[f32], k: f32) {
    let kv = _mm256_set1_ps(k);
    let (oh, ot) = dst.as_chunks_mut::<8>();
    let (sh, stl) = src.as_chunks::<8>();
    for (o, s) in oh.iter_mut().zip(sh) {
        st(o, _mm256_mul_ps(ld(s), kv));
    }
    scalar::scale_to(ot, stl, k);
}

/// The inverse RCT: `vpaddd`/`vpsubd` wrap as `wrapping_*` does and `vpsrad` is i32 `>>`.
#[target_feature(enable = "avx2")]
pub(super) fn rct(c0: &mut [i32], c1: &mut [i32], c2: &mut [i32]) {
    let (h0, t0) = c0.as_chunks_mut::<8>();
    let (h1, t1) = c1.as_chunks_mut::<8>();
    let (h2, t2) = c2.as_chunks_mut::<8>();
    for ((a, b), c) in h0.iter_mut().zip(h1).zip(h2) {
        let (y, u, v) = (ldm(a), ldm(b), ldm(c));
        let g = _mm256_sub_epi32(y, _mm256_srai_epi32::<2>(_mm256_add_epi32(u, v)));
        stm(a, _mm256_add_epi32(v, g));
        stm(b, g);
        stm(c, _mm256_add_epi32(u, g));
    }
    scalar::rct(t0, t1, t2);
}

/// `d -= (u + v + 2) >> 2`, wrapping.
#[target_feature(enable = "avx2")]
pub(super) fn lift53_sub(d: &mut [i32], u: &[i32], v: &[i32]) {
    let two = _mm256_set1_epi32(2);
    let (dh, dt) = d.as_chunks_mut::<8>();
    let (uh, ut) = u.as_chunks::<8>();
    let (vh, vt) = v.as_chunks::<8>();
    for ((d, u), v) in dh.iter_mut().zip(uh).zip(vh) {
        let t = _mm256_add_epi32(_mm256_add_epi32(ldm(u), ldm(v)), two);
        stm(d, _mm256_sub_epi32(ldm(d), _mm256_srai_epi32::<2>(t)));
    }
    scalar::lift53_sub(dt, ut, vt);
}

/// `d += (u + v) >> 1`, wrapping.
#[target_feature(enable = "avx2")]
pub(super) fn lift53_add(d: &mut [i32], u: &[i32], v: &[i32]) {
    let (dh, dt) = d.as_chunks_mut::<8>();
    let (uh, ut) = u.as_chunks::<8>();
    let (vh, vt) = v.as_chunks::<8>();
    for ((d, u), v) in dh.iter_mut().zip(uh).zip(vh) {
        let t = _mm256_add_epi32(ldm(u), ldm(v));
        stm(d, _mm256_add_epi32(ldm(d), _mm256_srai_epi32::<1>(t)));
    }
    scalar::lift53_add(dt, ut, vt);
}

/// `scalar::dequant_rev`: each flag bit moved to bit 31 and spread over the lane is the mask the
/// scalar loop builds; the bias is a blend of the two.
#[target_feature(enable = "avx2")]
pub(super) fn dequant_rev(o: &mut [i32], flags: &[u32], sh: u32, rb: [i32; 2]) {
    let (b0, b1) = (_mm256_set1_epi32(rb[0]), _mm256_set1_epi32(rb[1]));
    let count = _mm_cvtsi32_si128(sh as i32);
    let (oh, ot) = o.as_chunks_mut::<8>();
    let (fh, ft) = flags.as_chunks::<8>();
    for (o, f) in oh.iter_mut().zip(fh) {
        let f = _mm256_srl_epi32(ldi(f), count);
        let pi = _mm256_srai_epi32::<31>(_mm256_slli_epi32::<PI_UP>(f));
        let m = _mm256_srai_epi32::<31>(_mm256_slli_epi32::<CHI_UP>(f));
        let sig = _mm256_srai_epi32::<31>(_mm256_slli_epi32::<SIG_UP>(f));
        let v = _mm256_add_epi32(ldm(o), _mm256_blendv_epi8(b0, b1, pi));
        let v = _mm256_sub_epi32(_mm256_xor_si256(v, m), m);
        stm(o, _mm256_and_si256(v, sig));
    }
    scalar::dequant_rev(ot, ft, sh, rb);
}
