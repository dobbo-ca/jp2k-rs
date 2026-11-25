//! NEON kernels, 4 lanes. Each runs whole groups of 4 and leaves the tail to the scalar kernel.
//! Callers (`super`) run them only for `Tier::Neon`, which exists only if `neon` was detected.
//! Slice lengths are equal.

use super::{CHI_UP, PI_UP, Rounding, SIG_UP, scalar};
use crate::dsp::{ICT_CB_B, ICT_CB_G, ICT_CR_G, ICT_CR_R};
use std::arch::aarch64::*;

// The only `unsafe` in this tier: unaligned loads and stores of whole arrays.
#[allow(unsafe_code)]
mod mem {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) fn ld(p: &[f32; 4]) -> float32x4_t {
        // SAFETY: `p` is 16 valid bytes and `vld1q_f32` has no alignment requirement.
        unsafe { vld1q_f32(p.as_ptr()) }
    }

    #[target_feature(enable = "neon")]
    pub(super) fn st(p: &mut [f32; 4], v: float32x4_t) {
        // SAFETY: `p` is 16 valid writable bytes and `vst1q_f32` has no alignment requirement.
        unsafe { vst1q_f32(p.as_mut_ptr(), v) }
    }

    #[target_feature(enable = "neon")]
    pub(super) fn ldi(p: &[u32; 4]) -> uint32x4_t {
        // SAFETY: `p` is 16 valid bytes and `vld1q_u32` has no alignment requirement.
        unsafe { vld1q_u32(p.as_ptr()) }
    }

    #[target_feature(enable = "neon")]
    pub(super) fn ldm(p: &[i32; 4]) -> int32x4_t {
        // SAFETY: `p` is 16 valid bytes and `vld1q_s32` has no alignment requirement.
        unsafe { vld1q_s32(p.as_ptr()) }
    }

    #[target_feature(enable = "neon")]
    pub(super) fn st16(p: &mut [u16; 4], v: uint16x4_t) {
        // SAFETY: `p` is 8 valid writable bytes and `vst1_u16` has no alignment requirement.
        unsafe { vst1_u16(p.as_mut_ptr(), v) }
    }
}
use mem::{ld, ldi, ldm, st, st16};

fn rounding(fused: bool) -> Rounding {
    if fused {
        Rounding::Fused
    } else {
        Rounding::Separate
    }
}

/// `d = c * (u + v) + d`.
#[target_feature(enable = "neon")]
pub(super) fn lift(d: &mut [f32], u: &[f32], v: &[f32], c: f32, fused: bool) {
    let cv = vdupq_n_f32(c);
    let (dh, dt) = d.as_chunks_mut::<4>();
    let (uh, ut) = u.as_chunks::<4>();
    let (vh, vt) = v.as_chunks::<4>();
    for ((d, u), v) in dh.iter_mut().zip(uh).zip(vh) {
        let s = vaddq_f32(ld(u), ld(v));
        let x = ld(d);
        st(
            d,
            if fused {
                vfmaq_f32(x, s, cv)
            } else {
                vaddq_f32(vmulq_f32(cv, s), x)
            },
        );
    }
    scalar::lift(dt, ut, vt, c, rounding(fused));
}

#[target_feature(enable = "neon")]
pub(super) fn scale(x: &mut [f32], k: f32) {
    let kv = vdupq_n_f32(k);
    let (xh, xt) = x.as_chunks_mut::<4>();
    for x in xh {
        st(x, vmulq_f32(ld(x), kv));
    }
    scalar::scale(xt, k);
}

#[target_feature(enable = "neon")]
pub(super) fn ict(c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], fused: bool) {
    let (crr, cbg) = (vdupq_n_f32(ICT_CR_R), vdupq_n_f32(ICT_CB_G));
    let (crg, cbb) = (vdupq_n_f32(ICT_CR_G), vdupq_n_f32(ICT_CB_B));
    let (h0, t0) = c0.as_chunks_mut::<4>();
    let (h1, t1) = c1.as_chunks_mut::<4>();
    let (h2, t2) = c2.as_chunks_mut::<4>();
    for ((a, b), c) in h0.iter_mut().zip(h1).zip(h2) {
        let (y, u, v) = (ld(a), ld(b), ld(c));
        let (r, g, bl);
        if fused {
            r = vfmaq_f32(y, v, crr);
            g = vfmsq_f32(vfmsq_f32(y, u, cbg), v, crg);
            bl = vfmaq_f32(y, u, cbb);
        } else {
            r = vaddq_f32(vmulq_f32(v, crr), y);
            g = vsubq_f32(vsubq_f32(y, vmulq_f32(u, cbg)), vmulq_f32(v, crg));
            bl = vaddq_f32(vmulq_f32(u, cbb), y);
        }
        st(a, r);
        st(b, g);
        st(c, bl);
    }
    scalar::ict(t0, t1, t2, rounding(fused));
}

#[target_feature(enable = "neon")]
pub(super) fn dequant(dst: &mut [f32], mag: &[i32], flags: &[u32], sh: u32, fb: [f32; 2], d: f32) {
    let (b0, b1, dv) = (vdupq_n_f32(fb[0]), vdupq_n_f32(fb[1]), vdupq_n_f32(d));
    let count = vdupq_n_s32(-(sh as i32));
    let sign = vdupq_n_u32(0x8000_0000);
    let (oh, ot) = dst.as_chunks_mut::<4>();
    let (mh, mt) = mag.as_chunks::<4>();
    let (fh, ft) = flags.as_chunks::<4>();
    for ((o, m), f) in oh.iter_mut().zip(mh).zip(fh) {
        let f = vshlq_u32(ldi(f), count);
        let pi = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_u32(vshlq_n_u32::<
            PI_UP,
        >(f))));
        let bias = vbslq_f32(pi, b1, b0);
        let v = vmulq_f32(vaddq_f32(vcvtq_f32_s32(ldm(m)), bias), dv);
        let neg = vandq_u32(vshlq_n_u32::<CHI_UP>(f), sign);
        let v = veorq_u32(vreinterpretq_u32_f32(v), neg);
        let sig = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_u32(vshlq_n_u32::<
            SIG_UP,
        >(f))));
        st(o, vreinterpretq_f32_u32(vandq_u32(v, sig)));
    }
    scalar::dequant(ot, mt, ft, sh, fb, d);
}

/// `scalar::pack_u16` for `precision` in [1, 16]: clamp (a NaN goes to the low end, as the
/// scalar compare-and-select does), round to even by the 1.5 * 2^23 trick, add the level shift,
/// narrow.
#[target_feature(enable = "neon")]
pub(super) fn pack_u16(dst: &mut [u16], src: &[f32], precision: i32) {
    let shift = 1i32 << (precision - 1);
    let lo = vdupq_n_f32(-(shift as f32));
    let hi = vdupq_n_f32(((1i32 << precision) - 1 - shift) as f32);
    let magic = vdupq_n_f32(12_582_912.0);
    let sh = vdupq_n_s32(shift);
    let (dh, dt) = dst.as_chunks_mut::<4>();
    let (sh4, st4) = src.as_chunks::<4>();
    for (d, s) in dh.iter_mut().zip(sh4) {
        let v = ld(s);
        let c = vbslq_f32(vcgtq_f32(v, lo), v, lo);
        let c = vbslq_f32(vcltq_f32(c, hi), c, hi);
        let r = vsubq_f32(vaddq_f32(c, magic), magic);
        st16(d, vqmovun_s32(vaddq_s32(vcvtq_s32_f32(r), sh)));
    }
    scalar::pack_u16(dt, st4, precision);
}

/// `dst = c * (u + v) + d`: `lift` with a destination of its own.
#[target_feature(enable = "neon")]
pub(super) fn lift_to(dst: &mut [f32], d: &[f32], u: &[f32], v: &[f32], c: f32, fused: bool) {
    let cv = vdupq_n_f32(c);
    let (oh, ot) = dst.as_chunks_mut::<4>();
    let (dh, dt) = d.as_chunks::<4>();
    let (uh, ut) = u.as_chunks::<4>();
    let (vh, vt) = v.as_chunks::<4>();
    for (((o, d), u), v) in oh.iter_mut().zip(dh).zip(uh).zip(vh) {
        let s = vaddq_f32(ld(u), ld(v));
        let x = ld(d);
        st(
            o,
            if fused {
                vfmaq_f32(x, s, cv)
            } else {
                vaddq_f32(vmulq_f32(cv, s), x)
            },
        );
    }
    scalar::lift_to(ot, dt, ut, vt, c, rounding(fused));
}

#[target_feature(enable = "neon")]
pub(super) fn scale_to(dst: &mut [f32], src: &[f32], k: f32) {
    let kv = vdupq_n_f32(k);
    let (oh, ot) = dst.as_chunks_mut::<4>();
    let (sh, stl) = src.as_chunks::<4>();
    for (o, s) in oh.iter_mut().zip(sh) {
        st(o, vmulq_f32(ld(s), kv));
    }
    scalar::scale_to(ot, stl, k);
}
