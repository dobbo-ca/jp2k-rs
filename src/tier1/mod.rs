//! Decodes JPEG 2000 code-blocks: the MQ arithmetic decoder and the EBCOT coding passes of
//! ISO/IEC 15444-1 Annexes C and D.

mod mq;
mod passes;

use crate::error::Result;
use mq::{MQ_W, MqDecoder, RawDecoder};
use std::sync::OnceLock;

// Context slots, T.800 Table D.7 / Annex C.2: 9 zero-coding contexts, 5 sign-coding, 3
// magnitude-refinement, plus run-length and uniform.
const CTX_RUN_LENGTH: usize = 17; // "AGG" in the reference: initial state I=3
const CTX_UNIFORM: usize = 18; // "UNI": initial state I=46

/// T.800's implicit cap on code-block area (64x64).
pub const MAX_CODE_BLOCK_SAMPLES: usize = 4096;

// Code-block style bits, T.800 Table A.19 (SPcod/SPcoc byte 4). CB_BYPASS is the selective
// arithmetic-coding bypass bit. 0x10 (predictable termination) changes no decode output.
pub const CB_BYPASS: u8 = 0x01; // selective arithmetic coding bypass (raw SPP/MRP)
const CB_RESET: u8 = 0x02; // reset context probabilities at every pass boundary
pub const CB_TERM_ALL: u8 = 0x04; // terminate the codeword on every coding pass
const CB_CAUSAL: u8 = 0x08; // vertically causal context formation
const CB_SEG_SYM: u8 = 0x20; // segmentation symbols at the end of each cleanup pass
pub const CB_HT_MASK: u8 = 0xC0; // HT block coding, not supported

/// The 19-context state array with the T.800 Table D.7 initial states: ZC context 0 starts at
/// state 4, AGG at 3, UNI at 46; every other context starts at state 0 with MPS 0.
const INIT_CTX: [u64; 32] = {
    let mut c = [MQ_W[0]; 32];
    c[0] = MQ_W[4 << 1];
    c[CTX_RUN_LENGTH] = MQ_W[3 << 1];
    c[CTX_UNIFORM] = MQ_W[46 << 1];
    c
};

/// The zero-coding context rule for LL/LH subbands (T.800 Table D.1), h/v/d the count of
/// significant horizontal/vertical/diagonal neighbours.
fn zc_ll_lh(h: usize, v: usize, d: usize) -> usize {
    if h == 2 {
        8
    } else if h == 1 && v >= 1 {
        7
    } else if h == 1 && d >= 1 {
        6
    } else if h == 1 {
        5
    } else if v == 2 {
        4
    } else if v == 1 {
        3
    } else if d >= 2 {
        2
    } else if d == 1 {
        1
    } else {
        0
    }
}

/// The zero-coding context rule for the HH subband (T.800 Table D.1), keyed by diagonal count
/// and h+v.
fn zc_hh(d: usize, hv: usize) -> usize {
    if d >= 3 {
        8
    } else if d == 2 && hv >= 1 {
        7
    } else if d == 2 {
        6
    } else if d == 1 && hv >= 2 {
        5
    } else if d == 1 && hv == 1 {
        4
    } else if d == 1 {
        3
    } else if hv >= 2 {
        2
    } else if hv == 1 {
        1
    } else {
        0
    }
}

/// The zero-coding context for a sample given its band orientation (0=LL, 1=HL, 2=LH, 3=HH)
/// and its neighbour significance counts. HL reuses the LL/LH rule with h and v swapped.
fn zc_context(orientation: usize, h: usize, v: usize, d: usize) -> usize {
    match orientation {
        1 => zc_ll_lh(v, h, d),
        3 => zc_hh(d, h + v),
        _ => zc_ll_lh(h, v, d),
    }
}

// Tier-1 state is one u32 per column of a 4-row stripe, OpenJPEG 2.x's t1.h layout: one load
// tells whether a column has anything to do in a pass, and a sample's contexts are shifts of
// that word. Bits 0..17 are the significance of the column's 3-wide, 6-tall neighbourhood
// (rows -1..4, columns x-1..x+1, row-major), so row ci's 3x3 is bits 3ci..3ci+8. The rest hold
// each row's sign (chi, also for rows -1 and 4), mu (refined before) and pi (visited by this
// bit-plane's SPP). Vertically causal mode (0x08) is applied on write: a stripe's row 0 never
// marks the stripe above, so that stripe's row 3 never sees it.
pub(crate) const T1_SIG: u32 = 1 << 4; // row 0's own significance
const T1_SIG_NB: u32 = 0x1EF; // row 0's 8 neighbours' significance
pub(crate) const T1_CHI: u32 = 19; // bit of row 0's sign; row ci's is T1_CHI + 3ci
const T1_MU: u32 = 1 << 20; // row 0 refined before
pub(crate) const T1_PI: u32 = 1 << 21; // row 0 visited this bit-plane
const T1_SIG_COL: u32 = T1_SIG | T1_SIG << 3 | T1_SIG << 6 | T1_SIG << 9;
const T1_PI_COL: u32 = T1_PI | T1_PI << 3 | T1_PI << 6 | T1_PI << 9;

/// `zc[orientation][nb]` maps orientation and a row's 3x3 significance bits to the zero-coding
/// context; `sc[index]` maps the sign index to the sign-coding context and XOR bit packed as
/// `ctx << 1 | xor`.
struct Luts {
    zc: [[u8; 512]; 4],
    sc: [u8; 256],
}

fn luts() -> &'static Luts {
    static L: OnceLock<Luts> = OnceLock::new();
    L.get_or_init(|| {
        let mut l = Luts {
            zc: [[0; 512]; 4],
            sc: [0; 256],
        };
        for nb in 0..512usize {
            let h = (nb >> 3 & 1) + (nb >> 5 & 1); // W, E
            let v = (nb >> 1 & 1) + (nb >> 7 & 1); // N, S
            let d = (nb & 1) + (nb >> 2 & 1) + (nb >> 6 & 1) + (nb >> 8 & 1);
            for or in 0..4 {
                l.zc[or][nb] = zc_context(or, h, v, d) as u8;
            }
        }
        // The T.800 Table D.3 per-axis contribution of a neighbour pair: both significant and
        // agreeing = +/-1, disagreeing = 0, one significant = its sign, none = 0.
        let contrib = |sa: i32, na: i32, sb: i32, nb: i32| -> i32 {
            if sa != 0 && sb != 0 {
                1 - na - nb
            } else if sa != 0 {
                1 - 2 * na
            } else if sb != 0 {
                1 - 2 * nb
            } else {
                0
            }
        };
        for lu in 0..256i32 {
            // Index bits: 0 W sign, 1 N sig, 2 E sign, 3 W sig, 4 N sign, 5 E sig, 6 S sign,
            // 7 S sig.
            let horiz = contrib(lu >> 5 & 1, lu >> 2 & 1, lu >> 3 & 1, lu & 1);
            let vert = contrib(lu >> 1 & 1, lu >> 4 & 1, lu >> 7 & 1, lu >> 6 & 1);
            let total = 3 * horiz + vert;
            let (ctx, xor) = if total < 0 {
                (9 - total, 1)
            } else {
                (9 + total, 0)
            };
            l.sc[lu as usize] = ((ctx << 1) | xor) as u8;
        }
        l
    })
}

/// Gathers row `ci`'s four direct neighbours' significance and sign into the sign LUT's index;
/// `wf` and `ef` are the west and east columns' words.
#[inline(always)]
fn sc_index(f: u32, wf: u32, ef: u32, ci: u32) -> usize {
    let s = (3 * ci) & 15;
    let mut lu = (f >> s) & (1 << 1 | 1 << 3 | 1 << 5 | 1 << 7)
        | (wf >> ((T1_CHI + s) & 31)) & 1
        | (ef >> ((T1_CHI - 2 + s) & 31)) & 4
        | (f >> (T1_CHI - 3 + s)) & 64;
    if ci == 0 {
        lu |= (f >> (T1_CHI - 5)) & 16;
    } else {
        lu |= (f >> (T1_CHI - 7 + s)) & 16;
    }
    (lu & 0xFF) as usize
}

/// The working flag grid for one code-block: one word per stripe column, padded by one column
/// each side and one stripe above and below so neighbour updates never need bounds checks.
pub struct State {
    w: usize,
    h: usize,
    /// Flags stride: w + 2.
    fs: usize,
    flags: Vec<u32>,
    /// 19 used; 32 lets a masked index skip bounds checks.
    ctxs: [u64; 32],
}

impl Default for State {
    fn default() -> Self {
        State {
            w: 0,
            h: 0,
            fs: 0,
            flags: Vec::new(),
            ctxs: INIT_CTX,
        }
    }
}

impl State {
    /// The flags index of column `x` of the stripe holding row `y`.
    fn col(&self, x: usize, y: usize) -> usize {
        (y / 4 + 1) * self.fs + x + 1
    }

    /// Sizes the state for a `w` x `h` code-block, reusing its backing array when it is already
    /// big enough.
    fn reset(&mut self, w: usize, h: usize) {
        let fs = w + 2;
        let n = fs * (h.div_ceil(4) + 2);
        self.flags.clear();
        self.flags.resize(n, 0);
        self.w = w;
        self.h = h;
        self.fs = fs;
    }

    /// Marks row `ci` of column `i` significant with sign `s` (1 = negative): in `f`, the
    /// column's word the caller holds, and in the words of the columns and stripes around it.
    /// `vsc` is set only for row 0 under vertically causal mode.
    #[inline(always)]
    fn sig(&mut self, f: u32, i: usize, ci: u32, s: u32, vsc: bool) -> u32 {
        let fs = self.fs;
        let fl = &mut self.flags;
        let sh = (3 * ci) & 15;
        fl[i - 1] |= 1 << 5 << sh;
        fl[i + 1] |= 1 << 3 << sh;
        if ci == 0 && !vsc {
            let n = i - fs;
            fl[n] |= s << 31 | 1 << 16;
            fl[n - 1] |= 1 << 17;
            fl[n + 1] |= 1 << 15;
        }
        if ci == 3 {
            let n = i + fs;
            fl[n] |= s << 18 | 1 << 1;
            fl[n - 1] |= 1 << 2;
            fl[n + 1] |= 1;
        }
        f | (s << T1_CHI | T1_SIG) << sh
    }

    /// The selective-bypass (0x01) counterpart of `significance_propagation_pass`: the ZC
    /// decision is a plain bit and, on a hit, the sign is a second plain bit with no
    /// sign-prediction XOR (T.800 D.4 / JJ2000 rawSigProgPass).
    fn significance_propagation_pass_raw(
        &mut self,
        raw: &mut RawDecoder,
        out: &mut [i32],
        plane: i32,
        vsc: bool,
    ) {
        let (w, h) = (self.w, self.h);
        let one = 1i32 << plane;
        let mut y0 = 0;
        while y0 < h {
            let rows = (h - y0).min(4) as u32;
            let i0 = self.col(0, y0);
            for x in 0..w {
                let i = i0 + x;
                let mut f = self.flags[i];
                if f == 0 {
                    continue;
                }
                for ci in 0..rows {
                    let sh = (3 * ci) & 15;
                    if (f >> sh) & (T1_SIG | T1_PI) != 0 || (f >> sh) & T1_SIG_NB == 0 {
                        continue;
                    }
                    if raw.read_bit() == 1 {
                        let s = raw.read_bit();
                        f = self.sig(f, i, ci, s, vsc && ci == 0);
                        out[(y0 + ci as usize) * w + x] |= one;
                    }
                    f |= T1_PI << sh;
                }
                self.flags[i] = f;
            }
            y0 += 4;
        }
    }

    /// The selective-bypass counterpart of the refinement pass: one raw bit per refined sample,
    /// no context. It still sets mu, matching OpenJPEG's opj_t1_dec_refpass_step_raw: if a
    /// later MQ MR pass ever refines a sample that already went through a raw MR pass, it must
    /// pick context 16.
    fn magnitude_refinement_pass_raw(&mut self, raw: &mut RawDecoder, out: &mut [i32], plane: i32) {
        let (w, h) = (self.w, self.h);
        let mut y0 = 0;
        while y0 < h {
            let rows = (h - y0).min(4) as u32;
            let i0 = self.col(0, y0);
            for x in 0..w {
                let i = i0 + x;
                let mut f = self.flags[i];
                if f & T1_SIG_COL == 0 {
                    continue;
                }
                for ci in 0..rows {
                    let sh = (3 * ci) & 15;
                    if (f >> sh) & (T1_SIG | T1_PI) != T1_SIG {
                        continue;
                    }
                    out[(y0 + ci as usize) * w + x] |= (raw.read_bit() as i32) << (plane & 31);
                    f |= T1_MU << sh;
                }
                self.flags[i] = f;
            }
            y0 += 4;
        }
    }

    /// Runs the coding passes for one code-block, ORing magnitude bits into `out` and leaving
    /// significance, sign and pi in the flags. Returns the pass type and bit-plane it stopped
    /// at, or `None` for a block with no coded bit-planes (`out` and the flags then untouched).
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        out: &mut [i32],
        segs: &[Segment],
        width: usize,
        height: usize,
        orientation: usize,
        zero_bit_planes: i32,
        mb: i32,
        style: u8,
    ) -> Result<Option<(i32, i32)>> {
        let numbps = mb as i64 - zero_bit_planes as i64;
        let total_passes: usize = segs.iter().map(|s| s.passes).sum();
        if numbps <= 0 || total_passes == 0 {
            return Ok(None);
        }
        if numbps > 31 {
            return Err(invalid!(
                "code-block bit-plane count {numbps} exceeds int32 sign-magnitude capacity"
            ));
        }
        let numbps = numbps as i32;
        out.fill(0); // the passes OR magnitude bits in

        self.reset(width, height);
        self.ctxs = INIT_CTX;

        let bypass = style & CB_BYPASS != 0;
        let reset = style & CB_RESET != 0;
        let vsc = style & CB_CAUSAL != 0;
        let segsym = style & CB_SEG_SYM != 0;

        let mut plane = numbps - 1;
        let mut pass_type = 2; // T.800 D.2: the first bit-plane starts with a cleanup pass
        'seg: for seg in segs {
            if plane < 0 {
                break;
            }
            // T.800 D.4 / OpenJPEG t1.c: bypass only ever applies to SPP/MRP (pass_type < 2)
            // once the bit-plane is at least the 5th coded one; the cleanup pass always stays
            // MQ. Tier-2's segmentation only ever splits segments at these exact transitions,
            // so this holds for every pass in the segment, not just its first.
            let is_raw = bypass && plane < numbps - 4 && pass_type < 2;
            let mut mq = None;
            let mut raw = None;
            if is_raw {
                raw = Some(RawDecoder::new(seg.data));
            } else {
                mq = Some(MqDecoder::new(seg.data));
            }
            for _ in 0..seg.passes {
                if plane < 0 {
                    break 'seg;
                }
                if is_raw && pass_type == 2 {
                    return Err(invalid!("raw segment reached a cleanup pass"));
                }
                match (pass_type, &mut mq, &mut raw) {
                    (0, _, Some(r)) => self.significance_propagation_pass_raw(r, out, plane, vsc),
                    (0, Some(m), _) => {
                        self.significance_propagation_pass(m, out, orientation, plane, vsc)
                    }
                    (1, _, Some(r)) => self.magnitude_refinement_pass_raw(r, out, plane),
                    (1, Some(m), _) => self.magnitude_refinement_pass(m, out, plane),
                    (2, Some(m), _) => {
                        // Segsym mismatch: decode anyway, don't fail.
                        self.cleanup_pass(m, out, orientation, plane, vsc, segsym);
                    }
                    _ => return Err(invalid!("raw segment reached a cleanup pass")),
                }
                if reset && !is_raw {
                    // T.800 Table A.19 0x02: reset context probabilities after every pass (raw
                    // passes have no contexts to reset).
                    self.ctxs = INIT_CTX;
                }
                pass_type += 1;
                if pass_type == 3 {
                    pass_type = 0;
                    plane -= 1;
                }
            }
        }
        Ok(Some((pass_type, plane)))
    }

    /// Decodes a code-block, then writes each coefficient one of three ways: with `out_f` None
    /// and `delta` 0, sign-magnitude into `out` and q into `qs`; with `qs` None and `out_f`
    /// None, the reversible coefficient into `out`; with `qs` None, the irreversible one (step
    /// `delta`) into `out_f`, `out` then holding scratch. Fusing the dequantization into the
    /// walk over significant samples saves a pass over the whole block.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &mut self,
        out: &mut [i32],
        qs: Option<&mut [i32]>,
        out_f: Option<&mut [f32]>,
        delta: f64,
        segs: &[Segment],
        width: usize,
        height: usize,
        orientation: usize,
        zero_bit_planes: i32,
        mb: i32,
        style: u8,
    ) -> Result<()> {
        // T.800 allows at most 1024 on a side and 4096 samples; the slices must hold them all.
        if width == 0 || height == 0 || width > 1024 || height > 1024 {
            return Err(invalid!(
                "code-block dimensions {width}x{height} out of range"
            ));
        }
        let n = width * height;
        if n > MAX_CODE_BLOCK_SAMPLES {
            return Err(invalid!(
                "code-block {width}x{height} exceeds the T.800 maximum of {MAX_CODE_BLOCK_SAMPLES} samples"
            ));
        }
        if out.len() < n
            || qs.as_deref().is_some_and(|q| q.len() < n)
            || out_f.as_deref().is_some_and(|f| f.len() < n)
        {
            return Err(invalid!(
                "code-block output buffers are shorter than {width}x{height} samples"
            ));
        }
        let st = self.decode(
            out,
            segs,
            width,
            height,
            orientation,
            zero_bit_planes,
            mb,
            style,
        )?;
        let mut qs = qs;
        if let Some(q) = qs.as_deref_mut() {
            q.fill(0);
        }
        let mut out_f = out_f;
        let Some((pass_type, plane)) = st else {
            // No coded bit-planes: an all-zero block, no bias.
            out.fill(0);
            if let Some(f) = out_f.as_deref_mut() {
                f.fill(0.0);
            }
            return Ok(());
        };
        // `plane` holds the last bit-plane index whose passes never fully ran (-1 if every
        // plane down to 0 was fully decoded): the count of undecoded low bit-planes is normally
        // that index plus one. Two exceptions need a per-sample count instead of one shared by
        // the whole block:
        //
        //   - pass_type == 2 (cleanup pending): significance-propagation and
        //     magnitude-refinement both ran for `plane`, so every currently significant sample
        //     already has its `plane` bit decided. Only cleanup-only candidates are still
        //     unresolved, and those stay at magnitude 0.
        //   - pass_type == 1 (magnitude-refinement pending): SPP ran for `plane` but MR did
        //     not. A sample that just became significant in that SPP has its `plane` bit
        //     decided, so only bits below `plane` are undecided: q = plane. A sample that was
        //     already significant before `plane` missed its MR visit for `plane`, so that bit
        //     is still undecided too: q = plane + 1. pi distinguishes the two.
        //
        // pass_type == 0 (SPP pending) needs no per-sample split.
        // q of a sample whose pi is clear, and of one whose pi is set.
        let (mut q_no, mut q_pi) = (plane + 1, plane + 1);
        match pass_type {
            2 => {
                q_no = plane;
                q_pi = plane;
            }
            1 => q_pi = plane,
            _ => {}
        }
        // The reversible dequantization's bias: 0.5 truncates to 0.
        let rev_bias = |q: i32| -> i32 { if q > 0 && q < 32 { 1 << (q - 1) } else { 0 } };
        // The irreversible dequantization's arithmetic exactly.
        let irr_bias = |q: i32| -> f32 {
            if q > 0 {
                (1u64 << (q as u32 & 63)) as f32 * 0.5
            } else {
                0.5
            }
        };
        let rb = [rev_bias(q_no), rev_bias(q_pi)];
        let fb = [irr_bias(q_no), irr_bias(q_pi)];
        let d = delta as f32;
        if let Some(qs) = qs {
            let mut y0 = 0;
            while y0 < height {
                let rows = (height - y0).min(4) as u32;
                let i0 = self.col(0, y0);
                for x in 0..width {
                    let f = self.flags[i0 + x];
                    for ci in 0..rows {
                        let sh = (3 * ci) & 15;
                        if (f >> sh) & T1_SIG == 0 {
                            continue;
                        }
                        let o = (y0 + ci as usize) * width + x;
                        out[o] = (out[o] as u32 | ((f >> (T1_CHI + sh)) & 1) << 31) as i32;
                        let q = if (f >> sh) & T1_PI != 0 { q_pi } else { q_no };
                        qs[o] = q.max(0);
                    }
                }
                y0 += 4;
            }
            return Ok(());
        }
        // Row by row and branch-free: whether a sample is significant is as unpredictable as
        // its sign, and insignificant ones come out 0. Bit 4 of a row's shifted flags is
        // T1_SIG, bit 21 T1_PI.
        for y in 0..height {
            let sh = (3 * (y & 3)) as u32;
            let c0 = self.col(0, y);
            let frow = &self.flags[c0..c0 + width];
            let orow = &mut out[y * width..y * width + width];
            match out_f.as_deref_mut() {
                None => crate::dsp::simd::dequant_rev(orow, frow, sh, rb),
                Some(of) => {
                    let frow2 = &mut of[y * width..y * width + width];
                    crate::dsp::simd::dequant(frow2, orow, frow, sh, fb, d);
                }
            }
        }
        Ok(())
    }
}

/// One termination segment's coded bytes and the number of coding passes it carries, in the
/// order tier-2 reported them (layer order, then segment order within a layer).
#[derive(Clone, Copy)]
pub struct Segment<'a> {
    pub data: &'a [u8],
    pub passes: usize,
}

/// Validates a code-block's dimensions and style before the caller slices its worker buffers
/// sized `MAX_CODE_BLOCK_SAMPLES`.
pub fn validate_code_block_dims(width: i64, height: i64, style: u8) -> Result<()> {
    if style & CB_HT_MASK != 0 {
        return Err(unsupported!(
            "code-block uses unsupported HT block coding (style 0x{style:02x})"
        ));
    }
    if width <= 0 || height <= 0 {
        return Err(invalid!(
            "code-block has non-positive dimensions {width}x{height}"
        ));
    }
    if width
        .checked_mul(height)
        .is_none_or(|n| n > MAX_CODE_BLOCK_SAMPLES as i64)
    {
        return Err(invalid!(
            "code-block {width}x{height} exceeds the T.800 maximum of {MAX_CODE_BLOCK_SAMPLES} samples"
        ));
    }
    Ok(())
}

/// EBCOT-decodes one code-block's coding-pass segments into sign-magnitude i32 coefficients:
/// bit 31 is the sign (1 = negative), bits 0..30 are the magnitude, already left-aligned to
/// bit-plane 0. `orientation` is the subband's band index (0=LL, 1=HL, 2=LH, 3=HH). `mb` is the
/// subband's Mb = guard bits + quantization exponent - 1 (T.800 E.1); `zero_bit_planes` comes
/// from the tier-2 packet header. The second result is one undecoded-low-bit-plane count per
/// coefficient, which dequantization needs to reproduce T.800's r=0.5 midpoint reconstruction
/// for truncated code-blocks.
pub fn decode_code_block(
    segs: &[Segment],
    width: i64,
    height: i64,
    orientation: usize,
    zero_bit_planes: i32,
    mb: i32,
    style: u8,
) -> Result<(Vec<i32>, Vec<i32>)> {
    validate_code_block_dims(width, height, style)?;
    let n = (width * height) as usize;
    let mut out = vec![0i32; n];
    let mut qs = vec![0i32; n];
    let mut t = State::default();
    t.finish(
        &mut out,
        Some(&mut qs),
        None,
        0.0,
        segs,
        width as usize,
        height as usize,
        orientation,
        zero_bit_planes,
        mb,
        style,
    )?;
    Ok((out, qs))
}

/// Go's FuzzMQ: 10000 decisions of the MQ decoder on contexts `i % 19`, then 10000 bits of the
/// raw decoder, both on `data`. Returns a hash of the bits, so nothing is optimised away.
pub(crate) fn mq_decisions(data: &[u8]) -> u32 {
    let mut mq = MqDecoder::new(data);
    let mut ctxs = INIT_CTX;
    let mut h = 0u32;
    for i in 0..10_000usize {
        h = h
            .wrapping_mul(3)
            .wrapping_add(mq.decode_bit(&mut ctxs, i % 19));
    }
    let mut raw = RawDecoder::new(data);
    for _ in 0..10_000 {
        h = h.wrapping_mul(3).wrapping_add(raw.read_bit());
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The raw MR pass must leave mu set: OpenJPEG's opj_t1_dec_refpass_step_raw sets T1_MU_THIS
    /// unconditionally, so a later MQ MR pass on the same coefficient must see it as
    /// already-refined (context 16).
    #[test]
    fn magnitude_refinement_pass_raw_sets_mu() {
        let mut st = State::default();
        st.reset(1, 1);
        let i = st.col(0, 0);
        let f = st.flags[i];
        st.flags[i] = st.sig(f, i, 0, 0, false);
        assert_eq!(
            st.flags[i] & T1_MU,
            0,
            "a newly significant sample must not have mu set"
        );
        let mut raw = RawDecoder::new(&[0x00]);
        st.magnitude_refinement_pass_raw(&mut raw, &mut [0i32; 1], 0);
        assert_ne!(st.flags[i] & T1_MU, 0);
    }

    /// A raw segment reaching a cleanup pass must error.
    #[test]
    fn raw_segment_cleanup_guard() {
        let d20 = [0u8; 20];
        let d10 = [0u8; 10];
        let segs = [
            Segment {
                data: &d20,
                passes: 10,
            }, // walks plane down into the raw window
            Segment {
                data: &d10,
                passes: 3,
            }, // raw: SPP, MRP, then a cleanup pass
        ];
        assert!(decode_code_block(&segs, 4, 4, 0, 0, 9, CB_BYPASS).is_err());
    }

    #[test]
    fn rejects_bad_dimensions_and_styles() {
        assert!(validate_code_block_dims(64, 64, 0).is_ok());
        assert!(validate_code_block_dims(0, 4, 0).is_err());
        assert!(validate_code_block_dims(65, 64, 0).is_err());
        assert!(validate_code_block_dims(4, 4, 0x40).is_err(), "HT");
        assert!(decode_code_block(&[], 4, 4, 0, 0, 9, 0x80).is_err());
    }

    /// A code-block with no passes decodes to zeros, whatever its bit-plane count.
    #[test]
    fn empty_block_is_zero() {
        let (out, qs) = decode_code_block(&[], 8, 8, 0, 0, 9, 0).unwrap();
        assert!(out.iter().all(|&v| v == 0) && qs.iter().all(|&v| v == 0));
    }

    /// More than 31 bit-planes is an error, not an overflow.
    #[test]
    fn too_many_bit_planes() {
        let d = [0u8; 8];
        let segs = [Segment {
            data: &d,
            passes: 1,
        }];
        assert!(decode_code_block(&segs, 4, 4, 0, 0, 32, 0).is_err());
    }

    /// Decoding arbitrary bytes under every style combination never panics.
    #[test]
    fn random_streams_do_not_panic() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for style in 0..0x40u8 {
            for orientation in 0..4 {
                let data: Vec<u8> = (0..300).map(|_| next() as u8).collect();
                let segs: Vec<Segment> = if style & (CB_TERM_ALL | CB_BYPASS) != 0 {
                    data.chunks(60)
                        .map(|c| Segment { data: c, passes: 3 })
                        .collect()
                } else {
                    vec![Segment {
                        data: &data,
                        passes: 40,
                    }]
                };
                let _ = decode_code_block(&segs, 16, 13, orientation, 1, 12, style);
            }
        }
    }

    /// A size whose product overflows i64 is an error, not a panic or a wrapped-to-zero pass.
    #[test]
    fn overflowing_dims_are_rejected() {
        assert!(validate_code_block_dims(1 << 33, 1 << 31, 0).is_err());
        assert!(validate_code_block_dims(i64::MAX, i64::MAX, 0).is_err());
        let segs = [Segment {
            data: &[0],
            passes: 1,
        }];
        assert!(decode_code_block(&segs, 1 << 33, 1 << 31, 0, 0, 5, 0).is_err());
    }

    /// `finish` checks the dimensions and the buffer lengths itself.
    #[test]
    fn finish_checks_dims_and_buffers() {
        let segs = [Segment {
            data: &[0xFF, 0xFF],
            passes: 1,
        }];
        let mut st = State::default();
        let mut short = [0i32; 4];
        assert!(
            st.finish(&mut short, None, None, 0.0, &segs, 64, 64, 0, 0, 5, 0)
                .is_err()
        );
        let mut out = vec![0i32; 4096];
        let mut short_qs = [0i32; 4];
        assert!(
            st.finish(
                &mut out,
                Some(&mut short_qs),
                None,
                0.0,
                &segs,
                64,
                64,
                0,
                0,
                5,
                0
            )
            .is_err()
        );
        let mut short_f = [0f32; 4];
        assert!(
            st.finish(
                &mut out,
                None,
                Some(&mut short_f),
                1.0,
                &segs,
                64,
                64,
                0,
                0,
                5,
                0
            )
            .is_err()
        );
        let mut big = vec![0i32; 1 << 22];
        for (w, h) in [(0, 4), (4, 0), (2048, 2), (1, 4097), (usize::MAX, 2)] {
            assert!(
                st.finish(&mut big, None, None, 0.0, &segs, w, h, 0, 0, 5, 0)
                    .is_err(),
                "{w}x{h}"
            );
        }
    }

    /// Go's FuzzMQ seeds: empty, short and 0xFF-heavy inputs decode 10000 decisions without a
    /// panic or a hang.
    #[test]
    fn mq_decisions_on_odd_inputs() {
        for data in [
            &[][..],
            &[0x00, 0x00, 0x00],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            &[0xFF, 0x90],
        ] {
            let _ = mq_decisions(data);
        }
    }
}
