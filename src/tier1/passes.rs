//! The MQ coding passes (significance propagation, magnitude refinement, cleanup), ported from
//! Go's generated `tier1_passes.go`. Each pass holds the MQ decoder's `a`, `c` and `ct` in
//! locals, open-codes the decision with `decide_w!` (a call would spill every live register) and
//! unrolls a full stripe's four rows, so every shift and neighbour update is a constant once the
//! compiler folds the literal row index.

use super::*;

/// DECODE (T.800 C.3.2, C.3.3) of the context word `$w` (read from slot `$cx` of `$ctxs`) on the
/// register copies `$a`, `$c`, `$ct`; the expression's value is the decoded bit.
///
/// Branch-free: `l` is set when `c` falls in the LPS sub-interval, `x` when the LPS is decoded
/// (the conditional exchange included). Without an exchange `a` stays >= 0x8000 > qe, so `x`,
/// and the shift `n` below, come out 0.
macro_rules! decide_w {
    ($mq:ident, $ctxs:ident, $a:ident, $c:ident, $ct:ident, $cx:expr, $w:expr) => {{
        let cx: usize = ($cx) & 31;
        let mut w: u64 = $w;
        let qe = w as u32;
        $a = $a.wrapping_sub(qe);
        let hi = (qe as u64) << 48;
        // The three picks below depend on the coded data, so a branch on them mispredicts often:
        // `select_unpredictable` asks for a conditional move, as Go's compiler emits for them.
        let lps = $c < hi;
        let l = lps as u32;
        $c = std::hint::select_unpredictable(lps, $c, $c.wrapping_sub(hi));
        let x = ($a < qe) as u32 ^ l;
        $a = std::hint::select_unpredictable(lps, qe, $a);
        let bit = ((w >> 48) as u32 & 1) ^ x;
        // Unconditional load: the successor is picked with a select, not a branch.
        let nw = MQ_W[((w >> (32 + 8 * (x & 1))) & 127) as usize];
        w = std::hint::select_unpredictable($a & 0x8000 == 0, nw, w);
        $ctxs[cx] = w;
        let n = ($a.leading_zeros().wrapping_sub(16)) & 15;
        if n > $ct {
            ($c, $ct) = $mq.fill_regs($c, $ct);
        }
        $a <<= n;
        $c <<= n;
        $ct -= n;
        bit
    }};
}

/// `decide_w!` on context `$cx`, loading its word.
macro_rules! decide {
    ($mq:ident, $ctxs:ident, $a:ident, $c:ident, $ct:ident, $cx:expr) => {{
        let cx: usize = ($cx) & 31;
        let w: u64 = $ctxs[cx];
        decide_w!($mq, $ctxs, $a, $c, $ct, cx, w)
    }};
}

/// `decide!` with a fast path for an MPS that needs no renormalization, which leaves the
/// context unchanged and so skips the store.
macro_rules! decide_fast {
    ($mq:ident, $ctxs:ident, $a:ident, $c:ident, $ct:ident, $cx:expr) => {{
        let cx: usize = ($cx) & 31;
        let w: u64 = $ctxs[cx];
        let qe = w as u32;
        let hi = (qe as u64) << 48;
        if $c >= hi && $a >= qe + 0x8000 {
            $a -= qe;
            $c -= hi;
            (w >> 48) as u32 & 1
        } else {
            decide_w!($mq, $ctxs, $a, $c, $ct, cx, w)
        }
    }};
}

/// Marks row `$ci` of column `$i` significant with sign `$s`: in `$f`, the column's word the
/// caller holds, and in the words of the columns and stripes around it. Row 0 does not mark the
/// stripe above under vertically causal mode `$vsc`.
macro_rules! sig_update {
    ($fl:ident, $fs:ident, $vsc:ident, $f:ident, $i:ident, $ci:ident, $s:ident) => {{
        let sh = (3 * $ci) & 15;
        $fl[$i - 1] |= 1 << 5 << sh;
        $fl[$i + 1] |= 1 << 3 << sh;
        if $ci == 0 && !$vsc {
            let n = $i - $fs;
            $fl[n] |= $s << 31 | 1 << 16;
            $fl[n - 1] |= 1 << 17;
            $fl[n + 1] |= 1 << 15;
        }
        if $ci == 3 {
            let n = $i + $fs;
            $fl[n] |= $s << 18 | 1 << 1;
            $fl[n - 1] |= 1 << 2;
            $fl[n + 1] |= 1;
        }
        $f |= ($s << T1_CHI | T1_SIG) << sh;
    }};
}

impl State {
    /// T.800 D.3.1: for every not-yet-significant sample with at least one significant
    /// neighbour, decode a zero-coding bit and, on a hit, a sign. Processed in 4-row stripes,
    /// column-major within a stripe (T.800 Figure D.6). Magnitude bits accumulate in `out`.
    pub(super) fn significance_propagation_pass(
        &mut self,
        mq: &mut MqDecoder,
        out: &mut [i32],
        orientation: usize,
        plane: i32,
        vsc: bool,
    ) {
        let l = luts();
        let zc = &l.zc[orientation & 3];
        let sc = &l.sc;
        let (w, h, fs) = (self.w, self.h, self.fs);
        let fl = &mut self.flags[..];
        let ctxs = &mut self.ctxs;
        let one = 1i32 << plane;
        let (mut a, mut c, mut ct) = (mq.a, mq.c, mq.ct);
        macro_rules! step {
            ($f:ident, $i:ident, $o:ident, $ci:expr) => {{
                let ci: u32 = $ci;
                let sh = (3 * ci) & 15;
                if ($f >> sh) & (T1_SIG | T1_PI) == 0 && ($f >> sh) & T1_SIG_NB != 0 {
                    let cx = (zc[(($f >> sh) & 0x1FF) as usize] & 31) as usize;
                    // The sign's context word is loaded before the zero-coding decision.
                    let e = sc[sc_index($f, fl[$i - 1], fl[$i + 1], ci)];
                    let sx = ((e >> 1) & 31) as usize;
                    let sw = ctxs[sx];
                    let bit = decide!(mq, ctxs, a, c, ct, cx);
                    if bit != 0 {
                        let sbit = decide_w!(mq, ctxs, a, c, ct, sx, sw);
                        let s = sbit ^ (e & 1) as u32;
                        sig_update!(fl, fs, vsc, $f, $i, ci, s);
                        out[$o + ci as usize * w] |= one;
                    }
                    $f |= T1_PI << sh;
                }
            }};
        }
        let mut y0 = 0;
        while y0 < h {
            let rows = (h - y0).min(4) as u32;
            let i0 = (y0 / 4 + 1) * fs + 1;
            for x in 0..w {
                let i = i0 + x;
                let o = y0 * w + x;
                let mut f = fl[i];
                // f>>17 moves each row's pi onto its sigma bit.
                if f == 0 || (f | f >> 17) & T1_SIG_COL == T1_SIG_COL {
                    continue;
                }
                if rows == 4 {
                    step!(f, i, o, 0);
                    step!(f, i, o, 1);
                    step!(f, i, o, 2);
                    step!(f, i, o, 3);
                } else {
                    for ci in 0..rows {
                        step!(f, i, o, ci);
                    }
                }
                fl[i] = f;
            }
            y0 += 4;
        }
        mq.a = a;
        mq.c = c;
        mq.ct = ct;
    }

    /// T.800 D.3.3: every sample already significant, but not newly so this plane, contributes
    /// one more magnitude bit.
    pub(super) fn magnitude_refinement_pass(
        &mut self,
        mq: &mut MqDecoder,
        out: &mut [i32],
        plane: i32,
    ) {
        let (w, h, fs) = (self.w, self.h, self.fs);
        let fl = &mut self.flags[..];
        let ctxs = &mut self.ctxs;
        let (mut a, mut c, mut ct) = (mq.a, mq.c, mq.ct);
        macro_rules! step {
            ($f:ident, $o:ident, $ci:expr) => {{
                let ci: u32 = $ci;
                let sh = (3 * ci) & 15;
                if ($f >> sh) & (T1_SIG | T1_PI) == T1_SIG {
                    // 14/15 are swapped vs T.800 Table D-2 (pdf.js labelling); a consistent
                    // bijection, decode-identical.
                    let mut cx = 16usize;
                    if ($f >> sh) & T1_MU == 0 {
                        cx = 15;
                        if ($f >> sh) & T1_SIG_NB != 0 {
                            cx = 14;
                        }
                    }
                    let bit = decide!(mq, ctxs, a, c, ct, cx);
                    out[$o + ci as usize * w] |= (bit as i32) << (plane & 31);
                    $f |= T1_MU << sh;
                }
            }};
        }
        let mut y0 = 0;
        while y0 < h {
            let rows = (h - y0).min(4) as u32;
            let i0 = (y0 / 4 + 1) * fs + 1;
            for x in 0..w {
                let i = i0 + x;
                let o = y0 * w + x;
                let mut f = fl[i];
                if f & T1_SIG_COL & !(f >> 17) == 0 {
                    continue; // no row significant and not yet visited
                }
                if rows == 4 {
                    step!(f, o, 0);
                    step!(f, o, 1);
                    step!(f, o, 2);
                    step!(f, o, 3);
                } else {
                    for ci in 0..rows {
                        step!(f, o, ci);
                    }
                }
                fl[i] = f;
            }
            y0 += 4;
        }
        mq.a = a;
        mq.c = c;
        mq.ct = ct;
    }

    /// T.800 D.3.4: codes everything the first two passes skipped, then clears pi for the next
    /// bit-plane. A full 4-row stripe column with no significance anywhere in its neighbourhood
    /// (a zero flags word) is coded with one run-length decision (context 17) instead of four
    /// zero-coding decisions; a hit is followed by a 2-bit uniform-context (18) row index, MSB
    /// first, that pinpoints the first significant sample without spending a further ZC bit on
    /// it. A partial last stripe never uses run-length coding. Returns true if `segsym` is set
    /// and the 4-bit marker decoded at the end doesn't match the expected 0b1010: per OpenJPEG
    /// that is a decode-and-discard signal, not a hard error.
    pub(super) fn cleanup_pass(
        &mut self,
        mq: &mut MqDecoder,
        out: &mut [i32],
        orientation: usize,
        plane: i32,
        vsc: bool,
        segsym: bool,
    ) -> bool {
        let l = luts();
        let zc = &l.zc[orientation & 3];
        let sc = &l.sc;
        let (w, h, fs) = (self.w, self.h, self.fs);
        let fl = &mut self.flags[..];
        let ctxs = &mut self.ctxs;
        let one = 1i32 << plane;
        let (mut a, mut c, mut ct) = (mq.a, mq.c, mq.ct);
        // Decodes row `$ci`'s sign after a zero-coding hit and marks it significant.
        macro_rules! sign {
            ($f:ident, $i:ident, $o:ident, $ci:expr) => {{
                let ci: u32 = $ci;
                let e = sc[sc_index($f, fl[$i - 1], fl[$i + 1], ci)];
                let bit = decide!(mq, ctxs, a, c, ct, (e >> 1) as usize);
                let s = bit ^ (e & 1) as u32;
                sig_update!(fl, fs, vsc, $f, $i, ci, s);
                out[$o + ci as usize * w] |= one;
            }};
        }
        macro_rules! step {
            ($f:ident, $i:ident, $o:ident, $ci:expr) => {{
                let ci: u32 = $ci;
                let sh = (3 * ci) & 15;
                if ($f >> sh) & (T1_SIG | T1_PI) == 0 {
                    let cx = (zc[(($f >> sh) & 0x1FF) as usize] & 31) as usize;
                    let bit = decide_fast!(mq, ctxs, a, c, ct, cx);
                    if bit != 0 {
                        sign!($f, $i, $o, ci);
                    }
                }
            }};
        }
        let mut y0 = 0;
        while y0 < h {
            let rows = (h - y0).min(4) as u32;
            let i0 = (y0 / 4 + 1) * fs + 1;
            for x in 0..w {
                let i = i0 + x;
                let o = y0 * w + x;
                let mut f = fl[i];
                let mut ci = 0u32;
                if rows == 4 {
                    if f != 0 {
                        if (f | f >> 17) & T1_SIG_COL != T1_SIG_COL {
                            step!(f, i, o, 0);
                            step!(f, i, o, 1);
                            step!(f, i, o, 2);
                            step!(f, i, o, 3);
                        }
                        fl[i] = f & !T1_PI_COL;
                        continue;
                    }
                    // MPS without renorm: context unchanged.
                    let rw = ctxs[CTX_RUN_LENGTH];
                    if (rw >> 48) & 1 == 0 {
                        let qe = rw as u32;
                        let hi = (qe as u64) << 48;
                        if c >= hi && a >= qe + 0x8000 {
                            a -= qe;
                            c -= hi;
                            continue;
                        }
                    }
                    let bit = decide!(mq, ctxs, a, c, ct, CTX_RUN_LENGTH);
                    if bit == 0 {
                        continue; // whole column stays insignificant
                    }
                    for _ in 0..2 {
                        let b = decide!(mq, ctxs, a, c, ct, CTX_UNIFORM);
                        ci = (ci << 1) | b;
                    }
                    sign!(f, i, o, ci);
                    ci += 1;
                }
                while ci < rows {
                    step!(f, i, o, ci);
                    ci += 1;
                }
                fl[i] = f & !T1_PI_COL;
            }
            y0 += 4;
        }
        mq.a = a;
        mq.c = c;
        mq.ct = ct;
        if segsym {
            let mut v = 0;
            for _ in 0..4 {
                v = (v << 1) | mq.decode_bit(ctxs, CTX_UNIFORM);
            }
            return v != 0x0a;
        }
        false
    }
}
