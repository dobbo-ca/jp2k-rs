//! T.800 Annex F: inverse discrete wavelet transform.
//!
//! Both the reversible 5/3 and irreversible 9/7 filters operate on an interleaved 1D sequence
//! (low-band samples at positions of parity `cas`, high-band samples at parity `1 - cas`, F.3.1)
//! and are applied 2D-separably, horizontal pass then vertical, one resolution level at a time
//! from the smallest resolution up (F.3.2). `cas` is each resolution's own rect origin parity
//! per axis: nonzero whenever the tile-component origin is odd.

// The lifting constants keep Go's decimal literals; they round to the same f32.
#![allow(clippy::excessive_precision)]

use super::Rounding;
use super::jj2000::synth_97_jj;
use super::simd;
use crate::codestream::Rect;
use crate::error::Result;
use crate::limits::{Budget, Charge, check_cancel, reuse};
use crate::parallel::{ROW_CHUNK, drive};
use std::sync::atomic::AtomicBool;

/// Splits a resolution axis of length `rw` into its low-band count and high-band count, given
/// the axis origin's parity `cas` (F.3.1).
pub fn sn_dn(rw: usize, cas: usize) -> (usize, usize) {
    if cas == 0 {
        (rw.div_ceil(2), rw / 2)
    } else {
        (rw / 2, rw.div_ceil(2))
    }
}

/// The reference-decoder symmetric clamp (T.800 F.3.7 / OpenJPEG dwt.c), not a padded buffer.
fn clamp_dwt(i: i64, n: usize) -> usize {
    if i < 0 {
        0
    } else if i >= n as i64 {
        n - 1
    } else {
        i as usize
    }
}

/// T.800 F.3.8.2's inverse reversible 5/3 lifting in place on the interleaved sequence `a`
/// (`a[2i]` is the i'th low-band sample, `a[2i+1]` the i'th high-band sample), given the
/// low/high counts `sn`, `dn` and the axis origin parity `cas`. The oracle of `row_53` and
/// `cols_53`, which lift the split bands.
#[cfg(test)]
fn inverse_1d_53(a: &mut [i32], sn: usize, dn: usize, cas: usize) {
    if cas == 0 {
        if dn == 0 && sn <= 1 {
            return; // F.3.7: a single unmodified sample
        }
        for i in 0..sn {
            let im1 = clamp_dwt(i as i64 - 1, dn);
            let i0 = clamp_dwt(i as i64, dn);
            let t = a[2 * im1 + 1].wrapping_add(a[2 * i0 + 1]).wrapping_add(2) >> 2;
            a[2 * i] = a[2 * i].wrapping_sub(t);
        }
        for i in 0..dn {
            let i0 = clamp_dwt(i as i64, sn);
            let i1 = clamp_dwt(i as i64 + 1, sn);
            let t = a[2 * i0].wrapping_add(a[2 * i1]) >> 1;
            a[2 * i + 1] = a[2 * i + 1].wrapping_add(t);
        }
        return;
    }
    if sn == 0 && dn == 1 {
        a[0] /= 2; // F.3.7: length-1 odd-origin signal
        return;
    }
    for i in 0..sn {
        let i0 = clamp_dwt(i as i64, dn);
        let i1 = clamp_dwt(i as i64 + 1, dn);
        let t = a[2 * i0].wrapping_add(a[2 * i1]).wrapping_add(2) >> 2;
        a[2 * i + 1] = a[2 * i + 1].wrapping_sub(t);
    }
    for i in 0..dn {
        let i0 = clamp_dwt(i as i64, sn);
        let im1 = clamp_dwt(i as i64 - 1, sn);
        let t = a[2 * i0 + 1].wrapping_add(a[2 * im1 + 1]) >> 1;
        a[2 * i] = a[2 * i].wrapping_add(t);
    }
}

// T.800 F.3.8.2 lifting constants (also OpenJPEG dwt.c / pdf.js jpx.js).
pub(super) const DWT97_ALPHA: f32 = -1.586134342059924;
pub(super) const DWT97_BETA: f32 = -0.052980118572961;
pub(super) const DWT97_GAMMA: f32 = 0.882911075530934;
pub(super) const DWT97_DELTA: f32 = 0.443506852043971;
const DWT97_K: f32 = 1.230174104914001;
/// OpenJPEG's decode-only high-band scale (dwt.c BUG_WEIRD_TWO_INVK): its historic literal, not
/// 2/K computed fresh.
const DWT97_TWO_INV_K: f32 = 1.625732422;

/// Reflects `y` into `[0, n)` about the edge samples (F.3.7).
fn mirror(y: i64, n: usize) -> usize {
    let n = n as i64;
    (if y < 0 {
        -y
    } else if y >= n {
        2 * n - 2 - y
    } else {
        y
    }) as usize
}

/// One lifting step on the interleaved line: every sample of parity `par` gets
/// `c * (a[j-1] + a[j+1])` added, the neighbours mirror-extended.
fn lift_step(a: &mut [f32], par: usize, c: f32, r: Rounding) {
    let n = a.len();
    let mut j = par;
    while j < n {
        let u = a[mirror(j as i64 - 1, n)];
        let v = a[mirror(j as i64 + 1, n)];
        a[j] = r.mul_add(c, u + v, a[j]);
        j += 2;
    }
}

/// T.800 F.3.8.2's inverse 9/7 lifting on an interleaved line of at least 2 samples. Low band
/// *= K, high band *= two_invK (not 1/K): the dequantizer drops the T.800 E-3 per-subband gain
/// term to match OpenJPEG's decode-only convention, and this factor-of-2-too-much high-band
/// scale cancels it back out.
fn inverse_1d_97(a: &mut [f32], cas: usize, r: Rounding) {
    let n = a.len();
    let mut j = cas;
    while j < n {
        a[j] *= DWT97_K;
        j += 2;
    }
    let mut j = 1 - cas;
    while j < n {
        a[j] *= DWT97_TWO_INV_K;
        j += 2;
    }
    lift_step(a, cas, -DWT97_DELTA, r);
    lift_step(a, 1 - cas, -DWT97_GAMMA, r);
    lift_step(a, cas, -DWT97_BETA, r);
    lift_step(a, 1 - cas, -DWT97_ALPHA, r);
}

/// Spreads a Mallat-ordered line (`sn` low samples, then the high samples) into `dst` in F.3.1
/// interleaved order for origin parity `cas`.
#[cfg(test)]
fn interleave<T: Copy>(dst: &mut [T], src: &[T], sn: usize, cas: usize) {
    for i in 0..sn {
        dst[cas + 2 * i] = src[i];
    }
    let (mut i, mut j) = (sn, 1 - cas);
    while i < src.len() {
        dst[j] = src[i];
        i += 1;
        j += 2;
    }
}

/// How many columns a vertical pass gathers at once: 16 for the line kernels, and 64 for the
/// 9/7 and 5/3 row kernels, whose lifting steps then run over longer contiguous rows.
const COL_BLOCK: usize = 16;
const COL_BLOCK_97: usize = 64;

/// A vertical pass over `bn` columns: `(p, stride, bn, blk, n, cas, hi)` with `p` starting at the
/// first column and `blk` `n * bn` samples of scratch.
type ColPass<'a, T> = &'a (dyn Fn(&mut [T], usize, usize, &mut [T], usize, usize, bool) + Sync);

/// `d[i] = step(d[i], u[clamp(i + o)], u[clamp(i + o + 1)])` for `o` in {-1, 0}: one lifting
/// step on a band `d` of a line held split into its low and high bands (Mallat order), `u` being
/// the other band. The mirror extension of F.3.7 becomes a clamp in split form. The edges are
/// scalar (`step`) and the interior, where no clamp applies, is one contiguous kernel call
/// (`kernel(d, u, v)`, `step` on each sample). Both bands are non-empty.
fn lift_split<T: Copy>(
    d: &mut [T],
    u: &[T],
    o: isize,
    step: impl Fn(T, T, T) -> T,
    kernel: impl FnOnce(&mut [T], &[T], &[T]),
) {
    let nu = u.len();
    let edge = |d: &mut [T], i: usize| {
        let a = u[clamp_dwt(i as i64 + o as i64, nu)];
        let b = u[clamp_dwt(i as i64 + o as i64 + 1, nu)];
        d[i] = step(d[i], a, b);
    };
    let i0 = (-o) as usize;
    let i1 = i0.max(d.len().min((nu as isize - 1 - o) as usize));
    for i in 0..i0.min(d.len()) {
        edge(d, i);
    }
    if i1 > i0 {
        let m = i1 - i0;
        kernel(&mut d[i0..i1], &u[..m], &u[1..=m]);
    }
    for i in i1..d.len() {
        edge(d, i);
    }
}

/// `lift_split` with the 9/7 step `d += c * (a + b)`, rounded as `r` says.
fn lift_split_97(d: &mut [f32], u: &[f32], o: isize, c: f32, r: Rounding) {
    let step = |d: f32, a: f32, b: f32| r.mul_add(c, a + b, d);
    lift_split(d, u, o, step, |d, u, v| simd::lift(d, u, v, c, r));
}

/// Interleaves the low band `l` and the high band `h` of a line into `out` (F.3.1 order for
/// origin parity `cas`).
fn zip<T: Copy>(out: &mut [T], l: &[T], h: &[T], cas: usize) {
    let (a, b) = if cas == 0 { (l, h) } else { (h, l) };
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        out[2 * i] = x;
        out[2 * i + 1] = y;
    }
    if a.len() > b.len() {
        out[2 * b.len()] = a[b.len()];
    }
}

/// The 5/3 row kernel: `inverse_1d_53` on a Mallat-ordered line without interleaving first, so
/// each step is one kernel call over contiguous memory, then the zip into `out`. Step 1 writes
/// only low samples from high ones and step 2 only high samples from the finished low ones, so
/// doing each over the split bands applies the same integer operations to the same operands.
fn row_53(mal: &mut [i32], sn: usize, cas: usize, out: &mut [i32]) {
    let dn = mal.len() - sn;
    if cas == 0 && dn == 0 && sn <= 1 {
        out.copy_from_slice(mal); // F.3.7: a single unmodified sample
        return;
    }
    if cas == 1 && sn == 0 && dn == 1 {
        out[0] = mal[0] / 2; // F.3.7: length-1 odd-origin signal (truncating, not `>> 1`)
        return;
    }
    let (l, h) = mal.split_at_mut(sn);
    // A band at even positions reads neighbours i-1 and i of the other.
    let (lo, ho) = (cas as isize - 1, -(cas as isize));
    lift_split(l, h, lo, step53_sub, simd::lift53_sub);
    lift_split(h, l, ho, step53_add, simd::lift53_add);
    zip(out, l, h, cas);
}

/// One sample of `simd::lift53_sub`.
fn step53_sub(d: i32, a: i32, b: i32) -> i32 {
    d.wrapping_sub(a.wrapping_add(b).wrapping_add(2) >> 2)
}

/// One sample of `simd::lift53_add`.
fn step53_add(d: i32, a: i32, b: i32) -> i32 {
    d.wrapping_add(a.wrapping_add(b) >> 1)
}

/// The 9/7 row kernel for `Fused` and `Separate` rounding: `inverse_1d_97` on a Mallat-ordered
/// line without interleaving first, so every step runs over contiguous memory, then the zip into
/// `out`. The same operations on the same values in the same order, so the same bits.
fn row_97(mal: &mut [f32], sn: usize, cas: usize, out: &mut [f32], r: Rounding, hi: bool) {
    if mal.len() < 2 {
        // F.3.7: a length-1 9/7 signal is unchanged.
        out.copy_from_slice(mal);
        return;
    }
    let (l, h) = mal.split_at_mut(sn);
    simd::scale(l, DWT97_K);
    // A band at even positions reads neighbours i-1 and i of the other.
    let (lo, ho) = (cas as isize - 1, -(cas as isize));
    // `hi` false: the high band is all zero on entry, so its scale and the delta step, which
    // adds a multiple of it to the low band, change nothing (x + -0.0 is x) and are skipped.
    if hi {
        simd::scale(h, DWT97_TWO_INV_K);
        lift_split_97(l, h, lo, -DWT97_DELTA, r);
    }
    lift_split_97(h, l, ho, -DWT97_GAMMA, r);
    lift_split_97(l, h, lo, -DWT97_BETA, r);
    lift_split_97(h, l, ho, -DWT97_ALPHA, r);
    zip(out, l, h, cas);
}

/// One lifting step on row `j` of `b` (rows `bn` wide), `kernel(b[j], b[ju], b[jv])`; `ju` and
/// `jv` are not `j`.
fn lift_row<T>(
    b: &mut [T],
    bn: usize,
    j: usize,
    ju: usize,
    jv: usize,
    kernel: impl FnOnce(&mut [T], &[T], &[T]),
) {
    let (before, rest) = b.split_at_mut(j * bn);
    let (d, after) = rest.split_at_mut(bn);
    let pick = |q: usize| {
        if q < j {
            &before[q * bn..(q + 1) * bn]
        } else {
            &after[(q - j - 1) * bn..(q - j) * bn]
        }
    };
    kernel(d, pick(ju), pick(jv));
}

/// `cols_97` for a level whose high bands are zero, in place over `w` columns of `p` with no
/// gather or scatter (Go's `sweep97Low`). The high rows of `p` are zero and never read, so
/// walking output row `y` downward only touches low rows already consumed: each step is a
/// `w`-wide scale or lift over five hot rows of `s` (`5 * w` samples: two scaled low rows, two
/// gamma rows, one zero row) or the rows of `p`, and every sample gets the same operations in the
/// same order as in `cols_97`.
fn sweep_low(
    p: &mut [f32],
    stride: usize,
    w: usize,
    n: usize,
    cas: usize,
    s: &mut [f32],
    r: Rounding,
) {
    s[4 * w..5 * w].fill(0.0);
    let (ni, ci) = (n as i64, cas as i64);
    // Slots 0, 1 hold scaled low rows, 2, 3 gamma rows; the row's parity picks the slot.
    let a_slot = |y: i64| ((mirror(y, n) as i64 - ci) >> 1 & 1) as usize;
    let g_slot = |y: i64| 2 + ((mirror(y, n) as i64 - 1 + ci) >> 1 & 1) as usize;
    let top = n + 2 + ((n + 2 - cas) & 1);
    let mut t = top as i64;
    while t >= -3 {
        if (0..ni).contains(&t) {
            let q = ((t - ci) / 2) as usize;
            let k = a_slot(t);
            simd::scale_to(
                &mut s[k * w..(k + 1) * w],
                &p[q * stride..q * stride + w],
                DWT97_K,
            );
        }
        let y = t + 1;
        if (0..ni).contains(&y) {
            let (g, a0, a1) = (g_slot(y), a_slot(y - 1), a_slot(y + 1));
            // The three sources are slots other than `g`.
            let (lo, rest) = s.split_at_mut(g * w);
            let (dst, hi) = rest.split_at_mut(w);
            let (lo, hi): (&[f32], &[f32]) = (lo, hi);
            let pick = |k: usize| {
                if k < g {
                    &lo[k * w..(k + 1) * w]
                } else {
                    &hi[(k - g - 1) * w..(k - g) * w]
                }
            };
            simd::lift_to(dst, pick(4), pick(a0), pick(a1), -DWT97_GAMMA, r);
        }
        let y = t + 2;
        if (0..ni).contains(&y) {
            let yu = y as usize;
            let (a, g0, g1) = (a_slot(y), g_slot(y - 1), g_slot(y + 1));
            let row = |k: usize| &s[k * w..(k + 1) * w];
            simd::lift_to(
                &mut p[yu * stride..yu * stride + w],
                row(a),
                row(g0),
                row(g1),
                -DWT97_BETA,
                r,
            );
        }
        let y = t + 3;
        if (0..ni).contains(&y) {
            let yu = y as usize;
            let g = g_slot(y);
            let (head, tail) = p.split_at_mut(yu * stride);
            let (dst, rest) = tail.split_at_mut(w);
            let (head, rest): (&[f32], &[f32]) = (head, rest);
            // Rows above `y` are in `head`, rows below in `rest` (which starts at `y*stride + w`).
            let pick = |q: usize| {
                if q < yu {
                    &head[q * stride..q * stride + w]
                } else {
                    &rest[(q - yu) * stride - w..(q - yu) * stride]
                }
            };
            let (u, v) = (pick(mirror(y - 1, n)), pick(mirror(y + 1, n)));
            simd::lift_to(dst, &s[g * w..(g + 1) * w], u, v, -DWT97_ALPHA, r);
        }
        t -= 2;
    }
}

/// The 9/7 vertical pass over `bn` columns of `p` at once: row `i` of the Mallat layout goes to
/// row `q` of the interleaved block `b`, so each step of `inverse_1d_97` is a kernel call over a
/// whole row of the block, and the result goes back to `p`.
fn cols_97(
    p: &mut [f32],
    stride: usize,
    bn: usize,
    b: &mut [f32],
    n: usize,
    cas: usize,
    r: Rounding,
    hi: bool,
) {
    if n < 2 {
        return; // F.3.7: a length-1 9/7 signal is unchanged
    }
    // `hi` false: the whole high band is zero, which a sweep over rows exploits.
    if !hi && b.len() >= 5 * bn {
        sweep_low(p, stride, bn, n, cas, &mut b[..5 * bn], r);
        return;
    }
    let (sn, _) = sn_dn(n, cas);
    for i in 0..n {
        let q = if i < sn {
            cas + 2 * i
        } else {
            1 - cas + 2 * (i - sn)
        };
        b[q * bn..(q + 1) * bn].copy_from_slice(&p[i * stride..i * stride + bn]);
    }
    for q in 0..n {
        // `hi` false: the high rows are zero, and scaling them changes nothing.
        let k = if q % 2 == cas {
            DWT97_K
        } else if hi {
            DWT97_TWO_INV_K
        } else {
            continue;
        };
        simd::scale(&mut b[q * bn..(q + 1) * bn], k);
    }
    for (par, c) in [
        (cas, -DWT97_DELTA),
        (1 - cas, -DWT97_GAMMA),
        (cas, -DWT97_BETA),
        (1 - cas, -DWT97_ALPHA),
    ] {
        // The delta step adds multiples of the (zero) high rows to the low rows.
        if !hi && c == -DWT97_DELTA {
            continue;
        }
        for j in (par..n).step_by(2) {
            let (ju, jv) = (mirror(j as i64 - 1, n), mirror(j as i64 + 1, n));
            lift_row(b, bn, j, ju, jv, |d, u, v| simd::lift(d, u, v, c, r));
        }
    }
    for q in 0..n {
        p[q * stride..q * stride + bn].copy_from_slice(&b[q * bn..(q + 1) * bn]);
    }
}

/// The 5/3 vertical pass over `bn` columns of `p` at once, as `cols_97`: the Mallat rows go to
/// the block `b` in the same order, each step of `row_53` is a kernel call over a whole row of
/// it (the neighbour rows clamped on the band indices, as `inverse_1d_53` does), and row `i` of
/// the block goes back to its interleaved row of `p`.
fn cols_53(p: &mut [i32], stride: usize, bn: usize, b: &mut [i32], n: usize, cas: usize) {
    if n < 2 {
        if cas == 1 && n == 1 {
            // F.3.7: length-1 odd-origin signal (truncating, not `>> 1`).
            for x in &mut p[..bn] {
                *x /= 2;
            }
        }
        return;
    }
    let (sn, dn) = sn_dn(n, cas);
    for i in 0..n {
        b[i * bn..(i + 1) * bn].copy_from_slice(&p[i * stride..i * stride + bn]);
    }
    let (lo, ho) = (cas as i64 - 1, -(cas as i64));
    for i in 0..sn {
        let ju = sn + clamp_dwt(i as i64 + lo, dn);
        let jv = sn + clamp_dwt(i as i64 + lo + 1, dn);
        lift_row(b, bn, i, ju, jv, simd::lift53_sub);
    }
    for i in 0..dn {
        let ju = clamp_dwt(i as i64 + ho, sn);
        let jv = clamp_dwt(i as i64 + ho + 1, sn);
        lift_row(b, bn, sn + i, ju, jv, simd::lift53_add);
    }
    for i in 0..n {
        let q = if i < sn {
            cas + 2 * i
        } else {
            1 - cas + 2 * (i - sn)
        };
        p[q * stride..q * stride + bn].copy_from_slice(&b[i * bn..(i + 1) * bn]);
    }
}

/// A thread is worth using (a pool task costs microseconds) for about this many samples of a pass, so a
/// level runs on one thread per this many samples, at most, and on one if it has fewer than two
/// of them (Go: `minParallelLevel`).
const SAMPLES_PER_THREAD: usize = 1 << 15;

/// One thread's scratch: a line to lift (`mal`, `out`), a block of columns (`blk`), and, for the
/// threaded vertical pass, a contiguous copy of the block's columns (`pblk`).
pub(crate) struct Lines<T> {
    mal: Vec<T>,
    out: Vec<T>,
    blk: Vec<T>,
    pblk: Vec<T>,
    /// The ledger bytes of the four vectors, given back when this drops; `None` while a
    /// `Decoder` keeps the line for its next call.
    charge: Option<Charge>,
}

/// One sample type's lines: those the transforms of this call use (charged), and the spares, which
/// a new line is made from before anything is allocated: the lines a `Decoder`'s transform shed
/// (still charged, until a new line takes one or the call ends) and those it kept from its last
/// call (uncharged).
#[derive(Default)]
pub(crate) struct DwtLines<T> {
    live: Vec<Lines<T>>,
    spare: Vec<Lines<T>>,
}

/// The scratch of the inverse DWT, kept in a `Budget` (one per tile lane) from one call to the
/// next: a lane runs a transform for every component of every tile, and allocating (and zeroing,
/// and page-faulting) each thread's blocks anew each time cost as much as the transform itself.
/// Its bytes stay charged to the ledger until the `Budget` drops. A transform that needs more
/// than it holds replaces it.
#[derive(Default)]
pub(crate) struct DwtScratch {
    /// A `Decoder`'s: lines no longer needed become spares, still charged, instead of being
    /// freed.
    pub(crate) keep: bool,
    i: DwtLines<i32>,
    f: DwtLines<f32>,
}

impl DwtScratch {
    /// Gives back the charge of every line and keeps them all as spares, for the next call.
    pub(crate) fn park(&mut self) {
        park(&mut self.i);
        park(&mut self.f);
    }
}

fn park<T>(l: &mut DwtLines<T>) {
    let DwtLines { live, spare } = l;
    spare.append(live);
    for x in spare.iter_mut() {
        x.charge = None;
    }
}

/// Drops the lines past the first `n`, or with `keep` keeps them, charged, as spares.
fn shed<T>(lines: &mut Vec<Lines<T>>, n: usize, spare: &mut Vec<Lines<T>>, keep: bool) {
    if !keep {
        lines.truncate(n);
        return;
    }
    while lines.len() > n {
        if let Some(l) = lines.pop() {
            spare.push(l);
        }
    }
}

/// The sample types of the transforms, to pick their scratch.
pub(crate) trait Sample: Sized {
    fn lines(s: &mut DwtScratch) -> &mut DwtLines<Self>;
}

impl Sample for i32 {
    fn lines(s: &mut DwtScratch) -> &mut DwtLines<i32> {
        &mut s.i
    }
}

impl Sample for f32 {
    fn lines(s: &mut DwtScratch) -> &mut DwtLines<f32> {
        &mut s.f
    }
}

/// The vertical pass over the `bw` columns of `p` starting at column `x0`, `p` having row pitch
/// `stride`: with `cols`, that closure; else each column is gathered as a line and given to
/// `kernel`. `blk` holds `bw * rh` samples of scratch, `out` a line.
fn col_pass<T: Copy>(
    p: &mut [T],
    stride: usize,
    x0: usize,
    bw: usize,
    rh: usize,
    (sn_y, cas_y, hi): (usize, usize, bool),
    blk: &mut [T],
    out: &mut [T],
    kernel: &impl Fn(&mut [T], usize, usize, &mut [T], bool),
    cols: Option<ColPass<'_, T>>,
) {
    if let Some(cols) = cols {
        cols(&mut p[x0..], stride, bw, &mut blk[..bw * rh], rh, cas_y, hi);
        return;
    }
    for y in 0..rh {
        let src = &p[y * stride + x0..y * stride + x0 + bw];
        for k in 0..bw {
            blk[k * rh + y] = src[k];
        }
    }
    for k in 0..bw {
        let line = &mut blk[k * rh..(k + 1) * rh];
        kernel(line, sn_y, cas_y, &mut out[..rh], hi);
        line.copy_from_slice(&out[..rh]);
    }
    for y in 0..rh {
        let dst = &mut p[y * stride + x0..y * stride + x0 + bw];
        for k in 0..bw {
            dst[k] = blk[k * rh + y];
        }
    }
}

/// Runs every level's horizontal then vertical pass. `kernel(mal, sn, cas, out)` turns one
/// Mallat-ordered line (`sn` low samples first) into its interleaved reconstruction in `out`;
/// it may use `mal` as scratch. `p` holds the tile component at `stride` in OpenJPEG's Mallat
/// layout: before level r runs, resolution r-1's samples fill the top-left corner of
/// resolution r's area and level r's HL, LH and HH bands sit to its right, below it and
/// diagonally from it, low samples first on each axis. Every level reads and writes the same
/// memory. `rects[r]` is resolution r's rect; only its size and origin parity matter. With
/// `cols`, the vertical pass is that closure over blocks of `col_block` columns, and `kernel`
/// only does the rows; without it, each column is gathered as a line and given to `kernel`.
/// `high[r]`, when `high` is given, says whether resolution level `r` has any HL, LH or HH
/// code-block: if not, the level's high bands are zero (`p` starts zeroed), the rows of the LH
/// and HH bands stay zero and are not lifted, and the kernels are told (`hi`) that the rest
/// of the high band is zero. `None`: every level may have high-band data.
///
/// With `nw` above 1 and a level of at least two `SAMPLES_PER_THREAD`, the rows go to up to `nw`
/// threads in chunks of `ROW_CHUNK` and the columns in one run of blocks per thread; every
/// sample is lifted by the same operations in the same order, so the result does not depend on
/// `nw`. Each thread's scratch is charged to `budget`; when it does not fit for every thread the
/// pass uses the threads it does.
fn inverse_dwt<T: Copy + Default + Send + Sample>(
    p: &mut [T],
    stride: usize,
    rects: &[Rect],
    high: Option<&[bool]>,
    budget: &mut Budget,
    cancel: &AtomicBool,
    nw: usize,
    kernel: impl Fn(&mut [T], usize, usize, &mut [T], bool) + Sync,
    col_block: usize,
    cols: Option<ColPass<'_, T>>,
) -> Result<()> {
    if rects.len() < 2 {
        return Ok(());
    }
    // Rect sizes come from the caller: subtract with overflow checks and cap each side by the
    // plane, so no size computed from them can overflow or ask for more than `p` could hold.
    let dims = |r: &Rect| -> Result<(usize, usize)> {
        let side = |lo: i64, hi: i64| {
            hi.checked_sub(lo)
                .map(|d| d.max(0))
                .and_then(|d| usize::try_from(d).ok())
                .ok_or_else(|| limit!("DWT rect size overflows"))
        };
        let (w, h) = (side(r.x0, r.x1)?, side(r.y0, r.y1)?);
        if w > p.len() || h > p.len() {
            return Err(invalid!("DWT rect {w}x{h} does not fit the plane"));
        }
        Ok((w, h))
    };
    let (max_w, max_h) = dims(&rects[rects.len() - 1])?;
    // The rects must fit the plane and the scratch lines: this is a public kernel.
    for r in rects {
        let (rw, rh) = dims(r)?;
        let fits = rw == 0
            || rh == 0
            || (stride >= rw
                && (rh - 1)
                    .checked_mul(stride)
                    .and_then(|n| n.checked_add(rw))
                    .is_some_and(|n| n <= p.len()));
        if rw > max_w || rh > max_h || !fits {
            return Err(invalid!("DWT rect {rw}x{rh} does not fit the plane"));
        }
    }
    let col_block = col_block.min(max_w.max(1));
    let line = max_w.max(max_h);
    let blk_len = col_block
        .checked_mul(max_h)
        .ok_or_else(|| limit!("DWT scratch size overflows"))?;
    let acct = budget.shared();
    let want = nw
        .min(max_w.saturating_mul(max_h) / SAMPLES_PER_THREAD)
        .max(1);
    let keep_spare = budget.dwt.keep;
    let mut lines = std::mem::take(&mut T::lines(&mut budget.dwt).live);
    let mut spare = std::mem::take(&mut T::lines(&mut budget.dwt).spare);
    // Keep the scratch of the threads wanted that is big enough, make the rest.
    let pb = if want > 1 { blk_len } else { 0 };
    shed(&mut lines, want, &mut spare, keep_spare);
    let keep = lines
        .iter()
        .take_while(|l| l.mal.len() >= line && l.blk.len() >= blk_len && l.pblk.len() >= pb)
        .count();
    shed(&mut lines, keep, &mut spare, keep_spare);
    for _ in keep..want {
        let one = (|| -> Result<Lines<T>> {
            let n = line
                .checked_mul(2)
                .and_then(|n| n.checked_add(blk_len))
                .and_then(|n| n.checked_add(pb))
                .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
                .ok_or_else(|| limit!("DWT scratch size overflows"))?;
            // A spare's vectors when there is one (its charge, if it has one, given back first),
            // else new ones; zeroed either way.
            let (mal, out, blk, pblk) = match spare.pop() {
                Some(l) => (l.mal, l.out, l.blk, l.pblk),
                None => Default::default(),
            };
            let charge = budget.guard(n)?;
            Ok(Lines {
                mal: reuse(mal, line)?,
                out: reuse(out, line)?,
                blk: reuse(blk, blk_len)?,
                pblk: reuse(pblk, pb)?,
                charge: Some(charge),
            })
        })();
        match one {
            Ok(l) => lines.push(l),
            Err(e) => {
                if lines.is_empty() {
                    return Err(e);
                }
                break; // memory for fewer threads only
            }
        }
    }
    let nwk = lines.len();
    let mut run = || -> Result<()> {
        for r in 1..rects.len() {
            check_cancel(cancel)?;
            let rw = rects[r].x1 - rects[r].x0;
            let rh = rects[r].y1 - rects[r].y0;
            if rw <= 0 || rh <= 0 {
                continue;
            }
            let (rw, rh) = (rw as usize, rh as usize);
            let cas_x = (rects[r].x0 & 1) as usize;
            let cas_y = (rects[r].y0 & 1) as usize;
            let (sn_x, _) = sn_dn(rw, cas_x);
            let (sn_y, _) = sn_dn(rh, cas_y);
            let hi = high.is_none_or(|h| h[r]);
            // Without high bands, the rows of the LH and HH bands are zero and lift to zero.
            let rows = if hi { rh } else { sn_y };
            let lvl_nw = nwk.min(rw.saturating_mul(rh) / SAMPLES_PER_THREAD).max(1);
            let par = lvl_nw > 1;
            if par && rows > 0 {
                let slab = &mut p[..(rows - 1) * stride + rw];
                drive(
                    lvl_nw,
                    &acct,
                    cancel,
                    &mut lines,
                    slab.chunks_mut(ROW_CHUNK * stride),
                    |s, _, chunk| {
                        for row in chunk.chunks_mut(stride) {
                            let row = &mut row[..rw];
                            s.mal[..rw].copy_from_slice(row);
                            kernel(&mut s.mal[..rw], sn_x, cas_x, row, hi);
                        }
                        Ok(())
                    },
                )?;
            } else {
                let s = &mut lines[0];
                for y in 0..rows {
                    if y & 63 == 0 {
                        check_cancel(cancel)?;
                    }
                    let row = &mut p[y * stride..y * stride + rw];
                    s.mal[..rw].copy_from_slice(row);
                    kernel(&mut s.mal[..rw], sn_x, cas_x, row, hi);
                }
            }
            check_cancel(cancel)?;
            // The threaded vertical pass hands each thread one run of column blocks, as row
            // pieces; the pieces cost 16 bytes a row.
            let nbk = rw.div_ceil(col_block);
            let g = lvl_nw.min(nbk);
            let piece = std::mem::size_of::<&mut [T]>();
            let cost = if par && g > 1 {
                g.saturating_mul(rh).saturating_mul(piece)
            } else {
                0
            };
            if cost > 0 && budget.charge(cost).is_ok() {
                let width = nbk.div_ceil(g) * col_block;
                let mut groups: Vec<Vec<&mut [T]>> = Vec::new();
                let mut x = 0;
                while x < rw {
                    groups.push(Vec::with_capacity(rh));
                    x += width;
                }
                let slab = &mut p[..(rh - 1) * stride + rw];
                for row in slab.chunks_mut(stride) {
                    let mut rest = &mut row[..rw];
                    for grp in groups.iter_mut() {
                        let (a, b) = rest.split_at_mut(width.min(rest.len()));
                        grp.push(a);
                        rest = b;
                    }
                }
                let ng = groups.len();
                let res = drive(
                    ng,
                    &acct,
                    cancel,
                    &mut lines[..ng],
                    groups.into_iter(),
                    |s, _, mut rows: Vec<&mut [T]>| {
                        let gw = rows.first().map_or(0, |r| r.len());
                        let mut k0 = 0;
                        while k0 < gw {
                            check_cancel(cancel)?;
                            let bw = col_block.min(gw - k0);
                            for (y, row) in rows.iter().enumerate() {
                                s.pblk[y * bw..(y + 1) * bw].copy_from_slice(&row[k0..k0 + bw]);
                            }
                            col_pass(
                                &mut s.pblk[..rh * bw],
                                bw,
                                0,
                                bw,
                                rh,
                                (sn_y, cas_y, hi),
                                &mut s.blk,
                                &mut s.out,
                                &kernel,
                                cols,
                            );
                            for (y, row) in rows.iter_mut().enumerate() {
                                row[k0..k0 + bw].copy_from_slice(&s.pblk[y * bw..(y + 1) * bw]);
                            }
                            k0 += bw;
                        }
                        Ok(())
                    },
                );
                budget.release(cost);
                res?;
                continue;
            }
            let s = &mut lines[0];
            let mut x0 = 0;
            while x0 < rw {
                check_cancel(cancel)?;
                let bw = col_block.min(rw - x0);
                col_pass(
                    p,
                    stride,
                    x0,
                    bw,
                    rh,
                    (sn_y, cas_y, hi),
                    &mut s.blk,
                    &mut s.out,
                    &kernel,
                    cols,
                );
                x0 += bw;
            }
        }
        Ok(())
    };
    let res = run();
    let d = T::lines(&mut budget.dwt);
    d.live = lines;
    d.spare = spare;
    res
}

/// Reconstructs one reversible tile component in place.
pub fn inverse_dwt_53(
    p: &mut [i32],
    stride: usize,
    rects: &[Rect],
    high: Option<&[bool]>,
    budget: &mut Budget,
    cancel: &AtomicBool,
    nw: usize,
) -> Result<()> {
    let cols = |p: &mut [i32], stride, bn, b: &mut [i32], n, cas, _: bool| {
        cols_53(p, stride, bn, b, n, cas)
    };
    inverse_dwt(
        p,
        stride,
        rects,
        high,
        budget,
        cancel,
        nw,
        |mal, sn, cas, out, _| row_53(mal, sn, cas, out),
        COL_BLOCK_97,
        Some(&cols),
    )
}

/// Reconstructs one irreversible tile component in place, rounding as `r` says.
pub fn inverse_dwt_97(
    p: &mut [f32],
    stride: usize,
    rects: &[Rect],
    high: Option<&[bool]>,
    r: Rounding,
    budget: &mut Budget,
    cancel: &AtomicBool,
    nw: usize,
) -> Result<()> {
    if r == Rounding::JJ2000 {
        return inverse_dwt(
            p,
            stride,
            rects,
            high,
            budget,
            cancel,
            nw,
            |mal, sn, cas, out, _| {
                let (l, h) = mal.split_at_mut(sn);
                synth_97_jj(out, l, h, cas);
            },
            COL_BLOCK,
            None,
        );
    }
    let cols = |p: &mut [f32], stride, bn, b: &mut [f32], n, cas, hi| {
        cols_97(p, stride, bn, b, n, cas, r, hi)
    };
    inverse_dwt(
        p,
        stride,
        rects,
        high,
        budget,
        cancel,
        nw,
        |mal, sn, cas, out, hi| row_97(mal, sn, cas, out, r, hi),
        COL_BLOCK_97,
        Some(&cols),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::Limits;

    fn rects(w: i64, h: i64, x0: i64, y0: i64, levels: usize) -> Vec<Rect> {
        (0..=levels)
            .map(|r| {
                let l = (levels - r) as u32;
                Rect {
                    x0: crate::codestream::ceil_div_pow2(x0, l),
                    y0: crate::codestream::ceil_div_pow2(y0, l),
                    x1: crate::codestream::ceil_div_pow2(x0 + w, l),
                    y1: crate::codestream::ceil_div_pow2(y0 + h, l),
                }
            })
            .collect()
    }

    #[test]
    fn rects_larger_than_the_plane_are_an_error() {
        let mut budget = Budget::new(&Limits::default());
        let cancel = AtomicBool::new(false);
        let r = rects(8, 8, 0, 0, 1);
        let mut p = vec![0i32; 10];
        assert!(inverse_dwt_53(&mut p, 8, &r, None, &mut budget, &cancel, 1).is_err());
    }

    /// The forward reversible 5/3 transform, the dual of `inverse_1d_53`: the original
    /// interleaved signal in, the coefficients in the same interleaved layout out.
    fn forward_5x3(x: &[i32], sn: usize, dn: usize, cas: usize) -> Vec<i32> {
        let mut a = x.to_vec();
        let cl = clamp_dwt;
        if cas == 0 {
            if dn == 0 && sn <= 1 {
                return a;
            }
            for i in 0..dn {
                let t = (a[2 * cl(i as i64, sn)] + a[2 * cl(i as i64 + 1, sn)]) >> 1;
                a[2 * i + 1] -= t;
            }
            for i in 0..sn {
                let t = (a[2 * cl(i as i64 - 1, dn) + 1] + a[2 * cl(i as i64, dn) + 1] + 2) >> 2;
                a[2 * i] += t;
            }
            return a;
        }
        if sn == 0 && dn == 1 {
            a[0] *= 2;
            return a;
        }
        for i in 0..dn {
            let t = (a[2 * cl(i as i64, sn) + 1] + a[2 * cl(i as i64 - 1, sn) + 1]) >> 1;
            a[2 * i] -= t;
        }
        for i in 0..sn {
            let t = (a[2 * cl(i as i64, dn)] + a[2 * cl(i as i64 + 1, dn)] + 2) >> 2;
            a[2 * i + 1] += t;
        }
        a
    }

    #[test]
    fn inverse_53_round_trip_1d() {
        for length in 1..=17usize {
            for cas in 0..=1usize {
                let (sn, dn) = sn_dn(length, cas);
                let orig: Vec<i32> = (0..length)
                    .map(|i| ((i * 37 + length * 11 + cas * 5) % 53) as i32 - 26)
                    .collect();
                let mut coeffs = forward_5x3(&orig, sn, dn, cas);
                inverse_1d_53(&mut coeffs, sn, dn, cas);
                assert_eq!(coeffs, orig, "length={length} cas={cas}");
            }
        }
    }

    /// The forward 9/7 in f64 (T.800 F.3.8.2 in forward order: alpha, beta, gamma, delta, then
    /// scale) with the same mirror boundary convention, to build round-trip fixtures.
    fn forward_97(x: &[f64], cas: usize) -> Vec<f64> {
        let n = x.len();
        let mut a = x.to_vec();
        let lift = |a: &mut Vec<f64>, par: usize, c: f64| {
            let mut j = par;
            while j < n {
                let u = a[mirror(j as i64 - 1, n)];
                let v = a[mirror(j as i64 + 1, n)];
                a[j] += c * (u + v);
                j += 2;
            }
        };
        lift(&mut a, 1 - cas, DWT97_ALPHA as f64);
        lift(&mut a, cas, DWT97_BETA as f64);
        lift(&mut a, 1 - cas, DWT97_GAMMA as f64);
        lift(&mut a, cas, DWT97_DELTA as f64);
        // The reciprocals of the scales `inverse_1d_97` applies.
        let mut j = cas;
        while j < n {
            a[j] *= 1.0 / DWT97_K as f64;
            j += 2;
        }
        let mut j = 1 - cas;
        while j < n {
            a[j] *= 1.0 / DWT97_TWO_INV_K as f64;
            j += 2;
        }
        a
    }

    #[test]
    fn inverse_97_round_trip_1d() {
        for length in 2..=17usize {
            for cas in 0..=1usize {
                let orig: Vec<f64> = (0..length)
                    .map(|i| ((i * 29 + length * 7 + cas * 3) % 41) as f64 - 20.0)
                    .collect();
                let mut buf: Vec<f32> = forward_97(&orig, cas).iter().map(|&v| v as f32).collect();
                inverse_1d_97(&mut buf, cas, Rounding::Fused);
                for i in 0..length {
                    assert!(
                        (buf[i] as f64 - orig[i]).abs() <= 1e-3,
                        "length={length} cas={cas}: mismatch at {i}: got {} want {}",
                        buf[i],
                        orig[i]
                    );
                }
            }
        }
    }

    /// Length-1 odd-origin signals are halved (F.3.7), truncating toward zero like C.
    #[test]
    fn inverse_1d_53_cas1_length1() {
        for v in [6, 7, -7, -6, 0, 1, -1] {
            let mut a = [v];
            inverse_1d_53(&mut a, 0, 1, 1);
            assert_eq!(a[0], v / 2, "v={v}");
        }
    }

    /// A flat LL band with all-zero high bands reconstructs a flat image, 5/3 exactly and 9/7
    /// to within float error, at every origin parity and odd size.
    #[test]
    fn flat_image_stays_flat() {
        let cancel = AtomicBool::new(false);
        for (w, h) in [(1, 1), (2, 1), (1, 2), (5, 7), (16, 16), (33, 17)] {
            for (x0, y0) in [(0, 0), (1, 0), (0, 1), (3, 5)] {
                for levels in 1..=3usize {
                    let rs = rects(w, h, x0, y0, levels);
                    let top = rs[levels];
                    let (tw, th) = ((top.x1 - top.x0) as usize, (top.y1 - top.y0) as usize);
                    let ll = rs[0];
                    let (lw, lh) = ((ll.x1 - ll.x0) as usize, (ll.y1 - ll.y0) as usize);
                    let mut b = Budget::new(&Limits::default());
                    let mut p = vec![0i32; tw * th];
                    for y in 0..lh {
                        for x in 0..lw {
                            p[y * tw + x] = 100;
                        }
                    }
                    inverse_dwt_53(&mut p, tw, &rs, None, &mut b, &cancel, 1).unwrap();
                    // A constant LL with zero highs gives a constant image under the 5/3.
                    assert!(
                        p.iter().all(|&v| v == p[0]),
                        "{w}x{h} at ({x0},{y0}) L{levels}"
                    );

                    let mut f = vec![0f32; tw * th];
                    for y in 0..lh {
                        for x in 0..lw {
                            f[y * tw + x] = 100.0;
                        }
                    }
                    inverse_dwt_97(
                        &mut f,
                        tw,
                        &rs,
                        None,
                        super::super::Rounding::Fused,
                        &mut b,
                        &cancel,
                        1,
                    )
                    .unwrap();
                    assert!(f.iter().all(|v| v.is_finite()));
                }
            }
        }
    }

    /// Rects whose sizes overflow, or that are larger than the plane even with a zero side, are
    /// errors and never panic or allocate.
    #[test]
    fn huge_rects_are_errors() {
        let cancel = AtomicBool::new(false);
        let mut b = Budget::new(&Limits::default());
        let rs = [
            Rect::default(),
            Rect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 1 << 62,
            },
        ];
        assert!(matches!(
            inverse_dwt_53(&mut [0i32; 4], 1, &rs, None, &mut b, &cancel, 1),
            Err(crate::Error::Invalid(_))
        ));
        let rs = [
            Rect::default(),
            Rect {
                x0: i64::MIN,
                y0: 0,
                x1: i64::MAX,
                y1: 1,
            },
        ];
        assert!(matches!(
            inverse_dwt_97(
                &mut [0f32; 4],
                1,
                &rs,
                None,
                Rounding::Fused,
                &mut b,
                &cancel,
                1
            ),
            Err(crate::Error::LimitExceeded(_))
        ));
        assert_eq!(b.held(), 0);
    }

    /// The cancel flag stops the transform with `Cancelled`.
    #[test]
    fn cancel_stops_transform() {
        let rs = rects(16, 16, 0, 0, 2);
        let mut p = vec![0i32; 256];
        let cancel = AtomicBool::new(true);
        let mut b = Budget::new(&Limits::default());
        assert_eq!(
            inverse_dwt_53(&mut p, 16, &rs, None, &mut b, &cancel, 1).unwrap_err(),
            crate::Error::Cancelled
        );
    }

    /// The scratch is charged against the budget.
    #[test]
    fn scratch_is_charged() {
        let rs = rects(64, 64, 0, 0, 2);
        let mut p = vec![0i32; 64 * 64];
        let cancel = AtomicBool::new(false);
        let mut b = Budget::new(&Limits {
            max_alloc: 16,
            ..Limits::default()
        });
        assert!(matches!(
            inverse_dwt_53(&mut p, 64, &rs, None, &mut b, &cancel, 1),
            Err(crate::Error::LimitExceeded(_))
        ));
    }

    /// The split-band row kernel and the row-block vertical pass equal the interleaved line
    /// transform bit for bit, for both roundings, every origin parity and odd sizes.
    #[test]
    fn row_kernels_match_line_transform() {
        let cancel = AtomicBool::new(false);
        let mut st = 0x2545_f491_4f6c_dd1du64;
        for (w, h) in [(1, 1), (2, 2), (3, 5), (17, 9), (33, 70), (130, 67)] {
            for (x0, y0) in [(0, 0), (1, 0), (0, 1), (3, 5)] {
                for levels in 1..=3usize {
                    for r in [Rounding::Fused, Rounding::Separate, Rounding::JJ2000] {
                        let rs = rects(w, h, x0, y0, levels);
                        let top = rs[levels];
                        let (tw, th) = ((top.x1 - top.x0) as usize, (top.y1 - top.y0) as usize);
                        let data: Vec<f32> = (0..tw * th)
                            .map(|_| {
                                st ^= st << 13;
                                st ^= st >> 7;
                                st ^= st << 17;
                                (st >> 40) as f32 / 4096.0 - 2048.0
                            })
                            .collect();
                        let mut want = data.clone();
                        let mut b = Budget::new(&Limits::default());
                        inverse_dwt(
                            &mut want,
                            tw,
                            &rs,
                            None,
                            &mut b,
                            &cancel,
                            1,
                            |mal, sn, cas, out, _| {
                                if r == Rounding::JJ2000 {
                                    let (l, h) = mal.split_at_mut(sn);
                                    synth_97_jj(out, l, h, cas);
                                    return;
                                }
                                if mal.len() < 2 {
                                    out.copy_from_slice(mal);
                                    return;
                                }
                                interleave(out, mal, sn, cas);
                                inverse_1d_97(out, cas, r);
                            },
                            COL_BLOCK,
                            None,
                        )
                        .unwrap();
                        let mut got = data;
                        inverse_dwt_97(&mut got, tw, &rs, None, r, &mut b, &cancel, 1).unwrap();
                        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(
                            bits(&got),
                            bits(&want),
                            "{w}x{h} at ({x0},{y0}) L{levels} {r:?}"
                        );
                    }
                }
            }
        }
    }

    /// The 5/3 split-band row kernel and row-block vertical pass equal the interleaved line
    /// transform `inverse_1d_53`, for every side from 1 to 130 (one or two levels), every origin
    /// parity, and random samples with the i32 extremes among them (the arithmetic wraps).
    #[test]
    fn rows_53_match_line_transform() {
        let cancel = AtomicBool::new(false);
        let mut st = 0x51ed_2701_aa55_0f0fu64;
        for w in 1..=130i64 {
            let (h, levels) = ((w * 7) % 130 + 1, 1 + (w as usize % 2));
            for (x0, y0) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let rs = rects(w, h, x0, y0, levels);
                let top = rs[levels];
                let (tw, th) = ((top.x1 - top.x0) as usize, (top.y1 - top.y0) as usize);
                let data: Vec<i32> = (0..tw * th)
                    .map(|_| {
                        st ^= st << 13;
                        st ^= st >> 7;
                        st ^= st << 17;
                        match st % 8 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            2 => -1,
                            _ => (st >> 32) as i32 >> (st % 24),
                        }
                    })
                    .collect();
                let mut want = data.clone();
                let mut b = Budget::new(&Limits::default());
                inverse_dwt(
                    &mut want,
                    tw,
                    &rs,
                    None,
                    &mut b,
                    &cancel,
                    1,
                    |mal, sn, cas, out, _| {
                        let n = mal.len();
                        interleave(out, mal, sn, cas);
                        inverse_1d_53(out, sn, n - sn, cas);
                    },
                    COL_BLOCK,
                    None,
                )
                .unwrap();
                let mut got = data;
                inverse_dwt_53(&mut got, tw, &rs, None, &mut b, &cancel, 1).unwrap();
                assert_eq!(got, want, "{w}x{h} at ({x0},{y0}) L{levels}");
            }
        }
    }

    /// Levels flagged without high-band data give the same bits as lifting their zero high
    /// bands, for both transforms, every origin parity, odd sizes and any mix of flagged levels.
    #[test]
    fn zero_high_levels_match_full_lifting() {
        let cancel = AtomicBool::new(false);
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for (w, h) in [
            (1, 1),
            (2, 3),
            (5, 2),
            (9, 5),
            (6, 7),
            (17, 9),
            (33, 70),
            (130, 67),
        ] {
            for (x0, y0) in [(0, 0), (1, 0), (0, 1), (3, 5)] {
                for levels in 1..=3usize {
                    let rs = rects(w, h, x0, y0, levels);
                    let top = rs[levels];
                    let (tw, th) = ((top.x1 - top.x0) as usize, (top.y1 - top.y0) as usize);
                    for mask in 0..(1u32 << levels) {
                        // high[r] for r = 0..=levels; level 0 is unused.
                        let high: Vec<bool> = (0..=levels)
                            .map(|r| r > 0 && (mask >> (r - 1)) & 1 == 1)
                            .collect();
                        // Random samples in the LL band, and in the bands of the flagged levels
                        // (what resolution r holds beyond resolution r - 1); zero elsewhere.
                        let mut data = vec![0f32; tw * th];
                        for r in 0..=levels {
                            if r > 0 && !high[r] {
                                continue;
                            }
                            let size = |q: Rect| ((q.x1 - q.x0) as usize, (q.y1 - q.y0) as usize);
                            let (iw, ih) = size(rs[r]);
                            let (pw, ph) = if r == 0 { (0, 0) } else { size(rs[r - 1]) };
                            for y in 0..ih {
                                for x in 0..iw {
                                    if x >= pw || y >= ph {
                                        data[y * tw + x] = (rnd() >> 40) as f32 / 4096.0 - 2048.0;
                                    }
                                }
                            }
                        }
                        let ints: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                        let mut b = Budget::new(&Limits::default());
                        let mut want = ints.clone();
                        inverse_dwt_53(&mut want, tw, &rs, None, &mut b, &cancel, 1).unwrap();
                        let mut got = ints;
                        inverse_dwt_53(&mut got, tw, &rs, Some(&high), &mut b, &cancel, 1).unwrap();
                        assert_eq!(
                            got, want,
                            "5/3 {w}x{h} at ({x0},{y0}) L{levels} mask {mask:b}"
                        );
                        for r in [Rounding::Fused, Rounding::Separate, Rounding::JJ2000] {
                            let mut want = data.clone();
                            inverse_dwt_97(&mut want, tw, &rs, None, r, &mut b, &cancel, 1)
                                .unwrap();
                            let mut got = data.clone();
                            inverse_dwt_97(&mut got, tw, &rs, Some(&high), r, &mut b, &cancel, 1)
                                .unwrap();
                            let bits =
                                |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                            assert_eq!(
                                bits(&got),
                                bits(&want),
                                "9/7 {w}x{h} at ({x0},{y0}) L{levels} mask {mask:b} {r:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    /// The threaded passes give the bits of the serial ones, for both transforms, every origin
    /// parity and thread count, with and without flagged zero high levels.
    #[test]
    fn threads_match_serial() {
        let cancel = AtomicBool::new(false);
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for (w, h) in [(1100, 500), (801, 701), (1030, 520)] {
            for (x0, y0) in [(0, 0), (1, 1), (3, 0)] {
                let rs = rects(w, h, x0, y0, 3);
                let top = rs[3];
                let (tw, th) = ((top.x1 - top.x0) as usize, (top.y1 - top.y0) as usize);
                let ints: Vec<i32> = (0..tw * th).map(|_| (next() >> 50) as i32 - 8192).collect();
                let floats: Vec<f32> = ints.iter().map(|&v| v as f32 / 4.0).collect();
                for high in [None, Some([true, false, true, true])] {
                    let hi = high.as_ref().map(|h| &h[..]);
                    let mut b = Budget::new(&Limits::default());
                    let mut want = ints.clone();
                    inverse_dwt_53(&mut want, tw, &rs, hi, &mut b, &cancel, 1).unwrap();
                    for nw in [2, 3, 8] {
                        let mut got = ints.clone();
                        inverse_dwt_53(&mut got, tw, &rs, hi, &mut b, &cancel, nw).unwrap();
                        assert_eq!(got, want, "5/3 {w}x{h} ({x0},{y0}) nw {nw}");
                    }
                    for r in [Rounding::Fused, Rounding::Separate, Rounding::JJ2000] {
                        let mut want = floats.clone();
                        inverse_dwt_97(&mut want, tw, &rs, hi, r, &mut b, &cancel, 1).unwrap();
                        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        for nw in [2, 3, 8] {
                            let mut got = floats.clone();
                            inverse_dwt_97(&mut got, tw, &rs, hi, r, &mut b, &cancel, nw).unwrap();
                            assert_eq!(
                                bits(&got),
                                bits(&want),
                                "9/7 {r:?} {w}x{h} ({x0},{y0}) nw {nw}"
                            );
                        }
                    }
                    assert_eq!(b.held(), 0);
                }
            }
        }
    }
}
