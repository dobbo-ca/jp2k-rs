//! The decode pipeline: tier-2, tier-1, the wavelet and colour stages per tile, and assembly of
//! the output image, whole or for a region.

use crate::codestream::jp2::{CdefEntry, JP2_SIGNATURE};
use crate::codestream::{
    self, Codestream, CodingStyle, Rect, Siz, ceil_div, ceil_div_pow2, effective_coding_style,
    effective_mct, quant_for, tile_overrides,
};
use crate::dsp::{self, Rounding};
use crate::error::{Error, Result};
use crate::limits::{Account, Budget, Charge, Limits, Pool, check_cancel, held_by, reuse};
use crate::parallel::{ROW_CHUNK, drive, drive_threads, lock, panic_error};
use crate::tier1::{self, Segment};
use crate::tier2::{self, CodeBlockRef, Cursor, PacketRef, TileGeom, TileWant};
use crate::{ColorModel, Config, Options, Plane, Raster};
use std::collections::HashMap;
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Runs `f`, turning a panic into `Error::Internal` (Go: `ErrInternal`).
fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => Err(panic_error(p)),
    }
}

// -- pipeline glue: tier2 -> tier1 -> dequant -> inverse DWT --

/// The most resolutions a tile component has (T.800: 32 decomposition levels).
const MAX_RES: usize = 33;

/// The most termination segments a code-block can have: one pass each, at most the passes tier 2
/// lets a code-block accumulate.
const MAX_SEGS: usize = 109;

/// An empty vector of another element type in `v`'s allocation, which must have the same size and
/// alignment: how the vectors of locks over a call's buffers, whose element types borrow those
/// buffers, are kept from one call to the next. std collects in place here (an optimisation it
/// does not promise; `tests/reuse_alloc.rs` checks it still does).
fn recycle<T, U>(mut v: Vec<T>) -> Vec<U> {
    v.clear();
    v.into_iter().filter_map(|_| None).collect()
}

/// `v` emptied and refilled with `n` copies of `x`.
fn fill<T: Clone>(v: &mut Vec<T>, n: usize, x: T) {
    v.clear();
    v.resize(n, x);
}

/// Where code-block `cb` of resolution `r` starts in comp's Mallat-layout plane: LL at r = 0
/// fills the corner, and at r > 0 HL sits right of the `sn` low columns, LH below the low
/// rows, HH diagonal. `None` when a malformed (e.g. fuzzed) COD/QCD describes a code-block that
/// would run outside its band's area of the plane.
fn cb_origin(tg: &TileGeom, comp: usize, r: usize, cb: &CodeBlockRef) -> Option<(i64, i64)> {
    if r >= tg.comps[comp].resolutions.len() {
        return None;
    }
    let res = &tg.comps[comp].resolutions[r];
    let b = res.bands.iter().find(|b| b.subband == cb.subband)?;
    let (mut x, mut y) = (0i64, 0i64);
    let mut bw = res.rect.x1 - res.rect.x0;
    let mut bh = res.rect.y1 - res.rect.y0;
    if r > 0 {
        let (sn_x, dn_x) = dsp::sn_dn(bw as usize, (res.rect.x0 & 1) as usize);
        let (sn_y, dn_y) = dsp::sn_dn(bh as usize, (res.rect.y0 & 1) as usize);
        let (sn_x, dn_x, sn_y, dn_y) = (sn_x as i64, dn_x as i64, sn_y as i64, dn_y as i64);
        bw = sn_x;
        bh = sn_y;
        if cb.subband == 1 || cb.subband == 3 {
            x = sn_x;
            bw = dn_x;
        }
        if cb.subband == 2 || cb.subband == 3 {
            y = sn_y;
            bh = dn_y;
        }
    }
    let (lx, ly) = (cb.x0 - b.rect.x0, cb.y0 - b.rect.y0);
    if lx < 0 || ly < 0 || lx + cb.x1 - cb.x0 > bw || ly + cb.y1 - cb.y0 > bh {
        return None;
    }
    Some((x + lx, y + ly))
}

/// One packet's bytes `[start, end)` of a code-block's segment. `ord` is its place among the
/// code-block's segments as gathered (the two `u32`s keep the record the size it had with a
/// `usize` layer).
struct CbSeg {
    layer: u32,
    ord: u32,
    seg_index: usize,
    start: usize,
    end: usize,
    passes: usize,
}

/// One code-block's per-packet segment contributions gathered from every layer, in layer order
/// (and, within a layer, tier-2 segment order), ready to be re-grouped by `seg_index`.
struct CbGroup {
    res: usize,
    /// Representative rect/subband; `zero_bit_planes` from the first inclusion.
    cb: CodeBlockRef,
    segs: Vec<CbSeg>,
}

/// The code-blocks of a tile component, gathered by `group_code_blocks`, and what it keeps for
/// the next one: the index, and the segment lists of the last groups, last first.
#[derive(Default)]
struct Groups {
    index: HashMap<(usize, usize, usize, usize, usize), usize>,
    groups: Vec<CbGroup>,
    spare: Vec<Vec<CbSeg>>,
}

impl Groups {
    /// Keeps the groups' segment lists for the next component, each for the same group.
    fn recycle(&mut self) {
        for g in self.groups.drain(..).rev() {
            let mut v = g.segs;
            v.clear();
            self.spare.push(v);
        }
    }

    /// The heap the lists hold, by capacity, the groups' segment lists included; the index's
    /// estimated from its capacity (a bucket of key, value and control byte for each 7/8 of a slot).
    fn heap(&self) -> usize {
        type Entry = ((usize, usize, usize, usize, usize), usize);
        held_by(&self.groups)
            + self.groups.iter().map(|g| held_by(&g.segs)).sum::<usize>()
            + held_by(&self.spare)
            + self.spare.iter().map(held_by).sum::<usize>()
            + self.index.capacity() * 8 / 7 * (size_of::<Entry>() + 1)
    }
}

/// Gathers comp's code-block segments across every packet (every layer, every progression
/// position) by code-block identity, into `gr.groups`. Segments are sorted by layer so
/// concatenation order matches the encoder's layer order regardless of the codestream's
/// progression. Returns the bytes charged to `budget` for them (the group, its map entry, its
/// job in `decode_component`, and twice each segment for Vec growth); the caller releases them.
fn group_code_blocks(
    pkts: &[PacketRef],
    comp: usize,
    budget: &mut Budget,
    gr: &mut Groups,
) -> Result<usize> {
    gr.index.clear();
    gr.recycle();
    let mut charged = 0usize;
    for p in pkts {
        if p.component != comp {
            continue;
        }
        for cb in &p.blocks {
            if cb.x1 - cb.x0 <= 0 || cb.y1 - cb.y0 <= 0 {
                continue;
            }
            let k = (p.resolution, p.precinct, cb.subband, cb.cbx, cb.cby);
            let seg_cost = 2 * size_of::<CbSeg>() * cb.segments.len();
            let group_cost = if gr.index.contains_key(&k) {
                0
            } else {
                size_of::<CbGroup>() + size_of::<CbJob>() + 2 * 64
            };
            budget.charge(seg_cost + group_cost)?;
            charged += seg_cost + group_cost;
            let gi = match gr.index.get(&k) {
                Some(&gi) => gi,
                None => {
                    gr.groups.push(CbGroup {
                        res: p.resolution,
                        cb: CodeBlockRef {
                            subband: cb.subband,
                            cbx: cb.cbx,
                            cby: cb.cby,
                            x0: cb.x0,
                            y0: cb.y0,
                            x1: cb.x1,
                            y1: cb.y1,
                            zero_bit_planes: cb.zero_bit_planes,
                            segments: Vec::new(),
                        },
                        segs: gr.spare.pop().unwrap_or_default(),
                    });
                    gr.index.insert(k, gr.groups.len() - 1);
                    gr.groups.len() - 1
                }
            };
            let segs = &mut gr.groups[gi].segs;
            for s in &cb.segments {
                segs.push(CbSeg {
                    layer: p.layer as u32,
                    ord: segs.len() as u32,
                    seg_index: s.seg_index,
                    start: s.start,
                    end: s.end,
                    passes: s.passes,
                });
            }
        }
    }
    for g in &mut gr.groups {
        // By layer, and gathering order within a layer: a style that splits one layer's
        // inclusion into several termination segments (TERMALL/bypass) appends them in tier-2's
        // own order, which must survive. (A stable sort would allocate.)
        g.segs.sort_unstable_by_key(|s| (s.layer, s.ord));
    }
    Ok(charged)
}

impl CbGroup {
    /// Re-checks every layer's byte range against the tile data length.
    fn segs_valid(&self, tile_data_len: usize) -> bool {
        self.segs
            .iter()
            .all(|s| s.end <= tile_data_len && s.start <= s.end)
    }
}

/// How one tier-1 segment's bytes are made: a range of the tile data, or the join of several
/// layers' contributions to the same termination segment (a range of the worker's `join`).
#[derive(Clone, Copy)]
enum SegSrc {
    Tile(usize, usize),
    Joined(usize, usize),
}

/// One tile component's working samples: `i` for the reversible 5/3 path, `f` for 9/7. After
/// `decode_component`, the finished `w` x `h` window starting at component sample (`x0`, `y0`)
/// sits at `i`/`f` from `off`, with row pitch `stride`.
struct PlaneBuf {
    rev: bool,
    i: Vec<i32>,
    f: Vec<f32>,
    stride: usize,
    off: usize,
    x0: i64,
    y0: i64,
    w: usize,
    h: usize,
    /// Bytes charged for `i`/`f`.
    charged: usize,
}

/// Caps a decode's tier-1 work, the sum over its code-blocks of coding passes times samples, at
/// `T1_WORK_PER_BYTE` per input byte plus `T1_WORK_SLACK`. That is over ten times the most any
/// legitimate file tested needs.
const T1_WORK_PER_BYTE: i64 = 1 << 18;
const T1_WORK_SLACK: i64 = 1 << 22;

/// A thread is worth using for about this many samples of a row pass: on a pool's workers a task
/// costs microseconds (Go starts a goroutine per item with no threshold).
const SAMPLES_PER_THREAD: usize = 1 << 15;

/// Bytes one tier-1 worker holds: its sample scratch (2 x 4096 x 4) and its flag grid (at most
/// 12312 for any legal block shape, with room for the `Vec`'s growth).
const T1_SCRATCH: usize = 64 << 10;

/// The most a `Decoder`'s lane keeps, uncharged, within a call of its row buffers and of a tile
/// component's code-block lists and jobs, by capacity: more is freed when the component is done.
const KEPT_SMALL: usize = 64 << 10;

/// The most a `Decoder`'s lane keeps of a tile's tier-2 lists, by capacity, charged, for its next
/// tile: more are freed when the tile is done.
const T2_KEPT: usize = 256 << 10;

/// A tier-1 thread is worth starting for about this many sample-passes of work (coding passes
/// times samples, summed over its code-blocks).
const T1_WORK_PER_THREAD: i64 = 1 << 14;

/// One tier-1 worker's scratch, reused for every code-block it decodes.
struct T1Worker {
    t: tier1::State,
    raw: Vec<i32>,
    f: Vec<f32>,
    /// A code-block's segments that span several layers, joined (charged while the block is
    /// decoded). A `Decoder`'s worker keeps it, uncharged, only while it holds at most
    /// `T1_SCRATCH` bytes.
    join: Vec<u8>,
    /// The ledger's share for the scratch, given back when the worker drops; `None` while a
    /// `Decoder` keeps the worker for its next call.
    charge: Option<Charge>,
}

impl T1Worker {
    /// A worker charged to `budget`: one a `Decoder` kept in `cold` if there is one, else a new one.
    fn new(budget: &Budget, cold: &mut Vec<T1Worker>) -> Result<T1Worker> {
        let charge = budget.guard(T1_SCRATCH)?;
        if let Some(mut w) = cold.pop() {
            w.charge = Some(charge);
            return Ok(w);
        }
        Ok(T1Worker {
            t: tier1::State::default(),
            raw: vec![0; tier1::MAX_CODE_BLOCK_SAMPLES],
            f: vec![0.0; tier1::MAX_CODE_BLOCK_SAMPLES],
            join: Vec::new(),
            charge: Some(charge),
        })
    }
}

/// A lane's spare tile-component planes, with the bytes charged for each. Making a plane for every
/// tile component, and freeing it after, has the allocator give the memory back to the system and
/// the next plane fault it in again (4 MiB for a 1024x1024 tile, 12 with the colour transform):
/// a fifth of the time of a single-threaded decode of big tiles. The bytes stay charged while a
/// plane is spare.
#[derive(Default)]
struct PlanePool {
    i: Vec<(Vec<i32>, usize)>,
    f: Vec<(Vec<f32>, usize)>,
    /// Bytes charged for the spare planes and the `done` ones.
    bytes: usize,
    /// A `Decoder`'s planes and wavelet levels the call is done with, past the spares, still
    /// charged (as the spares are) until a take reuses one or the call ends: a buffer the
    /// ledger released is not held while the call charges more.
    done_i: Vec<(Vec<i32>, usize)>,
    done_f: Vec<(Vec<f32>, usize)>,
    /// Planes a `Decoder` keeps between calls, uncharged: a plane charged as new is taken from
    /// these.
    cold_i: Pool<i32>,
    cold_f: Pool<f32>,
}

/// Rows of a tile component's plane under one lock while its code-blocks are copied in.
const PLANE_BAND_ROWS: usize = ROW_CHUNK;

/// Spare planes kept of each kind: the colour transform's three.
const POOL_PLANES: usize = 3;

impl PlanePool {
    /// A zeroed plane of `n` samples and the bytes charged for it: a spare one (by the bytes it
    /// was charged, not its capacity, which a kept plane can have more of), else a new one.
    fn take_i(&mut self, n: usize, budget: &mut Budget) -> Result<(Vec<i32>, usize)> {
        if let Some(k) = self.i.iter().position(|(_, c)| *c >= n.saturating_mul(4)) {
            let (mut v, c) = self.i.swap_remove(k);
            self.bytes -= c;
            v.clear();
            v.resize(n, 0);
            return Ok((v, c));
        }
        take_done(
            &mut self.done_i,
            &mut self.cold_i,
            &mut self.bytes,
            n,
            budget,
        )
    }

    fn take_f(&mut self, n: usize, budget: &mut Budget) -> Result<(Vec<f32>, usize)> {
        if let Some(k) = self.f.iter().position(|(_, c)| *c >= n.saturating_mul(4)) {
            let (mut v, c) = self.f.swap_remove(k);
            self.bytes -= c;
            v.clear();
            v.resize(n, 0.0);
            return Ok((v, c));
        }
        take_done(
            &mut self.done_f,
            &mut self.cold_f,
            &mut self.bytes,
            n,
            budget,
        )
    }

    /// Keeps the plane of a finished tile component, or if enough are kept frees it and gives
    /// its bytes back (with `keep` it keeps it charged in `done` instead).
    fn give(&mut self, p: PlaneBuf, budget: &mut Budget, keep: bool) {
        if p.charged == 0 {
            return;
        }
        if p.rev && self.i.len() < POOL_PLANES {
            self.i.push((p.i, p.charged));
        } else if !p.rev && self.f.len() < POOL_PLANES {
            self.f.push((p.f, p.charged));
        } else if keep {
            if p.rev {
                self.done_i.push((p.i, p.charged));
            } else {
                self.done_f.push((p.f, p.charged));
            }
        } else {
            let c = p.charged;
            drop(p);
            budget.release(c);
            return;
        }
        self.bytes += p.charged;
    }

    /// Makes every spare and `done` plane a cold one, at the end of a `Decoder`'s call (the
    /// lane's ledger handle, which holds their bytes, is dropped then).
    fn park(&mut self) {
        for (v, _) in self.i.drain(..).chain(self.done_i.drain(..)) {
            self.cold_i.give(v, true);
        }
        for (v, _) in self.f.drain(..).chain(self.done_f.drain(..)) {
            self.cold_f.give(v, true);
        }
        self.bytes = 0;
    }
}

/// A plane or level of `n` samples and the bytes charged for it: the `done` one charged with the
/// fewest bytes that cover it, keeping its charge, else one charged as new and taken from `cold`.
fn take_done<T: Clone + Default>(
    done: &mut Vec<(Vec<T>, usize)>,
    cold: &mut Pool<T>,
    bytes: &mut usize,
    n: usize,
    budget: &mut Budget,
) -> Result<(Vec<T>, usize)> {
    let need = n.saturating_mul(std::mem::size_of::<T>());
    let best = (done.iter().enumerate())
        .filter(|(_, (_, c))| *c >= need)
        .min_by_key(|(_, (_, c))| *c)
        .map(|(k, _)| k);
    if let Some(k) = best {
        let (v, c) = done.swap_remove(k);
        *bytes -= c;
        return Ok((reuse(v, n)?, c));
    }
    Ok((budget.take_vec(n, || cold.pick(n))?, need))
}

/// A wavelet level of `c` charged bytes the call is done with: kept charged in `done` with
/// `keep`, else freed and its bytes given back.
fn give_done<T>(
    done: &mut Vec<(Vec<T>, usize)>,
    bytes: &mut usize,
    v: Vec<T>,
    c: usize,
    budget: &mut Budget,
    keep: bool,
) {
    if keep {
        done.push((v, c));
        *bytes += c;
    } else {
        drop(v);
        budget.release(c);
    }
}

/// What a lane keeps from one tile component to the next, and a `Decoder` from one call to the
/// next (`keep`): everything it allocates, so a second call of the same shape allocates nothing.
#[derive(Default)]
struct LaneScratch {
    /// A `Decoder`'s lane: buffers no longer needed are kept for later, not freed (`docs/limits.md`
    /// says which stay charged).
    keep: bool,
    workers: Vec<T1Worker>,
    /// Tier-1 workers a `Decoder` keeps between calls, uncharged.
    cold: Vec<T1Worker>,
    planes: PlanePool,
    /// The lane's DWT scratch between calls (during one it is in the lane's `Budget`).
    dwt: dsp::DwtScratch,
    /// The highest resolution decoded of each component of the tile, and its window, and the
    /// window lists of a region's `TileWant` not in use.
    tops: Vec<usize>,
    wins: Vec<Rect>,
    want_spare: Vec<Vec<Rect>>,
    /// Tier 2's packets and geometry of the tile, and the code-blocks of its component.
    t2: tier2::TileScratch,
    /// A `Decoder`'s: the heap `t2` holds, charged after its tile until the lane's next tile takes
    /// it or the call ends.
    t2_bytes: usize,
    groups: Groups,
    /// `decode_component`'s level buffers, code-block jobs and plane locks, kept empty.
    lvi: Vec<Vec<i32>>,
    lvf: Vec<Vec<f32>>,
    jobs: Vec<CbJob>,
    bands_i: Vec<Mutex<&'static mut [i32]>>,
    bands_f: Vec<Mutex<&'static mut [f32]>>,
    /// `emit`'s row buffers.
    rows: Vec<Vec<i32>>,
    row_pool: Pool<i32>,
}

impl LaneScratch {
    /// Gives back what the lane holds charged and keeps it uncharged for the next call.
    fn park(&mut self) {
        for mut w in self.workers.drain(..) {
            w.charge = None;
            self.cold.push(w);
        }
        self.planes.park();
        self.dwt.park();
        self.t2_bytes = 0;
        // Between calls a lane keeps at most `T2_KEPT` of tier-2 lists, whatever kept them.
        if self.t2.heap() > T2_KEPT {
            self.t2 = tier2::TileScratch::default();
        }
    }
}

/// One code-block ready for tier-1: where it lands in the plane, and its subband's step size.
struct CbJob {
    g: usize,
    /// The level buffer it lands in (always 0 for a whole-resolution plane).
    lv: usize,
    x: i64,
    y: i64,
    /// The part of the block that is needed, block-local `[sx0, sx1) x [sy0, sy1)`.
    sx0: usize,
    sy0: usize,
    sx1: usize,
    sy1: usize,
    delta: f64,
    mb: i32,
}

/// Runs tier-1, dequantization and the inverse DWT for one tile component up to resolution
/// `top`, for output window `win` of that resolution. Code-blocks that tier-2 left out (those
/// outside what `win` needs) read 0. A `win` short of the whole resolution gets one buffer per
/// level and a windowed inverse DWT (`dwt_levels`, `partial_dwt`), else one plane transformed in
/// place; either way `win` comes out as Go's decoder computes it. Malformed packets/code-blocks
/// (out-of-range resolution, subband, or byte range) are skipped rather than causing a panic. `style` is comp's effective coding style
/// (T.800 A.6: tile/main COC overrides COD).
///
/// The code-blocks go to up to `nw` threads (`scratch` is the lane's), each writing its
/// block into the shared planes under a lock; the inverse DWT gets `nw` too. The bytes are the
/// same for every `nw`.
#[allow(clippy::too_many_arguments)]
fn decode_component(
    cs: &Codestream,
    tg: &TileGeom,
    tile_data: &[u8],
    groups: &[CbGroup],
    tile_idx: u16,
    comp: usize,
    style: &CodingStyle,
    top: usize,
    win: Rect,
    round: Rounding,
    work: &AtomicI64,
    scratch: &mut LaneScratch,
    nw: usize,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<PlaneBuf> {
    let q = quant_for(cs, tile_idx, comp);
    let precision = cs.siz.components[comp].precision as i32;
    let cb_style = style.code_block_style;
    let tc = &tg.comps[comp];
    let rev = style.transformation == 1;
    let keep = scratch.keep;
    let mut p = PlaneBuf {
        rev,
        i: Vec::new(),
        f: Vec::new(),
        stride: 0,
        off: 0,
        x0: 0,
        y0: 0,
        w: 0,
        h: 0,
        charged: 0,
    };
    if win.empty() {
        return Ok(p);
    }
    if top >= MAX_RES {
        return Err(invalid!(
            "tile {tile_idx} component {comp} has more than {MAX_RES} resolutions"
        ));
    }
    p.x0 = win.x0;
    p.y0 = win.y0;
    p.w = (win.x1 - win.x0) as usize;
    p.h = (win.y1 - win.y0) as usize;
    // A window short of the whole resolution gets one buffer per level, each covering only what
    // the level above reads (`dwt_windows`); else one plane in Mallat layout, transformed in
    // place.
    let mut wbuf = [Rect::default(); MAX_RES];
    let wins = dwt_levels(tc, top, win, rev, &mut wbuf)?;
    let top_rect = tc.resolutions[top].rect;
    let dim = |r: &Rect| -> Result<(usize, usize)> {
        Ok((
            usize::try_from(r.x1 - r.x0)
                .map_err(|_| limit!("tile component plane size overflows"))?,
            usize::try_from(r.y1 - r.y0)
                .map_err(|_| limit!("tile component plane size overflows"))?,
        ))
    };
    // Each level's size (one for a whole resolution), and the bytes charged for a window's.
    let mut sizes = [(0usize, 0usize); MAX_RES];
    let mut chg = [0usize; MAX_RES];
    let nl = match wins {
        Some(ws) => {
            for (k, r) in ws.iter().enumerate() {
                sizes[k] = dim(r)?;
            }
            ws.len()
        }
        None => {
            sizes[0] = dim(&top_rect)?;
            1
        }
    };
    let mut lvi = std::mem::take(&mut scratch.lvi);
    let mut lvf = std::mem::take(&mut scratch.lvf);
    lvi.clear();
    lvf.clear();
    // The plane of a whole resolution comes from the lane's spares, with its charge.
    let mut pooled = 0usize;
    for (k, &(w, h)) in sizes[..nl].iter().enumerate() {
        let n = w
            .checked_mul(h)
            .ok_or_else(|| limit!("tile component plane size overflows"))?;
        if wins.is_none() {
            if rev {
                let (v, c) = scratch.planes.take_i(n, budget)?;
                lvi.push(v);
                pooled = c;
            } else {
                let (v, c) = scratch.planes.take_f(n, budget)?;
                lvf.push(v);
                pooled = c;
            }
        } else if rev {
            let pl = &mut scratch.planes;
            let (v, c) = take_done(&mut pl.done_i, &mut pl.cold_i, &mut pl.bytes, n, budget)?;
            lvi.push(v);
            chg[k] = c;
        } else {
            let pl = &mut scratch.planes;
            let (v, c) = take_done(&mut pl.done_f, &mut pl.cold_f, &mut pl.bytes, n, budget)?;
            lvf.push(v);
            chg[k] = c;
        }
    }
    // Slot of level `l` in `lvi`/`lvf`, and its row pitch.
    let slot = |l: usize| if wins.is_some() { l } else { 0 };
    let pitch = |l: usize| sizes[slot(l)].0;
    p.charged = if wins.is_none() { pooled } else { chg[top] };
    if let Some(ws) = wins {
        let e = ws[top];
        p.stride = pitch(top);
        p.off = ((win.y0 - e.y0) as usize) * p.stride + (win.x0 - e.x0) as usize;
    } else {
        p.stride = pitch(top);
        p.off = ((win.y0 - top_rect.y0) as usize) * p.stride + (win.x0 - top_rect.x0) as usize;
    }

    // Every code-block in a subband shares one step size (E.1.1.1); indexed by res*4+subband.
    let mut steps = [None::<(f64, i32)>; MAX_RES * 4];
    let mut jobs = std::mem::take(&mut scratch.jobs);
    jobs.clear();
    jobs.reserve(groups.len());
    let mut wk: i64 = 0;
    for (gi, g) in groups.iter().enumerate() {
        let cb = &g.cb;
        if g.res > top || !g.segs_valid(tile_data.len()) {
            continue;
        }
        let (lv, x, y, sx0, sy0, sx1, sy1);
        match wins {
            None => {
                let Some((cx, cy)) = cb_origin(tg, comp, g.res, cb) else {
                    continue;
                };
                (lv, x, y) = (top, cx, cy);
                (sx0, sy0, sx1, sy1) = (0, 0, (cb.x1 - cb.x0) as usize, (cb.y1 - cb.y0) as usize);
            }
            Some(ws) => {
                let (bw, ox, oy) = tier2::band_window(ws, g.res, cb.subband);
                let (x0, y0) = (cb.x0.max(bw.x0), cb.y0.max(bw.y0));
                let (x1, y1) = (cb.x1.min(bw.x1), cb.y1.min(bw.y1));
                if x0 >= x1 || y0 >= y1 {
                    continue;
                }
                (lv, x, y) = (g.res, ox + x0 - bw.x0, oy + y0 - bw.y0);
                (sx0, sy0) = ((x0 - cb.x0) as usize, (y0 - cb.y0) as usize);
                (sx1, sy1) = ((x1 - cb.x0) as usize, (y1 - cb.y0) as usize);
            }
        }
        let si = g.res * 4 + cb.subband;
        let (delta, mb) = match steps[si] {
            Some(s) => s,
            None => {
                let (mut delta, mb) = dsp::step_size(q, g.res, cb.subband, precision, rev)?;
                if round == Rounding::JJ2000 {
                    // JJ2000 keeps T.800 E-3's gain: 2 for HL, LH, 4 for HH.
                    delta *= (1i64 << cb.subband.div_ceil(2)) as f64;
                }
                steps[si] = Some((delta, mb));
                (delta, mb)
            }
        };
        tier1::validate_code_block_dims(cb.x1 - cb.x0, cb.y1 - cb.y0, cb_style)?;
        jobs.push(CbJob {
            g: gi,
            lv,
            x,
            y,
            sx0,
            sy0,
            sx1,
            sy1,
            delta,
            mb,
        });
        for sg in &g.segs {
            wk += sg.passes as i64 * (cb.x1 - cb.x0) * (cb.y1 - cb.y0);
        }
    }
    if work.fetch_sub(wk, Ordering::Relaxed) - wk < 0 {
        return Err(limit!(
            "tile {tile_idx}: tier-1 work exceeds the budget for the input size"
        ));
    }

    if !jobs.is_empty() {
        let LaneScratch {
            workers,
            cold,
            bands_i: kept_i,
            bands_f: kept_f,
            ..
        } = &mut *scratch;
        if workers.is_empty() {
            workers.push(T1Worker::new(budget, cold)?);
        }
        // One thread per `T1_WORK_PER_THREAD` of work, at most `nw`; scratch for fewer if the
        // ledger has no room for all of them. The lane keeps its workers for its next tile
        // components: making and zeroing 14 of them for each cost a quarter of the decode.
        let by_work = usize::try_from(wk / T1_WORK_PER_THREAD).unwrap_or(usize::MAX);
        let want = nw.min(jobs.len()).min(by_work).max(1);
        while workers.len() < want {
            match T1Worker::new(budget, cold) {
                Ok(w) => workers.push(w),
                Err(_) => break,
            }
        }
        let n_ws = want.min(workers.len());
        let acct = budget.shared();
        let ledger: &Budget = budget;
        // The planes as bands of rows with a lock each, not one lock for all of them: a block's
        // rows are copied in under the bands' locks, from the band its column picks, so threads
        // finishing blocks side by side are not all at the same band (one lock made every copy,
        // 16 KiB a block, wait for the one before it). Level `l`'s bands start at `boff[l]`.
        let mut boff = [0usize; MAX_RES];
        let mut bands_i: Vec<Mutex<&mut [i32]>> = recycle(std::mem::take(kept_i));
        let mut bands_f: Vec<Mutex<&mut [f32]>> = recycle(std::mem::take(kept_f));
        plane_bands(&mut lvi, &sizes, &mut bands_i, &mut boff);
        plane_bands(&mut lvf, &sizes, &mut bands_f, &mut boff);
        let r = drive(
            want,
            &acct,
            cancel,
            &mut workers[..n_ws],
            jobs.iter(),
            |w, _, j| {
                let g = &groups[j.g];
                let cb = &g.cb;
                let segs = &g.segs;
                let (cw, ch) = ((cb.x1 - cb.x0) as usize, (cb.y1 - cb.y0) as usize);
                // Re-group the code-block's per-packet byte ranges by tier-2 segment index,
                // concatenating every contribution to the same segment (a segment can span
                // several layers under the default style) into one tier-1 segment. Segments stay
                // distinct entries, one per MQ/raw decoder run. The joined segments are charged
                // before they are copied (all at once, so `join` grows once) and given back when
                // the block is done.
                let run_end = |a: usize| {
                    let mut b = a + 1;
                    while b < segs.len() && segs[b].seg_index == segs[a].seg_index {
                        b += 1;
                    }
                    b
                };
                let (mut joined, mut a) = (0usize, 0);
                while a < segs.len() {
                    let b = run_end(a);
                    if b > a + 1 {
                        joined += segs[a..b].iter().map(|x| x.end - x.start).sum::<usize>();
                    }
                    a = b;
                }
                let _held = match joined {
                    0 => None,
                    n => Some(ledger.guard(n)?),
                };
                w.join.clear();
                w.join
                    .try_reserve_exact(joined)
                    .map_err(|_| Error::OutOfMemory)?;
                let mut plan = [(SegSrc::Tile(0, 0), 0usize); MAX_SEGS];
                let mut np = 0;
                let mut a = 0;
                while a < segs.len() {
                    let s = &segs[a];
                    let b = run_end(a);
                    if np == MAX_SEGS {
                        w.join = Vec::new();
                        return Err(invalid!(
                            "a code-block has more than {MAX_SEGS} termination segments"
                        ));
                    }
                    if b > a + 1 {
                        let start = w.join.len();
                        let mut passes = 0;
                        for pp in &segs[a..b] {
                            w.join.extend_from_slice(&tile_data[pp.start..pp.end]);
                            passes += pp.passes;
                        }
                        plan[np] = (SegSrc::Joined(start, w.join.len()), passes);
                    } else {
                        plan[np] = (SegSrc::Tile(s.start, s.end), s.passes);
                    }
                    np += 1;
                    a = b;
                }
                let mut sgs = [Segment {
                    data: &[],
                    passes: 0,
                }; MAX_SEGS];
                for (k, &(src, passes)) in plan[..np].iter().enumerate() {
                    sgs[k] = Segment {
                        data: match src {
                            SegSrc::Tile(a, b) => &tile_data[a..b],
                            SegSrc::Joined(a, b) => &w.join[a..b],
                        },
                        passes,
                    };
                }
                let segs = &sgs[..np];
                let raw = &mut w.raw[..cw * ch];
                let r = if rev {
                    w.t.finish(
                        raw,
                        None,
                        None,
                        j.delta,
                        segs,
                        cw,
                        ch,
                        cb.subband,
                        cb.zero_bit_planes,
                        j.mb,
                        cb_style,
                    )
                } else {
                    let f = &mut w.f[..cw * ch];
                    w.t.finish(
                        raw,
                        None,
                        Some(f),
                        j.delta,
                        segs,
                        cw,
                        ch,
                        cb.subband,
                        cb.zero_bit_planes,
                        j.mb,
                        cb_style,
                    )
                };
                // A `Decoder` keeps the join for its next block only up to a worker's fixed
                // scratch size: a bigger one would stay resident, no longer charged, while the
                // call charges up to `max_alloc` again. Freed before `r` is returned, so an error
                // of the block does not keep it either.
                if !keep || w.join.capacity() > T1_SCRATCH {
                    w.join = Vec::new();
                }
                r?;
                let (stride, n) = (pitch(j.lv), j.sx1 - j.sx0);
                let (l, r0, h) = (slot(j.lv), j.y as usize, j.sy1 - j.sy0);
                if h == 0 {
                    return Ok(());
                }
                // Block rows `r0..r0 + h` of level `l`, band by band from the band this column
                // picks.
                let (b0, cnt) = (
                    r0 / PLANE_BAND_ROWS,
                    (r0 + h - 1) / PLANE_BAND_ROWS + 1 - r0 / PLANE_BAND_ROWS,
                );
                for i in 0..cnt {
                    let b = b0 + (i + (j.x as usize >> 6)) % cnt;
                    let (ra, rb) = (
                        (b * PLANE_BAND_ROWS).max(r0),
                        ((b + 1) * PLANE_BAND_ROWS).min(r0 + h),
                    );
                    let at = |r: usize| (r - b * PLANE_BAND_ROWS) * stride + j.x as usize;
                    let from = |r: usize| (j.sy0 + r - r0) * cw + j.sx0;
                    if rev {
                        let mut g = lock(&bands_i[boff[l] + b]);
                        for r in ra..rb {
                            g[at(r)..at(r) + n].copy_from_slice(&w.raw[from(r)..from(r) + n]);
                        }
                    } else {
                        let mut g = lock(&bands_f[boff[l] + b]);
                        for r in ra..rb {
                            g[at(r)..at(r) + n].copy_from_slice(&w.f[from(r)..from(r) + n]);
                        }
                    }
                }
                Ok(())
            },
        );
        // A free function's containers are freed here, as they always were.
        if keep {
            *kept_i = recycle(bands_i);
            *kept_f = recycle(bands_f);
        }
        r?;
    }
    // No code-block: every sample of the component is zero, and stays zero through the
    // transform (the buffers were zeroed above).
    // A level without any HL, LH or HH code-block has zero high bands (the buffers started
    // zeroed), so its lifting can skip them.
    let mut high = [false; MAX_RES];
    for j in &jobs {
        high[groups[j.g].res] = true;
    }
    let no_jobs = jobs.is_empty();
    if keep {
        scratch.jobs = jobs;
    }
    if let Some(ws) = wins {
        if !no_jobs {
            if rev {
                partial_dwt(&mut lvi, ws, &high, budget, cancel, nw, dsp::inverse_dwt_53)?;
            } else {
                partial_dwt(
                    &mut lvf,
                    ws,
                    &high,
                    budget,
                    cancel,
                    nw,
                    |f, st, rs, h, b, c, n| dsp::inverse_dwt_97(f, st, rs, h, round, b, c, n),
                )?;
            }
        }
        // The top level is the result; the levels below it are done with.
        if rev {
            p.i = lvi.pop().unwrap_or_default();
        } else {
            p.f = lvf.pop().unwrap_or_default();
        }
        let pl = &mut scratch.planes;
        for (v, &c) in lvi.drain(..).zip(&chg) {
            give_done(&mut pl.done_i, &mut pl.bytes, v, c, budget, keep);
        }
        for (v, &c) in lvf.drain(..).zip(&chg) {
            give_done(&mut pl.done_f, &mut pl.bytes, v, c, budget, keep);
        }
        if keep {
            scratch.lvi = lvi;
            scratch.lvf = lvf;
        }
        return Ok(p);
    }
    if rev {
        p.i = lvi.pop().unwrap_or_default();
    } else {
        p.f = lvf.pop().unwrap_or_default();
    }
    if keep {
        scratch.lvi = lvi;
        scratch.lvf = lvf;
    }
    if no_jobs {
        return Ok(p);
    }
    let mut rects = [Rect::default(); MAX_RES];
    for (k, r) in tc.resolutions.iter().take(top + 1).enumerate() {
        rects[k] = r.rect;
    }
    if rev {
        dsp::inverse_dwt_53(
            &mut p.i,
            p.stride,
            &rects[..=top],
            Some(&high[..=top]),
            budget,
            cancel,
            nw,
        )?;
    } else {
        dsp::inverse_dwt_97(
            &mut p.f,
            p.stride,
            &rects[..=top],
            Some(&high[..=top]),
            round,
            budget,
            cancel,
            nw,
        )?;
    }
    Ok(p)
}

/// The levels of `planes` (level `l` at `sizes[l].0` samples a row) cut into bands of
/// `PLANE_BAND_ROWS` rows, each under its own lock, appended to `out`; level `l`'s bands start
/// at `off[l]`.
fn plane_bands<'a, T>(
    planes: &'a mut [Vec<T>],
    sizes: &[(usize, usize)],
    out: &mut Vec<Mutex<&'a mut [T]>>,
    off: &mut [usize],
) {
    let rows = |l: usize| (sizes[l].0 * PLANE_BAND_ROWS).max(1);
    // Room for all of them at once, as one list per level had.
    out.reserve_exact(
        planes
            .iter()
            .enumerate()
            .map(|(l, v)| v.len().div_ceil(rows(l)))
            .sum(),
    );
    for (l, v) in planes.iter_mut().enumerate() {
        off[l] = out.len();
        out.extend(v.chunks_mut(rows(l)).map(Mutex::new));
    }
}

/// Runs the inverse DWT one level at a time over the windows from `dwt_windows`. `lv[r]` holds
/// level r's window in Mallat order with its high-pass bands already filled; each level first
/// copies in its low-pass quadrant from the finished level below. The top level, last, is the
/// result.
fn partial_dwt<T: Copy>(
    lv: &mut [Vec<T>],
    wins: &[Rect],
    high: &[bool],
    budget: &mut Budget,
    cancel: &AtomicBool,
    nw: usize,
    idwt: impl Fn(
        &mut [T],
        usize,
        &[Rect],
        Option<&[bool]>,
        &mut Budget,
        &AtomicBool,
        usize,
    ) -> Result<()>,
) -> Result<()> {
    for r in 1..wins.len() {
        check_cancel(cancel)?;
        let (e, prev) = (wins[r], wins[r - 1]);
        let (w, pw) = ((e.x1 - e.x0) as usize, (prev.x1 - prev.x0) as usize);
        let (lx0, ly0) = ((e.x0 + 1) >> 1, (e.y0 + 1) >> 1);
        let lw = (((e.x1 + 1) >> 1) - lx0) as usize;
        let lh = (((e.y1 + 1) >> 1) - ly0) as usize;
        let (below, here) = lv.split_at_mut(r);
        for y in 0..lh {
            let src = (ly0 - prev.y0) as usize * pw + y * pw + (lx0 - prev.x0) as usize;
            here[0][y * w..y * w + lw].copy_from_slice(&below[r - 1][src..src + lw]);
        }
        if w == 0 || e.y1 <= e.y0 {
            continue; // an empty level has nothing to lift (the kernel skips it too)
        }
        idwt(
            &mut here[0],
            w,
            &[Rect::default(), e],
            Some(&high[r - 1..=r]),
            budget,
            cancel,
            nw,
        )?;
    }
    Ok(())
}

// -- output assembly (T.800 Annex G/I plus this crate's chosen mapping) --

/// Picks the output pixel format from SIZ alone (so `decode_config`, which reads no further than
/// SIZ, can compute it too). Component subsampling does not affect the type: a subsampled
/// component is upsampled to the reference grid by sample replication. Cross-component
/// precision/signedness mismatches are rejected. Signed components are biased by
/// +2^(precision-1) into the same range an unsigned component of the same precision uses.
/// 2-component images are treated as gray+alpha (component 0 replicated into R/G/B, component 1
/// as alpha), matching opj_decompress's convention.
pub(crate) fn classify_output(siz: &Siz) -> Result<ColorModel> {
    let n = siz.components.len();
    if !(1..=4).contains(&n) {
        return Err(unsupported!("{n} components not supported (only 1-4)"));
    }
    let c0 = &siz.components[0];
    for (i, c) in siz.components.iter().enumerate() {
        if c.precision != c0.precision || c.signed != c0.signed {
            return Err(unsupported!(
                "component {i} precision/signedness does not match component 0 (mixed per-component precision not supported)"
            ));
        }
    }
    if c0.precision > 16 {
        return Err(unsupported!(
            "component precision {} exceeds 16",
            c0.precision
        ));
    }
    let eight = c0.precision == 8;
    Ok(match n {
        1 => {
            if eight {
                ColorModel::Gray
            } else {
                ColorModel::Gray16
            }
        }
        3 => {
            if eight {
                ColorModel::Rgba
            } else {
                ColorModel::Rgba64
            }
        }
        _ => {
            if eight {
                ColorModel::Nrgba
            } else {
                ColorModel::Nrgba64
            }
        }
    })
}

/// How samples become pixels of the output image.
#[derive(Clone, Copy)]
struct OutFmt {
    stride: usize,
    /// Bytes per pixel.
    bpp: usize,
    /// 16-bit big-endian channels.
    wide: bool,
    /// 2 components: channel 0 is replicated into R, G and B.
    gray_alpha: bool,
    /// 3 components: alpha is always the maximum.
    opaque: bool,
    precision: u8,
}

/// A decode's result being filled in place: finished samples go straight into `pix`.
struct OutImage {
    fmt: OutFmt,
    pix: Vec<u8>,
}

/// Rows per band of the image being written (see `Bands`): the rows of one task of a row pass.
const BAND_ROWS: usize = ROW_CHUNK;

/// The image's pixels as bands of `BAND_ROWS` rows, each under its own lock, for the time the
/// lanes run: a tile writes its rows band by band, so lanes writing tiles side by side wait only
/// for a band, not for each other's whole tile component (every lane started at the same
/// row, one after another, otherwise: converting a tile component takes about a millisecond).
struct Bands<'a> {
    fmt: OutFmt,
    rows: Vec<Mutex<&'a mut [u8]>>,
}

impl OutFmt {
    /// Stores samples `v` (each in [0, 2^precision)) as output channel `ch` of the pixels of
    /// `pix` starting at byte `off`. Odd precisions (e.g. 12-bit) are expanded into the full 16
    /// bits of the wide types.
    fn put_row(&self, pix: &mut [u8], off: usize, ch: usize, v: &[i32]) {
        self.put_chan(pix, off, ch, v);
        if ch != 0 {
            return;
        }
        if self.gray_alpha {
            self.put_chan(pix, off, 1, v);
            self.put_chan(pix, off, 2, v);
        }
        if self.opaque {
            self.fill_alpha(pix, off, v.len());
        }
    }

    /// `put_row` for a 16-bit wide format and 16-bit float samples `v` (`Fused`/`Separate`
    /// rounding): the same bytes as `samples_f` then `put_row`, without the integer row.
    fn put_row_f16(&self, pix: &mut [u8], off: usize, ch: usize, v: &[f32]) {
        self.put_chan_f16(pix, off, ch, v);
        if ch != 0 {
            return;
        }
        if self.gray_alpha {
            self.put_chan_f16(pix, off, 1, v);
            self.put_chan_f16(pix, off, 2, v);
        }
        if self.opaque {
            self.fill_alpha(pix, off, v.len());
        }
    }

    /// `put_row` for an 8-bit 4-byte pixel format (`ch` below 4) and float samples `v` of
    /// precision `prec` (`Fused`/`Separate` rounding): the same bytes as `samples_f` then
    /// `put_row`, without the integer row.
    fn put_row_u8x4(&self, pix: &mut [u8], ch: usize, v: &[f32], prec: i32) {
        // The kernel reads the pixels it merges into, and the image may be pages not yet touched
        // (`Budget::take_zeroed`): a read would map the shared zero page and the store then copy
        // it, two faults a page. Storing one pixel's byte every 4 KiB first, and the last one,
        // makes each page's first touch a store, one fault, as the byte stores of `put_chan` do.
        for i in (0..v.len()).step_by(1024).chain(v.len().checked_sub(1)) {
            dsp::pack_u8x4(&mut pix[4 * i..4 * i + 4], &v[i..=i], prec, ch);
        }
        dsp::pack_u8x4(pix, v, prec, ch);
        if ch != 0 {
            return;
        }
        if self.gray_alpha {
            dsp::pack_u8x4(pix, v, prec, 1);
            dsp::pack_u8x4(pix, v, prec, 2);
        }
        if self.opaque {
            self.fill_alpha(pix, 0, v.len());
        }
    }

    fn put_chan_f16(&self, pix: &mut [u8], off: usize, ch: usize, v: &[f32]) {
        let mut tmp = [0u16; 256];
        let mut at = off + 2 * ch;
        for v in v.chunks(tmp.len()) {
            let t = &mut tmp[..v.len()];
            dsp::pack_u16(t, v, 16);
            for (px, &s) in pix[at..].chunks_mut(self.bpp).zip(t.iter()) {
                px[..2].copy_from_slice(&s.to_be_bytes());
            }
            at += v.len() * self.bpp;
        }
    }

    fn fill_alpha(&self, pix: &mut [u8], off: usize, n: usize) {
        if self.bpp == 4 {
            let p = &mut pix[off..off + 4 * n];
            for i in (3..p.len()).step_by(4) {
                p[i] = 0xff;
            }
            return;
        }
        let p = &mut pix[off..off + 8 * n];
        for i in (7..p.len()).step_by(8) {
            p[i - 1] = 0xff;
            p[i] = 0xff;
        }
    }

    fn put_chan(&self, pix: &mut [u8], off: usize, ch: usize, v: &[i32]) {
        let bpp = self.bpp;
        if !self.wide {
            for (px, &s) in pix[off + ch..].chunks_mut(bpp).zip(v) {
                px[0] = s as u8;
            }
            return;
        }
        let prec = self.precision as u32;
        for (px, &s) in pix[off + 2 * ch..].chunks_mut(bpp).zip(v) {
            let e = dsp::expand16(s as u32, prec);
            px[..2].copy_from_slice(&e.to_be_bytes());
        }
    }
}

/// Where the lanes of a decode write: the image (while the lanes run, in `Bands` instead), or planes
/// of raw samples, and the canvases of subsampled components. Locked by a lane while it writes a
/// finished tile component.
struct OutState<'a> {
    o: Option<OutImage>,
    out_planes: &'a mut Vec<Option<(Rect, Vec<u16>)>>,
    canvas: &'a mut Vec<Option<Vec<i32>>>,
}

/// One lane's state: its handle on the ledger, its tier-1 workers, and its share of the
/// call's threads, the lane's own thread included (see `decode_image_once`).
struct Lane {
    budget: Budget,
    scratch: LaneScratch,
    nw: usize,
}

/// Tier 2 of the tiles runs in tile order, one tile at a time, whichever lane has it: the shared
/// cursor's budgets (and a PPM stream) are consumed in order, so errors do not depend on timing.
/// Tier 2 never waits on the pool, and a lane takes a tile only after the one before it was
/// taken, so a lane waiting for its turn is always waiting for a thread that is running: progress
/// never depends on a job being scheduled.
struct TurnState {
    /// The position in the tile order whose tier 2 runs next.
    next: usize,
    /// A tile failed before or in its tier 2: the tiles after it do not run theirs.
    stopped: bool,
    cur: Cursor,
}

struct Turn {
    m: Mutex<TurnState>,
    cv: Condvar,
}

/// Marks the turn stopped if a lane leaves before finishing its tile's tier 2 (a panic), so the
/// lanes waiting for the next turn do not wait for ever.
struct Abandon<'a> {
    turn: &'a Turn,
    done: bool,
}

impl Drop for Abandon<'_> {
    fn drop(&mut self) {
        if !self.done {
            lock(&self.turn.m).stopped = true;
            self.turn.cv.notify_all();
        }
    }
}

/// Runs `f` on the shared tier-2 cursor when it is the turn of tile position `k`, then passes
/// the turn on. If an earlier tile failed, or `cancel` is set while it waits, returns
/// `Error::Cancelled` without running `f`: the earlier tile's error is the one reported.
fn in_turn<R>(
    turn: &Turn,
    k: usize,
    cancel: &AtomicBool,
    f: impl FnOnce(&mut Cursor) -> Result<R>,
) -> Result<R> {
    let mut st = lock(&turn.m);
    loop {
        if st.stopped {
            return Err(Error::Cancelled);
        }
        if st.next == k {
            break;
        }
        if cancel.load(Ordering::Relaxed) {
            // The tile before may never run its tier 2: `drive` skips items once cancelled.
            st.stopped = true;
            drop(st);
            turn.cv.notify_all();
            return Err(Error::Cancelled);
        }
        st = turn
            .cv
            .wait_timeout(st, Duration::from_millis(20))
            .map(|(g, _)| g)
            .unwrap_or_else(|e| e.into_inner().0);
    }
    let r =
        catch_unwind(AssertUnwindSafe(|| f(&mut st.cur))).unwrap_or_else(|p| Err(panic_error(p)));
    if r.is_ok() {
        st.next = k + 1;
    } else {
        st.stopped = true;
    }
    drop(st);
    turn.cv.notify_all();
    r
}

/// Tile lanes get at most this much working memory for those past the first, and at most half
/// of what the ledger has left. Go stops at 64 MB and a quarter of the output (`laneBudget`), which
/// for 1024x1024 tiles on 16 cores is 2 lanes of 8 workers each; here a lane's nested passes cost a
/// pool wake and wait each (tens of microseconds, 20 to 30 of them per tile), so tiles in flight
/// are the cheaper parallelism and the ledger, not the output size, is what bounds them.
const LANE_BUDGET: u128 = 512 << 20;
/// A lane is worth a thread for this many bytes of codestream.
const LANE_MIN_BYTES: usize = 2 << 10;

/// How many tiles to decode at once (Go: `tileLanes`): not with PPM or an opened `Image`'s
/// planned decode, and no more than `nw`, the tiles, what the ledger has room for (about 8 bytes
/// a sample of plane per lane past the first), or the codestream's size justifies.
fn tile_lanes(
    cs: &Codestream,
    n: usize,
    reduce: u32,
    nw: usize,
    data_len: usize,
    remaining: usize,
) -> usize {
    let siz = &cs.siz;
    let planes: u128 = if siz.components.len() >= 3 { 3 } else { 1 };
    let (tw, th) = (
        ceil_div_pow2(siz.xtsiz as i64, reduce) as u128,
        ceil_div_pow2(siz.ytsiz as i64, reduce) as u128,
    );
    let tile_bytes = (tw * th * planes * 8).max(1);
    let room = LANE_BUDGET.min(remaining as u128 / 2);
    let lanes = (1 + room / tile_bytes)
        .min(n as u128)
        .min(nw as u128)
        .min((data_len / LANE_MIN_BYTES) as u128);
    let lanes = lanes.max(1) as usize;
    // The tiles go to the lanes in rounds, and the last round runs on fewer lanes: 32 tiles on
    // 15 lanes make 3 rounds, the third of 2 tiles. Fewer lanes make the same rounds with none idle.
    n.div_ceil(n.div_ceil(lanes).max(1)).clamp(1, lanes)
}
/// The reference-grid rect `[x0, x1) x [y0, y1)`, absolute (XOsiz/YOsiz included) and already at
/// 1/2^reduce scale, that `decode_image` produces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Region {
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    reduce: u32,
}

/// The whole image at full size, what `decode` produces.
fn full_region(siz: &Siz) -> Region {
    Region {
        x0: siz.xosiz as i64,
        y0: siz.yosiz as i64,
        x1: siz.xsiz as i64,
        y1: siz.ysiz as i64,
        reduce: 0,
    }
}

/// The part of component `c` that `reg` reads, in the component's own sample units at `reg`'s
/// scale: reference-grid sample X shows component sample floor(X / XRsiz), clipped to the
/// samples the component has.
fn comp_window(siz: &Siz, c: usize, reg: &Region) -> Rect {
    let xr = siz.components[c].xrsiz as i64;
    let yr = siz.components[c].yrsiz as i64;
    let s = reg.reduce;
    let cx0 = ceil_div_pow2(ceil_div(siz.xosiz as i64, xr), s);
    let cx1 = ceil_div_pow2(ceil_div(siz.xsiz as i64, xr), s);
    let cy0 = ceil_div_pow2(ceil_div(siz.yosiz as i64, yr), s);
    let cy1 = ceil_div_pow2(ceil_div(siz.ysiz as i64, yr), s);
    let clamp = |v: i64, lo: i64, hi: i64| v.max(lo).min(hi);
    Rect {
        x0: clamp(reg.x0 / xr, cx0, cx1),
        x1: clamp((reg.x1 - 1) / xr + 1, cx0, cx1),
        y0: clamp(reg.y0 / yr, cy0, cy1),
        y1: clamp((reg.y1 - 1) / yr + 1, cy0, cy1),
    }
}

/// Whether tile `tile` holds any sample of the component windows, without building its
/// geometry. It is per component because a subsampled component's window can reach a sample of
/// a tile the reference-grid region itself misses.
fn tile_touches(siz: &Siz, tile: u16, comp_win: &[Rect], reduce: u32) -> bool {
    let p = tile as i64 % siz.tiles_wide as i64;
    let q = tile as i64 / siz.tiles_wide as i64;
    let tx0 = (siz.xtosiz as i64 + p * siz.xtsiz as i64).max(siz.xosiz as i64);
    let tx1 = (siz.xtosiz as i64 + (p + 1) * siz.xtsiz as i64).min(siz.xsiz as i64);
    let ty0 = (siz.ytosiz as i64 + q * siz.ytsiz as i64).max(siz.yosiz as i64);
    let ty1 = (siz.ytosiz as i64 + (q + 1) * siz.ytsiz as i64).min(siz.ysiz as i64);
    for (c, w) in comp_win.iter().enumerate() {
        let xr = siz.components[c].xrsiz as i64;
        let yr = siz.components[c].yrsiz as i64;
        let t = Rect {
            x0: ceil_div_pow2(ceil_div(tx0, xr), reduce),
            y0: ceil_div_pow2(ceil_div(ty0, yr), reduce),
            x1: ceil_div_pow2(ceil_div(tx1, xr), reduce),
            y1: ceil_div_pow2(ceil_div(ty1, yr), reduce),
        };
        if !t.intersect(w).empty() {
            return true;
        }
    }
    false
}

/// For resolutions 0..=top of `tc`, the window each inverse DWT level computes so that `win`, a
/// window of resolution `top`, comes out exact. A level computed on a window instead of the
/// whole resolution is wrong only within `reach` samples of a window edge that is not a real
/// resolution edge, so each window is the one above grown by `reach`, and the level below must
/// supply its low-pass half. `wins` has `top + 1` entries.
fn dwt_windows(tc: &tier2::TileComponent, top: usize, win: Rect, reach: i64, wins: &mut [Rect]) {
    let mut n = win;
    for r in (1..=top).rev() {
        let e = Rect {
            x0: n.x0 - reach,
            y0: n.y0 - reach,
            x1: n.x1 + reach,
            y1: n.y1 + reach,
        }
        .intersect(&tc.resolutions[r].rect);
        wins[r] = e;
        n = Rect {
            x0: (e.x0 + 1) >> 1,
            y0: (e.y0 + 1) >> 1,
            x1: (e.x1 + 1) >> 1,
            y1: (e.y1 + 1) >> 1,
        };
    }
    wins[0] = n;
}

/// `dwt_windows` for a component window, in `buf`: `None` when `win` is the whole resolution
/// `top`, an empty slice when `win` is empty.
fn dwt_levels<'b>(
    tc: &tier2::TileComponent,
    top: usize,
    win: Rect,
    rev: bool,
    buf: &'b mut [Rect; MAX_RES],
) -> Result<Option<&'b [Rect]>> {
    if win.empty() {
        return Ok(Some(&buf[..0]));
    }
    if win == tc.resolutions[top].rect {
        return Ok(None);
    }
    let reach = if rev { 2 } else { 4 }; // 9/7 lifting steps
    let ws = buf
        .get_mut(..=top)
        .ok_or_else(|| invalid!("more than {MAX_RES} resolutions"))?;
    dwt_windows(tc, top, win, reach, ws);
    Ok(Some(&*ws))
}

/// For each component of tile `tile_idx`, the highest resolution a decode of the component
/// windows reads (in `tops`), its window of that resolution (in `wins`), and what tier-2 must
/// keep of the tile (`None` when full), filled in `want` when there is one, its window lists
/// taken from and given to `spare`. `rev(c)`: component `c` uses the 5/3 filter.
#[allow(clippy::too_many_arguments)]
fn tile_windows(
    tg: &TileGeom,
    tile_idx: u16,
    rev: &dyn Fn(usize) -> bool,
    comp_win: &[Rect],
    reduce: u32,
    full: bool,
    tops: &mut Vec<usize>,
    wins: &mut Vec<Rect>,
    want: Option<TileWant>,
    spare: &mut Vec<Vec<Rect>>,
) -> Result<Option<TileWant>> {
    let num_comps = tg.comps.len();
    fill(tops, num_comps, 0);
    fill(wins, num_comps, Rect::default());
    for c in 0..num_comps {
        let nr = tg.comps[c].num_resolutions as i64;
        let t = nr - 1 - reduce as i64;
        if t < 0 {
            return Err(invalid!(
                "tile {tile_idx} component {c} has fewer than {reduce} decomposition levels"
            ));
        }
        tops[c] = t as usize;
        let w = comp_win[c].intersect(&tg.comps[c].resolutions[tops[c]].rect);
        wins[c] = if w.empty() { Rect::default() } else { w };
    }
    if full {
        return Ok(None);
    }
    let mut want = want.unwrap_or(TileWant {
        top: Vec::new(),
        wins: Vec::new(),
    });
    want.top.clear();
    want.top.extend_from_slice(tops);
    want.wins.truncate(num_comps);
    for c in 0..num_comps {
        let mut buf = [Rect::default(); MAX_RES];
        let ws = dwt_levels(&tg.comps[c], tops[c], wins[c], rev(c), &mut buf)?;
        if c == want.wins.len() {
            want.wins.push(None);
        }
        let slot = &mut want.wins[c];
        match ws {
            None => spare.extend(slot.take()),
            Some(w) => {
                let mut v = slot.take().or_else(|| spare.pop()).unwrap_or_default();
                v.clear();
                v.extend_from_slice(w);
                *slot = Some(v);
            }
        }
    }
    Ok(Some(want))
}

/// Replicates a subsampled component's samples onto the `w` x `h` reference grid
/// (opj_decompress's -upsample writes the same reference-grid pixel for every source sample it
/// maps to). Reference pixel (x, y), absolute image coordinate (x + XOsiz, y + YOsiz), maps to
/// source sample floor((x + XOsiz) / xr) - compOX; a coordinate that floor-divides to before
/// the component's first sample reads as 0, and one past the last sample clamps to it.
fn upsample_component(
    src: &[i32],
    src_w: usize,
    src_h: usize,
    comp_ox: i64,
    comp_oy: i64,
    xr: i64,
    yr: i64,
    w: usize,
    h: usize,
    x_osiz: i64,
    y_osiz: i64,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<Vec<i32>> {
    // The source index of reference coordinate `v`, -1 before the first sample; computed inline
    // so nothing sized by `w` or `h` is held besides the charged output.
    let src_index = |v: usize, osiz: i64, r: i64, comp_o: i64, len: usize| {
        let l = (v as i64 + osiz) / r - comp_o;
        if l < 0 { -1 } else { l.min(len as i64 - 1) }
    };
    let n = w
        .checked_mul(h)
        .ok_or_else(|| limit!("upsampled plane size overflows"))?;
    let mut out = budget.vec(n, 0i32)?;
    for y in 0..h {
        check_cancel(cancel)?;
        let ly = src_index(y, y_osiz, yr, comp_oy, src_h);
        if ly < 0 {
            continue; // leading rows before the component's first sample: zero
        }
        let di = y * w;
        for x in 0..w {
            let lx = src_index(x, x_osiz, xr, comp_ox, src_w);
            if lx < 0 {
                continue;
            }
            out[di + x] = src[ly as usize * src_w + lx as usize];
        }
    }
    Ok(out)
}

/// Checks `comps`, a request for those components as planes, and sets in `dec` the components a
/// decode of them runs: `comps`, and 0-2 too when a multiple component transform mixes those.
fn component_set(cs: &Codestream, comps: &[usize], dec: &mut Vec<bool>) -> Result<()> {
    let n = cs.siz.components.len();
    fill(dec, n, false);
    for &c in comps {
        if c >= n {
            return Err(invalid!("component {c} out of range (image has {n})"));
        }
        if dec[c] {
            return Err(invalid!("component {c} requested twice"));
        }
        dec[c] = true;
    }
    if n >= 3 && (dec[0] || dec[1] || dec[2]) {
        // Any tile's COD may turn it on.
        let mut mct = cs.cod.multiple_component_transform != 0;
        for tp in &cs.tile_parts {
            mct = mct
                || tp
                    .cod()
                    .is_some_and(|c| c.multiple_component_transform != 0);
        }
        if mct {
            dec[0] = true;
            dec[1] = true;
            dec[2] = true;
        }
    }
    for (c, &ok) in dec.iter().enumerate() {
        let p = cs.siz.components[c].precision;
        if ok && p > 16 {
            return Err(unsupported!("component {c} precision {p} exceeds 16"));
        }
    }
    Ok(())
}

/// The tiles a decode of the component windows visits, in order. The PPM stream is ordered by
/// the codestream APPEARANCE of the tile-parts it serves (T.800 A.7.4), which need not match
/// tile index order. Tiles go in the order their first tile-part appears in the codestream so
/// the shared PPM cursor consumes the right stretch of the stream for each tile; output is
/// written by tile geometry, so it is order-independent. `order` is the result, `seen` scratch.
fn decode_order(
    cs: &Codestream,
    full: bool,
    comp_win: &[Rect],
    reduce: u32,
    order: &mut Vec<u16>,
    seen: &mut Vec<bool>,
) {
    let siz = &cs.siz;
    let num_tiles = siz.tiles_wide as usize * siz.tiles_high as usize;
    order.clear();
    order.reserve(num_tiles);
    fill(seen, num_tiles, false);
    for tp in &cs.tile_parts {
        let ti = tp.sot.isot;
        if (ti as usize) < num_tiles && !seen[ti as usize] {
            seen[ti as usize] = true;
            order.push(ti);
        }
    }
    for t in 0..num_tiles {
        if !seen[t] {
            order.push(t as u16);
        }
    }
    if !full && cs.ppm.is_empty() {
        // Without PPM an untouched tile needs no work at all.
        order.retain(|&t| tile_touches(siz, t, comp_win, reduce));
    }
}

/// A decode's output: an image, or planes of raw samples.
struct Decoded {
    raster: Option<Raster>,
    planes: Vec<Plane>,
}

/// What a decode keeps for the next one: the vectors every call fills, the lanes' scratch, the
/// ledger and the last output. A `Decoder` holds one with `keep` set; the free functions pass a
/// new one without, so they allocate and free as they always did. Every buffer is charged to the
/// ledger where and as much as a new one would be, then taken from here instead of allocated, so
/// the limits, the errors and the bytes are the same either way.
#[derive(Default)]
pub(crate) struct Scratch {
    keep: bool,
    /// The last call's ledger, reset in place when nothing else holds it.
    acct: Option<Arc<Account>>,
    /// The pixels of the last result, and the samples of its planes.
    pix: Vec<u8>,
    u16s: Pool<u16>,
    /// The list the planes of a result go in, kept empty.
    planes: Vec<Plane>,
    /// The lanes of a call, kept empty, and what each lane keeps between calls.
    lanes: Vec<Lane>,
    parts: Vec<LaneScratch>,
    comp_w: Vec<usize>,
    comp_h: Vec<usize>,
    comp_origin: Vec<(i64, i64)>,
    comp_win: Vec<Rect>,
    order: Vec<u16>,
    seen: Vec<bool>,
    dec: Vec<bool>,
    out_ch: Vec<usize>,
    out_planes: Vec<Option<(Rect, Vec<u16>)>>,
    canvas: Vec<Option<Vec<i32>>>,
    bands: Vec<Mutex<&'static mut [u8]>>,
    /// The vectors of the last parse, and the region calls' buffers.
    parse: codestream::ParseBufs,
    region: RegionBufs,
}

/// The buffers of a region call: `sparse_file`'s compact copy (`out`) and its parts, and the
/// component windows `check_region` uses.
#[derive(Default)]
struct RegionBufs {
    prefix: Vec<u8>,
    hdr: Vec<u8>,
    parts: Vec<(usize, usize)>,
    out: Vec<u8>,
    comps: Vec<codestream::SizComponent>,
    wins: Vec<Rect>,
}

impl Scratch {
    /// Keeps the call's ledger for the next call, when this is a `Decoder`'s.
    fn keep_ledger(&mut self, b: &Budget) {
        if self.keep {
            self.acct = Some(b.shared());
        }
    }
}

/// Runs the full pipeline (tier-2, tier-1, dequantization, inverse DWT, MCT, DC shift/clamp,
/// component upsampling) for every tile `reg` touches and assembles `reg`. Each component's
/// coding style (code-block style, DWT filter) is resolved per tile, so a tile-part COD or a
/// COC override on any component is honored. With `comps` set it decodes only the components
/// `component_set` gives and returns them as planes of raw samples, on each component's own
/// grid, instead of an image. `plan` is a call on an opened `Image`: Go's `Image` runs tier 2 of
/// every tile a call reads before it decodes any, so a tier-2 error of a later tile is reported
/// ahead of a decode error of an earlier one.
#[allow(clippy::too_many_arguments)]
fn decode_image(
    cs: &Codestream,
    data: &[u8],
    cdef: Option<&[CdefEntry]>,
    reg: Region,
    comps: Option<&[usize]>,
    opts: &Options,
    plan: bool,
    cancel: &AtomicBool,
    s: &mut Scratch,
    budget: &mut Budget,
) -> Result<Decoded> {
    // The result must not depend on the thread count: what more threads and lanes hold (scratch,
    // tile lanes, row buffers, tier-2 of the next tile) is optional, so a decode that ran out of
    // room with them is run again on one thread, from the same ledger state, and fails only if
    // that fails.
    let mark = budget.held();
    let r = decode_image_once(cs, data, cdef, reg, comps, opts, plan, cancel, s, budget);
    if matches!(r, Err(Error::LimitExceeded(_)))
        && opts.limits.max_threads > 1
        && budget.account().denied()
    {
        budget.release_to(mark);
        let mut one = opts.clone();
        one.limits.max_threads = 1;
        return decode_image_once(cs, data, cdef, reg, comps, &one, plan, cancel, s, budget);
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn decode_image_once(
    cs: &Codestream,
    data: &[u8],
    cdef: Option<&[CdefEntry]>,
    reg: Region,
    comps: Option<&[usize]>,
    opts: &Options,
    plan: bool,
    cancel: &AtomicBool,
    s: &mut Scratch,
    budget: &mut Budget,
) -> Result<Decoded> {
    let Scratch {
        keep,
        pix: kept_pix,
        u16s,
        planes: kept_planes,
        lanes,
        parts,
        comp_w,
        comp_h,
        comp_origin,
        comp_win,
        order: tile_order,
        seen,
        dec: dec_buf,
        out_ch,
        out_planes,
        canvas,
        bands: kept_bands,
        ..
    } = s;
    let keep = *keep;
    let limits = &opts.limits;
    let round = opts.rounding();
    let siz = &cs.siz;
    let (kind, cdef_perm, dec): (ColorModel, Option<Vec<usize>>, Option<&[bool]>) = match comps {
        None => {
            let kind = classify_output(siz)?;
            // A cdef box reassigns which component plays which output channel; resolve that
            // into a permutation now so a bad cdef is a clean error before any tile is
            // decoded, not a silent mis-render after.
            let perm = codestream::jp2::cdef_permutation(cdef, siz.components.len())?;
            (kind, perm, None)
        }
        Some(c) => {
            component_set(cs, c, dec_buf)?;
            (ColorModel::Gray16, None, Some(&dec_buf[..]))
        }
    };
    // RGN (region of interest) is parsed but has no tier-1/dequant consumer here; decoding past
    // it would silently shift the affected code-blocks' magnitudes instead of erroring.
    if !cs.rgn.is_empty() {
        return Err(unsupported!("region-of-interest (RGN) not supported"));
    }
    let full = reg == full_region(siz) && dec.is_none_or(|d| !d.contains(&false));
    let num_comps = siz.components.len();
    let w64 = reg.x1 - reg.x0;
    let h64 = reg.y1 - reg.y0;
    if w64 > limits.max_width as i64
        || h64 > limits.max_height as i64
        || (w64 as u128) * (h64 as u128) > limits.max_pixels as u128
    {
        return Err(limit!(
            "declared image {w64}x{h64} exceeds the {}-pixel decode limit",
            limits.max_pixels
        ));
    }
    let (w, h) = (w64 as usize, h64 as usize);
    let precision = siz.components[0].precision;
    // A subsampled component (XRsiz/YRsiz != 1) is assembled on its own canvas at its own size
    // and upsampled after the last tile; every other component is written straight into the
    // output. `comp_origin` is the canvas origin in the component's own sample units.
    fill(comp_w, num_comps, 0);
    fill(comp_h, num_comps, 0);
    fill(comp_origin, num_comps, (0, 0));
    fill(comp_win, num_comps, Rect::default());
    for c in 0..num_comps {
        let mut cw = comp_window(siz, c, &reg);
        match dec {
            None => {}
            Some(d) if !d[c] => cw = Rect::default(),
            Some(_) => {
                // Planes map reg by ceil, as T.800 B.3 does.
                cw.x0 = cw
                    .x0
                    .max(ceil_div(reg.x0, siz.components[c].xrsiz as i64))
                    .min(cw.x1);
                cw.y0 = cw
                    .y0
                    .max(ceil_div(reg.y0, siz.components[c].yrsiz as i64))
                    .min(cw.y1);
            }
        }
        comp_win[c] = cw;
        comp_w[c] = (cw.x1 - cw.x0) as usize;
        comp_h[c] = (cw.y1 - cw.y0) as usize;
        comp_origin[c] = (cw.x0, cw.y0);
    }
    let (comp_w, comp_h, comp_origin, comp_win): (&[usize], &[usize], &[(i64, i64)], &[Rect]) =
        (comp_w, comp_h, comp_origin, comp_win);
    for tp in &cs.tile_parts {
        // A region decode reads those tiles it touches.
        if !tp.rgn().is_empty()
            && (full || !plan || tile_touches(siz, tp.sot.isot, comp_win, reg.reduce))
        {
            return Err(unsupported!("region-of-interest (RGN) not supported"));
        }
    }
    decode_order(cs, full, comp_win, reg.reduce, tile_order, seen);
    let tile_order: &[u16] = tile_order;
    let data_len = data.len();

    let mut out_bytes = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(kind.bytes_per_pixel()))
        .ok_or_else(|| limit!("output size overflows"))?;
    if let Some(cs_) = comps {
        let mut n: u128 = 0;
        for &c in cs_ {
            n += 2 * comp_w[c] as u128 * comp_h[c] as u128;
        }
        // No more than the largest image output.
        if n > limits.max_output as u128 {
            return Err(limit!("{n} bytes of planes exceeds the decode limit"));
        }
        out_bytes = n as usize;
    } else if out_bytes > limits.max_output {
        return Err(limit!(
            "{out_bytes} bytes of output exceeds the decode limit"
        ));
    }

    // The output: an image, or planes by component, `None` for those not asked for.
    let mut o: Option<OutImage> = None;
    let mut bounds = Rect::default();
    out_planes.clear();
    match dec {
        None => {
            // Bounds in reduced coordinates relative to the (reduced) image origin.
            let bx = reg.x0 - ceil_div_pow2(siz.xosiz as i64, reg.reduce);
            let by = reg.y0 - ceil_div_pow2(siz.yosiz as i64, reg.reduce);
            bounds = Rect {
                x0: bx,
                y0: by,
                x1: bx + w as i64,
                y1: by + h as i64,
            };
            let bpp = kind.bytes_per_pixel();
            o = Some(OutImage {
                fmt: OutFmt {
                    stride: w * bpp,
                    bpp,
                    wide: matches!(
                        kind,
                        ColorModel::Gray16 | ColorModel::Rgba64 | ColorModel::Nrgba64
                    ),
                    gray_alpha: num_comps == 2,
                    opaque: num_comps == 3,
                    precision,
                },
                pix: budget.take_zeroed(out_bytes, || std::mem::take(kept_pix))?,
            });
        }
        Some(_) => {
            out_planes.resize_with(num_comps, || None);
            let s = reg.reduce;
            for &c in comps.unwrap_or(&[]) {
                let cw = comp_win[c];
                let ox = ceil_div_pow2(
                    ceil_div(siz.xosiz as i64, siz.components[c].xrsiz as i64),
                    s,
                );
                let oy = ceil_div_pow2(
                    ceil_div(siz.yosiz as i64, siz.components[c].yrsiz as i64),
                    s,
                );
                let b = Rect {
                    x0: cw.x0 - ox,
                    y0: cw.y0 - oy,
                    x1: cw.x1 - ox,
                    y1: cw.y1 - oy,
                };
                let n = comp_w[c] * comp_h[c];
                out_planes[c] = Some((b, budget.take_zeroed(n, || u16s.pick(n))?));
            }
        }
    }
    // out_ch maps a component to its output channel (identity unless a cdef box reorders them).
    out_ch.clear();
    out_ch.extend(0..num_comps);
    if let Some(perm) = &cdef_perm {
        for (pos, &src) in perm.iter().enumerate() {
            out_ch[src] = pos;
        }
    }
    if num_comps == 2 {
        // Gray+alpha: channel 1 is alpha, stored in A, not G.
        for ch in out_ch.iter_mut() {
            if *ch == 1 {
                *ch = 3;
            }
        }
    }
    let out_ch: &[usize] = out_ch;
    canvas.clear();
    canvas.resize_with(num_comps, || None);
    if dec.is_none() {
        for c in 0..num_comps {
            if siz.components[c].xrsiz != 1 || siz.components[c].yrsiz != 1 {
                canvas[c] = Some(budget.vec(comp_w[c] * comp_h[c], 0i32)?);
            }
        }
    }
    let out = Mutex::new(OutState {
        o,
        out_planes,
        canvas,
    });

    // The image's pixels go to the lanes as bands; `out` keeps the other destinations.
    let mut img = lock(&out).o.take();
    let bands: Option<Bands> = img.as_mut().map(|im| {
        let mut rows: Vec<Mutex<&mut [u8]>> = recycle(std::mem::take(kept_bands));
        rows.extend(
            im.pix
                .chunks_mut((BAND_ROWS * im.fmt.stride).max(1))
                .map(Mutex::new),
        );
        Bands { fmt: im.fmt, rows }
    });

    // The tiles run in lanes, each one tile at a time, the call's threads split across the lanes
    // (Go's `lane`): tier 2 in tile order under `turn`, then tier 1, the wavelet, the colour
    // transform and the output on the lane's threads.
    let acct = budget.shared();
    let turn = Turn {
        m: Mutex::new(TurnState {
            next: 0,
            stopped: false,
            cur: Cursor::new(), // the shared PPM cursor and budgets, threaded across every tile in order
        }),
        cv: Condvar::new(),
    };
    let work = AtomicI64::new(T1_WORK_PER_BYTE * data_len as i64 + T1_WORK_SLACK);
    let nw_total = acct.cap_threads(limits.max_threads.max(1));
    let n_lanes = if plan || !cs.ppm.is_empty() {
        1
    } else {
        tile_lanes(
            cs,
            tile_order.len(),
            reg.reduce,
            nw_total,
            data_len,
            acct.remaining(),
        )
    };
    // The call's threads are shared out over the lanes as Go does: `nw_total / lanes` each, the
    // first `nw_total % lanes` lanes one more. With as many lanes as threads each lane has one, so
    // every stage of it is serial and the lanes are items of `drive` on the pool: they never wait
    // in rayon. With fewer lanes (but more than one) a lane waits for its nested passes on the
    // pool, and a pool worker that waits takes other jobs, another call's lane included, so those
    // lanes run on the caller and `n_lanes - 1` `std::thread::scope` threads started once for the
    // call (`drive_threads`), which only block. The threads add up to `nw_total` (a lane thread
    // each, and `nw - 1` pool workers for the nested passes of each lane). A `Decoder`'s lanes take
    // what its earlier calls kept.
    lanes.clear();
    for i in 0..n_lanes {
        let mut part = parts.pop().unwrap_or_default();
        part.keep = keep;
        let mut lb = budget.child();
        lb.dwt = std::mem::take(&mut part.dwt);
        lb.dwt.keep = keep;
        lanes.push(Lane {
            budget: lb,
            scratch: part,
            nw: nw_total / n_lanes + usize::from(i < nw_total % n_lanes),
        });
    }

    // The tiles of `tile_order` whose tier 2 has run, and the one being decoded (one lane only,
    // for `plan`).
    let (planned, at) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let tile = |lane: &mut Lane, k: usize, &tile_idx: &u16| -> Result<()> {
        at.store(k, Ordering::Relaxed);
        let mut abandon = Abandon {
            turn: &turn,
            done: false,
        };
        let mark = lane.budget.held();
        let spare = lane.scratch.planes.bytes;
        // What a `Decoder`'s kept tier-2 lists are charged as the tile begins.
        let kept = lane.scratch.t2_bytes;
        if !full && !tile_touches(siz, tile_idx, comp_win, reg.reduce) {
            // The shared PPM cursor still has to pass this tile.
            let r = in_turn(&turn, k, cancel, |cur| {
                tier2::decode_tile(cs, data, tile_idx, cur, None, cancel, &mut lane.budget)
                    .map(|_| ())
            });
            abandon.done = true;
            r?;
            planned.store(k + 1, Ordering::Relaxed);
            lane.budget.release_to(mark);
            return Ok(());
        }
        let (tile_cod, tile_coc, _) = tile_overrides(cs, tile_idx);
        let style = |c: usize| effective_coding_style(cs, tile_cod, &tile_coc, c);
        // `tops` is the highest resolution decoded, `wins` the part of it `reg` reads.
        let (tops, wins) = (&mut lane.scratch.tops, &mut lane.scratch.wins);
        // The tier-2 lists of the lane's last tile are charged until this tile's tier 2 takes them
        // and charges what it uses, as new ones.
        let mut t2 = std::mem::take(&mut lane.scratch.t2);
        lane.budget
            .release(std::mem::take(&mut lane.scratch.t2_bytes));
        t2.keep = keep;
        let (mut kept_want, want_spare) = (t2.want.take(), &mut lane.scratch.want_spare);
        let mut want_of = |tg: &TileGeom| -> Result<Option<TileWant>> {
            tile_windows(
                tg,
                tile_idx,
                &|c| style(c).transformation == 1,
                comp_win,
                reg.reduce,
                full,
                tops,
                wins,
                kept_want.take(),
                want_spare,
            )
        };
        let r = in_turn(&turn, k, cancel, |cur| {
            tier2::decode_tile_in(
                cs,
                data,
                tile_idx,
                cur,
                Some(&mut want_of),
                cancel,
                &mut lane.budget,
                &mut t2,
            )
        });
        abandon.done = true;
        let tile_data = r?;
        let (pkts, tg) = (&t2.pkts, &t2.tg);
        planned.store(k + 1, Ordering::Relaxed);
        let mct = effective_mct(cs, tile_cod);
        if mct == 2 {
            return Err(unsupported!(
                "tile {tile_idx}: array-based multiple component transform (MCT=2) not supported"
            ));
        }
        let decode_into = |c: usize, lane: &mut Lane| -> Result<PlaneBuf> {
            let mut groups = std::mem::take(&mut lane.scratch.groups);
            let charged = group_code_blocks(pkts, c, &mut lane.budget, &mut groups)?;
            // A free call drops the index here, as it always did: only its groups are charged
            // while the component decodes. So does a `Decoder` when the lists already hold over
            // `KEPT_SMALL`, which it frees after the component.
            let small = keep && groups.heap() <= KEPT_SMALL;
            if !small {
                groups.index = HashMap::new();
            }
            let (top, win) = (lane.scratch.tops[c], lane.scratch.wins[c]);
            let p = decode_component(
                cs,
                tg,
                &tile_data,
                &groups.groups,
                tile_idx,
                c,
                style(c),
                top,
                win,
                round,
                &work,
                &mut lane.scratch,
                lane.nw,
                &mut lane.budget,
                cancel,
            );
            lane.budget.release(charged);
            // A `Decoder` keeps the lists and the jobs for the next component, uncharged, only
            // when they hold at most `KEPT_SMALL`; it frees bigger ones, as a free call does.
            if keep {
                groups.recycle();
                if small && groups.heap() + held_by(&lane.scratch.jobs) <= KEPT_SMALL {
                    lane.scratch.groups = groups;
                } else {
                    lane.scratch.jobs = Vec::new();
                }
            }
            p
        };
        let nwl = lane.nw;

        let mut first = 0;
        // OpenJPEG applies MCT only when it actually has 3 components to mix, silently skipping
        // it rather than erroring otherwise, and picks RCT vs ICT from component 0's filter
        // alone. A component 1/2 whose effective filter disagrees with component 0 can't be
        // mixed by either transform, so that's an error rather than a silent guess.
        if mct == 1 && num_comps >= 3 {
            // The inverse transforms mix components 0-2 sample-for-sample, so they need
            // equal-size windows.
            let wins = &lane.scratch.wins;
            let n0 = wins[0].x1 - wins[0].x0;
            let m0 = wins[0].y1 - wins[0].y0;
            for c in 1..3 {
                if wins[c].x1 - wins[c].x0 != n0 || wins[c].y1 - wins[c].y0 != m0 {
                    return Err(invalid!(
                        "tile {tile_idx} has mismatched component dimensions across components 0-2 (differing XRsiz/YRsiz) with the multiple component transform enabled"
                    ));
                }
            }
            let reversible = style(0).transformation == 1;
            if (style(1).transformation == 1) != reversible
                || (style(2).transformation == 1) != reversible
            {
                return Err(invalid!(
                    "tile {tile_idx} has mismatched reversible/irreversible filters across components 0-2 with the multiple component transform enabled"
                ));
            }
            let mut p0 = decode_into(0, &mut *lane)?;
            let mut p1 = decode_into(1, &mut *lane)?;
            let mut p2 = decode_into(2, &mut *lane)?;
            mct_rows(
                [&mut p0, &mut p1, &mut p2],
                n0 as usize,
                m0 as usize,
                reversible,
                round,
                nwl,
                &acct,
                cancel,
            )?;
            for (c, p) in [&p0, &p1, &p2].into_iter().enumerate() {
                emit(
                    c,
                    p,
                    siz,
                    round,
                    &out,
                    bands.as_ref(),
                    comp_w,
                    comp_origin,
                    out_ch,
                    nwl,
                    k,
                    &mut lane.scratch,
                    &mut lane.budget,
                    cancel,
                )?;
            }
            for p in [p0, p1, p2] {
                lane.scratch.planes.give(p, &mut lane.budget, keep);
            }
            first = 3;
        }
        for c in first..num_comps {
            let p = decode_into(c, &mut *lane)?;
            emit(
                c,
                &p,
                siz,
                round,
                &out,
                bands.as_ref(),
                comp_w,
                comp_origin,
                out_ch,
                nwl,
                k,
                &mut lane.scratch,
                &mut lane.budget,
                cancel,
            )?;
            lane.scratch.planes.give(p, &mut lane.budget, keep);
        }
        // What the spare planes hold stays charged; the rest goes back to where the tile began.
        lane.budget
            .release_to(mark - kept - spare + lane.scratch.planes.bytes);
        if keep {
            // The tier-2 lists, for the lane's next tile, up to `T2_KEPT`: after the call's last
            // tile on one lane they are kept uncharged, as between calls (the lanes end with that
            // tile). Else they are kept charged what they hold, only where the call's ledger has
            // already been as high, so the charge does not raise its peak. Else freed, as a free
            // call frees them.
            t2.recycle();
            let held = t2.heap();
            let ledger = lane.budget.account();
            if held <= T2_KEPT {
                if n_lanes == 1 && k + 1 == tile_order.len() {
                    lane.scratch.t2 = t2;
                } else if ledger.used() + held <= ledger.peak() && lane.budget.charge(held).is_ok()
                {
                    lane.scratch.t2 = t2;
                    lane.scratch.t2_bytes = held;
                }
            }
        }
        Ok(())
    };
    let res = if n_lanes == nw_total {
        drive(
            n_lanes,
            &acct,
            cancel,
            &mut lanes[..],
            tile_order.iter(),
            tile,
        )
    } else {
        drive_threads(
            n_lanes,
            &acct,
            cancel,
            &mut lanes[..],
            tile_order.iter(),
            tile,
        )
    };
    if let Some(b) = bands {
        *kept_bands = recycle(b.rows);
    }
    lock(&out).o = img;
    // The lanes end here; a `Decoder` keeps what each holds, uncharged, for its next call. Not
    // when the call still charges the ledger after this (upsampling subsampled components, a
    // retry on one thread, tier 2 of the tiles left after an error): a buffer the ledger released
    // is not held while the call charges more, so those lanes are freed as a free call's are.
    let park = keep && res.is_ok() && !lock(&out).canvas.iter().any(Option::is_some);
    for mut l in lanes.drain(..) {
        if park {
            let mut part = std::mem::take(&mut l.scratch);
            part.dwt = std::mem::take(&mut l.budget.dwt);
            part.park();
            parts.push(part);
        }
    }
    if let Err(e) = res {
        let done = planned.load(Ordering::Relaxed);
        if plan
            && matches!(e, Error::Invalid(_) | Error::Unsupported(_))
            && done == at.load(Ordering::Relaxed) + 1
        {
            // The error came after tier 2 of this tile: run tier 2 of the tiles still to come,
            // as Go's Image did before it decoded anything.
            let mut cur = turn.m.into_inner().unwrap_or_else(|e| e.into_inner()).cur;
            for &t in &tile_order[done..] {
                let mark = budget.held();
                plan_tile(
                    cs, data, t, comp_win, reg.reduce, full, &mut cur, cancel, budget,
                )?;
                budget.release_to(mark);
            }
        }
        return Err(e);
    }

    let OutState {
        o,
        out_planes,
        canvas,
    } = out.into_inner().unwrap_or_else(|e| e.into_inner());
    if dec.is_some() {
        let mut planes = std::mem::take(kept_planes);
        planes.clear();
        for &c in comps.unwrap_or(&[]) {
            if let Some((b, pix)) = out_planes[c].take() {
                planes.push(Plane { bounds: b, pix });
            }
        }
        return Ok(Decoded {
            raster: None,
            planes,
        });
    }

    let Some(mut o) = o else {
        return Err(Error::Internal("no output image".into()));
    };
    for c in 0..num_comps {
        let Some(cv) = canvas[c].take() else {
            continue;
        };
        let xr = siz.components[c].xrsiz as i64;
        let yr = siz.components[c].yrsiz as i64;
        let up = upsample_component(
            &cv,
            comp_w[c],
            comp_h[c],
            comp_origin[c].0,
            comp_origin[c].1,
            xr,
            yr,
            w,
            h,
            reg.x0,
            reg.y0,
            budget,
            cancel,
        )?;
        let fmt = o.fmt;
        for y in 0..h {
            check_cancel(cancel)?;
            fmt.put_row(
                &mut o.pix,
                y * fmt.stride,
                out_ch[c],
                &up[y * w..(y + 1) * w],
            );
        }
    }
    Ok(Decoded {
        raster: Some(Raster {
            color_model: kind,
            bounds,
            stride: o.fmt.stride,
            pix: o.pix,
        }),
        planes: Vec::new(),
    })
}

/// Tier 2 of tile `tile_idx` for a decode of `comp_win`, its result dropped: the planning pass
/// of `decode_image` for an opened `Image`. It reads what the decode would, so it fails as the
/// decode's tier 2 does.
#[allow(clippy::too_many_arguments)]
fn plan_tile(
    cs: &Codestream,
    data: &[u8],
    tile_idx: u16,
    comp_win: &[Rect],
    reduce: u32,
    full: bool,
    cur: &mut Cursor,
    cancel: &AtomicBool,
    budget: &mut Budget,
) -> Result<()> {
    let siz = &cs.siz;
    if !full && !tile_touches(siz, tile_idx, comp_win, reduce) {
        tier2::decode_tile(cs, data, tile_idx, cur, None, cancel, budget)?;
        return Ok(());
    }
    let (tile_cod, tile_coc, _) = tile_overrides(cs, tile_idx);
    let rev = |c: usize| effective_coding_style(cs, tile_cod, &tile_coc, c).transformation == 1;
    let (mut tops, mut wins) = (Vec::new(), Vec::new());
    let mut want_of = |tg: &TileGeom| -> Result<Option<TileWant>> {
        tile_windows(
            tg,
            tile_idx,
            &rev,
            comp_win,
            reduce,
            full,
            &mut tops,
            &mut wins,
            None,
            &mut Vec::new(),
        )
    };
    tier2::decode_tile(cs, data, tile_idx, cur, Some(&mut want_of), cancel, budget)?;
    Ok(())
}

/// The `a`, `b` and `c` planes of an inverse multiple component transform, `n` samples wide and
/// `m` rows high from each plane's `off`, in rows of `ROW_CHUNK` on up to `nw` threads.
fn mct_rows(
    [p0, p1, p2]: [&mut PlaneBuf; 3],
    n: usize,
    m: usize,
    reversible: bool,
    round: Rounding,
    nw: usize,
    acct: &Account,
    cancel: &AtomicBool,
) -> Result<()> {
    if n == 0 || m == 0 {
        return Ok(());
    }
    let nw = nw.min(n * m / SAMPLES_PER_THREAD).max(1);
    let (s0, s1, s2) = (p0.stride, p1.stride, p2.stride);
    let (o0, o1, o2) = (p0.off, p1.off, p2.off);
    let mut units = vec![(); nw];
    if reversible {
        let it = p0.i[o0..o0 + (m - 1) * s0 + n]
            .chunks_mut(ROW_CHUNK * s0)
            .zip(p1.i[o1..o1 + (m - 1) * s1 + n].chunks_mut(ROW_CHUNK * s1))
            .zip(p2.i[o2..o2 + (m - 1) * s2 + n].chunks_mut(ROW_CHUNK * s2));
        drive(nw, acct, cancel, &mut units, it, |_, _, ((a, b), c)| {
            for y in 0..a.len().div_ceil(s0) {
                dsp::inverse_rct(
                    &mut a[y * s0..y * s0 + n],
                    &mut b[y * s1..y * s1 + n],
                    &mut c[y * s2..y * s2 + n],
                );
            }
            Ok(())
        })
    } else {
        let it = p0.f[o0..o0 + (m - 1) * s0 + n]
            .chunks_mut(ROW_CHUNK * s0)
            .zip(p1.f[o1..o1 + (m - 1) * s1 + n].chunks_mut(ROW_CHUNK * s1))
            .zip(p2.f[o2..o2 + (m - 1) * s2 + n].chunks_mut(ROW_CHUNK * s2));
        drive(nw, acct, cancel, &mut units, it, |_, _, ((a, b), c)| {
            for y in 0..a.len().div_ceil(s0) {
                dsp::inverse_ict(
                    &mut a[y * s0..y * s0 + n],
                    &mut b[y * s1..y * s1 + n],
                    &mut c[y * s2..y * s2 + n],
                    round,
                );
            }
            Ok(())
        })
    }
}

/// Converts `th` rows of samples to a destination `slab` of rows `stride` elements apart, the
/// first element of each row's `tw` samples at `ox`: `each(dst, row, y)` stores row `y` at the
/// start of `dst`, with `row` a buffer of its own to convert through. Rows go in chunks of
/// `ROW_CHUNK` to up to `bufs.len()` threads.
fn rows_to<T: Send>(
    slab: &mut [T],
    stride: usize,
    ox: usize,
    nw: usize,
    acct: &Account,
    cancel: &AtomicBool,
    bufs: &mut [Vec<i32>],
    each: &(impl Fn(&mut [T], &mut [i32], usize) + Sync),
) -> Result<()> {
    drive(
        nw,
        acct,
        cancel,
        bufs,
        slab.chunks_mut(ROW_CHUNK * stride),
        |row, i, chunk| {
            for (j, dst) in chunk.chunks_mut(stride).enumerate() {
                each(&mut dst[ox..], &mut row[..], i * ROW_CHUNK + j);
            }
            Ok(())
        },
    )
}

/// Writes tile component `c`, finished in `p`, to its destination, converting its rows on up to
/// `nw` threads (a row buffer each, charged to `budget`, from the lane's `scratch`): a plane of
/// samples or a subsampled component's canvas under the lock on `out`, or the image's bands, one
/// lock each, starting at band `rot` of the tile's so that lanes on tiles side by side are not at
/// the same one.
#[allow(clippy::too_many_arguments)]
fn emit(
    c: usize,
    p: &PlaneBuf,
    siz: &Siz,
    round: Rounding,
    out: &Mutex<OutState<'_>>,
    img: Option<&Bands>,
    comp_w: &[usize],
    comp_origin: &[(i64, i64)],
    out_ch: &[usize],
    nw: usize,
    rot: usize,
    scratch: &mut LaneScratch,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<()> {
    let (to_canvas, to_plane) = {
        let o = lock(out);
        (
            o.canvas[c].is_some(),
            o.out_planes.get(c).is_some_and(|x| x.is_some()),
        )
    };
    if !to_canvas && !to_plane && img.is_none() {
        return Ok(());
    }
    let (tw, th) = (p.w, p.h);
    if tw == 0 || th == 0 {
        return Ok(());
    }
    let ox = (p.x0 - comp_origin[c].0) as usize;
    let oy = (p.y0 - comp_origin[c].1) as usize;
    let prec = siz.components[c].precision as i32;
    let signed = siz.components[c].signed;
    let conv = |row: &mut [i32], y: usize| {
        let a = p.off + y * p.stride;
        if p.rev {
            dsp::samples_r(row, &p.i[a..a + tw], prec, signed);
        } else if round == Rounding::JJ2000 {
            dsp::samples_jj(row, &p.f[a..a + tw], prec);
        } else {
            dsp::samples_f(row, &p.f[a..a + tw], prec);
        }
    };
    // 9/7 samples that need no integer row: they go from float to the output in one pass.
    let flt = !p.rev && round != Rounding::JJ2000;
    let acct = budget.shared();
    let mark = budget.held();
    let want = nw.min(tw * th / SAMPLES_PER_THREAD).max(1);
    let mut bufs = std::mem::take(&mut scratch.rows);
    bufs.clear();
    let pool = &mut scratch.row_pool;
    bufs.push(budget.take_vec(tw, || pool.pick(tw))?);
    for _ in 1..want {
        match budget.take_vec(tw, || pool.pick(tw)) {
            Ok(b) => bufs.push(b),
            Err(_) => break,
        }
    }
    let n = bufs.len();
    let r = if to_canvas || to_plane {
        let mut o = lock(out);
        if let Some(cv) = o.canvas[c].as_mut() {
            let stride = comp_w[c];
            let slab = &mut cv[oy * stride..oy * stride + (th - 1) * stride + ox + tw];
            let each = |dst: &mut [i32], row: &mut [i32], y: usize| {
                conv(row, y);
                dst[..tw].copy_from_slice(row);
            };
            rows_to(slab, stride, ox, n, &acct, cancel, &mut bufs, &each)
        } else if let Some(Some((b, pix))) = o.out_planes.get_mut(c) {
            let stride = (b.x1 - b.x0) as usize;
            let slab = &mut pix[oy * stride..oy * stride + (th - 1) * stride + ox + tw];
            let each = |dst: &mut [u16], row: &mut [i32], y: usize| {
                if flt {
                    let a = p.off + y * p.stride;
                    dsp::pack_u16(&mut dst[..tw], &p.f[a..a + tw], prec);
                    return;
                }
                conv(row, y);
                for (d, &v) in dst.iter_mut().zip(row.iter()) {
                    *d = v as u16;
                }
            };
            rows_to(slab, stride, ox, n, &acct, cancel, &mut bufs, &each)
        } else {
            Ok(())
        }
    } else if let Some(img) = img {
        let fmt = img.fmt;
        let (stride, ch) = (fmt.stride, out_ch[c]);
        let wide16 = flt && fmt.wide && prec == 16 && fmt.precision == 16;
        let rgba8 = flt && !fmt.wide && fmt.bpp == 4 && ch < 4;
        let each = |dst: &mut [u8], row: &mut [i32], y: usize| {
            if wide16 {
                let a = p.off + y * p.stride;
                fmt.put_row_f16(dst, 0, ch, &p.f[a..a + tw]);
                return;
            }
            if rgba8 {
                let a = p.off + y * p.stride;
                fmt.put_row_u8x4(dst, ch, &p.f[a..a + tw], prec);
                return;
            }
            conv(row, y);
            fmt.put_row(dst, 0, ch, row);
        };
        let (first, count) = (
            oy / BAND_ROWS,
            (oy + th - 1) / BAND_ROWS + 1 - oy / BAND_ROWS,
        );
        // One band an item, from band `rot` of the tile's on; a band's rows are converted by the
        // thread that has it.
        drive(
            n,
            &acct,
            cancel,
            &mut bufs,
            (0..count).map(|i| first + (i + rot) % count),
            |row, _, b| {
                // The tile's rows `ya..yb` (image rows) are in band `b`.
                let (ya, yb) = ((b * BAND_ROWS).max(oy), ((b + 1) * BAND_ROWS).min(oy + th));
                let mut band = lock(&img.rows[b]);
                let at = (ya - b * BAND_ROWS) * stride;
                let slab = &mut band[at..at + (yb - ya - 1) * stride + (ox + tw) * fmt.bpp];
                for (j, dst) in slab.chunks_mut(stride).enumerate() {
                    each(&mut dst[ox * fmt.bpp..], &mut row[..], ya - oy + j);
                }
                Ok(())
            },
        )
    } else {
        Ok(())
    };
    // A `Decoder` keeps row buffers, uncharged, only up to `KEPT_SMALL`; the rest are freed.
    for b in bufs.drain(..) {
        let pooled: usize = scratch.row_pool.0.iter().map(Vec::capacity).sum();
        let small = (pooled + b.capacity()) * size_of::<i32>() <= KEPT_SMALL;
        scratch.row_pool.give(b, scratch.keep && small);
    }
    if scratch.keep {
        scratch.rows = bufs;
    }
    budget.release_to(mark);
    r
}

// -- regions --

/// `rect` at `reduce` as a region of the image.
fn region_of(siz: &Siz, rect: Rect, reduce: u32) -> Result<Region> {
    let xo = siz.xosiz as i64;
    let yo = siz.yosiz as i64;
    let rect = rect.intersect(&Rect {
        x0: 0,
        y0: 0,
        x1: siz.xsiz as i64 - xo,
        y1: siz.ysiz as i64 - yo,
    });
    if rect.empty() {
        return Err(invalid!("region does not intersect the image"));
    }
    let reg = Region {
        x0: ceil_div_pow2(rect.x0 + xo, reduce),
        y0: ceil_div_pow2(rect.y0 + yo, reduce),
        x1: ceil_div_pow2(rect.x1 + xo, reduce),
        y1: ceil_div_pow2(rect.y1 + yo, reduce),
        reduce,
    };
    if reg.x0 >= reg.x1 || reg.y0 >= reg.y1 {
        return Err(invalid!("region is empty at reduce {reduce}"));
    }
    Ok(reg)
}

/// `rect` at `reduce` as a region of `cs`, or an error if `cs` cannot give it. `wins` is scratch
/// for the component windows.
fn check_region(cs: &Codestream, rect: Rect, reduce: u32, wins: &mut Vec<Rect>) -> Result<Region> {
    let reg = region_of(&cs.siz, rect, reduce)?;
    // OpenJPEG checks every COD/COC it reads, used by a tile or not, and never reads the
    // tile-parts of tiles the region misses; skip those here too. The main COD, the main COCs,
    // then those of the tile-parts, in that order.
    wins.clear();
    wins.extend((0..cs.siz.components.len()).map(|c| comp_window(&cs.siz, c, &reg)));
    let check = |s: &CodingStyle| -> Result<()> {
        if reduce > s.num_decomp_levels as u32 {
            return Err(invalid!(
                "reduce {reduce} exceeds the {} decomposition levels of a component",
                s.num_decomp_levels
            ));
        }
        Ok(())
    };
    check(&cs.cod.coding_style)?;
    for c in &cs.coc {
        check(&c.coding_style)?;
    }
    for tp in &cs.tile_parts {
        if !tile_touches(&cs.siz, tp.sot.isot, wins, reduce) {
            continue;
        }
        if let Some(c) = tp.cod() {
            check(&c.coding_style)?;
        }
        for c in tp.coc() {
            check(&c.coding_style)?;
        }
    }
    Ok(reg)
}

// -- reading and parsing --

/// `n` bytes of `data` at `off`, charged to `budget` first, or `None` if `data` does not hold
/// them: Go's `read` in `sparseFile`.
fn sparse_read<'a>(
    data: &'a [u8],
    budget: &mut Budget,
    off: usize,
    n: usize,
) -> Result<Option<&'a [u8]>> {
    let Some(b) = off.checked_add(n).and_then(|e| data.get(off..e)) else {
        return Ok(None);
    };
    budget.charge(n)?;
    Ok(Some(b))
}

/// Go's `sparseFile`: for a JP2 file or raw codestream in `data`, a compact copy (the boxes
/// `jp2_codestream` reads, the main header, the tile-parts of the tiles `rect` at `reduce`
/// touches, EOC), found by walking the SOTs by Psot without parsing tile-part headers, in
/// `rb.out` (`true`). `false` means parse all of `data`: a PPM codestream, an invalid region,
/// Psot below 14, no EOC, no tile-part to keep, or a layout it does not walk. Charges `budget`
/// where Go reserves, and keeps the charge on `false`, as Go's ticket does. `rb`'s other vectors
/// are its scratch.
fn sparse_file(
    data: &[u8],
    rect: Rect,
    reduce: u32,
    budget: &mut Budget,
    cancel: &AtomicBool,
    rb: &mut RegionBufs,
) -> Result<bool> {
    use codestream::jp2::{BOX_JP, BOX_JP2C, BOX_JP2H};
    let RegionBufs {
        prefix,
        hdr,
        parts,
        out,
        comps,
        wins,
    } = rb;
    prefix.clear();
    hdr.clear();
    parts.clear();
    out.clear();
    let mut off = 0usize;
    if sparse_read(data, budget, 0, JP2_SIGNATURE.len())? == Some(&JP2_SIGNATURE[..]) {
        loop {
            check_cancel(cancel)?;
            let Some(h) = sparse_read(data, budget, off, 16)? else {
                return Ok(false);
            };
            let mut n = u64::from(codestream::be32(h));
            let typ = codestream::be32(&h[4..]);
            let mut hl = 8usize;
            if n == 1 {
                n = u64::from(codestream::be32(&h[8..])) << 32
                    | u64::from(codestream::be32(&h[12..]));
                hl = 16;
            }
            if typ == BOX_JP2C {
                off += hl;
                break;
            }
            if n < hl as u64 || n > i32::MAX as u64 {
                return Ok(false); // to-end or malformed box before jp2c
            }
            let n = n as usize;
            if typ == BOX_JP || typ == BOX_JP2H {
                let Some(b) = sparse_read(data, budget, off, n)? else {
                    return Ok(false);
                };
                budget.charge(2 * n)?;
                prefix.try_reserve(n).map_err(|_| Error::OutOfMemory)?;
                prefix.extend_from_slice(b);
            }
            off += n;
        }
        // The jp2c box as the parser then sees it: no length, so it runs to the end.
        prefix.extend_from_slice(&[0, 0, 0, 0, b'j', b'p', b'2', b'c']);
    }
    // Main header: SOC, then marker segments up to the first SOT.
    let Some(soc) = sparse_read(data, budget, off, 2)? else {
        return Ok(false);
    };
    if codestream::be16(soc) != codestream::MARKER_SOC {
        return Ok(false);
    }
    hdr.extend_from_slice(soc);
    let mut pos = off + 2;
    loop {
        check_cancel(cancel)?;
        let Some(m) = data.get(pos..pos + 4) else {
            return Ok(false);
        };
        if m[0] != 0xFF || codestream::be16(m) == codestream::MARKER_PPM {
            return Ok(false);
        }
        if codestream::be16(m) == codestream::MARKER_SOT {
            break;
        }
        let n = codestream::be16(&m[2..]) as usize;
        if n < 2 {
            return Ok(false);
        }
        budget.charge(2 * (n + 2))?;
        let Some(seg) = data.get(pos..pos + n + 2) else {
            return Ok(false);
        };
        hdr.try_reserve(n + 2).map_err(|_| Error::OutOfMemory)?;
        hdr.extend_from_slice(seg);
        pos += n + 2;
    }
    let Ok((siz, _)) = codestream::parse_siz_in(hdr, 2, std::mem::take(comps)) else {
        return Ok(false);
    };
    let Ok(reg) = region_of(&siz, rect, reduce) else {
        return Ok(false);
    };
    wins.clear();
    wins.extend((0..siz.components.len()).map(|c| comp_window(&siz, c, &reg)));
    // Walk the SOTs by their Psot, noting the tile-parts to keep.
    let mut total = prefix.len() + hdr.len() + 2;
    loop {
        check_cancel(cancel)?;
        let sot = data.get(pos..).map_or(&[][..], |d| &d[..d.len().min(12)]);
        if sot.len() >= 2 && codestream::be16(sot) == codestream::MARKER_EOC {
            break;
        }
        if sot.len() < 12
            || codestream::be16(sot) != codestream::MARKER_SOT
            || codestream::be16(&sot[2..]) != 10
        {
            return Ok(false);
        }
        let psot = codestream::be32(&sot[6..]) as usize;
        if psot < 14 {
            return Ok(false);
        }
        if tile_touches(&siz, codestream::be16(&sot[4..]), wins, reduce) {
            budget.charge(32)?;
            parts.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
            parts.push((pos, psot));
            let Some(t) = total.checked_add(psot) else {
                return Ok(false);
            };
            total = t;
        }
        let Some(p) = pos.checked_add(psot) else {
            return Ok(false);
        };
        pos = p;
    }
    *comps = siz.components;
    if parts.is_empty() {
        return Ok(false); // the parser wants a tile-part; the region's tiles have none
    }
    budget.charge(total)?;
    out.try_reserve_exact(total)
        .map_err(|_| Error::OutOfMemory)?;
    out.extend_from_slice(prefix);
    out.extend_from_slice(hdr);
    for &(p, n) in parts.iter() {
        let Some(b) = data.get(p..p + n) else {
            out.clear();
            return Ok(false);
        };
        out.extend_from_slice(b);
    }
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(true)
}

/// Parses a whole JP2 file or raw codestream held in `data`, into the vectors of `bufs`.
fn parse_file<'a>(
    data: &'a [u8],
    limits: &Limits,
    budget: &mut Budget,
    mode: codestream::Parse,
    cancel: &AtomicBool,
    bufs: &mut codestream::ParseBufs,
) -> Result<(Codestream, &'a [u8], Option<Vec<CdefEntry>>)> {
    let mut stream = data;
    let mut cdef = None;
    if data.starts_with(&JP2_SIGNATURE) {
        let (s, c) = codestream::jp2::jp2_codestream(data)?;
        stream = s;
        cdef = c;
    }
    // The parse checks its allocations against what the ledger has left (bytes the call already
    // holds count), before making them; `parse_cost` is charged once the parse is done.
    let limits = Limits {
        max_alloc: budget.account().remaining(),
        ..*limits
    };
    let cs = codestream::parse_tiles_in(stream, false, &limits, mode, cancel, bufs)?;
    budget.charge(cs.parse_cost.max(0) as usize)?;
    Ok((cs, stream, cdef))
}

/// `held` is memory the caller already holds for `data`, charged first. `s` is a `Decoder`'s
/// scratch, or a new one.
fn decode_whole(
    data: &[u8],
    comps: Option<&[usize]>,
    opts: &Options,
    cancel: &AtomicBool,
    held: usize,
    s: &mut Scratch,
) -> Result<Decoded> {
    guard(|| {
        check_cancel(cancel)?;
        with_ledger(opts, data.len(), s, |s, budget| {
            budget.charge(held)?;
            let (cs, stream, cdef) = parse_file(
                data,
                &opts.limits,
                budget,
                codestream::Parse::Whole,
                cancel,
                &mut s.parse,
            )?;
            let reg = full_region(&cs.siz);
            let r = decode_image(
                &cs,
                stream,
                cdef.as_deref(),
                reg,
                comps,
                opts,
                false,
                cancel,
                s,
                budget,
            );
            if s.keep {
                s.parse.reclaim(cs);
            }
            r
        })
    })
}

/// Runs `f` on a new ledger for a call under `opts` on `input_len` bytes, then keeps the ledger
/// in `s` when it is a `Decoder`'s.
fn with_ledger<T>(
    opts: &Options,
    input_len: usize,
    s: &mut Scratch,
    f: impl FnOnce(&mut Scratch, &mut Budget) -> Result<T>,
) -> Result<T> {
    // Lanes a panic left behind are freed before the call charges anything.
    s.lanes.clear();
    let mut budget = Budget::for_call_in(opts, input_len, &mut s.acct);
    let r = f(s, &mut budget);
    s.keep_ledger(&budget);
    r
}

fn raster_of(d: Decoded) -> Result<Raster> {
    d.raster
        .ok_or_else(|| Error::Internal("no output image".into()))
}

/// Decodes a JPEG 2000 image: either a raw codestream (starting with the SOC marker) or a JP2
/// file, whose jp2c box holds the codestream (its ihdr and colr boxes are otherwise unused).
///
/// The pixel format depends on component count and precision, see [`ColorModel`]. Signed
/// components are supported: their true (negative-capable) values are biased by +2^(precision-1)
/// into the same output range an unsigned component of the same precision would use. Component
/// subsampling (XRsiz/YRsiz != 1) is supported: a subsampled component is upsampled to the
/// reference grid by sample replication. Any other component count, mismatched per-component
/// precision or signedness, precision above 16 bits, or a region-of-interest (RGN) marker is an
/// error.
///
/// Limits (`opts.limits`): `max_alloc` caps the peak working memory of the call, all threads
/// together; `max_threads` caps its threads, the calling thread included (1 spawns none, a pooled call,
/// the shared pool included, uses at most that many and at most the pool's size minus 1 when the
/// pool has more than 2 threads, and the bytes are the same for every value); `max_pixels`, `max_width`, `max_height` and `max_output` cap
/// the size of the result. `cancel` is checked per tile, code-block and row on every thread.
///
/// Call this from a plain thread when `cancel` matters. From a worker thread of another rayon pool
/// (inside that pool's `par_iter`, say), a wait inside the call may run that pool's other jobs,
/// such as another decode, so the time to return after `cancel` is set is not bounded there.
///
/// # Errors
/// [`Error::LimitExceeded`] over any of those limits, [`Error::Cancelled`] when `cancel` is set,
/// [`Error::Invalid`] or [`Error::Unsupported`] for a malformed or unhandled stream,
/// [`Error::OutOfMemory`] when an allocation fails, [`Error::Internal`] when a panic was caught.
pub fn decode(data: &[u8], opts: &Options, cancel: &AtomicBool) -> Result<Raster> {
    raster_of(decode_whole(
        data,
        None,
        opts,
        cancel,
        0,
        &mut Scratch::default(),
    )?)
}

/// `decode` over a reader: reads it to the end (at most `opts.limits.max_input` bytes, charged
/// to `max_alloc`), checking `cancel` between reads.
///
/// Limits (`opts.limits`): `max_input` caps the bytes read, charged to `max_alloc`, which caps the peak working memory of the call, all threads
/// together; `max_threads` caps its threads, the calling thread included (1 spawns none, a pooled call,
/// the shared pool included, uses at most that many and at most the pool's size minus 1 when the
/// pool has more than 2 threads, and the bytes are the same for every value); `max_pixels`, `max_width`, `max_height` and `max_output` cap
/// the size of the result. `cancel` is checked per tile, code-block and row on every thread.
///
/// # Errors
/// [`Error::LimitExceeded`] over any of those limits, [`Error::Cancelled`] when `cancel` is set,
/// [`Error::Invalid`] or [`Error::Unsupported`] for a malformed or unhandled stream,
/// [`Error::OutOfMemory`] when an allocation fails, [`Error::Internal`] when a panic was caught.
/// [`Error::Io`] when the reader fails.
pub fn decode_reader<R: Read>(r: R, opts: &Options, cancel: &AtomicBool) -> Result<Raster> {
    let data = read_all(r, &opts.limits, cancel)?;
    raster_of(decode_whole(
        &data,
        None,
        opts,
        cancel,
        data.capacity(),
        &mut Scratch::default(),
    )?)
}

/// Reads `r` to the end within the limits, in chunks so a lying size cannot drive an
/// allocation.
fn read_all<R: Read>(mut r: R, limits: &Limits, cancel: &AtomicBool) -> Result<Vec<u8>> {
    let mut data: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 1 << 16];
    loop {
        check_cancel(cancel)?;
        let n = match r.read(&mut chunk) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        };
        if n == 0 {
            data.shrink_to_fit();
            return Ok(data);
        }
        let total = data.len() + n;
        if total > limits.max_input {
            return Err(limit!(
                "input is longer than the {}-byte input limit",
                limits.max_input
            ));
        }
        if total > limits.max_alloc {
            return Err(limit!(
                "input needs more than the {} bytes of working memory the limit allows",
                limits.max_alloc
            ));
        }
        if total > data.capacity() {
            // Doubling, but never past what the limits allow: capacity is what is held, and
            // growing may copy, holding the old buffer and the new together.
            let cap = (2 * data.capacity())
                .max(total)
                .min(limits.max_input)
                .min(limits.max_alloc.saturating_sub(data.capacity()));
            if cap < total {
                return Err(limit!(
                    "input needs more than the {} bytes of working memory the limit allows, growing the buffer holds the old one too",
                    limits.max_alloc
                ));
            }
            data.try_reserve_exact(cap - data.len())
                .map_err(|_| Error::OutOfMemory)?;
        }
        data.extend_from_slice(&chunk[..n]);
    }
}

/// Decodes only the listed components of a JPEG 2000 image, each into its own [`Plane`], in the
/// order of `comps`.
///
/// `comps` are codestream component indices (a JP2 cdef box is ignored). An empty `comps`, an
/// index out of range, one given twice, or a component of more than 16 bits is an error.
/// Component count and mixed precision or signedness, which `decode` refuses, are fine here.
/// A plane covers the whole component on its own sample grid; a subsampled component's plane is
/// its own size, with no upsampling. Components not asked for cost no tier-1 decoding,
/// dequantization or inverse wavelet. When a tile uses the multiple component transform and
/// `comps` holds any of 0-2, all three are decoded and transformed.
///
/// Limits (`opts.limits`): `max_alloc` caps the peak working memory of the call, all threads
/// together; `max_threads` caps its threads, the calling thread included (1 spawns none, a pooled call,
/// the shared pool included, uses at most that many and at most the pool's size minus 1 when the
/// pool has more than 2 threads, and the bytes are the same for every value); `max_pixels`, `max_width`, `max_height` and `max_output` cap
/// the size of the result. `cancel` is checked per tile, code-block and row on every thread.
///
/// # Errors
/// [`Error::LimitExceeded`] over any of those limits, [`Error::Cancelled`] when `cancel` is set,
/// [`Error::Invalid`] or [`Error::Unsupported`] for a malformed or unhandled stream,
/// [`Error::OutOfMemory`] when an allocation fails, [`Error::Internal`] when a panic was caught.
pub fn decode_components(
    data: &[u8],
    comps: &[usize],
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<Vec<Plane>> {
    if comps.is_empty() {
        return Err(invalid!("no components requested"));
    }
    Ok(decode_whole(data, Some(comps), opts, cancel, 0, &mut Scratch::default())?.planes)
}

fn decode_region_inner(
    data: &[u8],
    rect: Rect,
    reduce: i32,
    comps: Option<&[usize]>,
    opts: &Options,
    cancel: &AtomicBool,
    s: &mut Scratch,
) -> Result<Decoded> {
    guard(|| {
        check_cancel(cancel)?;
        if !(0..=32).contains(&reduce) {
            return Err(invalid!("reduce {reduce} out of range"));
        }
        with_ledger(opts, data.len(), s, |s, budget| {
            // Go reads the file sparsely, keeping the tile-parts the region reads, and parses
            // that copy strictly; a layout it does not walk is parsed whole.
            let mut rb = std::mem::take(&mut s.region);
            let compact = sparse_file(data, rect, reduce as u32, budget, cancel, &mut rb)?;
            let (cs, stream, cdef) = parse_file(
                if compact { &rb.out[..] } else { data },
                &opts.limits,
                budget,
                codestream::Parse::Whole,
                cancel,
                &mut s.parse,
            )?;
            let reg = check_region(&cs, rect, reduce as u32, &mut rb.wins)?;
            let r = decode_image(
                &cs,
                stream,
                cdef.as_deref(),
                reg,
                comps,
                opts,
                false,
                cancel,
                s,
                budget,
            );
            if s.keep {
                s.parse.reclaim(cs);
                s.region = rb;
            }
            r
        })
    })
}

/// Decodes only `rect` of the image at 1/2^`reduce` scale.
///
/// `rect` is in the coordinates of `decode`'s bounds and is clipped to the image; an empty
/// intersection is an error. `reduce` drops that many of the highest resolution levels (0 is
/// full size) and may not exceed the smallest decomposition level count of any component, as in
/// OpenJPEG. The result covers exactly `rect` at the reduced scale, and its bounds are in
/// reduced coordinates: each edge `e` becomes ceil(e/2^reduce) (for an image with a nonzero
/// XOsiz/YOsiz origin, ceil((e+XOsiz)/2^reduce) - ceil(XOsiz/2^reduce)). Adjacent regions
/// therefore tile without gaps or overlap, and `decode_region(full, 0)` equals `decode`. Pixel
/// values equal the same crop of a whole-image decode at the same reduce. The pixel limit
/// applies to the region, not the whole image.
///
/// Limits (`opts.limits`): `max_alloc` caps the peak working memory of the call, all threads
/// together; `max_threads` caps its threads, the calling thread included (1 spawns none, a pooled call,
/// the shared pool included, uses at most that many and at most the pool's size minus 1 when the
/// pool has more than 2 threads, and the bytes are the same for every value); `max_pixels`, `max_width`, `max_height` and `max_output` cap
/// the size of the result. `cancel` is checked per tile, code-block and row on every thread.
///
/// # Errors
/// [`Error::LimitExceeded`] over any of those limits, [`Error::Cancelled`] when `cancel` is set,
/// [`Error::Invalid`] or [`Error::Unsupported`] for a malformed or unhandled stream,
/// [`Error::OutOfMemory`] when an allocation fails, [`Error::Internal`] when a panic was caught.
pub fn decode_region(
    data: &[u8],
    rect: Rect,
    reduce: i32,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<Raster> {
    raster_of(decode_region_inner(
        data,
        rect,
        reduce,
        None,
        opts,
        cancel,
        &mut Scratch::default(),
    )?)
}

/// `decode_components` for `rect` at 1/2^`reduce` scale, as `decode_region` takes them, with the
/// same errors. For a component without subsampling, each plane covers exactly what
/// `decode_region`'s result does. A subsampled component's plane covers the same region on its
/// own sample grid: each edge `e` of the reduced-coordinate rect (in absolute coordinates,
/// XOsiz included) becomes ceil(e/XRsiz) (ceil(e/YRsiz) vertically), less the component's origin
/// at that scale; it may be empty.
///
/// Limits (`opts.limits`): `max_alloc` caps the peak working memory of the call, all threads
/// together; `max_threads` caps its threads, the calling thread included (1 spawns none, a pooled call,
/// the shared pool included, uses at most that many and at most the pool's size minus 1 when the
/// pool has more than 2 threads, and the bytes are the same for every value); `max_pixels`, `max_width`, `max_height` and `max_output` cap
/// the size of the result. `cancel` is checked per tile, code-block and row on every thread.
///
/// # Errors
/// [`Error::LimitExceeded`] over any of those limits, [`Error::Cancelled`] when `cancel` is set,
/// [`Error::Invalid`] or [`Error::Unsupported`] for a malformed or unhandled stream,
/// [`Error::OutOfMemory`] when an allocation fails, [`Error::Internal`] when a panic was caught.
pub fn decode_region_components(
    data: &[u8],
    rect: Rect,
    reduce: i32,
    comps: &[usize],
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<Vec<Plane>> {
    if comps.is_empty() {
        return Err(invalid!("no components requested"));
    }
    Ok(decode_region_inner(
        data,
        rect,
        reduce,
        Some(comps),
        opts,
        cancel,
        &mut Scratch::default(),
    )?
    .planes)
}

/// A decoder to keep and reuse, one per worker thread: it keeps what a call allocates (the output,
/// the sample planes, the tier-1 and wavelet scratch, the tile geometry, the parsed headers) and
/// reuses it on the next call, so a second decode of the same shape allocates nothing (with
/// `max_threads` 1; see `docs/performance.md`). The result of a call is borrowed from the
/// `Decoder` until its next call: clone it, or write it out, first.
///
/// Each call takes its own [`Options`] and returns what the free function of the same name
/// returns, with the same errors and the same bytes. The limits are those of the free function,
/// enforced the same way: every buffer is charged to the call's ledger where and as much as a new
/// one would be, then taken from the `Decoder` instead of allocated, so `max_alloc` caps the
/// call's working memory as before. Within a call nothing the ledger released stays held but
/// four buffers of fixed bound: a tier-1 worker's joined segments up to 64 KiB, a lane's row
/// buffers up to 64 KiB, a tile component's code-block lists up to 64 KiB (by capacity; bigger
/// ones are freed), and the room a lane's kept tier-2 lists hold beyond what the tile that takes
/// them charges (at most 256 KiB). Planes past a lane's three spares, a region's lower wavelet
/// levels and the wavelet lines a threaded transform sheds stay charged until the call takes them
/// again or ends, and a lane's tier-2 lists stay charged what they hold, by capacity, until its next tile
/// takes them, when that is at most 256 KiB and within the ledger's peak so far (else they are
/// freed); so there (tiles that grow, regions that leave out part of a tile, and by at most
/// 256 KiB a lane, other lanes' tiles or a region's skipped tiles while the tier-2 lists are
/// charged) the call's ledger peak, and the `max_alloc` it needs, can be higher than the free
/// function's, by at most those buffers. A call that upsamples subsampled components, fails or
/// retries frees its lanes' buffers first, and lanes a panic left are freed when the next call
/// starts. What the `Decoder` keeps between calls is not charged to any call; it is at most the
/// lanes of a call times the buffers a lane holds at once times the largest plane any call took,
/// plus the last call's output and parsed headers, plus fixed bounds a lane: its tier-2 lists up
/// to 256 KiB by capacity (bigger ones are freed when the call ends), its code-block lists and
/// their index up to 64 KiB, each tier-1 worker's join up to 64 KiB and its row buffers up to
/// 64 KiB; it is usually about what
/// the largest call it made needed (see `docs/limits.md`). It is freed when the `Decoder` drops.
///
/// ```
/// use jp2k::{Decoder, Limits, Options};
/// use std::sync::atomic::AtomicBool;
///
/// let data = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/g64_53.j2k"));
/// let opts = Options {
///     limits: Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() },
///     ..Options::default()
/// };
/// let cancel = AtomicBool::new(false);
/// let mut dec = Decoder::new();
/// for _ in 0..3 {
///     // After the first call, no allocation: the buffers of the one before are reused.
///     let img = dec.decode(data, &opts, &cancel)?;
///     assert_eq!((img.bounds.x1, img.bounds.y1), (64, 64));
/// }
/// assert!(dec.decode(data, &opts, &cancel)? == &jp2k::decode(data, &opts, &cancel)?);
/// # Ok::<(), jp2k::Error>(())
/// ```
pub struct Decoder {
    s: Scratch,
    raster: Option<Raster>,
    planes: Vec<Plane>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// A decoder holding nothing yet. It enforces no limit and allocates nothing.
    pub fn new() -> Decoder {
        Decoder {
            s: Scratch {
                keep: true,
                ..Scratch::default()
            },
            raster: None,
            planes: Vec::new(),
        }
    }

    /// Takes the last result back into the scratch, for the next call to fill.
    fn start(&mut self) {
        if let Some(r) = self.raster.take() {
            self.s.pix = r.pix;
        }
        for p in self.planes.drain(..) {
            self.s.u16s.give(p.pix, true);
        }
        self.s.planes = std::mem::take(&mut self.planes);
    }

    fn image(&mut self, d: Result<Decoded>) -> Result<&Raster> {
        let r = raster_of(d?)?;
        Ok(&*self.raster.insert(r))
    }

    fn planes(&mut self, d: Result<Decoded>) -> Result<&[Plane]> {
        self.planes = d?.planes;
        // Room to take the planes back on the next call without growing the list.
        let _ = self.s.u16s.0.try_reserve(self.planes.len());
        Ok(self.planes.as_slice())
    }

    /// [`decode`](crate::decode), reusing this decoder's buffers.
    ///
    /// Limits (`opts.limits`): as `decode`: `max_alloc` caps the peak working memory of the call,
    /// all threads together; `max_threads` caps its threads, the calling thread included (1
    /// spawns none); `max_pixels`, `max_width`, `max_height` and `max_output` cap the size of the
    /// result. `cancel` is checked per tile, code-block and row on every thread.
    ///
    /// # Errors
    /// As [`decode`](crate::decode): [`Error::LimitExceeded`] over any of those limits,
    /// [`Error::Cancelled`] when `cancel` is set, [`Error::Invalid`] or [`Error::Unsupported`]
    /// for a malformed or unhandled stream, [`Error::OutOfMemory`] when an allocation fails,
    /// [`Error::Internal`] when a panic was caught.
    pub fn decode(&mut self, data: &[u8], opts: &Options, cancel: &AtomicBool) -> Result<&Raster> {
        self.start();
        let d = decode_whole(data, None, opts, cancel, 0, &mut self.s);
        self.image(d)
    }

    /// [`decode_components`](crate::decode_components), reusing this decoder's buffers.
    ///
    /// Limits (`opts.limits`): as `decode_components`: `max_alloc`, `max_threads`, `max_pixels`,
    /// `max_width`, `max_height` and `max_output`; `cancel` as for [`Decoder::decode`].
    ///
    /// # Errors
    /// As [`decode_components`](crate::decode_components).
    pub fn decode_components(
        &mut self,
        data: &[u8],
        comps: &[usize],
        opts: &Options,
        cancel: &AtomicBool,
    ) -> Result<&[Plane]> {
        self.start();
        if comps.is_empty() {
            return Err(invalid!("no components requested"));
        }
        let d = decode_whole(data, Some(comps), opts, cancel, 0, &mut self.s);
        self.planes(d)
    }

    /// [`decode_region`](crate::decode_region), reusing this decoder's buffers.
    ///
    /// Limits (`opts.limits`): as `decode_region`: `max_alloc`, `max_threads`, `max_pixels`,
    /// `max_width`, `max_height` and `max_output`; `cancel` as for [`Decoder::decode`].
    ///
    /// # Errors
    /// As [`decode_region`](crate::decode_region).
    pub fn decode_region(
        &mut self,
        data: &[u8],
        rect: Rect,
        reduce: i32,
        opts: &Options,
        cancel: &AtomicBool,
    ) -> Result<&Raster> {
        self.start();
        let d = decode_region_inner(data, rect, reduce, None, opts, cancel, &mut self.s);
        self.image(d)
    }

    /// [`decode_region_components`](crate::decode_region_components), reusing this decoder's
    /// buffers.
    ///
    /// Limits (`opts.limits`): as `decode_region_components`: `max_alloc`, `max_threads`,
    /// `max_pixels`, `max_width`, `max_height` and `max_output`; `cancel` as for
    /// [`Decoder::decode`].
    ///
    /// # Errors
    /// As [`decode_region_components`](crate::decode_region_components).
    pub fn decode_region_components(
        &mut self,
        data: &[u8],
        rect: Rect,
        reduce: i32,
        comps: &[usize],
        opts: &Options,
        cancel: &AtomicBool,
    ) -> Result<&[Plane]> {
        self.start();
        if comps.is_empty() {
            return Err(invalid!("no components requested"));
        }
        let d = decode_region_inner(data, rect, reduce, Some(comps), opts, cancel, &mut self.s);
        self.planes(d)
    }

    /// [`Image::decode_region`] on an opened file, reusing this decoder's buffers: an `Image` can
    /// be shared by several threads, each with its own `Decoder`.
    ///
    /// Limits: those `img` was opened with, as for [`Image::decode_region`]: `max_alloc` (peak,
    /// all threads), `max_threads`, `max_pixels`, `max_width`, `max_height`, `max_output`.
    ///
    /// # Errors
    /// As [`Image::decode_region`].
    pub fn decode_image_region(
        &mut self,
        img: &Image<'_>,
        rect: Rect,
        reduce: i32,
        cancel: &AtomicBool,
    ) -> Result<&Raster> {
        self.start();
        let d = img.region(rect, reduce, None, cancel, &mut self.s);
        self.image(d)
    }

    /// [`Image::decode_region_components`] on an opened file, reusing this decoder's buffers.
    ///
    /// Limits: those `img` was opened with, as for [`Image::decode_region_components`].
    ///
    /// # Errors
    /// As [`Image::decode_region_components`].
    pub fn decode_image_region_components(
        &mut self,
        img: &Image<'_>,
        rect: Rect,
        reduce: i32,
        comps: &[usize],
        cancel: &AtomicBool,
    ) -> Result<&[Plane]> {
        self.start();
        if comps.is_empty() {
            return Err(invalid!("no components requested"));
        }
        let d = img.region(rect, reduce, Some(comps), cancel, &mut self.s);
        self.planes(d)
    }
}

/// Whether `e` is the limit error of a header parse whose cost outgrew the size of its input,
/// which Go reports as a plain error.
fn is_header_flood(e: &Error) -> bool {
    matches!(e, Error::LimitExceeded(m) if m.contains("need more memory than their size allows"))
}

/// A JPEG 2000 file opened for serving regions of it, such as the tiles of a tile server: its
/// headers parsed and checked once. Go's `Image` also keeps tile indexes across calls and
/// reads through an `io.ReaderAt`; this one holds the bytes it was given and parses a tile's
/// packet headers on every region call.
pub struct Image<'a> {
    data: &'a [u8],
    /// `None` after Go's fallback: the headers after the main header did not parse, so each call
    /// decodes as `decode_region` does.
    cs: Option<Codestream>,
    stream: &'a [u8],
    cdef: Option<Vec<CdefEntry>>,
    cfg: Config,
    levels: usize,
    opts: Options,
}

impl<'a> Image<'a> {
    /// Parses and checks the headers of the JPEG 2000 file (a JP2 file or raw codestream) in
    /// `data`. It reads no tile data.
    ///
    /// Limits (`opts.limits`, kept for every call on the image): `max_alloc` caps the working memory of
    /// this call and of each later one, `max_threads` the threads each call uses (a pooled call, the shared pool included, at most
    /// the pool's size minus 1 when the pool has more than 2 threads). `cancel` is checked
    /// while parsing.
    ///
    /// # Errors
    /// [`Error::LimitExceeded`] over `max_alloc`, [`Error::Cancelled`] when `cancel` is set,
    /// [`Error::Invalid`] or [`Error::Unsupported`] for a malformed or unhandled file,
    /// [`Error::OutOfMemory`] when an allocation fails, [`Error::Internal`] when a panic was caught.
    pub fn open(data: &'a [u8], opts: &Options, cancel: &AtomicBool) -> Result<Image<'a>> {
        guard(|| {
            check_cancel(cancel)?;
            let mut budget = Budget::with_pool(&opts.limits, opts.pool.clone());
            let (cs, stream, cdef) = match parse_file(
                data,
                &opts.limits,
                &mut budget,
                codestream::Parse::Open,
                cancel,
                &mut codestream::ParseBufs::default(),
            ) {
                Ok(p) => p,
                Err(e @ (Error::Invalid(_) | Error::Unsupported(_))) => {
                    return Self::fallback(data, opts, e);
                }
                // Go's parse of headers larger than their size allows fails with a plain error,
                // not its memory-limit one, so its `Open` falls back on it as well.
                Err(e) if is_header_flood(&e) => return Self::fallback(data, opts, e),
                Err(e) => return Err(e),
            };
            let kind = classify_output(&cs.siz)?;
            let mut levels = cs.cod.coding_style.num_decomp_levels as usize;
            for c in &cs.coc {
                levels = levels.min(c.coding_style.num_decomp_levels as usize);
            }
            for tp in &cs.tile_parts {
                if let Some(c) = tp.cod() {
                    levels = levels.min(c.coding_style.num_decomp_levels as usize);
                }
                for c in tp.coc() {
                    levels = levels.min(c.coding_style.num_decomp_levels as usize);
                }
            }
            let cfg = Config {
                color_model: kind,
                width: cs.siz.xsiz - cs.siz.xosiz,
                height: cs.siz.ysiz - cs.siz.yosiz,
            };
            Ok(Image {
                data,
                cs: Some(cs),
                stream,
                cdef,
                cfg,
                levels: levels + 1,
                opts: opts.clone(),
            })
        })
    }

    /// Go's `Open` when the headers after the main header fail with `err`: if the main header
    /// parses, an Image that decodes each region as `decode_region` does; otherwise `err`.
    fn fallback(data: &'a [u8], opts: &Options, err: Error) -> Result<Image<'a>> {
        let stream = if data.starts_with(&JP2_SIGNATURE) {
            codestream::jp2::jp2_codestream(data)
                .map_err(|_| err.clone())?
                .0
        } else {
            data
        };
        let (siz, cod, coc, _) =
            codestream::parse_main(stream, &opts.limits).map_err(|_| err.clone())?;
        let kind = classify_output(&siz).map_err(|_| err.clone())?;
        let levels = coc
            .iter()
            .fold(cod.coding_style.num_decomp_levels as usize, |l, c| {
                l.min(c.coding_style.num_decomp_levels as usize)
            });
        Ok(Image {
            data,
            cs: None,
            stream,
            cdef: None,
            cfg: Config {
                color_model: kind,
                width: siz.xsiz - siz.xosiz,
                height: siz.ysiz - siz.yosiz,
            },
            levels: levels + 1,
            opts: opts.clone(),
        })
    }

    /// The image's width, height and color model, as `decode_config` gives them.
    ///
    /// Enforces no limit: it reads what `open` parsed, and cannot fail.
    pub fn config(&self) -> Config {
        self.cfg
    }

    /// How many resolution levels every tile of the image has: `decode_region` takes `reduce`
    /// from 0 to `levels() - 1`.
    ///
    /// Enforces no limit: it reads what `open` parsed, and cannot fail.
    pub fn levels(&self) -> usize {
        self.levels
    }

    fn region(
        &self,
        rect: Rect,
        reduce: i32,
        comps: Option<&[usize]>,
        cancel: &AtomicBool,
        s: &mut Scratch,
    ) -> Result<Decoded> {
        guard(|| {
            check_cancel(cancel)?;
            let Some(cs) = &self.cs else {
                return decode_region_inner(self.data, rect, reduce, comps, &self.opts, cancel, s);
            };
            if !(0..=32).contains(&reduce) {
                return Err(invalid!("reduce {reduce} out of range"));
            }
            let reg = check_region(cs, rect, reduce as u32, &mut s.region.wins)?;
            with_ledger(&self.opts, self.data.len(), s, |s, budget| {
                budget.charge(cs.parse_cost.max(0) as usize)?;
                decode_image(
                    cs,
                    self.stream,
                    self.cdef.as_deref(),
                    reg,
                    comps,
                    &self.opts,
                    true,
                    cancel,
                    s,
                    budget,
                )
            })
        })
    }

    /// `decode_region` on the opened file, with the same result.
    ///
    /// Limits: those `open` was given, as for [`decode_region`](crate::decode_region): `max_alloc`
    /// (peak, all threads), `max_threads`, `max_pixels`, `max_width`, `max_height`, `max_output`.
    ///
    /// # Errors
    /// As [`decode_region`](crate::decode_region).
    pub fn decode_region(&self, rect: Rect, reduce: i32, cancel: &AtomicBool) -> Result<Raster> {
        raster_of(self.region(rect, reduce, None, cancel, &mut Scratch::default())?)
    }

    /// `decode_region_components` on the opened file, with the same result.
    ///
    /// Limits: those `open` was given, as for [`decode_region_components`](crate::decode_region_components).
    ///
    /// # Errors
    /// As [`decode_region_components`](crate::decode_region_components).
    pub fn decode_region_components(
        &self,
        rect: Rect,
        reduce: i32,
        comps: &[usize],
        cancel: &AtomicBool,
    ) -> Result<Vec<Plane>> {
        if comps.is_empty() {
            return Err(invalid!("no components requested"));
        }
        Ok(self
            .region(rect, reduce, Some(comps), cancel, &mut Scratch::default())?
            .planes)
    }
}

// -- config --

/// The longest a SIZ-only prefix of a raw codestream can be: 2 + 2 + 2 + 65535.
const JP2C_CONFIG_WINDOW: usize = 65541;

/// Reads into `buf` until it is full or `r` ends; returns the bytes read. Checks `cancel` before
/// every read, so a trickling or interrupted reader stays cancellable.
fn read_full<R: Read>(r: &mut R, buf: &mut [u8], cancel: &AtomicBool) -> Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        check_cancel(cancel)?;
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(m) => n += m,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(n)
}

/// Reads `n` bytes into `buf`, failing with truncation when `r` ends first.
fn read_exact_or_trunc<R: Read>(r: &mut R, buf: &mut [u8], cancel: &AtomicBool) -> Result<()> {
    let n = read_full(r, buf, cancel)?;
    if n < buf.len() {
        return Err(invalid!("box runs past end of data"));
    }
    Ok(())
}

/// Discards `n` bytes of `r`, in chunks so a skip stays interruptible. With `seekable` (an
/// in-memory input) a skip past the end succeeds, as `Seek` does on Go's `bytes.Reader`; the next
/// read then finds the end.
fn skip<R: Read>(
    r: &mut R,
    mut n: u64,
    cancel: &AtomicBool,
    seekable: bool,
    budget: &Budget,
) -> Result<()> {
    let len = n.min(32 << 10) as usize;
    let _held = budget.guard(len)?;
    let mut buf = vec![0u8; len];
    while n > 0 {
        check_cancel(cancel)?;
        let k = n.min(buf.len() as u64) as usize;
        let m = read_full(r, &mut buf[..k], cancel)?;
        if m < k {
            if seekable {
                return Ok(());
            }
            return Err(invalid!("box runs past end of data"));
        }
        n -= k as u64;
    }
    Ok(())
}

/// Reads `LBox(4) TBox(4) [XLBox(8)]`: the box type, header length, and payload length (-1 for
/// a box that runs to EOF). `Ok(None)` on a clean EOF before the header.
fn read_box_header<R: Read>(r: &mut R, cancel: &AtomicBool) -> Result<Option<(u32, i64, i64)>> {
    let mut hdr = [0u8; 16];
    let n = read_full(r, &mut hdr[..8], cancel)?;
    if n == 0 {
        return Ok(None);
    }
    if n < 8 {
        return Err(invalid!("box runs past end of data"));
    }
    let typ = codestream::be32(&hdr[4..8]);
    let mut length = codestream::be32(&hdr[0..4]) as i64;
    let mut header_len = 8i64;
    match length {
        1 => {
            read_exact_or_trunc(r, &mut hdr[8..16], cancel)?;
            let mut b = [0u8; 8];
            b.copy_from_slice(&hdr[8..16]);
            length = u64::from_be_bytes(b) as i64;
            header_len = 16;
        }
        0 => return Ok(Some((typ, header_len, -1))),
        _ => {}
    }
    if length < header_len {
        return Err(invalid!("box length shorter than its header"));
    }
    Ok(Some((typ, header_len, length - header_len)))
}

/// Streams the jp2h children of a JP2 file unbuffered.
fn walk_jp2h_children<R: Read>(
    budget: &mut Budget,
    r: &mut R,
    mut total: i64,
    child: &mut codestream::jp2::Jp2hChild,
    cancel: &AtomicBool,
    seekable: bool,
) -> Result<()> {
    while total > 0 {
        check_cancel(cancel)?;
        let Some((typ, header_len, mut payload)) = read_box_header(r, cancel)? else {
            return Err(invalid!("box runs past end of data"));
        };
        let remaining = total - header_len;
        if remaining < 0 {
            return Err(invalid!("box header runs past end of jp2h"));
        }
        if payload < 0 {
            payload = remaining; // LBox == 0: runs to end of jp2h
        } else if payload > remaining {
            return Err(invalid!("box runs past end of jp2h"));
        }
        let want = child.peek(typ).min(payload);
        let _held = budget.guard(want as usize)?;
        let mut buf = vec![0u8; want as usize];
        read_exact_or_trunc(r, &mut buf, cancel)?;
        skip(r, (payload - want) as u64, cancel, seekable, budget)?;
        child.finish(typ, &buf, want < payload)?;
        total -= header_len + payload;
    }
    Ok(())
}

/// Streams the jp2c prefix out of a JP2 file; the jp2h children validate the same as
/// `jp2_codestream`. The 8-byte signature box header is already read.
fn jp2_config_stream<R: Read>(
    budget: &mut Budget,
    r: &mut R,
    cancel: &AtomicBool,
    seekable: bool,
) -> Result<Vec<u8>> {
    let mut jp_payload = [0u8; 4]; // jP's LBox is fixed at 12
    read_exact_or_trunc(r, &mut jp_payload, cancel)?;
    if jp_payload != codestream::jp2::JP_MAGIC {
        return Err(invalid!("missing or malformed jP signature box"));
    }
    let mut child = codestream::jp2::Jp2hChild::default();
    loop {
        check_cancel(cancel)?;
        let Some((typ, _, payload)) = read_box_header(r, cancel)? else {
            return Err(invalid!("no jp2c box found")); // clean EOF, not truncation
        };
        match typ {
            codestream::jp2::BOX_JP2H => {
                if payload < 0 {
                    return Err(invalid!("no jp2c box found"));
                }
                walk_jp2h_children(budget, r, payload, &mut child, cancel, seekable)?;
            }
            codestream::jp2::BOX_JP2C => {
                if !child.saw_ihdr {
                    return Err(invalid!(
                        "jp2c box appears before a jp2h box with an ihdr child"
                    ));
                }
                let mut n = payload;
                if n < 0 || n > JP2C_CONFIG_WINDOW as i64 {
                    n = JP2C_CONFIG_WINDOW as i64;
                }
                // The prefix stays charged: the caller holds it to the end.
                budget.charge(n as usize)?;
                let mut buf = vec![0u8; n as usize];
                let m = read_full(r, &mut buf, cancel)?;
                if m == 0 {
                    return Err(invalid!("missing SOC marker at offset 0"));
                }
                buf.truncate(m);
                return Ok(buf);
            }
            _ => {
                if payload < 0 {
                    return Err(invalid!("no jp2c box found"));
                }
                skip(r, payload as u64, cancel, seekable, budget)?;
            }
        }
    }
}

/// The width, height, and color model of a JPEG 2000 image without decoding pixel data. For a
/// raw codestream it reads no more than the SOC and SIZ marker segments' worth of `data` (2 +
/// 2 + 2 + 65535, the largest a conforming SIZ segment can be), so a truncated-after-SIZ stream
/// still succeeds; for a JP2 file it walks the boxes, then reads the same SIZ-only codestream
/// prefix.
///
/// Limits: `max_alloc` caps the buffers it reads into, which are charged as they are made. It
/// starts no thread, so `max_threads` does not apply.
///
/// # Errors
/// [`Error::LimitExceeded`] over `max_alloc`, [`Error::Invalid`] or [`Error::Unsupported`] for a
/// malformed or unhandled header, [`Error::Io`] if the data ends before the header does,
/// [`Error::OutOfMemory`] when an allocation fails, [`Error::Internal`] when a panic was caught.
pub fn decode_config(mut data: &[u8], limits: &Limits) -> Result<Config> {
    decode_config_from(&mut data, limits, &AtomicBool::new(false), true)
}

/// `decode_config` over a reader: it reads no more than the headers, checking `cancel` between
/// reads. `Read::read` may return fewer bytes than asked, so this loops.
///
/// Limits: `max_input` caps the bytes read, `max_alloc` the buffers it reads into, which are charged
/// as they are made. It starts no thread, so `max_threads` does not apply.
///
/// # Errors
/// [`Error::LimitExceeded`] over `max_input` or `max_alloc`, [`Error::Cancelled`] when `cancel` is
/// set, [`Error::Invalid`] or [`Error::Unsupported`] for a malformed or unhandled header,
/// [`Error::Io`] when the reader fails or ends before the header does, [`Error::OutOfMemory`] when
/// an allocation fails, [`Error::Internal`] when a panic was caught.
pub fn decode_config_reader<R: Read>(r: R, limits: &Limits, cancel: &AtomicBool) -> Result<Config> {
    decode_config_from(r, limits, cancel, false)
}

/// Reads `r` up to `left` bytes; past that it fails and sets `hit`, so a caller can tell the
/// input limit from an I/O error. At the limit a reader that has ended still reads as ended.
struct Counted<R> {
    r: R,
    left: usize,
    hit: bool,
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.left == 0 {
            let mut one = [0u8; 1];
            let n = self.r.read(&mut one)?;
            if n > 0 {
                self.hit = true;
                return Err(std::io::Error::other("input limit"));
            }
            return Ok(0);
        }
        let k = buf.len().min(self.left);
        let n = self.r.read(&mut buf[..k])?;
        self.left -= n;
        Ok(n)
    }
}

/// `seekable`: the input is in memory, so skipping a box can go past the end (see `skip`).
/// Otherwise `limits.max_input` caps the bytes read.
fn decode_config_from<R: Read>(
    r: R,
    limits: &Limits,
    cancel: &AtomicBool,
    seekable: bool,
) -> Result<Config> {
    let mut r = Counted {
        r,
        left: if seekable {
            usize::MAX
        } else {
            limits.max_input
        },
        hit: false,
    };
    let out = guard(|| {
        check_cancel(cancel)?;
        // Every buffer is charged as it is made; the SIZ prefix stays charged to the end.
        let mut budget = Budget::new(limits);
        let mut head = [0u8; 8];
        let n = read_full(&mut r, &mut head, cancel)?;
        let is_jp2 = n == 8 && head == JP2_SIGNATURE;
        let stream: Vec<u8> = if is_jp2 {
            jp2_config_stream(&mut budget, &mut r, cancel, seekable)?
        } else {
            check_cancel(cancel)?;
            budget.charge(JP2C_CONFIG_WINDOW)?;
            let mut s = vec![0u8; JP2C_CONFIG_WINDOW];
            s[..n].copy_from_slice(&head[..n]);
            let m = read_full(&mut r, &mut s[n..], cancel)?;
            s.truncate(n + m);
            s
        };
        if stream.is_empty() {
            return Err(Error::Io(
                std::io::ErrorKind::UnexpectedEof,
                "unexpected EOF".into(),
            ));
        }
        if stream.len() < 2 || codestream::be16(&stream[0..2]) != codestream::MARKER_SOC {
            return Err(invalid!("missing SOC marker at offset 0"));
        }
        if stream.len() < 4 || codestream::be16(&stream[2..4]) != codestream::MARKER_SIZ {
            return Err(invalid!(
                "expected SIZ marker immediately after SOC at offset 2"
            ));
        }
        let (siz, _) = codestream::parse_siz_segment(&stream, 2)?;
        let kind = classify_output(&siz)?;
        Ok(Config {
            color_model: kind,
            width: siz.xsiz - siz.xosiz,
            height: siz.ysiz - siz.yosiz,
        })
    });
    if r.hit && out.is_err() {
        return Err(limit!(
            "input is longer than the {}-byte input limit",
            limits.max_input
        ));
    }
    out
}
