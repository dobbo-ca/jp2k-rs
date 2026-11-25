//! Reads JPEG 2000 packets (ISO/IEC 15444-1 Annex B): tile and precinct geometry, progression
//! orders, tag trees and packet headers, down to each code-block's coding-pass segments.

mod tagtree;
mod want;

pub use want::{TileWant, band_window};

use crate::codestream::{
    Cod, Codestream, Rect, ceil_div, ceil_div_pow2, effective_coding_style, floor_div_pow2,
    quant_for, tile_overrides,
};
use crate::error::{Error, Result};
use crate::limits::{Budget, held_by};
use crate::tier1::{CB_BYPASS, CB_HT_MASK, CB_TERM_ALL};
use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use tagtree::{BitSource, TagTree};

fn floor_log2(n: usize) -> i64 {
    if n == 0 {
        -1
    } else {
        (usize::BITS - 1 - n.leading_zeros()) as i64
    }
}

/// Bounds precinct-grid allocation: it rejects a crafted SIZ/COD before the tile geometry
/// allocates gigabytes of empty precincts.
const MAX_PRECINCTS_PER_RESOLUTION: usize = 1 << 20;

/// Bounds the SUM of precincts allocated across every resolution, band, and component of one
/// tile.
pub const MAX_PRECINCTS_PER_TILE: usize = 1 << 22;

/// Bounds the SUM of cw*ch (code-blocks, and tag-tree nodes) across every precinct in the tile.
const MAX_CODE_BLOCKS_PER_TILE: i64 = 1 << 24;

/// Bounds cw*ch, the code-block (and tag-tree) grid of a single precinct. OpenJPEG guards this
/// exact quantity (tcd.c opj_tcd_init_tile, l_nb_code_blocks) before allocating.
const MAX_CODE_BLOCKS_PER_PRECINCT: usize = 1 << 20;

/// Bounds the materialized packet sequence: a degenerate progression (huge layer count, or an
/// RPCL precinct pitch of 1 driving a reference-grid walk of the whole tile) must not cost more
/// than the bytes actually available.
const MAX_PACKETS_PER_TILE: i64 = 1 << 20;

/// One code-block's termination segment: how many coding passes it has accumulated and its
/// capacity.
#[derive(Clone, Copy, Debug, Default)]
struct SegState {
    numpasses: i32,
    maxpasses: i32,
}

/// Tier-2 state that must persist for the life of a tile: Lblock only ever grows, a code-block
/// is only ever tag-tree-decoded for inclusion until its first inclusion layer, and its
/// termination segments only ever grow in number across every including layer.
#[derive(Clone, Debug, Default)]
pub struct CodeblockState {
    ever_included: bool,
    lblock: i32,
    zero_bit_planes: i32,
    /// The last segment: only it takes passes.
    seg: SegState,
    /// Segments begun, `seg` being number `nsegs - 1`.
    nsegs: i32,
    /// Cumulative passes across every including layer, never reset.
    total_passes: i32,
}

/// Guards against a crafted stream that repeats small-but-nonzero numnewpasses codewords across
/// many layers. Cumulative passes cannot legally exceed the bit-plane total (91 real passes);
/// OpenJPEG uses a flat 109.
const CODE_BLOCK_PASS_BUDGET: i32 = 109;

/// How many of a code-block's passes, summed over layers, tier-1 can read: `3*Mb - 2`
/// (T.800 D.3, E.2), with the tile's largest Mb standing in for each subband's.
pub fn passes_read(cs: &Codestream, tile: u16) -> usize {
    let mut mb = 0i64;
    for c in 0..cs.siz.components.len() {
        let q = quant_for(cs, tile, c);
        for &e in &q.exponents {
            mb = mb.max(q.guard_bits as i64 + e as i64 - 1);
        }
    }
    (3 * mb - 2).clamp(1, 109) as usize
}

/// Starts a new termination segment for a code-block (T.800/OpenJPEG t2.c opj_t2_init_seg):
/// TERMALL makes every pass its own segment; selective bypass makes the first segment 10
/// passes, then alternates length 2 and length 1; TERMALL wins over bypass when both are set.
fn init_seg(st: &mut CodeblockState, cb_style: u8, first: bool) {
    let maxp = if cb_style & CB_TERM_ALL != 0 {
        1
    } else if cb_style & CB_BYPASS != 0 {
        if first {
            10
        } else {
            let prev = st.seg.maxpasses;
            if prev == 1 || prev == 10 { 2 } else { 1 }
        }
    } else {
        109
    };
    st.seg = SegState {
        numpasses: 0,
        maxpasses: maxp,
    };
    st.nsegs += 1;
}

#[derive(Debug, Default)]
pub struct Precinct {
    /// Clipped to the owning subband.
    pub(crate) rect: Rect,
    /// Code-block grid dimensions (also the tag-tree size).
    cw: usize,
    ch: usize,
    /// Code-block grid origin.
    tlcx: i64,
    tlcy: i64,
    /// Code-block size exponents.
    cbw: u32,
    cbh: u32,
    // Made by `alloc` on the precinct's first packet of the tile (until then `incl` is empty): a
    // reduced or windowed decode never reaches most of a large tile's precincts. A precinct
    // rebuilt for the next tile keeps their memory.
    incl: TagTree,
    imsb: TagTree,
    cb: Vec<CodeblockState>,
}

impl Precinct {
    fn alloc(&mut self) -> Result<()> {
        self.incl.reset(self.cw, self.ch)?;
        self.imsb.reset(self.cw, self.ch)?;
        // Exactly the code-blocks' room when the vector is new, as the ledger charged.
        self.cb.clear();
        self.cb
            .try_reserve_exact(self.cw * self.ch)
            .map_err(|_| Error::OutOfMemory)?;
        self.cb.resize(self.cw * self.ch, CodeblockState::default());
        Ok(())
    }

    /// Code-block `cbno`'s rect (row-major), clipped to the precinct.
    fn cb_rect(&self, cbno: usize) -> Rect {
        let x0 = self.tlcx + (((cbno % self.cw) as i64) << self.cbw);
        let y0 = self.tlcy + (((cbno / self.cw) as i64) << self.cbh);
        Rect {
            x0: x0.max(self.rect.x0),
            x1: (x0 + (1i64 << self.cbw)).min(self.rect.x1),
            y0: y0.max(self.rect.y0),
            y1: (y0 + (1i64 << self.cbh)).min(self.rect.y1),
        }
    }
}

#[derive(Debug, Default)]
pub struct Band {
    /// 0 = LL, 1 = HL, 2 = LH, 3 = HH.
    pub subband: usize,
    pub rect: Rect,
    /// Shares the resolution's precinct grid: the first `np` (`pw * ph`) are this tile's; any
    /// after them are kept from a larger tile, for their memory.
    pub(crate) precincts: Vec<Precinct>,
    np: usize,
}

#[derive(Debug, Default)]
pub struct Resolution {
    pub rect: Rect,
    /// Precinct exponents at this resolution, un-halved.
    pub(crate) ppx: u32,
    pub(crate) ppy: u32,
    pub(crate) pw: usize,
    pub(crate) ph: usize,
    /// Code-block size exponents, the same in every precinct.
    pub cbw: u32,
    pub cbh: u32,
    /// 1 band (LL) at r == 0, else 3 (HL, LH, HH).
    pub bands: Vec<Band>,
}

#[derive(Debug, Default)]
pub struct TileComponent {
    pub tcx0: i64,
    pub tcx1: i64,
    pub tcy0: i64,
    pub tcy1: i64,
    pub num_resolutions: usize,
    pub resolutions: Vec<Resolution>,
}

#[derive(Debug, Default)]
pub struct TileGeom {
    tx0: i64,
    tx1: i64,
    ty0: i64,
    ty1: i64,
    pub comps: Vec<TileComponent>,
    /// Per layer, summed over components.
    precincts: i64,
    code_blocks: i64,
    /// `passes_read`.
    pub passes: usize,
}

/// The tile's rectangle in the reference grid.
pub(crate) fn tile_extent(cs: &Codestream, tile: u16) -> (i64, i64, i64, i64) {
    let siz = &cs.siz;
    let p = tile as i64 % siz.tiles_wide as i64;
    let q = tile as i64 / siz.tiles_wide as i64;
    let tx0 = (siz.xtosiz as i64 + p * siz.xtsiz as i64).max(siz.xosiz as i64);
    let tx1 = (siz.xtosiz as i64 + (p + 1) * siz.xtsiz as i64).min(siz.xsiz as i64);
    let ty0 = (siz.ytosiz as i64 + q * siz.ytsiz as i64).max(siz.yosiz as i64);
    let ty1 = (siz.ytosiz as i64 + (q + 1) * siz.ytsiz as i64).min(siz.ysiz as i64);
    (tx0, tx1, ty0, ty1)
}

/// Caps a tile's code-blocks by its bytes. A code-block can cost zero bits (one tag-tree bit
/// excludes a whole precinct), so this is a heuristic: 256K code-blocks (~30 MB of state) are
/// free, beyond that a real stream carries far more than a byte per 16 of them.
fn code_block_budget(bytes: usize) -> i64 {
    MAX_CODE_BLOCKS_PER_TILE.min((1i64 << 18).max(16 * bytes as i64))
}

/// Builds a tile's geometry with the default per-tile limits (tests and tools; decoding goes
/// through `decode_tile`, which also applies the codestream's byte budgets).
pub fn build_tile_geom(cs: &Codestream, tile: u16, budget: &mut Budget) -> Result<TileGeom> {
    let mut tg = TileGeom::default();
    build_tile_geom_limits(
        cs,
        tile,
        MAX_PACKETS_PER_TILE,
        MAX_CODE_BLOCKS_PER_TILE,
        budget,
        &mut tg,
    )?;
    Ok(tg)
}

impl Band {
    /// The code-block grid (columns, rows) of precinct `p` of this band.
    pub fn code_block_grid(&self, p: usize) -> (usize, usize) {
        (self.precincts[p].cw, self.precincts[p].ch)
    }

    pub fn num_precincts(&self) -> usize {
        self.np
    }
}

impl Resolution {
    /// How many precincts this resolution has.
    pub fn precinct_count(&self) -> usize {
        self.pw * self.ph
    }
}

/// Computes the full tile -> component -> resolution -> band -> precinct -> code-block
/// partition once, with the tile's packet budget and code-block budget checked before
/// allocation. Every precinct counts, POC-covered or not: allocating it costs the same. It is
/// reused for every packet in the tile, since tag-tree and Lblock state must persist across
/// layers. It is built in `tg`, whose memory (that of an earlier tile) it reuses.
fn build_tile_geom_limits(
    cs: &Codestream,
    tile: u16,
    max_packets: i64,
    max_code_blocks: i64,
    budget: &mut Budget,
    tg: &mut TileGeom,
) -> Result<()> {
    let siz = &cs.siz;
    let (tx0, tx1, ty0, ty1) = tile_extent(cs, tile);
    if tx1 <= tx0 || ty1 <= ty0 {
        return Err(invalid!("tile {tile} has empty extent"));
    }
    tg.tx0 = tx0;
    tg.tx1 = tx1;
    tg.ty0 = ty0;
    tg.ty1 = ty1;
    tg.precincts = 0;
    tg.code_blocks = 0;
    tg.passes = passes_read(cs, tile);
    let (tile_cod, tile_coc, _) = tile_overrides(cs, tile);
    let mut total_precincts = 0usize;
    tg.comps.truncate(siz.csiz as usize);

    for c in 0..siz.csiz as usize {
        let xr = siz.components[c].xrsiz as i64;
        let yr = siz.components[c].yrsiz as i64;

        let style = effective_coding_style(cs, tile_cod, &tile_coc, c);
        if style.code_block_style & CB_HT_MASK != 0 {
            return Err(unsupported!(
                "component {c} uses unsupported HT block coding (style 0x{:02x})",
                style.code_block_style
            ));
        }
        let num_res = style.num_decomp_levels as usize + 1;
        if num_res > 33 {
            return Err(invalid!(
                "component {c} declares {num_res} resolutions, exceeds the T.800 maximum of 33"
            ));
        }
        if let Some(ps) = &style.precinct_sizes
            && ps.len() < num_res
        {
            return Err(invalid!(
                "component {c} has {} custom precinct sizes for {num_res} resolutions",
                ps.len()
            ));
        }

        if c == tg.comps.len() {
            tg.comps.push(TileComponent::default());
        }
        let tcomp = &mut tg.comps[c];
        tcomp.tcx0 = ceil_div(tx0, xr);
        tcomp.tcx1 = ceil_div(tx1, xr);
        tcomp.tcy0 = ceil_div(ty0, yr);
        tcomp.tcy1 = ceil_div(ty1, yr);
        tcomp.num_resolutions = num_res;
        tcomp.resolutions.truncate(num_res);
        tcomp
            .resolutions
            .reserve_exact(num_res - tcomp.resolutions.len());

        for r in 0..num_res {
            let levelno = (num_res - 1 - r) as u32;
            if r == tcomp.resolutions.len() {
                tcomp.resolutions.push(Resolution::default());
            }
            let res = &mut tcomp.resolutions[r];
            res.rect = Rect {
                x0: ceil_div_pow2(tcomp.tcx0, levelno),
                x1: ceil_div_pow2(tcomp.tcx1, levelno),
                y0: ceil_div_pow2(tcomp.tcy0, levelno),
                y1: ceil_div_pow2(tcomp.tcy1, levelno),
            };
            res.ppx = 15;
            res.ppy = 15;
            res.pw = 0;
            res.ph = 0;
            res.cbw = 0;
            res.cbh = 0;
            if let Some(ps) = &style.precinct_sizes {
                res.ppx = ps[r].ppx as u32;
                res.ppy = ps[r].ppy as u32;
            }
            if r > 0 && (res.ppx == 0 || res.ppy == 0) {
                return Err(invalid!(
                    "component {c} resolution {r} has an illegal zero precinct exponent (Table A.21 forbids it for r>0)"
                ));
            }

            let tl_prc_x = floor_div_pow2(res.rect.x0, res.ppx) << res.ppx;
            let tl_prc_y = floor_div_pow2(res.rect.y0, res.ppy) << res.ppy;
            let br_prc_x = ceil_div_pow2(res.rect.x1, res.ppx) << res.ppx;
            let br_prc_y = ceil_div_pow2(res.rect.y1, res.ppy) << res.ppy;
            if res.rect.x0 != res.rect.x1 {
                res.pw = ((br_prc_x - tl_prc_x) >> res.ppx) as usize;
            }
            if res.rect.y0 != res.rect.y1 {
                res.ph = ((br_prc_y - tl_prc_y) >> res.ppy) as usize;
            }
            // A malformed SIZ (huge Xsiz/Ysiz) can drive pw*ph past range; allocating
            // precincts for it would exhaust memory before a single packet is read.
            let grid = match res.pw.checked_mul(res.ph) {
                Some(g) if g <= i64::MAX as usize => g,
                _ => {
                    return Err(invalid!(
                        "component {c} resolution {r} precinct grid {} x {} overflows",
                        res.pw,
                        res.ph
                    ));
                }
            };
            if grid > MAX_PRECINCTS_PER_RESOLUTION {
                return Err(limit!(
                    "component {c} resolution {r} has {grid} precincts, exceeds the sanity limit of {MAX_PRECINCTS_PER_RESOLUTION}"
                ));
            }
            tg.precincts += grid as i64;
            if tg.precincts > max_packets {
                return Err(limit!(
                    "tile {tile} has more than {max_packets} precincts per layer, more packets than its bytes can hold"
                ));
            }

            // Precincts subdivide the resolution; bands at r>0 are half that scale, so the
            // code-block-group grid used to place code-blocks within a band's precinct is
            // halved too.
            let (cbgwexp, cbghexp, tlcbgx, tlcbgy, num_bands);
            if r == 0 {
                cbgwexp = res.ppx;
                cbghexp = res.ppy;
                tlcbgx = tl_prc_x;
                tlcbgy = tl_prc_y;
                num_bands = 1;
            } else {
                cbgwexp = res.ppx - 1;
                cbghexp = res.ppy - 1;
                tlcbgx = ceil_div_pow2(tl_prc_x, 1);
                tlcbgy = ceil_div_pow2(tl_prc_y, 1);
                num_bands = 3;
            }
            let cblkwexp = (style.code_block_width_exp as u32).min(cbgwexp);
            let cblkhexp = (style.code_block_height_exp as u32).min(cbghexp);
            res.cbw = cblkwexp;
            res.cbh = cblkhexp;
            let pw = res.pw;
            res.bands.truncate(num_bands);

            for b in 0..num_bands {
                let mut subband_idx = 0usize;
                let mut rect = res.rect;
                if r > 0 {
                    subband_idx = b + 1; // 1=HL, 2=LH, 3=HH
                    let x0b = (subband_idx & 1) as i64;
                    let y0b = (subband_idx >> 1) as i64;
                    let shift = levelno + 1;
                    rect = Rect {
                        x0: ceil_div_pow2(tcomp.tcx0 - (x0b << levelno), shift),
                        x1: ceil_div_pow2(tcomp.tcx1 - (x0b << levelno), shift),
                        y0: ceil_div_pow2(tcomp.tcy0 - (y0b << levelno), shift),
                        y1: ceil_div_pow2(tcomp.tcy1 - (y0b << levelno), shift),
                    };
                }

                let num_prec = grid;
                total_precincts += num_prec;
                if total_precincts > MAX_PRECINCTS_PER_TILE {
                    return Err(limit!(
                        "tile {tile} has more than {MAX_PRECINCTS_PER_TILE} precincts across all resolutions and components, exceeds the sanity limit"
                    ));
                }
                budget.charge(num_prec * std::mem::size_of::<Precinct>())?;
                if b == res.bands.len() {
                    res.bands.push(Band::default());
                }
                let band = &mut res.bands[b];
                band.subband = subband_idx;
                band.rect = rect;
                band.np = 0;
                let precincts = &mut band.precincts;
                if precincts.len() < num_prec {
                    precincts
                        .try_reserve_exact(num_prec - precincts.len())
                        .map_err(|_| Error::OutOfMemory)?;
                }
                for precno in 0..num_prec {
                    let cbgx0 = tlcbgx + ((precno % pw) as i64) * (1i64 << cbgwexp);
                    let cbgy0 = tlcbgy + ((precno / pw) as i64) * (1i64 << cbghexp);

                    if precno == precincts.len() {
                        precincts.push(Precinct::default());
                    }
                    let prc = &mut precincts[precno];
                    prc.rect = Rect {
                        x0: cbgx0.max(rect.x0),
                        x1: (cbgx0 + (1i64 << cbgwexp)).min(rect.x1),
                        y0: cbgy0.max(rect.y0),
                        y1: (cbgy0 + (1i64 << cbghexp)).min(rect.y1),
                    };
                    prc.cw = 0;
                    prc.ch = 0;
                    prc.tlcx = 0;
                    prc.tlcy = 0;
                    prc.cbw = 0;
                    prc.cbh = 0;
                    prc.incl.clear();
                    if prc.rect.x0 < prc.rect.x1 && prc.rect.y0 < prc.rect.y1 {
                        let tlcbx = floor_div_pow2(prc.rect.x0, cblkwexp) << cblkwexp;
                        let tlcby = floor_div_pow2(prc.rect.y0, cblkhexp) << cblkhexp;
                        let brcbx = ceil_div_pow2(prc.rect.x1, cblkwexp) << cblkwexp;
                        let brcby = ceil_div_pow2(prc.rect.y1, cblkhexp) << cblkhexp;
                        prc.cw = ((brcbx - tlcbx) >> cblkwexp) as usize;
                        prc.ch = ((brcby - tlcby) >> cblkhexp) as usize;
                        if prc.cw != 0 && prc.ch > MAX_CODE_BLOCKS_PER_PRECINCT / prc.cw {
                            return Err(limit!(
                                "component {c} resolution {r} precinct {precno} has a {} x {} code-block grid, exceeds the sanity limit of {MAX_CODE_BLOCKS_PER_PRECINCT} code-blocks",
                                prc.cw,
                                prc.ch
                            ));
                        }
                        let ncb = (prc.cw * prc.ch) as i64;
                        tg.code_blocks += ncb;
                        if tg.code_blocks > max_code_blocks {
                            return Err(limit!(
                                "tile {tile} has more than {max_code_blocks} code-blocks across all precincts, exceeds the limit for its size"
                            ));
                        }
                        // The per-code-block state and the two tag trees are allocated on the
                        // precinct's first packet; charge them now, as if all were.
                        budget.charge(
                            ncb as usize * std::mem::size_of::<CodeblockState>()
                                + 2 * TagTree::size_for(prc.cw, prc.ch)?,
                        )?;
                        prc.tlcx = tlcbx;
                        prc.tlcy = tlcby;
                        prc.cbw = cblkwexp;
                        prc.cbh = cblkhexp;
                    }
                    band.np = precno + 1;
                }
            }
        }
    }

    Ok(())
}

// -- tile-part data assembly --

/// Concatenates a tile's tile-part bodies in TPsot order into one flat buffer. Concatenating
/// (rather than resuming across tile-part boundaries) removes "packet straddles a tile-part
/// boundary" as a case the bit reader has to know about.
///
/// A concatenation is charged to `budget` before it is allocated.
pub fn assemble_tile<'a>(
    cs: &Codestream,
    data: &'a [u8],
    tile: u16,
    budget: &mut Budget,
) -> Result<Cow<'a, [u8]>> {
    // One tile-part (TPsot 0) needs no list of them and no concatenation: skip both.
    if let &[j] = cs.tile_part_index(tile) {
        let tp = &cs.tile_parts[j as usize];
        if tp.sot.tpsot != 0 {
            return Err(invalid!(
                "tile {tile} tile-parts are not a dense 0..N-1 TPsot run (got TPsot {} at position 0)",
                tp.sot.tpsot
            ));
        }
        return match tp.data_offset.checked_add(tp.data_length) {
            Some(e) if e <= data.len() => Ok(Cow::Borrowed(&data[tp.data_offset..e])),
            _ => Err(invalid!("tile {tile} tile-part data range out of bounds")),
        };
    }
    let parts = tile_spans(cs, tile)?;
    if parts.is_empty() {
        return Ok(Cow::Borrowed(&[]));
    }
    let check = |sp: &Span| -> Result<()> {
        if sp
            .data_offset
            .checked_add(sp.data_length)
            .is_none_or(|e| e > data.len())
        {
            return Err(invalid!("tile {tile} tile-part data range out of bounds"));
        }
        Ok(())
    };
    let mut total = 0usize;
    for tp in &parts {
        check(tp)?;
        total = total
            .checked_add(tp.data_length)
            .ok_or_else(|| limit!("tile {tile} data size overflows"))?;
    }
    budget.charge(total)?;
    let mut b = Vec::new();
    b.try_reserve_exact(total).map_err(|_| Error::OutOfMemory)?;
    for tp in &parts {
        b.extend_from_slice(&data[tp.data_offset..tp.data_offset + tp.data_length]);
    }
    Ok(Cow::Owned(b))
}

/// Where a tile-part's data lies in the codestream.
#[derive(Clone, Copy)]
pub(crate) struct Span {
    pub data_offset: usize,
    pub data_length: usize,
    tpsot: u8,
}

/// Tile `tile`'s tile-part data in TPsot order, the order its data is joined in: none for a
/// tile whose components are all empty, else an error unless its TPsot run is dense.
pub(crate) fn tile_spans(cs: &Codestream, tile: u16) -> Result<Vec<Span>> {
    let idx = cs.tile_part_index(tile);
    if idx.is_empty() {
        if tile_components_empty(cs, tile) {
            // Every component's sample grid is empty for this tile after subsampling ceildiv,
            // so a real encoder legitimately emits no tile-part for it.
            return Ok(Vec::new());
        }
        return Err(invalid!("no tile-parts found for tile {tile}"));
    }
    let mut spans: Vec<Span> = idx
        .iter()
        .map(|&j| {
            let tp = &cs.tile_parts[j as usize];
            Span {
                data_offset: tp.data_offset,
                data_length: tp.data_length,
                tpsot: tp.sot.tpsot,
            }
        })
        .collect();
    spans.sort_by_key(|s| s.tpsot);
    for (i, sp) in spans.iter().enumerate() {
        if sp.tpsot as usize != i {
            return Err(invalid!(
                "tile {tile} tile-parts are not a dense 0..N-1 TPsot run (got TPsot {} at position {i})",
                sp.tpsot
            ));
        }
    }
    Ok(spans)
}

/// Whether every component's sample grid is empty for `tile`: ceildiv(tx, XRsiz) collapses to
/// zero width or height.
fn tile_components_empty(cs: &Codestream, tile: u16) -> bool {
    let siz = &cs.siz;
    let (tx0, tx1, ty0, ty1) = tile_extent(cs, tile);
    for c in 0..siz.csiz as usize {
        let xr = siz.components[c].xrsiz as i64;
        let yr = siz.components[c].yrsiz as i64;
        if ceil_div(tx0, xr) != ceil_div(tx1, xr) && ceil_div(ty0, yr) != ceil_div(ty1, yr) {
            return false;
        }
    }
    true
}

// -- packet bit reader (T.800 B.10.1, the FF bit-stuffing rule) --

pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    ct: u32,
}

impl BitSource for BitReader<'_> {
    fn read_bit(&mut self) -> Result<u32> {
        self.read_bit_impl()
    }
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            buf: 0,
            ct: 0,
        }
    }

    fn byte_in(&mut self) -> Result<()> {
        self.buf = (self.buf << 8) & 0xffff;
        // A previous 0xFF byte means its stuffed successor has 7 real bits.
        self.ct = if self.buf == 0xff00 { 7 } else { 8 };
        if self.pos >= self.data.len() {
            return Err(invalid!(
                "packet header read past end of tile data at offset {}",
                self.pos
            ));
        }
        self.buf |= self.data[self.pos] as u64;
        self.pos += 1;
        Ok(())
    }

    fn read_bit_impl(&mut self) -> Result<u32> {
        if self.ct == 0 {
            self.byte_in()?;
        }
        self.ct -= 1;
        Ok(((self.buf >> self.ct) & 1) as u32)
    }

    /// Reads `n` bits (n <= 32) most-significant bit first.
    fn read(&mut self, n: usize) -> Result<u32> {
        let mut v: u32 = 0;
        for _ in 0..n {
            let bit = self.read_bit_impl()?;
            v = (v << 1) | bit;
        }
        Ok(v)
    }

    /// Drops any partial byte and, if the last consumed byte was 0xFF, consumes its stuffed
    /// successor too (T.800 B.10.1).
    fn align(&mut self) -> Result<()> {
        if self.buf & 0xff == 0xff {
            self.byte_in()?;
        }
        self.ct = 0;
        Ok(())
    }
}

/// Resolves a code-block's zero-bitplane count via the tag tree (T.800 B.10.5.3): the actual
/// number of all-zero MSB bit-planes. Values of 74 and up are an error, as in OpenJPEG's
/// threshold loop.
fn decode_zero_bit_planes(tree: &mut TagTree, cbno: usize, br: &mut BitReader) -> Result<i32> {
    // One walk at the loop's final threshold (74) reads exactly the bits the
    // increasing-threshold loop would.
    let done = tree.decode(cbno, 74, br)?;
    if !done {
        return Err(invalid!(
            "zero-bitplane tag tree for code-block {cbno} did not resolve"
        ));
    }
    Ok(tree.value(cbno))
}

/// Decodes the number-of-coding-passes codeword (T.800 B.10.6 / OpenJPEG opj_t2_getnumpasses).
fn read_coding_passes(br: &mut BitReader) -> Result<usize> {
    if br.read(1)? == 0 {
        return Ok(1);
    }
    if br.read(1)? == 0 {
        return Ok(2);
    }
    let n = br.read(2)?;
    if n != 3 {
        return Ok(3 + n as usize);
    }
    let n = br.read(5)?;
    if n != 31 {
        return Ok(6 + n as usize);
    }
    let n = br.read(7)?;
    Ok(37 + n as usize)
}

// -- packet records --

/// One packet's contribution of bytes to one of a code-block's termination segments. `seg_index`
/// identifies which of the code-block's segments (across its whole lifetime) this is: a segment
/// can span several layers under the default style, so two segments from different packets with
/// the same index belong to the same decoder run and must be concatenated.
#[derive(Clone, Copy, Debug, Default)]
pub struct CodeBlockSegment {
    pub seg_index: usize,
    pub start: usize,
    pub end: usize,
    pub passes: usize,
}

/// One included code-block's tier-2 header result.
#[derive(Clone, Debug, Default)]
pub struct CodeBlockRef {
    /// 0 = LL, 1 = HL, 2 = LH, 3 = HH.
    pub subband: usize,
    /// Code-block index within its precinct's grid.
    pub cbx: usize,
    pub cby: usize,
    /// Clipped code-block rect, in subband coordinates.
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
    /// Number of all-zero MSB bit-planes (T.800 B.10.5).
    pub zero_bit_planes: i32,
    pub segments: Vec<CodeBlockSegment>,
}

/// One packet's identity and its included code-blocks.
#[derive(Clone, Debug, Default)]
pub struct PacketRef {
    pub layer: usize,
    pub resolution: usize,
    pub component: usize,
    pub precinct: usize,
    /// -1 if SOP markers are not in use.
    pub nsop: i64,
    pub empty: bool,
    /// Offset into the tile data, or into the packed PPM/PPT header stream.
    pub header_pos: usize,
    pub blocks: Vec<CodeBlockRef>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PacketIndex {
    layer: usize,
    res: usize,
    comp: usize,
    prec: usize,
}

// -- progression iterators (T.800 B.12) --

fn max_resolutions(tg: &TileGeom) -> usize {
    tg.comps
        .iter()
        .map(|c| c.num_resolutions)
        .max()
        .unwrap_or(0)
}

/// A tile's packet sequence. Every progression appends through `add`, so no order can outgrow
/// the tile's budget, nor the capacity reserved (and charged) for `seq` up front.
struct PacketSeq {
    seq: Vec<PacketIndex>,
    /// The entries reserved (and charged) for `seq`; it may have more room, from an earlier tile.
    cap: usize,
    /// With several POC records, the tuples already in `seq` (t2.c's shared include[] array), so
    /// a repeat is dropped as it is generated. Reserved for every tuple `seq` can hold.
    seen: Option<HashSet<PacketIndex>>,
    /// Entries still allowed.
    left: i64,
    /// RPCL/PCRL/CPRL position-walk steps still allowed.
    walk: i64,
}

impl PacketSeq {
    /// Takes an RPCL/PCRL/CPRL reference-grid walk's step estimate from the tile's walk budget
    /// before the walk runs: the walk costs CPU even at positions that emit no packet.
    fn charge(
        &mut self,
        name: &str,
        tg: &TileGeom,
        dx: i64,
        dy: i64,
        per_position: i64,
    ) -> Result<()> {
        if per_position <= 0 {
            return Ok(()); // empty window: the walk's inner loops never run
        }
        let steps = ((tg.tx1 - tg.tx0) / dx + 2)
            .saturating_mul((tg.ty1 - tg.ty0) / dy + 2)
            .saturating_mul(per_position);
        if steps > self.walk {
            return Err(limit!(
                "{name} precinct grid walk of an estimated {steps} steps (dx={dx}, dy={dy}) exceeds the tile's remaining walk budget of {}",
                self.walk
            ));
        }
        self.walk -= steps;
        Ok(())
    }

    fn add(&mut self, layer: usize, res: usize, comp: usize, prec: usize) -> Result<()> {
        if self.left <= 0 {
            return Err(limit!(
                "tile packet sequence exceeds its budget (sanity limit {MAX_PACKETS_PER_TILE}, or more packets than the codestream's bytes can hold)"
            ));
        }
        self.left -= 1;
        let pi = PacketIndex {
            layer,
            res,
            comp,
            prec,
        };
        // A repeat still costs `left` (Go counts it too) but is not kept.
        if self.seen.as_ref().is_some_and(|set| set.contains(&pi)) {
            return Ok(());
        }
        if self.seq.len() == self.cap {
            return Err(limit!(
                "tile packet sequence outgrew the memory reserved for it"
            ));
        }
        if let Some(set) = &mut self.seen {
            set.insert(pi);
        }
        self.seq.push(pi);
        Ok(())
    }
}

/// Caps a tile's packet count by what its bytes can carry: every packet needs at least one
/// header bit, so a tile holds at most 8 packets per byte of tile data plus PPM/PPT header data.
fn packet_budget(data_bytes: usize, hdr_bytes: usize) -> i64 {
    let n = 8 * (data_bytes as i64 + hdr_bytes as i64) + 8;
    n.min(MAX_PACKETS_PER_TILE)
}

// The generators below apply a record's layer/resolution/component window while generating, so
// a narrow POC record costs only what it emits.

fn lrcp_order(tg: &TileGeom, rec: &PocRecord, s: &mut PacketSeq) -> Result<()> {
    for l in 0..rec.layno1 {
        for r in rec.resno0..rec.resno1 {
            for c in rec.compno0..rec.compno1 {
                let comp = &tg.comps[c];
                if r >= comp.num_resolutions {
                    continue;
                }
                let res = &comp.resolutions[r];
                for p in 0..res.pw * res.ph {
                    s.add(l, r, c, p)?;
                }
            }
        }
    }
    Ok(())
}

fn rlcp_order(tg: &TileGeom, rec: &PocRecord, s: &mut PacketSeq) -> Result<()> {
    for r in rec.resno0..rec.resno1 {
        for l in 0..rec.layno1 {
            for c in rec.compno0..rec.compno1 {
                let comp = &tg.comps[c];
                if r >= comp.num_resolutions {
                    continue;
                }
                let res = &comp.resolutions[r];
                for p in 0..res.pw * res.ph {
                    s.add(l, r, c, p)?;
                }
            }
        }
    }
    Ok(())
}

/// The positional resolution-precinct-component-layer order (T.800 B.12.1.3). It needs the
/// precinct's spatial position, so it walks the reference grid in steps of the finest common
/// precinct pitch and computes each component's precinct index at that position (OpenJPEG pi.c
/// opj_pi_next_rpcl).
fn rpcl_order(cs: &Codestream, tg: &TileGeom, rec: &PocRecord, s: &mut PacketSeq) -> Result<()> {
    let all: Vec<usize> = (0..tg.comps.len()).collect();
    let (dx, dy) = precinct_step_xy(cs, tg, &all);
    if dx <= 0 || dy <= 0 {
        return Err(invalid!("RPCL precinct step computed as non-positive"));
    }
    s.charge(
        "RPCL",
        tg,
        dx,
        dy,
        (rec.resno1 as i64 - rec.resno0 as i64) * (rec.compno1 as i64 - rec.compno0 as i64),
    )?;

    for r in rec.resno0..rec.resno1 {
        let mut y = tg.ty0;
        while y < tg.ty1 {
            let mut x = tg.tx0;
            while x < tg.tx1 {
                for c in rec.compno0..rec.compno1 {
                    let Some(p) = precinct_at(cs, tg, c, r, x, y) else {
                        continue;
                    };
                    for l in 0..rec.layno1 {
                        s.add(l, r, c, p)?;
                    }
                }
                x += dx - x % dx;
            }
            y += dy - y % dy;
        }
    }
    Ok(())
}

/// The finest common precinct pitch (in reference-grid units) over `comps`, the position step
/// opj_pi_next_{rpcl,pcrl,cprl} all share: RPCL and PCRL take the min over every component,
/// CPRL recomputes it per component.
fn precinct_step_xy(cs: &Codestream, tg: &TileGeom, comps: &[usize]) -> (i64, i64) {
    let (mut dx, mut dy) = (-1i64, -1i64);
    for &ci in comps {
        let comp = &tg.comps[ci];
        let xr = cs.siz.components[ci].xrsiz as i64;
        let yr = cs.siz.components[ci].yrsiz as i64;
        for (r, res) in comp.resolutions.iter().enumerate() {
            let levelno = comp.num_resolutions - 1 - r;
            let step_x = xr << (res.ppx as usize + levelno);
            let step_y = yr << (res.ppy as usize + levelno);
            if dx == -1 || step_x < dx {
                dx = step_x;
            }
            if dy == -1 || step_y < dy {
                dy = step_y;
            }
        }
    }
    (dx, dy)
}

fn ceil_div64(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// The precinct index for tile-component (c, r) at reference-grid position (x, y), per pi.c's
/// shared RPCL/PCRL/CPRL body. `None` when (x, y) is not on this component/resolution's
/// precinct grid, or the tile-component-resolution is empty.
fn precinct_at(
    cs: &Codestream,
    tg: &TileGeom,
    c: usize,
    r: usize,
    x: i64,
    y: i64,
) -> Option<usize> {
    let comp = &tg.comps[c];
    if r >= comp.num_resolutions {
        return None;
    }
    let xr = cs.siz.components[c].xrsiz as i64;
    let yr = cs.siz.components[c].yrsiz as i64;
    let levelno = (comp.num_resolutions - 1 - r) as u32;
    let xr_lvl = xr << levelno;
    let yr_lvl = yr << levelno;
    let trx0 = ceil_div64(tg.tx0, xr_lvl);
    let try0 = ceil_div64(tg.ty0, yr_lvl);
    let trx1 = ceil_div64(tg.tx1, xr_lvl);
    let try1 = ceil_div64(tg.ty1, yr_lvl);
    let res = &comp.resolutions[r];
    let rpx = res.ppx + levelno;
    let rpy = res.ppy + levelno;

    let y_aligned = y % (yr << rpy) == 0 || (y == tg.ty0 && (try0 << levelno) % (1i64 << rpy) != 0);
    if !y_aligned {
        return None;
    }
    let x_aligned = x % (xr << rpx) == 0 || (x == tg.tx0 && (trx0 << levelno) % (1i64 << rpx) != 0);
    if !x_aligned {
        return None;
    }
    if res.pw == 0 || res.ph == 0 || trx0 == trx1 || try0 == try1 {
        return None;
    }
    let prci = floor_div_pow2(ceil_div64(x, xr_lvl), res.ppx) - floor_div_pow2(trx0, res.ppx);
    let prcj = floor_div_pow2(ceil_div64(y, yr_lvl), res.ppy) - floor_div_pow2(try0, res.ppy);
    Some((prci + prcj * res.pw as i64) as usize)
}

/// The positional precinct-component-resolution-layer order (T.800 B.12.1.4, OpenJPEG
/// opj_pi_next_pcrl). Same step and precinct-grid test as RPCL; only the loop nesting differs.
fn pcrl_order(cs: &Codestream, tg: &TileGeom, rec: &PocRecord, s: &mut PacketSeq) -> Result<()> {
    let all: Vec<usize> = (0..tg.comps.len()).collect();
    let (dx, dy) = precinct_step_xy(cs, tg, &all);
    if dx <= 0 || dy <= 0 {
        return Err(invalid!("PCRL precinct step computed as non-positive"));
    }
    s.charge(
        "PCRL",
        tg,
        dx,
        dy,
        (rec.resno1 as i64 - rec.resno0 as i64) * (rec.compno1 as i64 - rec.compno0 as i64),
    )?;

    let mut y = tg.ty0;
    while y < tg.ty1 {
        let mut x = tg.tx0;
        while x < tg.tx1 {
            for c in rec.compno0..rec.compno1 {
                for r in rec.resno0..rec.resno1 {
                    let Some(p) = precinct_at(cs, tg, c, r, x, y) else {
                        continue;
                    };
                    for l in 0..rec.layno1 {
                        s.add(l, r, c, p)?;
                    }
                }
            }
            x += dx - x % dx;
        }
        y += dy - y % dy;
    }
    Ok(())
}

/// The positional component-precinct-resolution-layer order (T.800 B.12.1.5, OpenJPEG
/// opj_pi_next_cprl). Unlike RPCL/PCRL, the step is recomputed per component from that
/// component's own resolutions alone, so components with different XRsiz/YRsiz walk the grid at
/// different pitches.
fn cprl_order(cs: &Codestream, tg: &TileGeom, rec: &PocRecord, s: &mut PacketSeq) -> Result<()> {
    for c in rec.compno0..rec.compno1 {
        let (dx, dy) = precinct_step_xy(cs, tg, &[c]);
        if dx <= 0 || dy <= 0 {
            return Err(invalid!(
                "CPRL precinct step computed as non-positive for component {c}"
            ));
        }
        s.charge("CPRL", tg, dx, dy, rec.resno1 as i64 - rec.resno0 as i64)?;
        let mut y = tg.ty0;
        while y < tg.ty1 {
            let mut x = tg.tx0;
            while x < tg.tx1 {
                for r in rec.resno0..rec.resno1 {
                    let Some(p) = precinct_at(cs, tg, c, r, x, y) else {
                        continue;
                    };
                    for l in 0..rec.layno1 {
                        s.add(l, r, c, p)?;
                    }
                }
                x += dx - x % dx;
            }
            y += dy - y % dy;
        }
    }
    Ok(())
}

// -- packet header / body decoding --

/// The state one decode shares across its tiles: the PPM read position and budgets drawn from
/// the whole codestream's bytes. A per-tile budget alone lets every tile claim the same PPM
/// bytes.
#[derive(Default)]
pub struct Cursor {
    ppm_pos: usize,
    started: bool,
    /// Precincts built plus packets generated.
    packets: i64,
    /// RPCL/PCRL/CPRL walk steps.
    walk: i64,
    code_blocks: i64,
    /// Packet-header inclusion work left, see `T2_WORK_PER_BYTE`.
    work: i64,
}

/// Caps tier-2 packet-header CPU per decode.
const T2_WORK_PER_BYTE: i64 = 1 << 10;
const T2_WORK_SLACK: i64 = 1 << 20;

impl Cursor {
    pub fn new() -> Cursor {
        Cursor::default()
    }

    /// Sets up the budgets for a decode of `data_len` codestream bytes.
    fn start(&mut self, cs: &Codestream, data_len: usize) {
        if self.started {
            return;
        }
        // 8 packets per byte as in `packet_budget`, plus its per-tile slack.
        self.started = true;
        self.packets =
            8 * data_len as i64 + 8 * cs.siz.tiles_wide as i64 * cs.siz.tiles_high as i64;
        self.walk = 16 * self.packets;
        self.code_blocks = (1i64 << 18).max(16 * data_len as i64); // code_block_budget, uncapped
        self.work = T2_WORK_PER_BYTE * data_len as i64 + T2_WORK_SLACK;
    }
}

/// What tier 2 of a tile leaves, and keeps for the next tile: the geometry and the packets
/// (`decode_tile_in` fills them), and the vectors it fills on the way. With `keep` (a `Decoder`'s)
/// `recycle` keeps the packets' vectors for the next tile, each for the same packet and
/// code-block, so a tile of the same shape allocates none.
#[derive(Default)]
pub(crate) struct TileScratch {
    pub(crate) keep: bool,
    pub(crate) tg: TileGeom,
    pub(crate) pkts: Vec<PacketRef>,
    /// What the tile's region decode read (`want_of`'s), for the next tile's to fill.
    pub(crate) want: Option<TileWant>,
    seq: Vec<PacketIndex>,
    cb_styles: Vec<u8>,
    queue: Vec<SegmentJob>,
    /// The packets' block lists and the blocks' segment lists of the last tile, last first.
    blocks: Vec<Vec<CodeBlockRef>>,
    segs: Vec<Vec<CodeBlockSegment>>,
}

impl TileScratch {
    /// Keeps the vectors of the packets for the next tile: popped in the order they were made.
    pub(crate) fn recycle(&mut self) {
        for mut p in self.pkts.drain(..).rev() {
            for b in p.blocks.drain(..).rev() {
                let mut v = b.segments;
                v.clear();
                self.segs.push(v);
            }
            self.blocks.push(p.blocks);
        }
    }

    /// The heap its vectors hold, by capacity, after `recycle`: the geometry, the packets, the
    /// sequence, queue and styles, the block and segment lists, and a region's windows.
    pub(crate) fn heap(&self) -> usize {
        let mut n = held_by(&self.tg.comps);
        for c in &self.tg.comps {
            n += held_by(&c.resolutions);
            for r in &c.resolutions {
                n += held_by(&r.bands);
                for b in &r.bands {
                    n += held_by(&b.precincts);
                    for p in &b.precincts {
                        n += p.incl.heap() + p.imsb.heap() + held_by(&p.cb);
                    }
                }
            }
        }
        n += held_by(&self.pkts) + held_by(&self.seq) + held_by(&self.cb_styles);
        n += held_by(&self.queue) + held_by(&self.blocks) + held_by(&self.segs);
        n += self.blocks.iter().map(held_by).sum::<usize>();
        n += self.segs.iter().map(held_by).sum::<usize>();
        if let Some(w) = &self.want {
            n += held_by(&w.top) + held_by(&w.wins);
            n += w.wins.iter().flatten().map(held_by).sum::<usize>();
        }
        n
    }
}

/// Assembles a tile's data, partitions it per T.800 Annex B, and decodes every packet header in
/// progression order. No coefficient data is decoded; each code-block's inclusion, zero
/// bit-planes, and segment byte ranges are recorded.
///
/// `cur` holds the merged main-header PPM stream's persistent read cursor; a caller that decodes
/// every tile of a codestream must thread the same cursor through each call, in the order tiles
/// are processed, so a PPM stream spanning several tiles' packet headers is read as one
/// continuous sequence. `want_of`, if set, is called once the geometry is built and says what a
/// region decode reads: only those code-blocks are recorded (every header is still parsed:
/// tag-tree state and body offsets depend on it), and reading stops after the last packet that
/// can hold one, except under PPM, whose shared cursor has to pass every packet. Budgets are
/// charged for the whole tile first, so nothing skipped here escapes them.
#[allow(clippy::type_complexity)]
pub fn decode_tile<'a>(
    cs: &Codestream,
    data: &'a [u8],
    tile: u16,
    cur: &mut Cursor,
    want_of: Option<&mut dyn FnMut(&TileGeom) -> Result<Option<TileWant>>>,
    cancel: &AtomicBool,
    budget: &mut Budget,
) -> Result<(Vec<PacketRef>, TileGeom, Cow<'a, [u8]>)> {
    let mut ts = TileScratch::default();
    let tile_data = decode_tile_in(cs, data, tile, cur, want_of, cancel, budget, &mut ts)?;
    Ok((ts.pkts, ts.tg, tile_data))
}

/// `decode_tile` into `ts`: the packets in `ts.pkts`, the geometry in `ts.tg`, in the memory of
/// the tile before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_tile_in<'a>(
    cs: &Codestream,
    data: &'a [u8],
    tile: u16,
    cur: &mut Cursor,
    want_of: Option<&mut dyn FnMut(&TileGeom) -> Result<Option<TileWant>>>,
    cancel: &AtomicBool,
    budget: &mut Budget,
    ts: &mut TileScratch,
) -> Result<Cow<'a, [u8]>> {
    let tile_data = assemble_tile(cs, data, tile, budget)?;
    decode_tile_data(
        cs,
        &tile_data,
        data.len(),
        tile,
        cur,
        want_of,
        cancel,
        budget,
        ts,
    )?;
    Ok(tile_data)
}

#[allow(clippy::too_many_arguments)]
fn decode_tile_data(
    cs: &Codestream,
    tile_data: &[u8],
    data_len: usize,
    tile: u16,
    cur: &mut Cursor,
    want_of: Option<&mut dyn FnMut(&TileGeom) -> Result<Option<TileWant>>>,
    cancel: &AtomicBool,
    budget: &mut Budget,
    ts: &mut TileScratch,
) -> Result<()> {
    cur.start(cs, data_len);
    let (hdr_data, use_hdr_src, mut hdr_pos) = packet_header_source(cs, tile, cur.ppm_pos)?;
    let hdr_bytes = if use_hdr_src {
        hdr_data.len().saturating_sub(hdr_pos)
    } else {
        0
    };
    let ch_budget = packet_budget(tile_data.len(), hdr_bytes);
    let ch_cb_budget = code_block_budget(tile_data.len() + hdr_bytes);
    let pkt_budget = ch_budget.min(cur.packets);

    let (tile_cod, tile_coc, _) = tile_overrides(cs, tile);
    // T.800 A.6.1: a COD in the tile's first tile-part overrides the main header's progression
    // order, layer count, and SOP/EPH use for this tile.
    let eff_cod: &Cod = tile_cod.unwrap_or(&cs.cod);

    let mut recs = [PocRecord::default(); MAX_POC_RECORDS];
    let n_recs = tile_poc_records(cs, tile, eff_cod, &mut recs)?;
    let records = &recs[..n_recs];
    let TileScratch {
        keep,
        tg,
        pkts,
        want: kept_want,
        seq: kept_seq,
        cb_styles,
        queue,
        blocks,
        segs,
    } = ts;
    build_tile_geom_limits(
        cs,
        tile,
        pkt_budget,
        ch_cb_budget.min(cur.code_blocks),
        budget,
        tg,
    )?;
    cur.packets -= tg.precincts;
    cur.code_blocks -= tg.code_blocks;
    // Real streams walk well under one step per budgeted packet; 16x is slack, and it keeps
    // walk CPU tied to the input's bytes.
    let mut s = PacketSeq {
        seq: std::mem::take(kept_seq),
        cap: 0,
        seen: None,
        left: pkt_budget.min(cur.packets),
        walk: (16 * pkt_budget).min(cur.walk),
    };
    s.seq.clear();
    let reserve = s
        .left
        .min(tg.precincts * eff_cod.number_of_layers as i64)
        .max(0) as usize;
    // With several records the include set is reserved for as many tuples as `seq` can hold (it
    // keeps a control byte and spare buckets: 3 entries' worth); both are charged before
    // they are made, and neither grows.
    let multi = records.len() > 1;
    let reserved = reserve * std::mem::size_of::<PacketIndex>() * if multi { 4 } else { 1 };
    budget.charge(reserved)?;
    s.seq
        .try_reserve_exact(reserve)
        .map_err(|_| Error::OutOfMemory)?;
    s.cap = reserve;
    if multi {
        let mut set = HashSet::new();
        set.try_reserve(reserve).map_err(|_| Error::OutOfMemory)?;
        s.seen = Some(set);
    }
    let (left, walk) = (s.left, s.walk);
    let mut seq = poc_sequence(cs, tg, records, &mut s, *keep)?;
    cur.packets -= left - s.left;
    cur.walk -= walk - s.walk;
    budget.release(reserved);
    budget.charge(seq.len() * std::mem::size_of::<PacketIndex>())?;

    let want = match want_of {
        Some(f) => f(&*tg)?,
        None => None,
    };
    let mut all = true;
    if let Some(w) = &want
        && cs.ppm.is_empty()
    {
        let mut n = 0;
        for (i, pi) in seq.iter().enumerate() {
            if w.packet(tg, pi.comp, pi.res, pi.prec) {
                n = i + 1;
            }
        }
        all = n == seq.len();
        seq.truncate(n);
    }

    cb_styles.clear();
    cb_styles.extend(
        (0..tg.comps.len())
            .map(|c| effective_coding_style(cs, tile_cod, &tile_coc, c).code_block_style),
    );

    let r = decode_packets(
        eff_cod,
        cb_styles,
        tg,
        tile_data,
        hdr_data,
        use_hdr_src,
        &mut hdr_pos,
        &seq,
        all,
        want.as_ref(),
        cancel,
        &mut cur.work,
        budget,
        Lists {
            out: pkts,
            queue,
            blocks,
            segs,
        },
    );
    *kept_seq = seq;
    *kept_want = want;
    if use_hdr_src && !cs.ppm.is_empty() {
        cur.ppm_pos = hdr_pos;
    }
    r
}

/// Resolves T.800 B.9/A.7.2's PPM/PPT precedence: if the main header carries PPM markers, EVERY
/// tile's packet headers come from the merged main-header stream, read via the caller's
/// persistent cursor; otherwise a tile with PPT markers gets its own merged per-tile stream with
/// a fresh cursor (PPT is scoped to one tile, T.800 A.7.5); otherwise packet headers live inline
/// in the tile data (`use_hdr_src` false), also with a fresh cursor since it is realigned to the
/// tile-data body cursor on every packet anyway.
fn packet_header_source(
    cs: &Codestream,
    tile: u16,
    ppm_pos: usize,
) -> Result<(&[u8], bool, usize)> {
    if !cs.ppm.is_empty() {
        let hdr = cs.ppm_headers()?;
        return Ok((hdr, true, ppm_pos));
    }
    match cs.ppt_headers(tile)? {
        None => Ok((&[], false, 0)),
        Some(h) => Ok((h, true, 0)),
    }
}

/// More POC records than a tile may have (`tile_poc_records` stops at 31).
const MAX_POC_RECORDS: usize = 32;

/// One progression-order-change entry driving a distinct chunk of a tile's packet sequence
/// (T.800 B.12.2 / POC marker segment). The no-POC case is a single record spanning the whole
/// tile at the tile's effective COD progression order.
#[derive(Clone, Copy, Default)]
struct PocRecord {
    prg: u8,
    resno0: usize,
    resno1: usize,
    compno0: usize,
    compno1: usize,
    layno1: usize,
}

/// Resolves T.800 A.6.1/A.6.2's POC precedence: OpenJPEG copies the main-header POC entries
/// into every tile's tcp as its default, and a tile-part POC segment appends further records
/// rather than replacing them. Records run in that order: every main-header POC change, then
/// every tile-part POC change for `tile` in TPsot order. They go in `recs`; returns how many.
fn tile_poc_records(
    cs: &Codestream,
    tile: u16,
    eff_cod: &Cod,
    recs: &mut [PocRecord; MAX_POC_RECORDS],
) -> Result<usize> {
    let mut changes = Vec::new();
    for poc in &cs.poc {
        changes.extend_from_slice(&poc.changes);
    }
    // Consumption order is TPsot order, not file order: all records of a tile share one
    // include[] array, so out-of-order tile-parts carrying POC would otherwise compose a
    // different packet sequence.
    let mut idx: Vec<u32> = cs
        .tile_part_index(tile)
        .iter()
        .copied()
        .filter(|&k| !cs.tile_parts[k as usize].poc().is_empty())
        .collect();
    idx.sort_by_key(|&k| cs.tile_parts[k as usize].sot.tpsot);
    for k in idx {
        for poc in cs.tile_parts[k as usize].poc() {
            changes.extend_from_slice(&poc.changes);
        }
    }
    // The per-header cap in the marker parser does not bound the tile: every tile-part may add
    // another 31 (OpenJPEG opj_j2k_read_poc caps the sum).
    if changes.len() >= MAX_POC_RECORDS {
        return Err(invalid!(
            "tile {tile} has {} POC changes across its headers, exceeds the limit of 31",
            changes.len()
        ));
    }
    if changes.is_empty() {
        recs[0] = PocRecord {
            prg: eff_cod.progression_order,
            resno0: 0,
            resno1: 33, // T.800 maximum; poc_sequence clamps to the tile
            compno0: 0,
            compno1: cs.siz.components.len(),
            layno1: eff_cod.number_of_layers as usize,
        };
        return Ok(1);
    }
    let num_comps = cs.siz.components.len();
    let num_layers = eff_cod.number_of_layers as usize;
    for (r, ch) in recs.iter_mut().zip(&changes) {
        *r = PocRecord {
            prg: ch.ppoc,
            resno0: ch.rspoc as usize,
            resno1: ch.repoc as usize,
            compno0: ch.cspoc as usize,
            compno1: (ch.cepoc as usize).min(num_comps),
            layno1: (ch.lyepoc as usize).min(num_layers),
        };
    }
    Ok(changes.len())
}

/// Appends one record's packets (T.800 B.12.1) for its Ppoc order, restricted to its
/// layer/resolution/component window (B.12.2).
fn poc_order(cs: &Codestream, tg: &TileGeom, rec: &PocRecord, s: &mut PacketSeq) -> Result<()> {
    match rec.prg {
        0 => lrcp_order(tg, rec, s),
        1 => rlcp_order(tg, rec, s),
        2 => rpcl_order(cs, tg, rec, s),
        3 => pcrl_order(cs, tg, rec, s),
        4 => cprl_order(cs, tg, rec, s),
        p => Err(invalid!("POC entry uses unsupported progression order {p}")),
    }
}

/// Concatenates every record's packet sequence in order, dropping any (layer, res, comp, prec)
/// tuple a previous record already emitted (t2.c's shared include[] array across POC records;
/// `s.seen` does the dropping as the packets are generated). `s`'s budgets are shared by every
/// record. With `keep` the sequence keeps its room for the next tile.
fn poc_sequence(
    cs: &Codestream,
    tg: &TileGeom,
    records: &[PocRecord],
    s: &mut PacketSeq,
    keep: bool,
) -> Result<Vec<PacketIndex>> {
    let num_res = max_resolutions(tg);
    for rec in records {
        let mut rec = *rec;
        rec.resno1 = rec.resno1.min(num_res);
        poc_order(cs, tg, &rec, s)?;
    }
    s.seen = None;
    let mut seq = std::mem::take(&mut s.seq);
    // The caller charges `seq.len()` entries, so give back the rest of the reservation. With
    // `keep` the room stays for the next tile, but not room this tile reserved and left empty
    // (POC records that leave packets out): the ledger no longer counts it.
    if !keep || seq.len() < s.cap {
        seq.shrink_to_fit();
    }
    Ok(seq)
}

/// A segment's Start/End fill-in, deferred: lengths are known only after the whole packet
/// header is parsed, since bodies follow the header in header order.
struct SegmentJob {
    block_idx: Option<usize>,
    seg_idx: Option<usize>,
    length: u32,
}

/// The vectors `decode_packets` fills: the packets, its segment queue, and the block and segment
/// lists of the last tile's packets (see `TileScratch::recycle`), which new ones are taken from.
struct Lists<'a> {
    out: &'a mut Vec<PacketRef>,
    queue: &'a mut Vec<SegmentJob>,
    blocks: &'a mut Vec<Vec<CodeBlockRef>>,
    segs: &'a mut Vec<Vec<CodeBlockSegment>>,
}

/// Decodes every packet header in `seq`, into `lists.out`. Packet headers come from `tile_data` unless
/// `use_hdr_src` is set, in which case they come from `hdr_data` (the merged PPM/PPT stream,
/// T.800 B.9): `hdr_pos` tracks that stream's own persistent cursor, independent of `pos`, the
/// tile-data body cursor (SOP markers and code-block bytes always live in `tile_data`). `all`
/// says `seq` is the tile's whole packet sequence, so its bodies must use up the tile data
/// exactly. `want`, if set, filters the code-blocks recorded.
#[allow(clippy::too_many_arguments)]
fn decode_packets(
    cod: &Cod,
    cb_styles: &[u8],
    tg: &mut TileGeom,
    tile_data: &[u8],
    hdr_data: &[u8],
    use_hdr_src: bool,
    hdr_pos: &mut usize,
    seq: &[PacketIndex],
    all: bool,
    want: Option<&TileWant>,
    cancel: &AtomicBool,
    work: &mut i64,
    budget: &mut Budget,
    lists: Lists<'_>,
) -> Result<()> {
    let Lists {
        out,
        queue,
        blocks,
        segs: free_segs,
    } = lists;
    let uses_sop = cod.uses_sop;
    let uses_eph = cod.uses_eph;
    let tg_passes = tg.passes as i32;

    let src = if use_hdr_src { hdr_data } else { tile_data };

    let mut pos = 0usize; // tile-data body cursor: SOP markers and code-block bytes
    out.clear();
    budget.charge(seq.len().saturating_mul(std::mem::size_of::<PacketRef>()))?;
    out.try_reserve_exact(seq.len())
        .map_err(|_| Error::OutOfMemory)?;
    // Reused across packets; `qcap` is the room charged for it (it may have more, from an
    // earlier tile).
    queue.clear();
    let mut qcap = 0usize;

    for (pkt_idx, pi) in seq.iter().enumerate() {
        if pkt_idx & 255 == 0 && cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let res = &mut tg.comps[pi.comp].resolutions[pi.res];
        let n_prec = res.pw * res.ph;
        if pi.prec >= n_prec {
            return Err(invalid!(
                "precinct {} out of range ({n_prec} precincts) for resolution {} component {}",
                pi.prec,
                pi.res,
                pi.comp
            ));
        }
        let cb_style = cb_styles[pi.comp];

        let mut pr = PacketRef {
            layer: pi.layer,
            resolution: pi.res,
            component: pi.comp,
            precinct: pi.prec,
            nsop: -1,
            blocks: blocks.pop().unwrap_or_default(),
            ..PacketRef::default()
        };

        // SOP is optional per packet even when Scod signals its use (T.800 A.8.1; OpenJPEG
        // warns rather than fails when one is missing or truncated). Only EPH, checked below,
        // is a hard error. SOP always lives in the tile data, even under PPM/PPT.
        if uses_sop
            && pos + 2 <= tile_data.len()
            && tile_data[pos] == 0xFF
            && tile_data[pos + 1] == 0x91
            && pos + 6 <= tile_data.len()
        {
            let nsop = u16::from_be_bytes([tile_data[pos + 4], tile_data[pos + 5]]) as usize;
            if nsop != pkt_idx % 65536 {
                return Err(invalid!(
                    "SOP Nsop mismatch at tile offset {pos}: got {nsop}, want {}",
                    pkt_idx % 65536
                ));
            }
            pr.nsop = nsop as i64;
            pos += 6;
        }

        if !use_hdr_src {
            *hdr_pos = pos; // one shared cursor: header and body interleave in tile_data
        }
        if *hdr_pos >= src.len() {
            return Err(invalid!(
                "packet header stream exhausted mid-sequence before packet layer={} res={} comp={} prec={}",
                pi.layer,
                pi.res,
                pi.comp,
                pi.prec
            ));
        }
        pr.header_pos = *hdr_pos;

        let mut br = BitReader::new(src, *hdr_pos);
        let present = br.read(1)?;

        if present == 1 {
            queue.clear();
            for band in res.bands.iter_mut() {
                if band.rect.empty() {
                    continue; // t2.c opj_tcd_is_band_empty: contributes no header bits
                }
                let subband = band.subband;
                let prc = &mut band.precincts[pi.prec];
                if prc.incl.is_empty() {
                    prc.alloc()?;
                }
                let ncb = prc.cw * prc.ch;
                *work -= ncb as i64;
                if *work < 0 {
                    return Err(invalid!(
                        "tile packet headers exceed the code-block inclusion work budget for the input size"
                    ));
                }
                for cbno in 0..ncb {
                    if cbno & 1023 == 0 && cancel.load(Ordering::Relaxed) {
                        return Err(Error::Cancelled);
                    }
                    let included;
                    if !prc.cb[cbno].ever_included {
                        let threshold = pi.layer as i32 + 1;
                        let incl = &mut prc.incl;
                        if incl.excluded(threshold) {
                            // root already proved nothing included
                            included = false;
                        } else {
                            included = incl.decode(cbno, threshold, &mut br)?;
                        }
                    } else {
                        included = br.read_bit_impl()? == 1;
                    }
                    if !included {
                        continue;
                    }

                    if !prc.cb[cbno].ever_included {
                        let zbp = decode_zero_bit_planes(&mut prc.imsb, cbno, &mut br)?;
                        let st = &mut prc.cb[cbno];
                        st.zero_bit_planes = zbp;
                        st.lblock = 3;
                        st.ever_included = true;
                    }

                    let numnewpasses = read_coding_passes(&mut br)?;
                    while br.read_bit_impl()? != 0 {
                        prc.cb[cbno].lblock += 1;
                    }

                    let rect = prc.cb_rect(cbno);
                    let cw = prc.cw;
                    let st = &mut prc.cb[cbno];
                    let mut r = CodeBlockRef {
                        subband,
                        cbx: cbno % cw,
                        cby: cbno / cw,
                        x0: rect.x0,
                        y0: rect.y0,
                        x1: rect.x1,
                        y1: rect.y1,
                        zero_bit_planes: st.zero_bit_planes,
                        segments: Vec::new(),
                    };
                    // Not wanted: lengths are still read, to skip its bodies.
                    let mut block_idx: Option<usize> = None;
                    if want.is_none_or(|w| w.block(pi.comp, pi.res, &r)) {
                        // Twice the record for Vec growth.
                        budget.charge(2 * size_of::<CodeBlockRef>())?;
                        r.segments = free_segs.pop().unwrap_or_default();
                        pr.blocks.push(r);
                        block_idx = Some(pr.blocks.len() - 1);
                    }

                    // T.800/OpenJPEG t2.c opj_t2_read_packet_header: the Lblock increment above
                    // is read once, but the newly included passes can spill across several
                    // termination segments (TERMALL/bypass styles), each with its own length
                    // field sized from the passes it actually holds.
                    if st.nsegs == 0 {
                        init_seg(st, cb_style, true);
                    }
                    if st.seg.numpasses == st.seg.maxpasses {
                        init_seg(st, cb_style, false);
                    }
                    let mut n = numnewpasses as i32;
                    while n > 0 {
                        let newp = (st.seg.maxpasses - st.seg.numpasses).min(n);
                        let bits = st.lblock as i64 + floor_log2(newp.max(0) as usize);
                        if bits > 32 {
                            return Err(invalid!(
                                "code-block {cbno} length field of {bits} bits exceeds 32"
                            ));
                        }
                        let length = br.read(bits.max(0) as usize)?;
                        st.seg.numpasses += newp;
                        st.total_passes += newp;
                        if st.total_passes > CODE_BLOCK_PASS_BUDGET {
                            return Err(limit!(
                                "code-block {cbno} accumulated more than {CODE_BLOCK_PASS_BUDGET} coding passes, exceeds the sanity limit"
                            ));
                        }
                        let mut seg_idx = None;
                        // Past passes_read only the bytes are skipped.
                        if let Some(bi) = block_idx
                            && st.total_passes - newp < tg_passes
                        {
                            budget.charge(2 * size_of::<CodeBlockSegment>())?;
                            let segs = &mut pr.blocks[bi].segments;
                            segs.push(CodeBlockSegment {
                                seg_index: (st.nsegs - 1) as usize,
                                start: 0,
                                end: 0,
                                passes: newp as usize,
                            });
                            seg_idx = Some(segs.len() - 1);
                        }
                        if queue.len() == qcap {
                            // Charge the doubling before it happens: jobs are kept for blocks
                            // the region does not want too.
                            let grow = qcap.max(64);
                            budget.charge(grow * size_of::<SegmentJob>())?;
                            queue
                                .try_reserve_exact(grow)
                                .map_err(|_| Error::OutOfMemory)?;
                            qcap += grow;
                        }
                        queue.push(SegmentJob {
                            block_idx,
                            seg_idx,
                            length,
                        });
                        n -= newp;
                        if n > 0 {
                            init_seg(st, cb_style, false);
                        }
                    }
                }
            }

            br.align()?;
            *hdr_pos = br.pos;

            if uses_eph {
                // EPH lives in the header stream (tile data, or the merged PPM/PPT stream when
                // one is in use), never in tile_data directly under PPM/PPT.
                if *hdr_pos + 2 > src.len() || src[*hdr_pos] != 0xFF || src[*hdr_pos + 1] != 0x92 {
                    return Err(invalid!("missing EPH marker at header offset {}", *hdr_pos));
                }
                *hdr_pos += 2;
            }
            if !use_hdr_src {
                pos = *hdr_pos;
            }

            // Lengths are known only after the whole header is parsed; bodies always come from
            // the tile data, in header order.
            for job in queue.iter() {
                if pos as u64 + job.length as u64 > tile_data.len() as u64 {
                    return Err(invalid!(
                        "code-block data segment [{pos},{}) exceeds tile data length {}",
                        pos as u64 + job.length as u64,
                        tile_data.len()
                    ));
                }
                let end = pos + job.length as usize;
                if let (Some(bi), Some(si)) = (job.block_idx, job.seg_idx) {
                    let seg = &mut pr.blocks[bi].segments[si];
                    seg.start = pos;
                    seg.end = end;
                }
                pos = end;
            }
        } else {
            pr.empty = true;
            br.align()?;
            *hdr_pos = br.pos;

            if uses_eph {
                if *hdr_pos + 2 > src.len() || src[*hdr_pos] != 0xFF || src[*hdr_pos + 1] != 0x92 {
                    return Err(invalid!("missing EPH marker at header offset {}", *hdr_pos));
                }
                *hdr_pos += 2;
            }
            if !use_hdr_src {
                pos = *hdr_pos;
            }
        }

        out.push(pr);
    }

    if all && pos != tile_data.len() {
        return Err(invalid!(
            "{} bytes left over after the expected packet sequence (consumed {pos} of {})",
            tile_data.len() - pos,
            tile_data.len()
        ));
    }

    Ok(())
}
