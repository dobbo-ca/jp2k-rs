use super::{CodeBlockRef, TileGeom};
use crate::codestream::Rect;

/// What a region decode reads of one tile, per component: the resolutions up to `top`, and of
/// those only what the DWT windows cover (all of them where the windows are `None`).
pub struct TileWant {
    pub top: Vec<usize>,
    pub wins: Vec<Option<Vec<Rect>>>,
}

impl TileWant {
    /// Whether code-block `cb` of component `comp`'s resolution `res` is read.
    pub fn block(&self, comp: usize, res: usize, cb: &CodeBlockRef) -> bool {
        if res > self.top[comp] {
            return false;
        }
        let wins = match &self.wins[comp] {
            None => return true,
            Some(w) => w,
        };
        if wins.is_empty() {
            return false;
        }
        let (bw, _, _) = band_window(wins, res, cb.subband);
        !bw.intersect(&Rect {
            x0: cb.x0,
            y0: cb.y0,
            x1: cb.x1,
            y1: cb.y1,
        })
        .empty()
    }

    /// Whether packet (comp, res, prec) of `tg` can hold a code-block that `block` reads.
    /// Out-of-range indices say yes and are left for tier-2 to reject.
    pub(crate) fn packet(&self, tg: &TileGeom, comp: usize, res: usize, prec: usize) -> bool {
        if comp >= self.top.len()
            || comp >= tg.comps.len()
            || res >= tg.comps[comp].resolutions.len()
        {
            return true;
        }
        if res > self.top[comp] {
            return false;
        }
        let wins = match &self.wins[comp] {
            None => return true,
            Some(w) => w,
        };
        if wins.is_empty() {
            return false;
        }
        let r = &tg.comps[comp].resolutions[res];
        if prec >= r.pw * r.ph {
            return true;
        }
        for b in &r.bands {
            let (bw, _, _) = band_window(wins, res, b.subband);
            if !bw.intersect(&b.precincts[prec].rect).empty() {
                return true;
            }
        }
        false
    }
}

/// The part of subband `sub` at resolution `r` that level `r`'s window reads, in band
/// coordinates, and where it starts in that level's Mallat-ordered buffer: low-pass samples of
/// each axis first, then high-pass. Resolution coordinate 2u is low-pass sample u, 2u+1 is
/// high-pass sample u.
pub fn band_window(wins: &[Rect], r: usize, sub: usize) -> (Rect, i64, i64) {
    let e = wins[r];
    if r == 0 {
        return (e, 0, 0);
    }
    let lo = Rect {
        x0: (e.x0 + 1) >> 1,
        y0: (e.y0 + 1) >> 1,
        x1: (e.x1 + 1) >> 1,
        y1: (e.y1 + 1) >> 1,
    };
    let mut bw = lo;
    let (mut ox, mut oy) = (0, 0);
    if sub & 1 != 0 {
        // HL, HH
        bw.x0 = e.x0 >> 1;
        bw.x1 = e.x1 >> 1;
        ox = lo.x1 - lo.x0;
    }
    if sub & 2 != 0 {
        // LH, HH
        bw.y0 = e.y0 >> 1;
        bw.y1 = e.y1 >> 1;
        oy = lo.y1 - lo.y0;
    }
    (bw, ox, oy)
}
