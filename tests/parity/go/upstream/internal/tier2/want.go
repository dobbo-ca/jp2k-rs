package tier2

import "github.com/dobbo-ca/jp2k-go/internal/codestream"

// TileWant is what a region decode reads of one tile, per component: the
// resolutions up to top, and of those only what dwtLevels' windows cover
// (all of them where the windows are nil).
type TileWant struct {
	Top  []int
	Wins [][]codestream.SubbandRect
}

// block reports whether code-block cb of comp's resolution res is read.
func (w *TileWant) block(comp, res int, cb *CodeBlockRef) bool {
	if res > w.Top[comp] || w.Wins[comp] != nil && len(w.Wins[comp]) == 0 {
		return false
	}
	if w.Wins[comp] == nil {
		return true
	}
	bw, _, _ := BandWindow(w.Wins[comp], res, cb.Subband)
	return !bw.Intersect(codestream.SubbandRect{X0: cb.X0, Y0: cb.Y0, X1: cb.X1, Y1: cb.Y1}).Empty()
}

// packet reports whether packet (comp, res, prec) of tg can hold a code-
// block that block reads. Out-of-range indices say yes and are left for
// tier-2 to reject.
func (w *TileWant) packet(tg *TileGeom, comp, res, prec int) bool {
	if comp >= len(w.Top) || comp >= len(tg.Comps) || res >= len(tg.Comps[comp].Resolutions) {
		return true
	}
	if res > w.Top[comp] || w.Wins[comp] != nil && len(w.Wins[comp]) == 0 {
		return false
	}
	r := &tg.Comps[comp].Resolutions[res]
	if w.Wins[comp] == nil || prec >= r.pw*r.ph {
		return true
	}
	for _, b := range r.Bands {
		bw, _, _ := BandWindow(w.Wins[comp], res, b.Subband)
		if !bw.Intersect(b.precincts[prec].rect).Empty() {
			return true
		}
	}
	return false
}

// BandWindow is the part of subband sub at resolution r that level r's
// window reads, in band coordinates, and where it starts in that level's
// Mallat-ordered buffer: low-pass samples of each axis first, then
// high-pass. Resolution coordinate 2u is low-pass sample u, 2u+1 is
// high-pass sample u.
func BandWindow(wins []codestream.SubbandRect, r, sub int) (bw codestream.SubbandRect, ox, oy int) {
	e := wins[r]
	if r == 0 {
		return e, 0, 0
	}
	lo := codestream.SubbandRect{X0: (e.X0 + 1) >> 1, Y0: (e.Y0 + 1) >> 1, X1: (e.X1 + 1) >> 1, Y1: (e.Y1 + 1) >> 1}
	bw = lo
	if sub&1 != 0 { // HL, HH
		bw.X0, bw.X1, ox = e.X0>>1, e.X1>>1, lo.X1-lo.X0
	}
	if sub&2 != 0 { // LH, HH
		bw.Y0, bw.Y1, oy = e.Y0>>1, e.Y1>>1, lo.Y1-lo.Y0
	}
	return bw, ox, oy
}
