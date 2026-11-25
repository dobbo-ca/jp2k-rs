// Package tier2 reads JPEG 2000 packets (ISO/IEC 15444-1 Annex B): tile
// and precinct geometry, progression orders, tag trees and packet
// headers, down to each code-block's coding-pass segments.
package tier2

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"sort"
	"unsafe"

	"github.com/dobbo-ca/jp2k-go/internal/codestream"
	"github.com/dobbo-ca/jp2k-go/internal/tier1"
)

func floorLog2(n int) int { return bits.Len(uint(n)) - 1 }

// maxPrecinctsPerResolution bounds precinct-grid allocation. No real stream
// needs anywhere near this many; it exists to reject a crafted SIZ/COD
// before BuildTileGeom allocates gigabytes of empty precincts.
const maxPrecinctsPerResolution = 1 << 20

// MaxPrecinctsPerTile bounds the SUM of precincts allocated across every
// resolution, band, and component of one tile. maxPrecinctsPerResolution
// alone is not enough: a stream with the legal maximum of 33 resolutions
// (T.800 allows NumDecompLevels up to 32) can have every single resolution
// sit just under that per-resolution cap, so the aggregate work of building
// (and tag-tree-allocating) all of them is still tens of millions of
// precincts, multi-second to multi-minute even though no individual check
// ever fires.
const MaxPrecinctsPerTile = 1 << 22

// maxCodeBlocksPerTile bounds the SUM of cw*ch (code-blocks, and tag-tree
// nodes) across every precinct in the tile. maxCodeBlocksPerPrecinct caps
// one precinct and MaxPrecinctsPerTile caps the precinct count, but neither
// bounds their product: many resolutions can each have a precinct count and
// a per-precinct code-block count that individually pass, while the tile's
// total code-block allocation (and tag-tree node allocation, proportional
// to it) is still tens of millions.
const maxCodeBlocksPerTile = 1 << 24

// maxCodeBlocksPerPrecinct bounds cw*ch, the code-block (and tag-tree) grid
// of a single precinct. Capping the precinct COUNT (above) does nothing to
// bound precinct SIZE: a single 16384x16384 precinct (PPx=PPy=15, NL=0) with
// 4x4 code-blocks is one precinct holding 16M code-blocks. OpenJPEG guards
// this exact quantity (tcd.c opj_tcd_init_tile, l_nb_code_blocks) before
// allocating.
const maxCodeBlocksPerPrecinct = 1 << 20

// maxPacketsPerTile bounds the materialized packet sequence. OpenJPEG's
// progression iterators (pi.c) yield one packet at a time and the T2 loop
// stops as soon as tile data runs out, so a degenerate progression (huge
// layer count, or an RPCL precinct pitch of 1 driving a reference-grid walk
// of the whole tile) never costs more than the bytes actually available.
// This package builds the sequence eagerly instead, so it needs an explicit
// ceiling in its place.
const maxPacketsPerTile = 1 << 20

// segState is one code-block's termination segment: how many coding passes
// it has accumulated and its capacity, per initSeg below.
type segState struct {
	numpasses, maxpasses int
}

// CodeblockState is Tier-2 state that must persist for the life of a tile:
// Lblock only ever grows, a code-block is only ever tag-tree-decoded for
// inclusion until its first inclusion layer, and its termination segments
// only ever grow in number across every including layer.
type CodeblockState struct {
	everIncluded  bool
	lblock        int
	zeroBitPlanes int32
	seg           segState // the last segment: only it takes passes
	nsegs         int      // segments begun, seg being number nsegs-1
	totalPasses   int      // cumulative passes across every including layer, never reset
}

// codeBlockPassBudget guards against a crafted stream that repeats
// small-but-nonzero numnewpasses codewords across many layers. Cumulative
// passes across layers cannot legally exceed the bit-plane total (numbps
// caps at 31, each bit-plane needs at most 3 passes, so 91 real passes);
// OpenJPEG uses a flat 109. Scaling by the layer count (uint16) would open
// a ~6M-segment, ~300MB allocation window from a few MB of crafted headers.
func codeBlockPassBudget(int) int {
	return 109
}

// PassesRead is how many of a code-block's passes, summed over layers,
// tier-1 can read: 3*Mb-2 (T.800 D.3, E.2), with the tile's largest Mb
// standing in for each subband's. Tier-2 keeps no segment past it, so
// a crafted stream cannot make it allocate for passes nothing reads,
// and the estimate relies on that.
func PassesRead(cs *codestream.Codestream, tileIdx uint16) int {
	if cs.QCD == nil {
		return 109 // built by hand; tier-1 needs one anyway
	}
	mb := 0
	for c := range cs.SIZ.Components {
		q := codestream.QuantFor(cs, tileIdx, c)
		for _, e := range q.Exponents {
			mb = max(mb, int(q.GuardBits)+int(e)-1)
		}
	}
	return min(109, max(1, 3*mb-2))
}

// initSeg starts a new termination segment for a code-block, T.800/OpenJPEG
// t2.c opj_t2_init_seg: TERMALL (0x04) makes every pass its own segment;
// selective bypass (0x01, "LAZY") makes the first segment 10 passes, then
// alternates length 2 (a raw SPP+MRP pair) and length 1 (an MQ cleanup);
// TERMALL wins over bypass when both are set. Default style is one
// effectively-unbounded (109-pass) segment, matching the earlier
// behaviour when codeBlockPassBudget caps it lower still.
func initSeg(st *CodeblockState, cbStyle uint8, first bool) {
	var maxp int
	switch {
	case cbStyle&tier1.CbTermAll != 0:
		maxp = 1
	case cbStyle&tier1.CbBypass != 0:
		if first {
			maxp = 10
		} else {
			prev := st.seg.maxpasses
			if prev == 1 || prev == 10 {
				maxp = 2
			} else {
				maxp = 1
			}
		}
	default:
		maxp = 109
	}
	st.seg = segState{maxpasses: maxp}
	st.nsegs++
}

type Precinct struct {
	rect       codestream.SubbandRect // clipped to the owning subband
	cw, ch     int                    // code-block grid dimensions (also the tag-tree size)
	tlcx, tlcy int                    // code-block grid origin
	cbw, cbh   uint                   // code-block size exponents
	// Allocated by alloc on the precinct's first packet: a reduced or
	// windowed decode never reaches most of a large tile's precincts.
	incl *tagTree
	imsb *tagTree
	cb   []CodeblockState // len cw*ch, persists across layers
}

// alloc sets up the per-code-block tier-2 state.
func (p *Precinct) alloc() {
	p.incl = newTagTree(p.cw, p.ch)
	p.imsb = newTagTree(p.cw, p.ch)
	p.cb = make([]CodeblockState, p.cw*p.ch)
}

// cbRect is code-block cbno's rect (row-major), clipped to the precinct.
func (p *Precinct) cbRect(cbno int) codestream.SubbandRect {
	x0 := p.tlcx + (cbno%p.cw)<<p.cbw
	y0 := p.tlcy + (cbno/p.cw)<<p.cbh
	return codestream.SubbandRect{
		X0: max(x0, p.rect.X0),
		X1: min(x0+1<<p.cbw, p.rect.X1),
		Y0: max(y0, p.rect.Y0),
		Y1: min(y0+1<<p.cbh, p.rect.Y1),
	}
}

type Band struct {
	Subband   int // 0=LL, 1=HL, 2=LH, 3=HH
	Rect      codestream.SubbandRect
	precincts []Precinct // shares the resolution's precinct grid, len pw*ph
}

type Resolution struct {
	Rect     codestream.SubbandRect
	ppx, ppy uint // precinct exponents at this resolution, un-halved
	pw, ph   int
	CbW, CbH uint   // code-block size exponents, the same in every precinct
	Bands    []Band // 1 band (LL) at r==0, else 3 (HL, LH, HH)
}

type TileComponent struct {
	Tcx0, Tcx1, Tcy0, Tcy1 int
	NumResolutions         int
	Resolutions            []Resolution
}

type TileGeom struct {
	tx0, tx1, ty0, ty1 int // tile rect in the reference grid
	Comps              []TileComponent
	precincts          int // per layer, summed over components
	codeBlocks         int
	passes             int // PassesRead
}

// BuildTileGeom computes the full tile -> component -> resolution -> band
// -> precinct -> code-block partition once. It is reused for every packet
// in the tile, since tag-tree and Lblock state must persist across layers.
func BuildTileGeom(cs *codestream.Codestream, tileIdx uint16) (*TileGeom, error) {
	return buildTileGeomLimits(cs, tileIdx, maxPacketsPerTile, maxCodeBlocksPerTile)
}

// CodeBlockBudget caps a tile's code-blocks by its bytes. Unlike packets,
// a code-block can cost zero bits (one tag-tree bit excludes a whole
// precinct), so this is a heuristic: 256K code-blocks (~30 MB of state)
// are free, beyond that a real stream carries far more than a byte per 16
// of them.
func CodeBlockBudget(bytes int) int {
	return int(min(maxCodeBlocksPerTile, max(1<<18, 16*int64(bytes))))
}

// buildTileGeomLimits is BuildTileGeom with the tile's packet budget (see
// PacketBudget) and code-block budget checked before allocation. Every
// precinct counts, POC-covered or not: allocating it costs the same.
func buildTileGeomLimits(cs *codestream.Codestream, tileIdx uint16, maxPackets, maxCodeBlocks int) (*TileGeom, error) {
	siz := cs.SIZ
	p := int(tileIdx) % int(siz.TilesWide)
	q := int(tileIdx) / int(siz.TilesWide)

	tx0 := max(int(siz.XTOsiz)+p*int(siz.XTsiz), int(siz.XOsiz))
	tx1 := min(int(siz.XTOsiz)+(p+1)*int(siz.XTsiz), int(siz.Xsiz))
	ty0 := max(int(siz.YTOsiz)+q*int(siz.YTsiz), int(siz.YOsiz))
	ty1 := min(int(siz.YTOsiz)+(q+1)*int(siz.YTsiz), int(siz.Ysiz))
	if tx1 <= tx0 || ty1 <= ty0 {
		return nil, fmt.Errorf("jp2k: tile %d has empty extent", tileIdx)
	}

	tg := &TileGeom{tx0: tx0, tx1: tx1, ty0: ty0, ty1: ty1, passes: PassesRead(cs, tileIdx)}
	tileCOD, tileCOC, _ := codestream.TileOverrides(cs, tileIdx)
	totalPrecincts := 0

	for c := 0; c < int(siz.Csiz); c++ {
		xr := int(siz.Components[c].XRsiz)
		yr := int(siz.Components[c].YRsiz)

		style := codestream.EffectiveCodingStyle(cs, tileCOD, tileCOC, c)
		if style.CodeBlockStyle&tier1.CbHTMask != 0 {
			return nil, fmt.Errorf("jp2k: component %d uses unsupported HT block coding (style 0x%02x)", c, style.CodeBlockStyle)
		}
		numRes := int(style.NumDecompLevels) + 1
		if numRes > 33 {
			return nil, fmt.Errorf("jp2k: component %d declares %d resolutions, exceeds the T.800 maximum of 33", c, numRes)
		}
		if style.PrecinctSizes != nil && len(style.PrecinctSizes) < numRes {
			return nil, fmt.Errorf("jp2k: component %d has %d custom precinct sizes for %d resolutions", c, len(style.PrecinctSizes), numRes)
		}

		tcomp := TileComponent{
			Tcx0:           codestream.CeilDiv(tx0, xr),
			Tcx1:           codestream.CeilDiv(tx1, xr),
			Tcy0:           codestream.CeilDiv(ty0, yr),
			Tcy1:           codestream.CeilDiv(ty1, yr),
			NumResolutions: numRes,
		}

		for r := 0; r < numRes; r++ {
			levelno := numRes - 1 - r
			res := Resolution{
				Rect: codestream.SubbandRect{
					X0: codestream.CeilDivPow2(tcomp.Tcx0, uint(levelno)),
					X1: codestream.CeilDivPow2(tcomp.Tcx1, uint(levelno)),
					Y0: codestream.CeilDivPow2(tcomp.Tcy0, uint(levelno)),
					Y1: codestream.CeilDivPow2(tcomp.Tcy1, uint(levelno)),
				},
				ppx: 15,
				ppy: 15,
			}
			if style.PrecinctSizes != nil {
				res.ppx = uint(style.PrecinctSizes[r].PPx)
				res.ppy = uint(style.PrecinctSizes[r].PPy)
			}
			if r > 0 && (res.ppx == 0 || res.ppy == 0) {
				return nil, fmt.Errorf("jp2k: component %d resolution %d has an illegal zero precinct exponent (Table A.21 forbids it for r>0)", c, r)
			}

			tlPrcX := codestream.FloorDivPow2(res.Rect.X0, res.ppx) << res.ppx
			tlPrcY := codestream.FloorDivPow2(res.Rect.Y0, res.ppy) << res.ppy
			brPrcX := codestream.CeilDivPow2(res.Rect.X1, res.ppx) << res.ppx
			brPrcY := codestream.CeilDivPow2(res.Rect.Y1, res.ppy) << res.ppy
			if res.Rect.X0 != res.Rect.X1 {
				res.pw = (brPrcX - tlPrcX) >> res.ppx
			}
			if res.Rect.Y0 != res.Rect.Y1 {
				res.ph = (brPrcY - tlPrcY) >> res.ppy
			}
			// A malformed SIZ (huge Xsiz/Ysiz) can drive pw*ph past int64
			// range (OpenJPEG tcd.c:998-1009 guards the same overflow) or,
			// short of overflow, past anything a real stream needs; either
			// way allocating precincts for it panics or exhausts memory
			// before a single packet is read.
			if res.pw != 0 && res.ph > math.MaxInt/res.pw {
				return nil, fmt.Errorf("jp2k: component %d resolution %d precinct grid %d x %d overflows", c, r, res.pw, res.ph)
			}
			if res.pw*res.ph > maxPrecinctsPerResolution {
				return nil, fmt.Errorf("jp2k: component %d resolution %d has %d precincts, exceeds the sanity limit of %d", c, r, res.pw*res.ph, maxPrecinctsPerResolution)
			}
			tg.precincts += res.pw * res.ph
			if tg.precincts > maxPackets {
				return nil, fmt.Errorf("jp2k: tile %d has more than %d precincts per layer, more packets than its bytes can hold", tileIdx, maxPackets)
			}

			// Precincts subdivide the resolution; bands at r>0 are half
			// that scale (a resolution-r+1 LL pixel maps to 2x2 pixels in
			// each r child band), so the code-block-group grid used to
			// place code-blocks within a band's precinct is halved too.
			var cbgwexp, cbghexp uint
			var tlcbgx, tlcbgy int
			numBands := 1
			if r == 0 {
				cbgwexp, cbghexp = res.ppx, res.ppy
				tlcbgx, tlcbgy = tlPrcX, tlPrcY
			} else {
				cbgwexp, cbghexp = res.ppx-1, res.ppy-1
				tlcbgx = codestream.CeilDivPow2(tlPrcX, 1)
				tlcbgy = codestream.CeilDivPow2(tlPrcY, 1)
				numBands = 3
			}
			cblkwexp := min(uint(style.CodeBlockWidthExp), cbgwexp)
			cblkhexp := min(uint(style.CodeBlockHeightExp), cbghexp)
			res.CbW, res.CbH = cblkwexp, cblkhexp

			for b := 0; b < numBands; b++ {
				subbandIdx := 0
				rect := res.Rect
				if r > 0 {
					subbandIdx = b + 1 // 1=HL, 2=LH, 3=HH
					x0b := subbandIdx & 1
					y0b := subbandIdx >> 1
					shift := uint(levelno) + 1
					rect = codestream.SubbandRect{
						X0: codestream.CeilDivPow2(tcomp.Tcx0-(x0b<<uint(levelno)), shift),
						X1: codestream.CeilDivPow2(tcomp.Tcx1-(x0b<<uint(levelno)), shift),
						Y0: codestream.CeilDivPow2(tcomp.Tcy0-(y0b<<uint(levelno)), shift),
						Y1: codestream.CeilDivPow2(tcomp.Tcy1-(y0b<<uint(levelno)), shift),
					}
				}

				bandG := Band{Subband: subbandIdx, Rect: rect}
				numPrec := res.pw * res.ph
				totalPrecincts += numPrec
				if totalPrecincts > MaxPrecinctsPerTile {
					return nil, fmt.Errorf("jp2k: tile %d has more than %d precincts across all resolutions and components, exceeds the sanity limit", tileIdx, MaxPrecinctsPerTile)
				}
				bandG.precincts = make([]Precinct, numPrec)
				for precno := 0; precno < numPrec; precno++ {
					cbgx0 := tlcbgx + (precno%res.pw)*(1<<cbgwexp)
					cbgy0 := tlcbgy + (precno/res.pw)*(1<<cbghexp)

					prc := Precinct{rect: codestream.SubbandRect{
						X0: max(cbgx0, rect.X0),
						X1: min(cbgx0+(1<<cbgwexp), rect.X1),
						Y0: max(cbgy0, rect.Y0),
						Y1: min(cbgy0+(1<<cbghexp), rect.Y1),
					}}
					if prc.rect.X0 < prc.rect.X1 && prc.rect.Y0 < prc.rect.Y1 {
						tlcbx := codestream.FloorDivPow2(prc.rect.X0, cblkwexp) << cblkwexp
						tlcby := codestream.FloorDivPow2(prc.rect.Y0, cblkhexp) << cblkhexp
						brcbx := codestream.CeilDivPow2(prc.rect.X1, cblkwexp) << cblkwexp
						brcby := codestream.CeilDivPow2(prc.rect.Y1, cblkhexp) << cblkhexp
						prc.cw = (brcbx - tlcbx) >> cblkwexp
						prc.ch = (brcby - tlcby) >> cblkhexp
						if prc.cw != 0 && prc.ch > maxCodeBlocksPerPrecinct/prc.cw {
							return nil, fmt.Errorf("jp2k: component %d resolution %d precinct %d has a %d x %d code-block grid, exceeds the sanity limit of %d code-blocks", c, r, precno, prc.cw, prc.ch, maxCodeBlocksPerPrecinct)
						}
						tg.codeBlocks += prc.cw * prc.ch
						if tg.codeBlocks > maxCodeBlocks {
							return nil, fmt.Errorf("jp2k: tile %d has more than %d code-blocks across all precincts, exceeds the limit for its size", tileIdx, maxCodeBlocks)
						}
						prc.tlcx, prc.tlcy, prc.cbw, prc.cbh = tlcbx, tlcby, cblkwexp, cblkhexp
					}
					bandG.precincts[precno] = prc
				}

				res.Bands = append(res.Bands, bandG)
			}

			tcomp.Resolutions = append(tcomp.Resolutions, res)
		}

		tg.Comps = append(tg.Comps, tcomp)
	}

	return tg, nil
}

// DropState frees the tag trees and code-block state a packet parse left
// on tg's precincts, leaving only geometry, and the precincts too unless
// keep is set: only AnyPacket reads them.
func (tg *TileGeom) DropState(keep bool) {
	for c := range tg.Comps {
		for r := range tg.Comps[c].Resolutions {
			for b := range tg.Comps[c].Resolutions[r].Bands {
				band := &tg.Comps[c].Resolutions[r].Bands[b]
				if !keep {
					band.precincts = nil
					continue
				}
				for i := range band.precincts {
					band.precincts[i].incl, band.precincts[i].imsb, band.precincts[i].cb = nil, nil, nil
				}
			}
		}
	}
}

// Size is about the heap tg holds after DropState.
func (tg *TileGeom) Size() int64 {
	n := int64(unsafe.Sizeof(*tg)) + int64(cap(tg.Comps))*int64(unsafe.Sizeof(TileComponent{}))
	for c := range tg.Comps {
		n += int64(cap(tg.Comps[c].Resolutions)) * int64(unsafe.Sizeof(Resolution{}))
		for _, r := range tg.Comps[c].Resolutions {
			n += int64(cap(r.Bands)) * int64(unsafe.Sizeof(Band{}))
			for _, b := range r.Bands {
				n += int64(cap(b.precincts)) * int64(unsafe.Sizeof(Precinct{}))
			}
		}
	}
	return n
}

// -- tile-part data assembly ---------------------------------------------

// AssembleTile concatenates a tile's tile-part bodies in TPsot order into
// one flat buffer. Concatenating (rather than resuming across tile-part
// boundaries) removes "packet straddles a tile-part boundary" as a case
// the bit reader has to know about.
func AssembleTile(cs *codestream.Codestream, data []byte, tileIdx uint16) ([]byte, error) {
	return assembleTileInto(cs, data, tileIdx, nil)
}

// assembleTileInto is AssembleTile, concatenating into *buf (grown and
// updated in place) when that is not nil, so a decode can reuse one
// buffer for every tile.
func assembleTileInto(cs *codestream.Codestream, data []byte, tileIdx uint16, buf *[]byte) ([]byte, error) {
	parts, err := TileSpans(cs, tileIdx)
	if err != nil || len(parts) == 0 {
		return nil, err
	}
	if len(parts) == 1 {
		tp := parts[0]
		if tp.DataOffset < 0 || tp.DataOffset+tp.DataLength > len(data) {
			return nil, fmt.Errorf("jp2k: tile %d tile-part data range out of bounds", tileIdx)
		}
		// One tile-part needs no concatenation: skip the copy.
		return data[tp.DataOffset : tp.DataOffset+tp.DataLength : tp.DataOffset+tp.DataLength], nil
	}
	var b []byte
	if buf != nil {
		b = (*buf)[:0]
	}
	for _, tp := range parts {
		if tp.DataOffset < 0 || tp.DataOffset+tp.DataLength > len(data) {
			return nil, fmt.Errorf("jp2k: tile %d tile-part data range out of bounds", tileIdx)
		}
		b = append(b, data[tp.DataOffset:tp.DataOffset+tp.DataLength]...)
	}
	if buf != nil {
		*buf = b
	}
	return b, nil
}

// Span is where a tile-part's data lies in the codestream.
type Span struct {
	DataOffset, DataLength int
	tpsot                  uint8
}

// TileSpans returns tileIdx's tile-part data in TPsot order, the order
// its data is joined in: none for a tile whose components are all
// empty, else an error unless its TPsot run is dense.
func TileSpans(cs *codestream.Codestream, tileIdx uint16) ([]Span, error) {
	idx := codestream.TilePartIndex(cs, tileIdx)
	if len(idx) == 0 {
		if tileComponentsEmpty(cs, tileIdx) {
			// Every component's sample grid is empty for this tile after
			// subsampling ceildiv (e.g. XRsiz/YRsiz larger than the tile),
			// so a real encoder legitimately emits no tile-part for it.
			return nil, nil
		}
		return nil, fmt.Errorf("jp2k: no tile-parts found for tile %d", tileIdx)
	}
	all := cs.Parts()
	spans := make([]Span, len(idx))
	for i, j := range idx {
		tp := &all[j]
		spans[i] = Span{tp.DataOffset, tp.DataLength, tp.SOT.TPsot}
	}
	sort.SliceStable(spans, func(i, j int) bool { return spans[i].tpsot < spans[j].tpsot })
	for i, sp := range spans {
		if int(sp.tpsot) != i {
			return nil, fmt.Errorf("jp2k: tile %d tile-parts are not a dense 0..N-1 TPsot run (got TPsot %d at position %d)", tileIdx, sp.tpsot, i)
		}
	}
	return spans, nil
}

// tileComponentsEmpty reports whether every component's sample grid is
// empty for tileIdx: ceildiv(tx,XRsiz) collapses to zero width or height.
// Mirrors BuildTileGeom's Tcx0/Tcx1/Tcy0/Tcy1 without building the full
// geometry, since AssembleTile needs it before a tile-part-missing error.
func tileComponentsEmpty(cs *codestream.Codestream, tileIdx uint16) bool {
	siz := cs.SIZ
	p := int(tileIdx) % int(siz.TilesWide)
	q := int(tileIdx) / int(siz.TilesWide)
	tx0 := max(int(siz.XTOsiz)+p*int(siz.XTsiz), int(siz.XOsiz))
	tx1 := min(int(siz.XTOsiz)+(p+1)*int(siz.XTsiz), int(siz.Xsiz))
	ty0 := max(int(siz.YTOsiz)+q*int(siz.YTsiz), int(siz.YOsiz))
	ty1 := min(int(siz.YTOsiz)+(q+1)*int(siz.YTsiz), int(siz.Ysiz))
	for c := 0; c < int(siz.Csiz); c++ {
		xr := int(siz.Components[c].XRsiz)
		yr := int(siz.Components[c].YRsiz)
		if codestream.CeilDiv(tx0, xr) != codestream.CeilDiv(tx1, xr) && codestream.CeilDiv(ty0, yr) != codestream.CeilDiv(ty1, yr) {
			return false
		}
	}
	return true
}

// -- packet bit reader (T.800 B.10.1, the FF bit-stuffing rule) ----------

type bitReader struct {
	data []byte
	pos  int
	buf  uint64
	ct   uint
}

func newBitReader(data []byte, pos int) *bitReader {
	return &bitReader{data: data, pos: pos}
}

func (b *bitReader) bytein() error {
	b.buf = (b.buf << 8) & 0xffff
	if b.buf == 0xff00 {
		b.ct = 7 // previous byte was 0xFF: its stuffed successor has 7 real bits
	} else {
		b.ct = 8
	}
	if b.pos < 0 || b.pos >= len(b.data) {
		return fmt.Errorf("jp2k: packet header read past end of tile data at offset %d", b.pos)
	}
	b.buf |= uint64(b.data[b.pos])
	b.pos++
	return nil
}

func (b *bitReader) readBit() (int, error) {
	if b.ct == 0 {
		if err := b.bytein(); err != nil {
			return 0, err
		}
	}
	b.ct--
	return int((b.buf >> b.ct) & 1), nil
}

// read reads n bits (n <= 32) into a uint32, most-significant bit first.
func (b *bitReader) read(n int) (uint32, error) {
	var v uint32
	for i := 0; i < n; i++ {
		bit, err := b.readBit()
		if err != nil {
			return 0, err
		}
		v = v<<1 | uint32(bit)
	}
	return v, nil
}

// align drops any partial byte and, if the last consumed byte was 0xFF,
// consumes its stuffed successor too (T.800 B.10.1).
func (b *bitReader) align() error {
	if b.buf&0xff == 0xff {
		if err := b.bytein(); err != nil {
			return err
		}
	}
	b.ct = 0
	return nil
}

// decodeZeroBitPlanes resolves a code-block's zero-bitplane count via the
// tag tree (T.800 B.10.5.3): the actual number of all-zero MSB
// bit-planes, not OpenJPEG's raw loop index "i" (one more). Values of 74
// and up are an error, as in OpenJPEG's threshold loop.
func decodeZeroBitPlanes(tree *tagTree, cbno int, readBit func() (int, error)) (int32, error) {
	// One walk at the loop's final threshold (74) reads exactly the bits
	// the increasing-threshold loop would: every node on the path
	// resolves before its child reads anything, whatever the threshold.
	done, err := tree.decode(cbno, 74, readBit)
	if err != nil {
		return 0, err
	}
	if !done {
		return 0, fmt.Errorf("jp2k: zero-bitplane tag tree for code-block %d did not resolve", cbno)
	}
	return tree.nodes[cbno].value, nil
}

// readCodingPasses decodes the number-of-coding-passes codeword (T.800
// B.10.6 / OpenJPEG t2.c opj_t2_getnumpasses).
func readCodingPasses(br *bitReader) (int, error) {
	bit, err := br.read(1)
	if err != nil {
		return 0, err
	}
	if bit == 0 {
		return 1, nil
	}
	bit, err = br.read(1)
	if err != nil {
		return 0, err
	}
	if bit == 0 {
		return 2, nil
	}
	n, err := br.read(2)
	if err != nil {
		return 0, err
	}
	if n != 3 {
		return int(3 + n), nil
	}
	n, err = br.read(5)
	if err != nil {
		return 0, err
	}
	if n != 31 {
		return int(6 + n), nil
	}
	n, err = br.read(7)
	if err != nil {
		return 0, err
	}
	return int(37 + n), nil
}

// -- packet records --------------------------------------------------------

// CodeBlockSegment is one packet's contribution of bytes to one of a
// code-block's termination segments. SegIndex identifies which of the
// code-block's segments (across its whole lifetime, T.800/OpenJPEG t2.c
// opj_t2_init_seg) this is: a segment can span several layers under the
// default style (each layer just appends more bytes to the same open MQ
// codeword until it hits its 109-pass cap), so two CodeBlockSegments from
// different packets with the same SegIndex belong to the same decoder run
// and must be concatenated, not decoded independently. TERMALL/bypass
// styles instead start a new SegIndex on (almost) every pass.
type CodeBlockSegment struct {
	SegIndex   int
	Start, End int
	Passes     int
}

// CodeBlockRef records one included code-block's Tier-2 header result: no
// coefficient data is decoded in the tier-1 stage.
type CodeBlockRef struct {
	Subband        int   // 0=LL/HL-band index within the packet's band order
	CbX, CbY       int   // code-block index within its precinct's grid
	X0, Y0, X1, Y1 int   // clipped code-block rect, in subband coordinates
	ZeroBitPlanes  int32 // number of all-zero MSB bit-planes (T.800 B.10.5), not "missing MSBs + 1"
	Segments       []CodeBlockSegment
}

// PacketRef records one packet's identity and its included code-blocks.
type PacketRef struct {
	Layer, Resolution, Component, Precinct int
	Nsop                                   int // -1 if SOP markers are not in use
	Empty                                  bool
	HeaderPos                              int // offset into the tile data, or into the packed PPM/PPT header stream
	Blocks                                 []CodeBlockRef
}

type PacketIndex struct{ layer, res, comp, prec int }

// -- progression iterators (T.800 B.12) -----------------------------------

func maxResolutions(tg *TileGeom) int {
	n := 0
	for _, c := range tg.Comps {
		if c.NumResolutions > n {
			n = c.NumResolutions
		}
	}
	return n
}

// packetSeq collects a tile's packet sequence. Every progression appends
// through add, so no order can outgrow the tile's budget.
type packetSeq struct {
	seq  []PacketIndex
	left int // entries still allowed
	walk int // RPCL/PCRL/CPRL position-walk steps still allowed
}

// charge takes an RPCL/PCRL/CPRL reference-grid walk's step estimate from
// the tile's walk budget before the walk runs: the walk costs CPU even at
// positions that emit no packet.
func (s *packetSeq) charge(name string, tg *TileGeom, dx, dy, perPosition int) error {
	if perPosition <= 0 {
		return nil // empty window: the walk's inner loops never run
	}
	steps := (int64(tg.tx1-tg.tx0)/int64(dx) + 2) * (int64(tg.ty1-tg.ty0)/int64(dy) + 2) * int64(perPosition)
	if steps > int64(s.walk) {
		return fmt.Errorf("jp2k: %s precinct grid walk of an estimated %d steps (dx=%d, dy=%d) exceeds the tile's remaining walk budget of %d", name, steps, dx, dy, s.walk)
	}
	s.walk -= int(steps)
	return nil
}

func (s *packetSeq) add(l, r, c, p int) error {
	if s.left <= 0 {
		return fmt.Errorf("jp2k: tile packet sequence exceeds its budget (sanity limit %d, or more packets than the codestream's bytes can hold)", maxPacketsPerTile)
	}
	s.left--
	s.seq = append(s.seq, PacketIndex{l, r, c, p})
	return nil
}

// PacketBudget caps a tile's packet count by what its bytes can carry:
// every packet needs at least one header bit, so a tile holds at most 8
// packets per byte of tile data plus PPM/PPT header data. The small
// constant is slack; maxPacketsPerTile still applies.
func PacketBudget(dataBytes, hdrBytes int) int {
	n := 8*(int64(dataBytes)+int64(hdrBytes)) + 8
	if n > maxPacketsPerTile {
		return maxPacketsPerTile
	}
	return int(n)
}

// The generators below apply rec's layer/resolution/component window
// while generating, so a narrow POC record costs only what it emits.

func lrcpOrder(tg *TileGeom, rec pocRecord, s *packetSeq) error {
	for l := 0; l < rec.layno1; l++ {
		for r := rec.resno0; r < rec.resno1; r++ {
			for c := rec.compno0; c < rec.compno1; c++ {
				comp := tg.Comps[c]
				if r >= comp.NumResolutions {
					continue
				}
				res := comp.Resolutions[r]
				for p := 0; p < res.pw*res.ph; p++ {
					if err := s.add(l, r, c, p); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func rlcpOrder(tg *TileGeom, rec pocRecord, s *packetSeq) error {
	for r := rec.resno0; r < rec.resno1; r++ {
		for l := 0; l < rec.layno1; l++ {
			for c := rec.compno0; c < rec.compno1; c++ {
				comp := tg.Comps[c]
				if r >= comp.NumResolutions {
					continue
				}
				res := comp.Resolutions[r]
				for p := 0; p < res.pw*res.ph; p++ {
					if err := s.add(l, r, c, p); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

// rpclOrder is the positional resolution-precinct-component-layer order
// (T.800 B.12.1.3). Unlike LRCP/RLCP it needs the precinct's spatial
// position, not just its index, so it walks the reference grid in steps of
// the finest common precinct pitch and computes each component's precinct
// index at that position (OpenJPEG pi.c opj_pi_next_rpcl).
func rpclOrder(cs *codestream.Codestream, tg *TileGeom, rec pocRecord, s *packetSeq) error {
	allComps := make([]int, len(tg.Comps))
	for i := range allComps {
		allComps[i] = i
	}
	dx, dy := precinctStepXY(cs, tg, allComps)
	if dx <= 0 || dy <= 0 {
		return fmt.Errorf("jp2k: RPCL precinct step computed as non-positive")
	}
	if err := s.charge("RPCL", tg, dx, dy, (rec.resno1-rec.resno0)*(rec.compno1-rec.compno0)); err != nil {
		return err
	}

	for r := rec.resno0; r < rec.resno1; r++ {
		// x/y as int64: avoids int32 overflow.
		for y64 := int64(tg.ty0); y64 < int64(tg.ty1); y64 += int64(dy) - y64%int64(dy) {
			y := int(y64)
			for x64 := int64(tg.tx0); x64 < int64(tg.tx1); x64 += int64(dx) - x64%int64(dx) {
				x := int(x64)
				for c := rec.compno0; c < rec.compno1; c++ {
					p, ok := precinctAt(cs, tg, c, r, x, y)
					if !ok {
						continue
					}
					for l := 0; l < rec.layno1; l++ {
						if err := s.add(l, r, c, p); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// precinctStepXY returns the finest common precinct pitch (in reference-grid
// units) over comps, the position step opj_pi_next_{rpcl,pcrl,cprl} all
// share (pi.c dx/dy, :513/:518): RPCL and PCRL take the min over every
// component, CPRL recomputes it per component (cprlOrder passes one comp).
func precinctStepXY(cs *codestream.Codestream, tg *TileGeom, comps []int) (dx, dy int) {
	// int64 avoids 386 32-bit shift wrap.
	dx64, dy64 := int64(-1), int64(-1)
	for _, ci := range comps {
		comp := tg.Comps[ci]
		xr := int64(cs.SIZ.Components[ci].XRsiz)
		yr := int64(cs.SIZ.Components[ci].YRsiz)
		for r, res := range comp.Resolutions {
			levelno := comp.NumResolutions - 1 - r
			stepX := xr << uint(int(res.ppx)+levelno)
			stepY := yr << uint(int(res.ppy)+levelno)
			if dx64 == -1 || stepX < dx64 {
				dx64 = stepX
			}
			if dy64 == -1 || stepY < dy64 {
				dy64 = stepY
			}
		}
	}
	return int(dx64), int(dy64)
}

// precinctAt computes the precinct index for tile-component (c, r) at
// reference-grid position (x,y), per pi.c's shared RPCL/PCRL/CPRL body
// (opj_pi_next_rpcl :430-443, identical in _pcrl/_cprl). ok is false when
// (x,y) is not on this component/resolution's precinct grid, or the
// tile-component-resolution is empty.
func precinctAt(cs *codestream.Codestream, tg *TileGeom, c, r int, x, y int) (p int, ok bool) {
	comp := tg.Comps[c]
	if r >= comp.NumResolutions {
		return 0, false
	}
	// int64 avoids 386 32-bit shift wrap.
	xr := int64(cs.SIZ.Components[c].XRsiz)
	yr := int64(cs.SIZ.Components[c].YRsiz)
	levelno := comp.NumResolutions - 1 - r
	xrLvl := xr << uint(levelno)
	yrLvl := yr << uint(levelno)
	trx0 := ceilDiv64(int64(tg.tx0), xrLvl)
	try0 := ceilDiv64(int64(tg.ty0), yrLvl)
	trx1 := ceilDiv64(int64(tg.tx1), xrLvl)
	try1 := ceilDiv64(int64(tg.ty1), yrLvl)
	res := comp.Resolutions[r]
	rpx := uint(int(res.ppx) + levelno)
	rpy := uint(int(res.ppy) + levelno)

	yAligned := int64(y)%(yr<<rpy) == 0 || (y == tg.ty0 && (try0<<uint(levelno))%(int64(1)<<rpy) != 0)
	if !yAligned {
		return 0, false
	}
	xAligned := int64(x)%(xr<<rpx) == 0 || (x == tg.tx0 && (trx0<<uint(levelno))%(int64(1)<<rpx) != 0)
	if !xAligned {
		return 0, false
	}
	if res.pw == 0 || res.ph == 0 || trx0 == trx1 || try0 == try1 {
		return 0, false
	}

	prci := codestream.FloorDivPow2(int(ceilDiv64(int64(x), xrLvl)), res.ppx) - codestream.FloorDivPow2(int(trx0), res.ppx)
	prcj := codestream.FloorDivPow2(int(ceilDiv64(int64(y), yrLvl)), res.ppy) - codestream.FloorDivPow2(int(try0), res.ppy)
	return prci + prcj*res.pw, true
}

// int64 avoids 32-bit shift wrap.
func ceilDiv64(a, b int64) int64 { return (a + b - 1) / b }

// pcrlOrder is the positional precinct-component-resolution-layer order
// (T.800 B.12.1.4, OpenJPEG opj_pi_next_pcrl). Same dx/dy step and
// precinct-grid test as RPCL; only the loop nesting (and hence packet
// order) differs.
func pcrlOrder(cs *codestream.Codestream, tg *TileGeom, rec pocRecord, s *packetSeq) error {
	allComps := make([]int, len(tg.Comps))
	for i := range allComps {
		allComps[i] = i
	}
	dx, dy := precinctStepXY(cs, tg, allComps)
	if dx <= 0 || dy <= 0 {
		return fmt.Errorf("jp2k: PCRL precinct step computed as non-positive")
	}
	if err := s.charge("PCRL", tg, dx, dy, (rec.resno1-rec.resno0)*(rec.compno1-rec.compno0)); err != nil {
		return err
	}

	// x/y as int64: avoids int32 overflow.
	for y64 := int64(tg.ty0); y64 < int64(tg.ty1); y64 += int64(dy) - y64%int64(dy) {
		y := int(y64)
		for x64 := int64(tg.tx0); x64 < int64(tg.tx1); x64 += int64(dx) - x64%int64(dx) {
			x := int(x64)
			for c := rec.compno0; c < rec.compno1; c++ {
				for r := rec.resno0; r < rec.resno1; r++ {
					p, ok := precinctAt(cs, tg, c, r, x, y)
					if !ok {
						continue
					}
					for l := 0; l < rec.layno1; l++ {
						if err := s.add(l, r, c, p); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// cprlOrder is the positional component-precinct-resolution-layer order
// (T.800 B.12.1.5, OpenJPEG opj_pi_next_cprl). Unlike RPCL/PCRL, dx/dy is
// recomputed per component from that component's own resolutions alone
// (pi.c :640-661), so components with different XRsiz/YRsiz walk the grid
// at different pitches.
func cprlOrder(cs *codestream.Codestream, tg *TileGeom, rec pocRecord, s *packetSeq) error {
	for c := rec.compno0; c < rec.compno1; c++ {
		dx, dy := precinctStepXY(cs, tg, []int{c})
		if dx <= 0 || dy <= 0 {
			return fmt.Errorf("jp2k: CPRL precinct step computed as non-positive for component %d", c)
		}
		if err := s.charge("CPRL", tg, dx, dy, rec.resno1-rec.resno0); err != nil {
			return err
		}
		// x/y as int64: avoids int32 overflow.
		for y64 := int64(tg.ty0); y64 < int64(tg.ty1); y64 += int64(dy) - y64%int64(dy) {
			y := int(y64)
			for x64 := int64(tg.tx0); x64 < int64(tg.tx1); x64 += int64(dx) - x64%int64(dx) {
				x := int(x64)
				for r := rec.resno0; r < rec.resno1; r++ {
					p, ok := precinctAt(cs, tg, c, r, x, y)
					if !ok {
						continue
					}
					for l := 0; l < rec.layno1; l++ {
						if err := s.add(l, r, c, p); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// -- packet header / body decoding -----------------------------------------

// Decode assembles a tile's data, partitions it per T.800 Annex B, and
// decodes every packet header in progression order. No coefficient data is
// decoded; each code-block's inclusion, zero bit-planes, and segment byte
// ranges are recorded.
//
// A standalone call always starts its own fresh PPM cursor. That is only
// correct for a codestream whose main header carries no PPM markers, or
// when this is the only tile being decoded: PPM is ONE continuous stream
// for the whole codestream (T.800 A.7.4), so decoding multiple tiles each
// needs to pick up the PPM cursor where the previous tile left it. The
// top-level Decode() pipeline uses DecodeCursor directly with one
// Cursor shared across every tile, in tile-processing order, instead of
// calling this function per tile.
func Decode(cs *codestream.Codestream, data []byte, tileIdx uint16) ([]PacketRef, error) {
	pkts, _, _, err := DecodeCursor(cs, data, tileIdx, new(Cursor))
	return pkts, err
}

// Cursor is the state one decode shares across its tiles: the PPM
// read position and budgets drawn from the whole codestream's bytes. A
// per-tile budget alone lets every tile claim the same PPM bytes.
type Cursor struct {
	ppmPos     int
	started    bool
	packets    int64 // precincts built plus packets generated
	walk       int64 // RPCL/PCRL/CPRL walk steps
	codeBlocks int64
	work       int64 // packet-header inclusion work left, see t2WorkPerByte
	// WorkSpent is work already drawn from a budget shared with cur.
	WorkSpent int64
	last      Charge
	// Done, when closed, stops the packet parse with ErrStopped.
	Done <-chan struct{}
}

// Charge is what parsing one tile draws from a Cursor's budgets, and
// the tile's own limits on it.
type Charge struct {
	budget, cbBudget                     int64 // the tile's packet and code-block limits
	precincts, codeBlocks, packets, walk int64
	work                                 int64 // inclusion work, see t2WorkPerByte
}

// start sets up cur's budgets for a decode of dataLen codestream bytes.
func (cur *Cursor) start(cs *codestream.Codestream, dataLen int) {
	if cur.started {
		return
	}
	// 8 packets per byte as in PacketBudget, plus its per-tile slack.
	cur.started = true
	cur.packets = 8*int64(dataLen) + 8*int64(cs.SIZ.TilesWide)*int64(cs.SIZ.TilesHigh)
	cur.walk = 16 * cur.packets
	cur.codeBlocks = max(1<<18, 16*int64(dataLen)) // CodeBlockBudget, uncapped
	cur.work = WorkBudget(dataLen) - cur.WorkSpent
}

// t2WorkPerByte and t2WorkSlack cap tier-2 packet-header CPU per decode.
const t2WorkPerByte, t2WorkSlack = 1 << 10, 1 << 20

// WorkBudget is the tier-2 packet-header work budget for dataLen bytes.
func WorkBudget(dataLen int) int64 { return t2WorkPerByte*int64(dataLen) + t2WorkSlack }

// Work returns cur's tier-2 work budget left.
func (cur *Cursor) Work() int64 { return cur.work }

// Started reports whether cur has begun drawing on its budgets.
func (cur *Cursor) Started() bool { return cur.started }

// Replay draws ch, from a parse of tileIdx under budgets no tighter,
// from cur as parsing the tile again would, and fails where that would.
func (cur *Cursor) Replay(cs *codestream.Codestream, dataLen int, tileIdx uint16, ch Charge) error {
	cur.start(cs, dataLen)
	budget := min(ch.budget, cur.packets)
	if ch.precincts > budget {
		return fmt.Errorf("jp2k: tile %d has more than %d precincts per layer, more packets than its bytes can hold", tileIdx, budget)
	}
	if lim := min(ch.cbBudget, cur.codeBlocks); ch.codeBlocks > lim {
		return fmt.Errorf("jp2k: tile %d has more than %d code-blocks across all precincts, exceeds the limit for its size", tileIdx, lim)
	}
	cur.packets -= ch.precincts
	cur.codeBlocks -= ch.codeBlocks
	if lim := min(16*budget, cur.walk); ch.walk > lim {
		return fmt.Errorf("jp2k: tile %d precinct grid walk of an estimated %d steps exceeds the tile's remaining walk budget of %d", tileIdx, ch.walk, lim)
	}
	if ch.packets > min(budget, cur.packets) {
		return fmt.Errorf("jp2k: tile packet sequence exceeds its budget (sanity limit %d, or more packets than the codestream's bytes can hold)", maxPacketsPerTile)
	}
	cur.packets -= ch.packets
	cur.walk -= ch.walk
	if cur.work -= ch.work; cur.work < 0 {
		return ErrWork
	}
	return nil
}

// ErrWork is the error of a parse past its tier-2 work budget.
var ErrWork = errors.New("jp2k: tile packet headers exceed the code-block inclusion work budget for the input size")

// ErrStopped is the error of a parse stopped by Cursor.Done.
var ErrStopped = errors.New("jp2k: packet parse stopped")

// DecodeCursor is Decode's implementation. cur holds the merged
// main-header PPM stream's persistent read cursor; callers that decode
// every tile of a codestream must thread the same cursor through each call,
// in the order tiles are processed, so a PPM stream spanning several
// tiles' packet headers is read as one continuous sequence instead of
// restarting at 0 for every tile. It also returns the tile's geometry and
// data so Decode does not build either twice.
func DecodeCursor(cs *codestream.Codestream, data []byte, tileIdx uint16, cur *Cursor) ([]PacketRef, *TileGeom, []byte, error) {
	return DecodeTile(cs, data, tileIdx, cur, nil, nil)
}

// DecodeTile is DecodeCursor for a region decode. tileBuf, if set,
// is reused for tile data that spans tile-parts. wantOf, if set, is
// called once the geometry is built and says what the region reads:
// only those code-blocks are recorded (every header is still parsed:
// tag-tree state and body offsets depend on it, and PLT lengths are
// input that nothing short of parsing can check), and reading stops
// after the last packet that can hold one, except under PPM, whose
// shared cursor has to pass every packet. Budgets are charged for the whole tile first, so
// nothing skipped here escapes them.
func DecodeTile(cs *codestream.Codestream, data []byte, tileIdx uint16, cur *Cursor, tileBuf *[]byte, wantOf func(*TileGeom) (*TileWant, error)) ([]PacketRef, *TileGeom, []byte, error) {
	tileData, err := assembleTileInto(cs, data, tileIdx, tileBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	pkts, tg, _, err := decodeTileData(cs, tileData, len(data), tileIdx, cur, wantOf, nil)
	return pkts, tg, tileData, err
}

// IndexTile parses every packet header of tileIdx, whose data from
// TileSpans is tileData, as DecodeTile would with dataLen bytes of
// codestream in hand, for an index kept across decodes. It passes the
// geometry to geom once built, then each packet to emit, which must not
// keep it. When a packet fails it also returns tail, the sequence from
// the first packet a decode needing any of it fails on: the one that
// failed, or the last when bytes are left over. ch is what the parse
// drew from cur, once it got as far as a packet sequence.
func IndexTile(cs *codestream.Codestream, tileData []byte, dataLen int, tileIdx uint16, cur *Cursor, geom func(*TileGeom) error, emit func(*PacketRef) error) (tail []PacketIndex, ch Charge, err error) {
	k := 0
	_, _, seq, err := decodeTileData(cs, tileData, dataLen, tileIdx, cur, func(tg *TileGeom) (*TileWant, error) {
		return nil, geom(tg)
	}, func(p *PacketRef) error {
		k++
		return emit(p)
	})
	if err != nil && seq != nil {
		if k == len(seq) {
			k = max(0, k-1)
		}
		tail = append([]PacketIndex(nil), seq[k:]...)
	}
	return tail, cur.last, err
}

// AnyPacket reports whether a decode reading w (nil for all) of tg
// parses any packet of tail, which DecodeTile would then fail on. An
// empty tail fails them all: its error was not a packet's.
func (w *TileWant) AnyPacket(tg *TileGeom, tail []PacketIndex) bool {
	if w == nil || len(tail) == 0 {
		return true
	}
	for _, p := range tail {
		if w.packet(tg, p.comp, p.res, p.prec) {
			return true
		}
	}
	return false
}

// stopped reports whether done is closed.
func stopped(done <-chan struct{}) bool {
	select {
	case <-done:
		return true
	default:
		return false
	}
}

// decodeTileData is DecodeTile after tile assembly. It also returns the
// packet sequence, when it got as far as parsing packets. emit, if set,
// takes each packet instead of the result.
func decodeTileData(cs *codestream.Codestream, tileData []byte, dataLen int, tileIdx uint16, cur *Cursor, wantOf func(*TileGeom) (*TileWant, error), emit func(*PacketRef) error) ([]PacketRef, *TileGeom, []PacketIndex, error) {
	cur.start(cs, dataLen)
	cur.last = Charge{}
	hdrData, useHdrSrc, hdrPos, err := packetHeaderSource(cs, tileIdx, &cur.ppmPos)
	if err != nil {
		return nil, nil, nil, err
	}
	hdrBytes := 0
	if useHdrSrc {
		hdrBytes = max(len(hdrData)-*hdrPos, 0)
	}
	ch := Charge{budget: int64(PacketBudget(len(tileData), hdrBytes)), cbBudget: int64(CodeBlockBudget(len(tileData) + hdrBytes))}
	budget := int(min(ch.budget, cur.packets))

	tileCOD, tileCOC, _ := codestream.TileOverrides(cs, tileIdx)
	// T.800 A.6.1: a COD in the tile's first tile-part overrides the main
	// header's progression order, layer count, and SOP/EPH use for this tile.
	effCOD := cs.COD
	if tileCOD != nil {
		effCOD = tileCOD
	}

	records, err := tilePOCRecords(cs, tileIdx, effCOD)
	if err != nil {
		return nil, nil, nil, err
	}
	tg, err := buildTileGeomLimits(cs, tileIdx, budget, int(min(ch.cbBudget, cur.codeBlocks)))
	if err != nil {
		return nil, nil, nil, err
	}
	cur.packets -= int64(tg.precincts)
	cur.codeBlocks -= int64(tg.codeBlocks)
	// Real streams walk well under one step per budgeted packet; 16x is
	// slack, and it keeps walk CPU tied to the input's bytes.
	s := &packetSeq{left: int(min(int64(budget), cur.packets)), walk: int(min(16*int64(budget), cur.walk))}
	// Sized up front, within the budget: append garbage sets a region
	// decode's peak heap.
	s.seq = make([]PacketIndex, 0, max(0, min(int64(s.left), int64(tg.precincts)*int64(effCOD.NumberOfLayers))))
	left, walk := s.left, s.walk
	seq, err := pocSequence(cs, tg, records, s)
	if err != nil {
		return nil, nil, nil, err
	}
	cur.packets -= int64(left - s.left)
	cur.walk -= int64(walk - s.walk)
	ch.precincts, ch.codeBlocks, ch.packets, ch.walk = int64(tg.precincts), int64(tg.codeBlocks), int64(left-s.left), int64(walk-s.walk)
	cur.last = ch

	var want *TileWant
	if wantOf != nil {
		if want, err = wantOf(tg); err != nil {
			return nil, nil, nil, err
		}
	}
	all := true
	if want != nil && len(cs.PPM) == 0 {
		n := 0
		for i, pi := range seq {
			if want.packet(tg, pi.comp, pi.res, pi.prec) {
				n = i + 1
			}
		}
		all = n == len(seq)
		seq = seq[:n]
	}

	cbStyles := make([]uint8, len(tg.Comps))
	for c := range tg.Comps {
		cbStyles[c] = codestream.EffectiveCodingStyle(cs, tileCOD, tileCOC, c).CodeBlockStyle
	}

	work := cur.work
	pkts, err := decodePackets(effCOD, cbStyles, tg, tileData, hdrData, useHdrSrc, hdrPos, seq, all, want, cur.Done, &cur.work, emit)
	cur.last.work = work - cur.work
	return pkts, tg, seq, err
}

// packetHeaderSource resolves T.800 B.9/A.7.2's PPM/PPT precedence: if the
// main header carries PPM markers, EVERY tile's packet headers come from
// the merged main-header stream, read via the caller's persistent ppmPos
// cursor; otherwise a tile with PPT markers gets its own merged per-tile
// stream with a fresh cursor (PPT is scoped to one tile, T.800 A.7.5);
// otherwise packet headers live inline in the tile data (useHdrSrc false,
// the plain path), also with a fresh cursor since it is
// realigned to the tile-data body cursor on every packet anyway.
func packetHeaderSource(cs *codestream.Codestream, tileIdx uint16, ppmPos *int) (hdrData []byte, useHdrSrc bool, hdrPos *int, err error) {
	if len(cs.PPM) > 0 {
		hdrData, err := cs.PPMHeaders()
		if err != nil {
			return nil, false, nil, err
		}
		return hdrData, true, ppmPos, nil
	}
	hdrData, ok, err := cs.PPTHeaders(tileIdx)
	if err != nil {
		return nil, false, nil, err
	}
	if !ok {
		return nil, false, new(int), nil
	}
	return hdrData, true, new(int), nil
}

// pocRecord is one progression-order-change entry driving a distinct chunk
// of a tile's packet sequence (T.800 B.12.2 / POC marker segment). The
// no-POC case is a single record spanning the whole tile at the tile's
// effective COD progression order (this unifies the POC and non-POC paths).
type pocRecord struct {
	prg              uint8
	resno0, resno1   int
	compno0, compno1 int
	layno1           int
}

// tilePOCRecords resolves T.800 A.6.1/A.6.2's POC precedence: OpenJPEG
// copies the main-header POC entries into every tile's tcp as its default
// (opj_j2k_copy_default_tcp_and_create_tcd), and a tile-part POC segment
// appends further records rather than replacing them
// (opj_j2k_read_poc's l_old_poc_nb). Records run in that order: every
// main-header POC change, then every tile-part POC change for tileIdx in
// tile-part order.
func tilePOCRecords(cs *codestream.Codestream, tileIdx uint16, effCOD *codestream.COD) ([]pocRecord, error) {
	var changes []codestream.POCChange
	for _, poc := range cs.POC {
		changes = append(changes, poc.Changes...)
	}
	// Consumption order is TPsot order, not file order: all records of a
	// tile share one include[] array, so out-of-order tile-parts carrying
	// POC would otherwise compose a different packet sequence.
	// Indices, not tile-part copies: a flood of tile-parts makes those
	// large.
	all := cs.Parts()
	var idx []int32
	for _, k := range codestream.TilePartIndex(cs, tileIdx) {
		if len(all[k].POC) > 0 {
			idx = append(idx, k)
		}
	}
	sort.SliceStable(idx, func(i, j int) bool { return all[idx[i]].SOT.TPsot < all[idx[j]].SOT.TPsot })
	for _, k := range idx {
		for _, poc := range all[k].POC {
			changes = append(changes, poc.Changes...)
		}
	}
	// The per-header cap in parseMarkers does not bound the tile: every
	// tile-part may add another 31 (OpenJPEG opj_j2k_read_poc caps the sum).
	if len(changes) >= 32 {
		return nil, fmt.Errorf("jp2k: tile %d has %d POC changes across its headers, exceeds the limit of 31", tileIdx, len(changes))
	}
	if len(changes) == 0 {
		return []pocRecord{{
			prg:     effCOD.ProgressionOrder,
			resno0:  0,
			resno1:  33, // T.800 maximum; pocSequence clamps to the tile
			compno0: 0,
			compno1: len(cs.SIZ.Components),
			layno1:  int(effCOD.NumberOfLayers),
		}}, nil
	}
	numComps := len(cs.SIZ.Components)
	numLayers := int(effCOD.NumberOfLayers)
	recs := make([]pocRecord, len(changes))
	for i, ch := range changes {
		layno1 := min(int(ch.LYEpoc), numLayers)
		compno1 := min(int(ch.CEpoc), numComps)
		recs[i] = pocRecord{
			prg:     ch.Ppoc,
			resno0:  int(ch.RSpoc),
			resno1:  int(ch.REpoc),
			compno0: int(ch.CSpoc),
			compno1: compno1,
			layno1:  layno1,
		}
	}
	return recs, nil
}

// pocOrder appends one record's packets (T.800 B.12.1) for its Ppoc
// order, restricted to its layer/resolution/component window (B.12.2).
// precno is left unbounded because every order's precinct loop already
// covers each precinct of the resolutions/components in the window.
func pocOrder(cs *codestream.Codestream, tg *TileGeom, rec pocRecord, s *packetSeq) error {
	switch rec.prg {
	case 0:
		return lrcpOrder(tg, rec, s)
	case 1:
		return rlcpOrder(tg, rec, s)
	case 2:
		return rpclOrder(cs, tg, rec, s)
	case 3:
		return pcrlOrder(cs, tg, rec, s)
	case 4:
		return cprlOrder(cs, tg, rec, s)
	}
	return fmt.Errorf("jp2k: POC entry uses unsupported progression order %d", rec.prg)
}

// pocSequence concatenates every record's packet sequence in order,
// dropping any (layer,res,comp,prec) tuple a previous record already
// emitted (t2.c's shared include[] array across POC records: a tuple is
// only ever sent once, however many records' windows cover it). s's
// budgets are shared by every record.
func pocSequence(cs *codestream.Codestream, tg *TileGeom, records []pocRecord, s *packetSeq) ([]PacketIndex, error) {
	numRes := maxResolutions(tg)
	for _, rec := range records {
		rec.resno1 = min(rec.resno1, numRes)
		if err := pocOrder(cs, tg, rec, s); err != nil {
			return nil, err
		}
	}
	if len(records) == 1 {
		// One record never repeats a tuple: skip the dedup map.
		return s.seq, nil
	}
	seen := make(map[PacketIndex]bool, len(s.seq))
	out := s.seq[:0]
	for _, pi := range s.seq {
		if !seen[pi] {
			seen[pi] = true
			out = append(out, pi)
		}
	}
	return out, nil
}

// segmentJob defers a segment's Start/End fill-in: lengths are known only
// after the whole packet header is parsed, since bodies follow the header
// in header order, but each segment's length field is read as its header
// bits are decoded. blockIdx/segIdx index into pr.Blocks[blockIdx].Segments,
// not pointers, since pr.Blocks keeps reallocating as the packet is built.
type segmentJob struct {
	blockIdx, segIdx int
	length           uint32 // not int: 32 bits wraps negative on 386/arm
}

// decodePackets decodes every packet header in seq. Packet headers come
// from tileData unless useHdrSrc is set, in which case they come from
// hdrData (the merged PPM/PPT stream, T.800 B.9): *hdrPos tracks that
// stream's own persistent cursor, independent of pos, the tile-data body
// cursor (SOP markers and code-block bytes always live in tileData, T.800
// A.7.2). hdrPos is a pointer, not a
// local, because a PPM stream can span several tiles' worth of packet
// headers: the caller threads the same cursor through every tile's call.
// all says seq is the tile's whole packet sequence, so its bodies must
// use up the tile data exactly. want, if set, filters the code-blocks
// recorded (see DecodeTile).
func decodePackets(cod *codestream.COD, cbStyles []uint8, tg *TileGeom, tileData []byte, hdrData []byte, useHdrSrc bool, hdrPos *int, seq []PacketIndex, all bool, want *TileWant, done <-chan struct{}, work *int64, emit func(*PacketRef) error) (out []PacketRef, err error) {
	usesSOP := cod.UsesSOP
	usesEPH := cod.UsesEPH

	src := tileData
	if useHdrSrc {
		src = hdrData
	}

	pos := 0 // tile-data body cursor: SOP markers and code-block bytes
	sopCounter := 0
	if emit == nil {
		out = make([]PacketRef, 0, len(seq))
	}
	var queue []segmentJob    // reused across packets
	var blocks []CodeBlockRef // reused across emitted packets

	for pktIdx, pi := range seq {
		if pktIdx&255 == 0 && stopped(done) {
			return out, ErrStopped
		}
		res := &tg.Comps[pi.comp].Resolutions[pi.res]
		if pi.prec < 0 || pi.prec >= res.pw*res.ph {
			return out, fmt.Errorf("jp2k: precinct %d out of range (%d precincts) for resolution %d component %d", pi.prec, res.pw*res.ph, pi.res, pi.comp)
		}
		cbStyle := cbStyles[pi.comp]

		pr := PacketRef{Layer: pi.layer, Resolution: pi.res, Component: pi.comp, Precinct: pi.prec, Nsop: -1}
		if emit != nil {
			pr.Blocks = blocks[:0]
		}

		// SOP is optional per packet even when Scod signals its use (T.800
		// A.8.1; OpenJPEG t2.c:1112-1113 warns rather than fails when one is
		// missing or truncated). Only EPH, checked below, is a hard error.
		// SOP always lives in the tile data, even under PPM/PPT.
		if usesSOP && pos+2 <= len(tileData) && tileData[pos] == 0xFF && tileData[pos+1] == 0x91 {
			if pos+6 <= len(tileData) {
				nsop := int(codestream.Be16(tileData[pos+4 : pos+6]))
				if nsop != sopCounter%65536 {
					return out, fmt.Errorf("jp2k: SOP Nsop mismatch at tile offset %d: got %d, want %d", pos, nsop, sopCounter%65536)
				}
				pr.Nsop = nsop
				pos += 6
			}
		}
		sopCounter++

		if !useHdrSrc {
			*hdrPos = pos // one shared cursor: header and body interleave in tileData
		}
		if *hdrPos < 0 {
			return out, fmt.Errorf("jp2k: negative packet header offset %d", *hdrPos)
		}
		if *hdrPos >= len(src) {
			return out, fmt.Errorf("jp2k: packet header stream exhausted mid-sequence before packet layer=%d res=%d comp=%d prec=%d", pi.layer, pi.res, pi.comp, pi.prec)
		}
		pr.HeaderPos = *hdrPos

		br := newBitReader(src, *hdrPos)
		present, err := br.read(1)
		if err != nil {
			return out, err
		}

		if present == 1 {
			queue = queue[:0]
			for _, bandG := range res.Bands {
				if bandG.Rect.Empty() {
					continue // t2.c opj_tcd_is_band_empty: contributes no header bits
				}
				prc := &bandG.precincts[pi.prec]
				if prc.incl == nil {
					prc.alloc()
				}
				if work != nil {
					*work -= int64(prc.cw * prc.ch)
					if *work < 0 {
						return out, ErrWork
					}
				}
				for cbno := 0; cbno < prc.cw*prc.ch; cbno++ {
					if cbno&1023 == 0 && stopped(done) {
						return out, ErrStopped
					}
					st := &prc.cb[cbno]

					var included bool
					if !st.everIncluded {
						threshold := int32(pi.layer + 1)
						if prc.incl.excluded(threshold) {
							// root already proved nothing included
							included = false
						} else {
							included, err = prc.incl.decode(cbno, threshold, br.readBit)
						}
					} else {
						var bit int
						bit, err = br.readBit()
						included = bit == 1
					}
					if err != nil {
						return out, err
					}
					if !included {
						continue
					}

					if !st.everIncluded {
						zbp, zerr := decodeZeroBitPlanes(prc.imsb, cbno, br.readBit)
						if zerr != nil {
							return out, zerr
						}
						st.zeroBitPlanes = zbp
						st.lblock = 3
						st.everIncluded = true
					}

					numnewpasses, perr := readCodingPasses(br)
					if perr != nil {
						return out, perr
					}
					for {
						bit, berr := br.readBit()
						if berr != nil {
							return out, berr
						}
						if bit == 0 {
							break
						}
						st.lblock++
					}

					rect := prc.cbRect(cbno)
					ref := CodeBlockRef{
						Subband:       bandG.Subband,
						CbX:           cbno % prc.cw,
						CbY:           cbno / prc.cw,
						X0:            rect.X0,
						Y0:            rect.Y0,
						X1:            rect.X1,
						Y1:            rect.Y1,
						ZeroBitPlanes: st.zeroBitPlanes,
					}
					blockIdx := -1 // not wanted: lengths are still read, to skip its bodies
					if want == nil || want.block(pi.comp, pi.res, &ref) {
						pr.Blocks = append(pr.Blocks, ref)
						blockIdx = len(pr.Blocks) - 1
					}

					// T.800/OpenJPEG t2.c opj_t2_read_packet_header: the
					// Lblock increment above is read once, but the newly
					// included passes can spill across several termination
					// segments (TERMALL/bypass styles), each with its own
					// length field sized from the passes it actually holds.
					if st.nsegs == 0 {
						initSeg(st, cbStyle, true)
					}
					if st.seg.numpasses == st.seg.maxpasses {
						initSeg(st, cbStyle, false)
					}
					n := numnewpasses
					for n > 0 {
						seg := &st.seg
						newp := seg.maxpasses - seg.numpasses
						if newp > n {
							newp = n
						}
						bits := st.lblock + floorLog2(newp)
						if bits > 32 {
							return out, fmt.Errorf("jp2k: code-block %d length field of %d bits exceeds 32", cbno, bits)
						}
						length, lerr := br.read(bits)
						if lerr != nil {
							return out, lerr
						}
						seg.numpasses += newp
						st.totalPasses += newp
						if budget := codeBlockPassBudget(int(cod.NumberOfLayers)); st.totalPasses > budget {
							return out, fmt.Errorf("jp2k: code-block %d accumulated more than %d coding passes, exceeds the sanity limit", cbno, budget)
						}
						segIdx := -1
						// Past PassesRead only the bytes are skipped.
						if blockIdx >= 0 && st.totalPasses-newp < tg.passes {
							pr.Blocks[blockIdx].Segments = append(pr.Blocks[blockIdx].Segments, CodeBlockSegment{SegIndex: st.nsegs - 1, Passes: newp})
							segIdx = len(pr.Blocks[blockIdx].Segments) - 1
						}
						queue = append(queue, segmentJob{blockIdx: blockIdx, segIdx: segIdx, length: length})
						n -= newp
						if n > 0 {
							initSeg(st, cbStyle, false)
						}
					}
				}
			}

			if err := br.align(); err != nil {
				return out, err
			}
			*hdrPos = br.pos

			if usesEPH {
				// EPH lives in the header stream (tile data, or the merged
				// PPM/PPT stream when one is in use), never in tileData
				// directly under PPM/PPT.
				if *hdrPos+2 > len(src) || src[*hdrPos] != 0xFF || src[*hdrPos+1] != 0x92 {
					return out, fmt.Errorf("jp2k: missing EPH marker at header offset %d", *hdrPos)
				}
				*hdrPos += 2
			}
			if !useHdrSrc {
				pos = *hdrPos
			}

			// Lengths are known only after the whole header is parsed;
			// bodies always come from the tile data, in header order.
			for _, job := range queue {
				if uint64(pos)+uint64(job.length) > uint64(len(tileData)) {
					return out, fmt.Errorf("jp2k: code-block data segment [%d,%d) exceeds tile data length %d", pos, uint64(pos)+uint64(job.length), len(tileData))
				}
				end := pos + int(job.length)
				if job.segIdx >= 0 {
					seg := &pr.Blocks[job.blockIdx].Segments[job.segIdx]
					seg.Start, seg.End = pos, end
				}
				pos = end
			}
		} else {
			pr.Empty = true
			if err := br.align(); err != nil {
				return out, err
			}
			*hdrPos = br.pos

			if usesEPH {
				if *hdrPos+2 > len(src) || src[*hdrPos] != 0xFF || src[*hdrPos+1] != 0x92 {
					return out, fmt.Errorf("jp2k: missing EPH marker at header offset %d", *hdrPos)
				}
				*hdrPos += 2
			}
			if !useHdrSrc {
				pos = *hdrPos
			}
		}

		if emit != nil {
			if err := emit(&pr); err != nil {
				return out, err
			}
			blocks = pr.Blocks
			continue
		}
		out = append(out, pr)
	}

	if all && pos != len(tileData) {
		return out, fmt.Errorf("jp2k: %d bytes left over after the expected packet sequence (consumed %d of %d)", len(tileData)-pos, pos, len(tileData))
	}

	return out, nil
}
