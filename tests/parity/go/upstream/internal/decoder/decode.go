// Package decoder is the decode pipeline behind the jp2k package: it
// reads the input under the memory budget, runs tier-2, tier-1, the
// wavelet and colour stages per tile, and assembles the output image,
// whole or for a region.
package decoder

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"io/fs"
	"math"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/dobbo-ca/jp2k-go/internal/codestream"
	"github.com/dobbo-ca/jp2k-go/internal/dsp"
	"github.com/dobbo-ca/jp2k-go/internal/parallel"
	"github.com/dobbo-ca/jp2k-go/internal/tier1"
	"github.com/dobbo-ca/jp2k-go/internal/tier2"
)

// -- pipeline glue: tier2 -> tier1 -> dequant -> inverse DWT --------------
//
// Shared by the production Decode path and pipeline_test.go's exact/
// tolerance tests and fuzzing.

// resRects returns comp's resolution rects, smallest first.
func resRects(tg *tier2.TileGeom, comp int) []codestream.SubbandRect {
	res := tg.Comps[comp].Resolutions
	rects := make([]codestream.SubbandRect, len(res))
	for r := range res {
		rects[r] = res[r].Rect
	}
	return rects
}

// cbOrigin returns where code-block cb of resolution r starts in comp's
// Mallat-layout plane (see InverseDWT53): LL at r=0 fills the corner, and
// at r>0 HL sits right of the sn low columns, LH below the low rows, HH
// diagonal. ok is false when a malformed (e.g. fuzzed) COD/QCD describes a
// code-block that would run outside its band's area of the plane.
func cbOrigin(tg *tier2.TileGeom, comp, r int, cb tier2.CodeBlockRef) (x, y int, ok bool) {
	if r < 0 || r >= len(tg.Comps[comp].Resolutions) {
		return 0, 0, false
	}
	res := tg.Comps[comp].Resolutions[r]
	var b *tier2.Band
	for i := range res.Bands {
		if res.Bands[i].Subband == cb.Subband {
			b = &res.Bands[i]
			break
		}
	}
	if b == nil {
		return 0, 0, false
	}
	bw, bh := res.Rect.X1-res.Rect.X0, res.Rect.Y1-res.Rect.Y0
	if r > 0 {
		snX, dnX := dsp.SnDn(bw, res.Rect.X0&1)
		snY, dnY := dsp.SnDn(bh, res.Rect.Y0&1)
		bw, bh = snX, snY
		if cb.Subband == int(dsp.SubbandHL) || cb.Subband == int(dsp.SubbandHH) {
			x, bw = snX, dnX
		}
		if cb.Subband == int(dsp.SubbandLH) || cb.Subband == int(dsp.SubbandHH) {
			y, bh = snY, dnY
		}
	}
	lx, ly := cb.X0-b.Rect.X0, cb.Y0-b.Rect.Y0
	if lx < 0 || ly < 0 || lx+cb.X1-cb.X0 > bw || ly+cb.Y1-cb.Y0 > bh {
		return 0, 0, false
	}
	return x + lx, y + ly, true
}

// cbKey identifies one code-block across every layer's packet, so its
// per-layer segments can be found and concatenated before decoding.
type cbKey struct{ res, prec, subband, cbx, cby int }

// cbGroup is one code-block's per-packet segment contributions gathered
// from every layer, in layer order (and, within a layer, tier-2 segment
// order), ready to be re-grouped by SegIndex and handed to DecodeCodeBlock.
type cbGroup struct {
	res  int
	cb   tier2.CodeBlockRef // representative rect/subband; ZeroBitPlanes from the first inclusion
	segs []cbSeg
}

// cbSeg is one packet's bytes [start, end) of a code-block's segment.
type cbSeg struct {
	layer      int
	segIndex   int
	start, end int
	passes     int
}

// groupCodeBlocks gathers comp's code-block segments across every packet
// (every layer, every progression position) by code-block identity, per
// DecodeCodeBlock's doc: "a code-block split across layers yields several
// CodeBlockRef segments; concatenate their bytes and sum their Passes
// before calling this." Segments are returned sorted by layer so
// concatenation order matches the encoder's layer order regardless of the
// codestream's progression (LRCP/RLCP/RPCL/PCRL/CPRL all interleave
// layers differently in packet order).
func groupCodeBlocks(pkts []tier2.PacketRef, comp int) []*cbGroup {
	groups := map[cbKey]*cbGroup{}
	var order []cbKey
	for _, p := range pkts {
		if p.Component != comp {
			continue
		}
		for _, cb := range p.Blocks {
			if cb.X1-cb.X0 <= 0 || cb.Y1-cb.Y0 <= 0 {
				continue
			}
			k := cbKey{p.Resolution, p.Precinct, cb.Subband, cb.CbX, cb.CbY}
			g, ok := groups[k]
			if !ok {
				g = &cbGroup{res: p.Resolution, cb: cb}
				groups[k] = g
				order = append(order, k)
			}
			for _, s := range cb.Segments {
				g.segs = append(g.segs, cbSeg{p.Layer, s.SegIndex, s.Start, s.End, s.Passes})
			}
		}
	}
	out := make([]*cbGroup, 0, len(order))
	for _, k := range order {
		g := groups[k]
		// SliceStable: a style that splits one layer's inclusion into
		// several termination segments (TERMALL/bypass) appends them here
		// in tier-2's own order, which sort.Slice's non-stable ties would
		// be free to scramble within a layer. SegIndex only ever grows as
		// later layers include a code-block, so after sorting by layer,
		// entries sharing a SegIndex are already contiguous.
		sort.SliceStable(g.segs, func(i, j int) bool { return g.segs[i].layer < g.segs[j].layer })
		out = append(out, g)
	}
	return out
}

// concatSegments re-groups a code-block's per-packet byte ranges (in
// groupCodeBlocks's layer order) by tier-2 SegIndex, concatenating every
// contribution to the same segment (a segment can span several layers
// under the default style, each just appending more bytes to the same
// open MQ codeword) into one DecodeCodeBlock Segment. Segments must stay
// distinct entries in the result, one per MQ/raw decoder run: flattening
// everything into a single buffer is only correct when there is exactly
// one segment (see Segment).
func (g *cbGroup) concatSegments(tileData []byte) []tier1.Segment {
	out, _ := g.concatSegmentsInto(tileData, nil, nil)
	return out
}

// concatSegmentsInto is concatSegments reusing out and appending joined
// bytes to buf, both returned for the next call: a tier-1 worker keeps
// them so a multi-layer code-block costs no allocation. A segment in one
// piece aliases tileData.
func (g *cbGroup) concatSegmentsInto(tileData []byte, out []tier1.Segment, buf []byte) ([]tier1.Segment, []byte) {
	out = out[:0]
	for i := 0; i < len(g.segs); {
		s := g.segs[i]
		seg := tier1.Segment{Data: tileData[s.start:s.end:s.end], Passes: s.passes}
		j := i + 1
		for j < len(g.segs) && g.segs[j].segIndex == s.segIndex {
			j++
		}
		if j > i+1 {
			st := len(buf)
			buf = append(buf, seg.Data...)
			for _, p := range g.segs[i+1 : j] {
				buf = append(buf, tileData[p.start:p.end]...)
				seg.Passes += p.passes
			}
			seg.Data = buf[st:len(buf):len(buf)]
		}
		out = append(out, seg)
		i = j
	}
	return out, buf
}

// cbSegmentsValid guards against a malformed packet-header byte range
// (out-of-order or out-of-bounds Start/End) before DecodeCodeBlock ever
// slices tileData; decodeComponentR/F skip such a code-block rather than
// panicking, since they also serve the fuzzed public Decode entry point.
func cbSegmentsValid(segs []tier2.CodeBlockSegment, tileDataLen int) bool {
	for _, s := range segs {
		if s.Start < 0 || s.End > tileDataLen || s.Start > s.End {
			return false
		}
	}
	return true
}

// segsValid re-checks every layer's byte range, not just the
// representative packet's: concatSegments slices tileData with g.segs, and
// on 32-bit GOARCHes a 0xFFFFFFFF length wraps negative past the uint32
// checks tier-2 does on 64-bit.
func (g *cbGroup) segsValid(tileDataLen int) bool {
	for _, s := range g.segs {
		if s.start < 0 || s.end > tileDataLen || s.start > s.end {
			return false
		}
	}
	return true
}

// plane is one tile component's working samples: i for the reversible
// 5/3 path, f for 9/7, plus the per-worker scratch that works on it.
// Everything is reused across the components and tiles of a decode.
// After decodeComponent, the finished w x h window starting at component
// sample (x0, y0) sits at i/f[off:] with row pitch stride.
type plane struct {
	i            []int32
	f            []float32
	rev          bool
	stride, off  int
	x0, y0, w, h int
	lvI          [][]int32 // per-level buffers; lvI[0] is the whole-tile plane
	lvF          [][]float32
	nw           int // worker cap, 0 for GOMAXPROCS
	dwtI         [][]int32
	dwtF         [][]float32
	t1           []*t1Worker
	done         <-chan struct{}            // closed to stop the decode
	work         *atomic.Int64              // tier-1 work left in the decode's budget, nil for none
	emit         func(y int, row []float32) // sink for the last DWT level's rows, set per lane/component
	packed       bool                       // true once emit packed the plane, in place of pack
	round        dsp.Rounding               // 9/7 lift and ICT rounding
}

// t1WorkPerByte and t1WorkSlack cap a decode's tier-1 work, the sum over
// its code-blocks of coding passes times samples, at t1WorkPerByte per
// input byte plus t1WorkSlack. That is over ten times the most any
// legitimate file tested needs: lossless 16-bit images of sparse points
// reach about 14000 per byte. A crafted file that signals every pass of
// empty code-blocks gets about 90000, but those passes are cheap.
const t1WorkPerByte, t1WorkSlack = 1 << 18, 1 << 22

// level sizes buffer l to n zeroed samples: code-blocks a stream leaves
// out (legal per T.800 A.6.2, or skipped here as malformed) must read 0.
func (p *plane) level(l, n int) {
	for len(p.lvI) <= l {
		p.lvI = append(p.lvI, nil)
		p.lvF = append(p.lvF, nil)
	}
	if p.rev {
		p.lvI[l] = zeroed(p.lvI[l], n)
	} else {
		p.lvF[l] = zeroed(p.lvF[l], n)
	}
}

func zeroed[T int32 | float32](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	s = s[:n]
	clear(s)
	return s
}

// cbJob is one code-block ready for tier-1: the part of it that is
// needed (block-local [sx0,sx1)x[sy0,sy1)), where that lands in level
// buffer lv, and its subband's step size.
type cbJob struct {
	g                  *cbGroup
	lv, x, y           int
	sx0, sy0, sx1, sy1 int
	delta              float64
	mb                 int
}

// t1Worker is one tier-1 worker's scratch, reused for every code-block
// it decodes.
type t1Worker struct {
	t    tier1.State
	raw  [tier1.MaxCodeBlockSamples]int32
	f    [tier1.MaxCodeBlockSamples]float32
	segs []tier1.Segment
	data []byte
}

// decodeComponent runs tier-1, dequantization and the inverse DWT for one
// tile component into p, up to resolution top and only as far as needed
// for win, a window of that resolution. Code-blocks are decoded on up to
// GOMAXPROCS workers, each writing its own disjoint rect of the plane.
// Malformed packets/code-blocks (out-of-range resolution, subband, or byte
// range) are skipped rather than causing a panic, since this also serves
// the fuzzed public Decode entry point. style is comp's effective coding
// style (T.800 A.6: tile/main COC overrides COD), not necessarily the main
// header's COD. groups are comp's code-blocks, from groupCodeBlocks.
// inverseDWT97NoEmit adapts InverseDWT97 to partialDWT's signature, a
// package-level value so partialDWT's windowed path allocates no
// closure per call.
func inverseDWT97NoEmit(f []float32, st int, rs []codestream.SubbandRect, h []bool, sc *[][]float32, nw int, d <-chan struct{}) {
	dsp.InverseDWT97(f, st, rs, h, sc, nw, d, nil, dsp.Fused)
}

// inverseDWT97SepNoEmit is inverseDWT97NoEmit with dsp.Separate.
func inverseDWT97SepNoEmit(f []float32, st int, rs []codestream.SubbandRect, h []bool, sc *[][]float32, nw int, d <-chan struct{}) {
	dsp.InverseDWT97(f, st, rs, h, sc, nw, d, nil, dsp.Separate)
}

// inverseDWT97JJNoEmit is inverseDWT97NoEmit with dsp.JJ2000.
func inverseDWT97JJNoEmit(f []float32, st int, rs []codestream.SubbandRect, h []bool, sc *[][]float32, nw int, d <-chan struct{}) {
	dsp.InverseDWT97(f, st, rs, h, sc, nw, d, nil, dsp.JJ2000)
}

func decodeComponent(cs *codestream.Codestream, tg *tier2.TileGeom, tileData []byte, groups []*cbGroup, tileIdx uint16, comp int, style codestream.CodingStyle, p *plane, top int, win codestream.SubbandRect) error {
	q := codestream.QuantFor(cs, tileIdx, comp)
	precision := int(cs.SIZ.Components[comp].Precision)
	cbStyle := style.CodeBlockStyle
	tc := &tg.Comps[comp]
	p.rev = style.Transformation == 1
	p.packed = false
	if win.Empty() {
		p.w, p.h = 0, 0
		return nil
	}
	p.x0, p.y0, p.w, p.h = win.X0, win.Y0, win.X1-win.X0, win.Y1-win.Y0
	// A window short of the whole resolution gets one buffer per level,
	// each covering only what the level above reads (dwtWindows); else
	// one plane in Mallat layout, transformed in place.
	wins := dwtLevels(tc, top, win, p.rev)
	if wins != nil {
		for l, e := range wins {
			p.level(top-l, (e.X1-e.X0)*(e.Y1-e.Y0))
		}
	} else {
		// slot 0 doubles as largest level.
		p.level(0, p.w*p.h)
	}
	levelW := func(l int) int {
		if wins == nil {
			return p.w
		}
		return wins[l].X1 - wins[l].X0
	}

	// Every code-block in a subband shares one step size (E.1.1.1);
	// indexed by res*4+subband.
	type step struct {
		delta float64
		mb    int
		have  bool
	}
	steps := make([]step, tc.NumResolutions*4)
	jobs := make([]cbJob, 0, len(groups))
	var work int64
	for _, g := range groups {
		cb := g.cb
		if g.res > top || !cbSegmentsValid(cb.Segments, len(tileData)) || !g.segsValid(len(tileData)) {
			continue
		}
		j := cbJob{g: g, sx1: cb.X1 - cb.X0, sy1: cb.Y1 - cb.Y0}
		if wins == nil {
			x, y, ok := cbOrigin(tg, comp, g.res, cb)
			if !ok {
				continue
			}
			j.lv, j.x, j.y = top, x, y
		} else {
			bw, ox, oy := tier2.BandWindow(wins, g.res, cb.Subband)
			x0, y0 := max(cb.X0, bw.X0), max(cb.Y0, bw.Y0)
			x1, y1 := min(cb.X1, bw.X1), min(cb.Y1, bw.Y1)
			if x0 >= x1 || y0 >= y1 {
				continue
			}
			j.lv, j.x, j.y = g.res, ox+x0-bw.X0, oy+y0-bw.Y0
			j.sx0, j.sy0, j.sx1, j.sy1 = x0-cb.X0, y0-cb.Y0, x1-cb.X0, y1-cb.Y0
		}
		s := &steps[g.res*4+cb.Subband]
		if !s.have {
			delta, mb, err := dsp.StepSize(q, g.res, dsp.SubbandType(cb.Subband), precision, p.rev)
			if err != nil {
				return err
			}
			if p.round == dsp.JJ2000 {
				// JJ2000 keeps T.800 E-3's gain: 2 for HL, LH, 4 for HH.
				delta *= float64(int(1) << ((cb.Subband + 1) / 2))
			}
			*s = step{delta, mb, true}
		}
		if err := tier1.ValidateCodeBlockDims(cb.X1-cb.X0, cb.Y1-cb.Y0, cbStyle); err != nil {
			return err
		}
		j.delta, j.mb = s.delta, s.mb
		jobs = append(jobs, j)
		for _, sg := range g.segs {
			work += int64(sg.passes) * int64(cb.X1-cb.X0) * int64(cb.Y1-cb.Y0)
		}
	}
	if p.work != nil && p.work.Add(-work) < 0 {
		return fmt.Errorf("jp2k: tile %d: tier-1 work exceeds the budget for the input size", tileIdx)
	}
	// A level with no HL/LH/HH block has its whole high band zeroed by
	// p.level, so its DWT lifting can skip straight to no-ops.
	high := make([]bool, top+1)
	for _, j := range jobs {
		high[j.g.res] = true
	}

	nw := parallel.Workers(len(jobs), p.nw)
	for len(p.t1) < nw {
		p.t1 = append(p.t1, new(t1Worker))
	}
	err := parallel.For(nw, len(jobs), func(w, i int) error {
		if parallel.FaultHook != nil {
			parallel.FaultHook("tier1")
		}
		if parallel.IsDone(p.done) {
			return parallel.ErrStopped
		}
		s := p.t1[w]
		j := jobs[i]
		cb := j.g.cb
		cw, ch := cb.X1-cb.X0, cb.Y1-cb.Y0
		raw := s.raw[:cw*ch]
		var f []float32
		if !p.rev {
			f = s.f[:cw*ch]
		}
		s.segs, s.data = j.g.concatSegmentsInto(tileData, s.segs, s.data[:0])
		if err := s.t.Finish(raw, nil, f, j.delta, s.segs, cw, ch, cb.Subband, cb.ZeroBitPlanes, j.mb, cbStyle); err != nil {
			return err
		}
		stride, n := levelW(j.lv), j.sx1-j.sx0
		for dy := j.sy0; dy < j.sy1; dy++ {
			o := (j.y+dy-j.sy0)*stride + j.x
			a := dy*cw + j.sx0
			if p.rev {
				copy(p.lvI[top-j.lv][o:o+n], raw[a:a+n])
			} else {
				copy(p.lvF[top-j.lv][o:o+n], f[a:a+n])
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	if wins == nil {
		p.stride, p.off = p.w, 0
		if p.rev {
			p.i = p.lvI[0]
			dsp.InverseDWT53(p.i, p.stride, resRects(tg, comp)[:top+1], high, &p.dwtI, p.nw, p.done)
		} else {
			p.f = p.lvF[0]
			p.packed = dsp.InverseDWT97(p.f, p.stride, resRects(tg, comp)[:top+1], high, &p.dwtF, p.nw, p.done, p.emit, p.round)
		}
	} else {
		if p.rev {
			p.i = partialDWT(byLevel(p.lvI, top), wins, high, &p.dwtI, p.nw, p.done, dsp.InverseDWT53)
		} else {
			idwt := inverseDWT97NoEmit
			switch p.round {
			case dsp.Separate:
				idwt = inverseDWT97SepNoEmit
			case dsp.JJ2000:
				idwt = inverseDWT97JJNoEmit
			}
			p.f = partialDWT(byLevel(p.lvF, top), wins, high, &p.dwtF, p.nw, p.done, idwt)
		}
		e := wins[top]
		p.stride = e.X1 - e.X0
		p.off = (win.Y0-e.Y0)*p.stride + win.X0 - e.X0
	}
	if parallel.IsDone(p.done) {
		return parallel.ErrStopped
	}
	return nil
}

// parallelRows calls fn(w, y0, y1) over [0,n) in RowChunk-sized pieces
// on at most nw workers, so w < nw.
func parallelRows(nw, n int, fn func(w, y0, y1 int)) {
	tasks := codestream.CeilDiv(n, dsp.RowChunk)
	_ = parallel.For(min(nw, tasks), tasks, func(w, t int) error {
		fn(w, t*dsp.RowChunk, min(n, (t+1)*dsp.RowChunk))
		return nil
	})
}

// -- output assembly (T.800 Annex G/I plus this package's chosen mapping) --

// outputKind is the image.Image concrete type Decode produces for a given
// component count and precision, per the table in Decode's doc comment.
type outputKind int

const (
	outGray outputKind = iota
	outGray16
	outRGBA
	outRGBA64
	outNRGBA
	outNRGBA64
)

func (k outputKind) bytesPerPixel() int {
	switch k {
	case outGray:
		return 1
	case outGray16:
		return 2
	case outRGBA, outNRGBA:
		return 4
	default:
		return 8
	}
}

func (k outputKind) colorModel() color.Model {
	switch k {
	case outGray:
		return color.GrayModel
	case outGray16:
		return color.Gray16Model
	case outRGBA:
		return color.RGBAModel
	case outRGBA64:
		return color.RGBA64Model
	case outNRGBA:
		return color.NRGBAModel
	case outNRGBA64:
		return color.NRGBA64Model
	default:
		panic("jp2k: unreachable output kind")
	}
}

// classifyOutput picks Decode's output image type from SIZ alone (so
// DecodeConfig, which reads no further than SIZ, can compute it too).
// Component subsampling (XRsiz/YRsiz) is allowed and does not affect the
// output type: decodeImage upsamples any subsampled component to the
// reference grid by sample replication before assembly (see
// upsampleComponent). Cross-component precision/signedness mismatches are
// still rejected (per-component precision scaling is
// unimplemented; see decodeImage's doc). Signed components are allowed:
// decodeImage biases their true (negative-capable) values by
// +2^(precision-1) at canvas-assembly time, mapping them into the same
// unsigned range as an unsigned component of the same precision, so the
// rest of this function's output-type table applies unchanged. 2-component
// images are treated as gray+alpha (component 0 replicated into R/G/B,
// component 1 as alpha), matching opj_decompress's convert.c convention.
func classifyOutput(siz *codestream.SIZ) (outputKind, error) {
	n := len(siz.Components)
	if n < 1 || n > 4 {
		return 0, fmt.Errorf("jp2k: %d components not supported (only 1-4)", n)
	}
	c0 := siz.Components[0]
	for i, c := range siz.Components {
		if c.Precision != c0.Precision || c.Signed != c0.Signed {
			return 0, fmt.Errorf("jp2k: component %d precision/signedness does not match component 0 (mixed per-component precision not supported)", i)
		}
	}
	if c0.Precision > 16 {
		return 0, fmt.Errorf("jp2k: component precision %d exceeds 16", c0.Precision)
	}
	eightBit := c0.Precision == 8
	switch n {
	case 1:
		if eightBit {
			return outGray, nil
		}
		return outGray16, nil
	case 2:
		if eightBit {
			return outNRGBA, nil
		}
		return outNRGBA64, nil
	case 3:
		if eightBit {
			return outRGBA, nil
		}
		return outRGBA64, nil
	default: // 4
		if eightBit {
			return outNRGBA, nil
		}
		return outNRGBA64, nil
	}
}

// outImage is Decode's result being filled in place: finished samples go
// straight into its Pix, with no intermediate per-component canvas.
type outImage struct {
	img       image.Image
	pix       []byte
	stride    int
	bpp       int  // bytes per pixel
	wide      bool // 16-bit big-endian channels
	grayAlpha bool // 2 components: channel 0 is replicated into R, G and B
	opaque    bool // 3 components: alpha is always the maximum
	precision uint8
}

// newOutImage makes an image with bounds r; putRow and friends take
// coordinates relative to r.Min.
func newOutImage(kind outputKind, r image.Rectangle, numComps int, precision uint8) *outImage {
	o := &outImage{grayAlpha: numComps == 2, opaque: numComps == 3, precision: precision}
	switch kind {
	case outGray:
		m := image.NewGray(r)
		o.img, o.pix, o.stride, o.bpp = m, m.Pix, m.Stride, 1
	case outGray16:
		m := image.NewGray16(r)
		o.img, o.pix, o.stride, o.bpp, o.wide = m, m.Pix, m.Stride, 2, true
	case outRGBA:
		m := image.NewRGBA(r)
		o.img, o.pix, o.stride, o.bpp = m, m.Pix, m.Stride, 4
	case outNRGBA:
		m := image.NewNRGBA(r)
		o.img, o.pix, o.stride, o.bpp = m, m.Pix, m.Stride, 4
	case outRGBA64:
		m := image.NewRGBA64(r)
		o.img, o.pix, o.stride, o.bpp, o.wide = m, m.Pix, m.Stride, 8, true
	case outNRGBA64:
		// Non-premultiplied: our samples are independent channel values,
		// and image.RGBA64's RGBA() assumes alpha-premultiplied storage
		// (it would misreport translucent 4-component pixels otherwise).
		m := image.NewNRGBA64(r)
		o.img, o.pix, o.stride, o.bpp, o.wide = m, m.Pix, m.Stride, 8, true
	default:
		panic("jp2k: unreachable output kind")
	}
	return o
}

// putRow stores samples v (each in [0, 2^precision)) as output channel ch
// of the pixels starting at (x0, y). Odd precisions (e.g. 12-bit) are
// expanded into the full 16 bits of the wide types (see Expand16).
func (o *outImage) putRow(ch, x0, y int, v []int32) {
	off := y*o.stride + x0*o.bpp
	o.putChan(off, ch, v)
	if ch != 0 {
		return
	}
	if o.grayAlpha {
		o.putChan(off, 1, v)
		o.putChan(off, 2, v)
	}
	if o.opaque {
		o.fillAlpha(off, len(v))
	}
}

// putRowF is SamplesF then putRow, for 9..16-bit output at o.precision,
// without the intermediate int32 row.
func (o *outImage) putRowF(ch, x0, y int, v []float32, precision int32) {
	off := y*o.stride + x0*o.bpp
	dsp.PackF16(o.pix[off+2*ch:], o.bpp, v, precision)
	if ch != 0 {
		return
	}
	if o.grayAlpha {
		dsp.PackF16(o.pix[off+2:], o.bpp, v, precision)
		dsp.PackF16(o.pix[off+4:], o.bpp, v, precision)
	}
	if o.opaque {
		o.fillAlpha(off, len(v))
	}
}

func (o *outImage) fillAlpha(off, n int) {
	// Per-width loops: the generic one cost ~1 ms.
	if o.bpp == 4 {
		p := o.pix[off : off+4*n]
		for i := 3; i < len(p); i += 4 {
			p[i] = 0xff
		}
		return
	}
	p := o.pix[off : off+8*n]
	for i := 7; i < len(p); i += 8 {
		p[i-1], p[i] = 0xff, 0xff
	}
}

func (o *outImage) putChan(off, ch int, v []int32) {
	if !o.wide {
		p := o.pix[off+ch:]
		for i, s := range v {
			p[i*o.bpp] = uint8(s)
		}
		return
	}
	p := o.pix[off+2*ch:]
	for i, s := range v {
		dsp.Put16BE(p[i*o.bpp:], dsp.Expand16(uint32(s), o.precision))
	}
}

// tileLanes is how many of a reduce-level decode's n tiles decodeImage
// decodes at once. More lanes let one tile's serial phases overlap
// another's work, but each holds a tile's working memory, so the lanes
// past the first stay within laneBudget and a quarter of the outBytes
// output. PPM needs tiles in order.
func tileLanes(cs *codestream.Codestream, n, reduce, outBytes int) int {
	if len(cs.PPM) > 0 {
		return 1
	}
	siz := cs.SIZ
	planes := int64(1)
	if len(siz.Components) >= 3 {
		planes = 3 // with MCT
	}
	// About 8 bytes of heap per sample of plane, measured.
	s := uint(reduce)
	tileBytes := max(1, int64(codestream.CeilDivPow2(int(siz.XTsiz), s))*int64(codestream.CeilDivPow2(int(siz.YTsiz), s))*planes*8)
	// Small outputs still get laneFloor, a negligible amount.
	budget := min(laneBudget, max(laneFloor, int64(outBytes)/4))
	return int(max(1, min(int64(n), int64(runtime.GOMAXPROCS(0)), 1+budget/tileBytes)))
}

// laneBudget bounds the working memory of the lanes past the first.
const laneBudget, laneFloor = 64 << 20, 1 << 20

// maxCanvasPixels bounds the w*h this package will allocate for Decode's
// output canvas. SIZ's Xsiz/Ysiz are attacker-controlled 32-bit values with
// no relation to the codestream's actual byte length, so without a bound a
// few hundred bytes can request terabytes (make([]int32, w*h) per
// component) before any tile is parsed or any existing sanity check (tile
// grid <= 65535, code-blocks per tile <= 16777216) runs. ~268M pixels caps
// one component's canvas at ~1GB.
const maxCanvasPixels = 1 << 28

// tileWindows returns, for each component of tile tileIdx with geometry
// tg and coding styles, the highest resolution a decode of the component
// windows compWin at reduce reads and its window of that resolution, and
// what tier-2 must keep of the tile for it: nil when full.
func tileWindows(tg *tier2.TileGeom, tileIdx uint16, styles []codestream.CodingStyle, compWin []codestream.SubbandRect, reduce int, full bool) (tops []int, wins []codestream.SubbandRect, want *tier2.TileWant, err error) {
	numComps := len(tg.Comps)
	tops = make([]int, numComps)
	wins = make([]codestream.SubbandRect, numComps)
	for c := 0; c < numComps; c++ {
		tops[c] = tg.Comps[c].NumResolutions - 1 - reduce
		if tops[c] < 0 {
			return nil, nil, nil, fmt.Errorf("jp2k: tile %d component %d has fewer than %d decomposition levels", tileIdx, c, reduce)
		}
		wins[c] = compWin[c].Intersect(tg.Comps[c].Resolutions[tops[c]].Rect)
		if wins[c].Empty() {
			wins[c] = codestream.SubbandRect{}
		}
	}
	if full {
		return tops, wins, nil, nil
	}
	want = &tier2.TileWant{Top: tops, Wins: make([][]codestream.SubbandRect, numComps)}
	for c := range want.Wins {
		want.Wins[c] = dwtLevels(&tg.Comps[c], tops[c], wins[c], styles[c].Transformation == 1)
	}
	return tops, wins, want, nil
}

// decodeImage runs the full pipeline (tier2, tier1, dequant,
// inverse DWT, MCT, DC shift/clamp, component upsampling) for every tile
// reg touches and assembles reg. Each component's coding style (code-block
// style, DWT filter) is resolved per tile via EffectiveCodingStyle, so a
// tile-part COD or a COC override on any component is honored rather than
// silently ignored. With hd not nil the tiles' packet headers come from
// hd's indexes, built as needed, and data is unused. With comps not nil
// it decodes only the components componentSet gives and returns comps
// as planes of raw samples, on each component's own grid, instead of
// an image.
func decodeImage(ctx context.Context, t *ticket, cs *codestream.Codestream, data []byte, cdef []codestream.CdefEntry, reg region, hd *Handle, comps []int, opts Options) (image.Image, []*image.Gray16, error) {
	if parallel.FaultHook != nil {
		parallel.FaultHook("caller")
	}
	if err := coordsFit(cs.SIZ, reg.reduce); err != nil {
		return nil, nil, err
	}
	var (
		kind     outputKind
		cdefPerm []int
		dec      []bool // components decoded, nil for all
		err      error
	)
	if comps == nil {
		if kind, err = classifyOutput(cs.SIZ); err != nil {
			return nil, nil, err
		}
		// A cdef box reassigns which component plays which output channel (e.g.
		// R/G/B swapped, or alpha on a component other than the last); resolve
		// that into a permutation now so a bad cdef is a clean error before any
		// tile is decoded, not a silent mis-render after.
		if cdefPerm, err = codestream.CdefPermutation(cdef, len(cs.SIZ.Components)); err != nil {
			return nil, nil, err
		}
	} else if dec, err = componentSet(cs, comps); err != nil {
		return nil, nil, err
	}
	// RGN (region of interest) is parsed but has no tier-1/dequant consumer
	// in this package; decoding past it would silently shift the affected
	// code-blocks' magnitudes instead of erroring, which is worse than
	// refusing the file outright.
	if len(cs.RGN) > 0 {
		return nil, nil, fmt.Errorf("jp2k: region-of-interest (RGN) not supported")
	}
	siz := cs.SIZ
	full := reg == fullRegion(siz) && !slices.Contains(dec, false)
	numComps := len(siz.Components)
	w64 := int64(reg.x1) - int64(reg.x0)
	h64 := int64(reg.y1) - int64(reg.y0)
	// w64/h64 > 0 is already guaranteed by ParseSIZSegment; divide rather
	// than multiply to check the bound without risking int64 overflow
	// (Xsiz/Ysiz are uint32, so w64*h64 can exceed int64's range).
	if w64 > maxCanvasPixels/h64 {
		return nil, nil, fmt.Errorf("jp2k: declared image %dx%d exceeds the %d-pixel decode limit", w64, h64, maxCanvasPixels)
	}
	// Up to 8 bytes per pixel overflows int on 386/arm.
	if w64*h64 > math.MaxInt/8 {
		return nil, nil, fmt.Errorf("jp2k: declared image %dx%d is too large for this platform", w64, h64)
	}
	w, h := int(w64), int(h64)
	precision := siz.Components[0].Precision
	// A subsampled component (XRsiz/YRsiz != 1) is assembled on its own
	// canvas at its own size (ceildiv(Xsiz,XRsiz) etc, T.800 Annex B.2, or
	// the part of that reg reads) and upsampled after the last tile (see
	// upsampleComponent); every other component is written straight into
	// o. compOrigin is the canvas origin in the component's own sample
	// units, matching TileGeom's Tcx0/Tcy0.
	compW := make([]int, numComps)
	compH := make([]int, numComps)
	compOrigin := make([][2]int, numComps)
	compWin := make([]codestream.SubbandRect, numComps)
	for c := range compWin {
		cw := compWindow(siz, c, reg)
		switch {
		case dec == nil:
		case !dec[c]:
			cw = codestream.SubbandRect{}
		default:
			// Planes map reg by ceil, as T.800 B.3 does.
			cw.X0 = min(max(cw.X0, codestream.CeilDiv(reg.x0, int(siz.Components[c].XRsiz))), cw.X1)
			cw.Y0 = min(max(cw.Y0, codestream.CeilDiv(reg.y0, int(siz.Components[c].YRsiz))), cw.Y1)
		}
		compWin[c] = cw
		compW[c] = cw.X1 - cw.X0
		compH[c] = cw.Y1 - cw.Y0
		compOrigin[c] = [2]int{cw.X0, cw.Y0}
	}
	for _, tp := range cs.Parts() {
		// A handle holds every tile-part; a region decode, those it reads.
		if len(tp.RGN) > 0 && (hd == nil || full || tileTouches(siz, tp.SOT.Isot, compWin, reg.reduce)) {
			return nil, nil, fmt.Errorf("jp2k: region-of-interest (RGN) not supported")
		}
	}
	tileOrder := decodeOrder(cs, full, compWin, reg.reduce)
	var plans []*tilePlan
	var built int64 // indexes this decode built, live but not yet reserved
	dataLen := len(data)
	if hd != nil {
		if !full {
			// Indexed, PPM tiles need no visit either.
			tileOrder = slices.DeleteFunc(tileOrder, func(t uint16) bool { return !tileTouches(siz, t, compWin, reg.reduce) })
		}
		// t holds nothing yet: a decode waiting on another's build of
		// an index must not.
		var release func()
		if plans, dataLen, built, release, err = hd.plan(ctx, tileOrder, compWin, reg, full); err != nil {
			return nil, nil, err
		}
		defer release()
	}
	outBytes := w * h * kind.bytesPerPixel()
	if dec != nil {
		n := int64(0)
		for _, c := range comps {
			n += 2 * int64(compWin[c].X1-compWin[c].X0) * int64(compWin[c].Y1-compWin[c].Y0)
		}
		// No more than the largest image output.
		if n > min(8*maxCanvasPixels, math.MaxInt) {
			return nil, nil, fmt.Errorf("jp2k: %d bytes of planes exceeds the decode limit", n)
		}
		outBytes = int(n)
	}
	procs := runtime.GOMAXPROCS(0)
	if opts.Workers > 0 {
		procs = min(procs, opts.Workers)
	}
	numLanes := min(tileLanes(cs, len(tileOrder), reg.reduce, outBytes), procs)
	// t holds the input, its parse and the garbage of both. The parse's
	// garbage goes back, counted until a GC; the estimate counts the
	// input and parse, and the read's garbage is added.
	t.drop(cs.ParseCost)
	estimate := func() (need, floor int64) {
		est, fl := estimateDecode(cs, len(data), reg, outBytes, dec != nil, compWin, tileOrder, numLanes, procs, full, plans)
		extra := max(0, t.held-int64(len(data))-cs.ParseCost) + built
		return est + extra, fl + extra
	}
	need, floor := estimate()
	if collectFor(need) {
		need, floor = estimate()
	}
	if hd != nil {
		// A hung read holds only its bytes.
		n := built
		for _, p := range plans {
			n += p.readCost()
		}
		if err := t.grow(n); err != nil {
			return nil, nil, err
		}
		for _, p := range plans {
			if err := hd.read(ctx, p); err != nil {
				return nil, nil, err
			}
		}
	}
	if err := t.run(need, floor); err != nil {
		return nil, nil, err
	}

	var (
		o         *outImage
		outPlanes []*image.Gray16 // by component, nil for those not asked for
	)
	if dec == nil {
		// Bounds in reduced coordinates relative to the (reduced) image origin.
		bx := reg.x0 - codestream.CeilDivPow2(int(siz.XOsiz), uint(reg.reduce))
		by := reg.y0 - codestream.CeilDivPow2(int(siz.YOsiz), uint(reg.reduce))
		o = newOutImage(kind, image.Rect(bx, by, bx+w, by+h), numComps, precision)
	} else {
		outPlanes = make([]*image.Gray16, numComps)
		s := uint(reg.reduce)
		for _, c := range comps {
			cw := compWin[c]
			ox := codestream.CeilDivPow2(codestream.CeilDiv(int(siz.XOsiz), int(siz.Components[c].XRsiz)), s)
			oy := codestream.CeilDivPow2(codestream.CeilDiv(int(siz.YOsiz), int(siz.Components[c].YRsiz)), s)
			outPlanes[c] = image.NewGray16(image.Rect(cw.X0-ox, cw.Y0-oy, cw.X1-ox, cw.Y1-oy))
		}
	}
	// outCh maps a component to its output channel (identity unless a
	// cdef box reorders them).
	outCh := make([]int, numComps)
	for c := range outCh {
		outCh[c] = c
	}
	for pos, src := range cdefPerm {
		outCh[src] = pos
	}
	if numComps == 2 {
		// Gray+alpha: channel 1 is alpha, stored in A, not G.
		for c := range outCh {
			if outCh[c] == 1 {
				outCh[c] = 3
			}
		}
	}
	canvas := make([][]int32, numComps)
	for c := range canvas {
		if dec == nil && (siz.Components[c].XRsiz != 1 || siz.Components[c].YRsiz != 1) {
			canvas[c] = make([]int32, compW[c]*compH[c])
		}
	}

	// emit writes tile component c, finished in p, to its destination,
	// using rowBufs as per-worker scratch.
	emit := func(c int, p *plane, rowBufs *[][]int32) {
		if o == nil && outPlanes[c] == nil || p.packed {
			return
		}
		tw, th := p.w, p.h
		ox, oy := p.x0-compOrigin[c][0], p.y0-compOrigin[c][1]
		prec := int32(siz.Components[c].Precision)
		signed := siz.Components[c].Signed
		// GOMAXPROCS can change mid-decode; size scratch from this call's nw.
		nw := parallel.Workers(th, p.nw)
		for len(*rowBufs) < nw {
			*rowBufs = append(*rowBufs, nil)
		}
		bufs := *rowBufs
		parallelRows(nw, th, func(wk, y0, y1 int) {
			if parallel.FaultHook != nil {
				parallel.FaultHook("output")
			}
			if cap(bufs[wk]) < tw {
				bufs[wk] = make([]int32, tw)
			}
			row := bufs[wk][:tw]
			jj := p.round == dsp.JJ2000
			fused := o != nil && !p.rev && !jj && canvas[c] == nil && o.wide && prec >= 9 && uint8(prec) == o.precision
			plane16 := o == nil && !p.rev && !jj && prec == 16 // Expand16 is identity at 16
			planeP := o == nil && !p.rev && !jj && canvas[c] == nil && prec >= 1 && prec < 16
			planeJJ := o == nil && !p.rev && jj && canvas[c] == nil
			for y := y0; y < y1; y++ {
				a := p.off + y*p.stride
				if fused {
					o.putRowF(outCh[c], ox, oy+y, p.f[a:a+tw], prec)
					continue
				}
				if plane16 {
					dsp.PackF16(outPlanes[c].Pix[(oy+y)*outPlanes[c].Stride+2*ox:], 2, p.f[a:a+tw], prec)
					continue
				}
				if planeP {
					dsp.PackP16(outPlanes[c].Pix[(oy+y)*outPlanes[c].Stride+2*ox:], p.f[a:a+tw], prec)
					continue
				}
				if planeJJ {
					dsp.PackJJ16(outPlanes[c].Pix[(oy+y)*outPlanes[c].Stride+2*ox:], p.f[a:a+tw], prec)
					continue
				}
				switch {
				case p.rev:
					dsp.SamplesR(row, p.i[a:a+tw], prec, signed)
				case jj:
					dsp.SamplesJJ(row, p.f[a:a+tw], prec)
				default:
					dsp.SamplesF(row, p.f[a:a+tw], prec)
				}
				switch {
				case canvas[c] != nil:
					copy(canvas[c][(oy+y)*compW[c]+ox:], row)
				case o == nil:
					pix := outPlanes[c].Pix[(oy+y)*outPlanes[c].Stride+2*ox:]
					for i, v := range row {
						dsp.Put16BE(pix[2*i:], uint16(v))
					}
				default:
					o.putRow(outCh[c], ox, oy+y, row)
				}
			}
		})
	}

	ppmPos := new(tier2.Cursor) // shared PPM cursor and budgets, threaded across every tile in order (T.800 A.7.4)
	ppmPos.Done = ctx.Done()
	// A lane decodes one tile at a time with its own scratch; lanes run
	// concurrently and split the workers, since per-tile serial phases
	// otherwise leave cores idle. One plane per component the MCT mixes,
	// else one reused for each component in turn: a lane's working memory
	// is one tile component's samples (three with MCT).
	type lane struct {
		planes  [3]plane
		rowBufs [][]int32
		tileBuf []byte
		// emit16, by component, packs the last zero-high 9/7 level
		// straight into outPlanes[c]; built once here, not per tile,
		// so the ordinary path allocates no closure per tile.
		emit16 []func(y int, row []float32)
	}
	lanes := make([]lane, numLanes)
	work := new(atomic.Int64)
	work.Store(t1WorkPerByte*int64(dataLen) + t1WorkSlack)
	for i := range lanes {
		nw := procs / len(lanes)
		if i < procs%len(lanes) {
			nw++
		}
		for j := range lanes[i].planes {
			lanes[i].planes[j].nw = nw
			lanes[i].planes[j].done = ctx.Done()
			lanes[i].planes[j].work = work
			lanes[i].planes[j].round = opts.Round
		}
		lanes[i].emit16 = make([]func(y int, row []float32), numComps)
		p := &lanes[i].planes[0]
		for c := 0; c < numComps && outPlanes != nil; c++ {
			if outPlanes[c] == nil || siz.Components[c].Precision != 16 {
				continue
			}
			op, ox0, oy0 := outPlanes[c], compOrigin[c][0], compOrigin[c][1]
			lanes[i].emit16[c] = func(y int, row []float32) {
				if parallel.FaultHook != nil {
					parallel.FaultHook("output")
				}
				ox, oy := p.x0-ox0, p.y0-oy0
				dsp.PackF16(op.Pix[(oy+y)*op.Stride+2*ox:], 2, row, 16)
			}
		}
	}
	// tier-2 shares ppmPos's cursor and budgets; the PPM stream also needs
	// tile order, which one lane gives.
	var tier2Mu sync.Mutex
	err = parallel.For(len(lanes), len(tileOrder), func(li, i int) error {
		// On a busy P a GC mark ends only when the decode yields or is
		// preempted, up to 10 ms later, and the heap grows unpaced till
		// then: many small tiles can triple it.
		runtime.Gosched()
		if ctx.Err() != nil {
			return parallel.ErrStopped
		}
		tileIdx := tileOrder[i]
		ln := &lanes[li]
		planes := &ln.planes
		if !full && !tileTouches(siz, tileIdx, compWin, reg.reduce) {
			// The shared PPM cursor still has to pass this tile.
			_, _, _, err := tier2.DecodeCursor(cs, data, tileIdx, ppmPos)
			return err
		}
		tileCOD, tileCOC, _ := codestream.TileOverrides(cs, tileIdx)
		styles := make([]codestream.CodingStyle, numComps)
		for c := 0; c < numComps; c++ {
			styles[c] = codestream.EffectiveCodingStyle(cs, tileCOD, tileCOC, c)
		}
		// top is the highest resolution decoded, win the part of it reg reads.
		var (
			tops     []int
			wins     []codestream.SubbandRect
			tg       *tier2.TileGeom
			tileData []byte
			groupsOf func(c int) []*cbGroup
			err      error
		)
		if hd != nil {
			pl := plans[i]
			tg, tops, wins = pl.ix.tg, pl.tops, pl.wins
			var groups [][]*cbGroup
			tileData, groups = pl.load()
			groupsOf = func(c int) []*cbGroup { return groups[c] }
		} else {
			wantOf := func(tg *tier2.TileGeom) (want *tier2.TileWant, err error) {
				tops, wins, want, err = tileWindows(tg, tileIdx, styles, compWin, reg.reduce, full)
				return want, err
			}
			var pkts []tier2.PacketRef
			pkts, tg, tileData, err = func() ([]tier2.PacketRef, *tier2.TileGeom, []byte, error) {
				tier2Mu.Lock()
				defer tier2Mu.Unlock() // a recovered panic must not strand other lanes
				if ctx.Err() != nil {
					return nil, nil, nil, parallel.ErrStopped
				}
				if parallel.FaultHook != nil {
					parallel.FaultHook("tier2")
				}
				return tier2.DecodeTile(cs, data, tileIdx, ppmPos, &ln.tileBuf, wantOf)
			}()
			if errors.Is(err, tier2.ErrStopped) {
				return parallel.ErrStopped
			}
			if err != nil {
				return err
			}
			groupsOf = func(c int) []*cbGroup { return groupCodeBlocks(pkts, c) }
		}

		mct := codestream.EffectiveMCT(cs, tileCOD)
		if mct == 2 {
			return fmt.Errorf("jp2k: tile %d: array-based multiple component transform (MCT=2) not supported", tileIdx)
		}
		decodeInto := func(c int, p *plane) error {
			return decodeComponent(cs, tg, tileData, groupsOf(c), tileIdx, c, styles[c], p, tops[c], wins[c])
		}

		first := 0
		// OpenJPEG applies MCT only when it actually has 3 components to mix
		// (tcd.c), silently skipping it rather than erroring otherwise, and
		// picks RCT vs ICT from component 0's filter alone (tcd.c). A
		// component 1/2 whose effective filter disagrees with component 0
		// can't be mixed by either transform, so that's an error rather
		// than a silent guess.
		if mct == 1 && numComps >= 3 {
			// InverseRCT/InverseICT mix components 0-2 sample-for-sample, so
			// they need equal-size windows; differing XRsiz/YRsiz among
			// those components (legal per SIZ, just not something a real
			// MCT-enabled encoder would produce) would otherwise index past
			// the shortest component's slice.
			n0 := wins[0].X1 - wins[0].X0
			m0 := wins[0].Y1 - wins[0].Y0
			for c := 1; c < 3; c++ {
				if wins[c].X1-wins[c].X0 != n0 || wins[c].Y1-wins[c].Y0 != m0 {
					return fmt.Errorf("jp2k: tile %d has mismatched component dimensions across components 0-2 (differing XRsiz/YRsiz) with the multiple component transform enabled", tileIdx)
				}
			}
			reversible := styles[0].Transformation == 1
			if (styles[1].Transformation == 1) != reversible || (styles[2].Transformation == 1) != reversible {
				return fmt.Errorf("jp2k: tile %d has mismatched reversible/irreversible filters across components 0-2 with the multiple component transform enabled", tileIdx)
			}
			for c := 0; c < 3; c++ {
				planes[c].emit = nil // MCT mixes samples; the pack fast path doesn't apply
				if err := decodeInto(c, &planes[c]); err != nil {
					return err
				}
			}
			p0, p1, p2 := &planes[0], &planes[1], &planes[2]
			parallelRows(parallel.Workers(m0, planes[0].nw), m0, func(_, y0, y1 int) {
				for y := y0; y < y1; y++ {
					a0, a1, a2 := p0.off+y*p0.stride, p1.off+y*p1.stride, p2.off+y*p2.stride
					if reversible {
						dsp.InverseRCT(p0.i[a0:a0+n0], p1.i[a1:a1+n0], p2.i[a2:a2+n0])
					} else {
						dsp.InverseICT(p0.f[a0:a0+n0], p1.f[a1:a1+n0], p2.f[a2:a2+n0], p0.round)
					}
				}
			})
			for c := 0; c < 3; c++ {
				emit(c, &planes[c], &ln.rowBufs)
			}
			first = 3
		}
		for c := first; c < numComps; c++ {
			planes[0].emit = ln.emit16[c]
			if err := decodeInto(c, &planes[0]); err != nil {
				return err
			}
			emit(c, &planes[0], &ln.rowBufs)
		}
		return nil
	})
	if errors.Is(err, parallel.ErrStopped) {
		return nil, nil, fmt.Errorf("jp2k: decode stopped: %w", ctx.Err())
	}
	if err != nil {
		return nil, nil, err
	}
	if dec != nil {
		out := make([]*image.Gray16, len(comps))
		for i, c := range comps {
			out[i] = outPlanes[c]
		}
		return nil, out, nil
	}

	for c := 0; c < numComps; c++ {
		if canvas[c] == nil {
			continue
		}
		xr := int(siz.Components[c].XRsiz)
		yr := int(siz.Components[c].YRsiz)
		up := upsampleComponent(canvas[c], compW[c], compH[c], compOrigin[c][0], compOrigin[c][1], xr, yr, w, h, reg.x0, reg.y0)
		canvas[c] = nil
		for y := 0; y < h; y++ {
			o.putRow(outCh[c], 0, y, up[y*w:(y+1)*w])
		}
	}
	return o.img, nil, nil
}

// componentSet checks comps, a request for those components as planes,
// and returns the components a decode of them runs: comps, and 0-2 too
// when a multiple component transform mixes those.
func componentSet(cs *codestream.Codestream, comps []int) ([]bool, error) {
	n := len(cs.SIZ.Components)
	dec := make([]bool, n)
	for _, c := range comps {
		if c < 0 || c >= n {
			return nil, fmt.Errorf("jp2k: component %d out of range (image has %d)", c, n)
		}
		if dec[c] {
			return nil, fmt.Errorf("jp2k: component %d requested twice", c)
		}
		dec[c] = true
	}
	if n >= 3 && (dec[0] || dec[1] || dec[2]) {
		// Any tile's COD may turn it on.
		mct := cs.COD.MultipleComponentTransform != 0
		for _, tp := range cs.Parts() {
			mct = mct || tp.COD != nil && tp.COD.MultipleComponentTransform != 0
		}
		if mct {
			dec[0], dec[1], dec[2] = true, true, true
		}
	}
	for c, ok := range dec {
		if p := cs.SIZ.Components[c].Precision; ok && p > 16 {
			return nil, fmt.Errorf("jp2k: component %d precision %d exceeds 16", c, p)
		}
	}
	return dec, nil
}

// decodeOrder is the tiles a decode of the component windows compWin
// visits, in order. The PPM stream is ordered by the codestream
// APPEARANCE of the tile-parts it serves (T.800 A.7.4), which need not
// match tile index order (a tile-part's Isot can be anything). Tiles go
// in the order their first tile-part appears in the codestream so the
// shared PPM cursor consumes the right stretch of the stream for each
// tile; output is written by tile geometry, so it is order-independent.
func decodeOrder(cs *codestream.Codestream, full bool, compWin []codestream.SubbandRect, reduce int) []uint16 {
	siz := cs.SIZ
	numTiles := int(siz.TilesWide) * int(siz.TilesHigh)
	tileOrder := make([]uint16, 0, numTiles)
	seenTile := make([]bool, numTiles)
	for _, tp := range cs.Parts() {
		ti := tp.SOT.Isot
		if int(ti) < numTiles && !seenTile[ti] {
			seenTile[ti] = true
			tileOrder = append(tileOrder, ti)
		}
	}
	for t := 0; t < numTiles; t++ {
		if !seenTile[t] {
			tileOrder = append(tileOrder, uint16(t))
		}
	}
	if !full && len(cs.PPM) == 0 {
		// Without PPM an untouched tile needs no work at all.
		n := 0
		for _, t := range tileOrder {
			if tileTouches(siz, t, compWin, reduce) {
				tileOrder[n] = t
				n++
			}
		}
		tileOrder = tileOrder[:n]
	}
	return tileOrder
}

// upsampleComponent replicates a subsampled component's samples onto the
// w x h reference grid (T.800 output convention: opj_decompress's -upsample
// writes the same reference-grid pixel for every source sample it maps to,
// per src/bin/jp2/opj_decompress.c upsample_image_components). Reference
// pixel (x,y), absolute image coordinate (x+XOsiz, y+YOsiz), maps to source
// sample floor((x+XOsiz)/xr) - compOX; a coordinate that floor-divides to
// before the component's first sample (possible when XOsiz % xr != 0) reads
// as 0, and one past the last sample clamps to it.
func upsampleComponent(src []int32, srcW, srcH, compOX, compOY, xr, yr, w, h, xOsiz, yOsiz int) []int32 {
	colSrc := make([]int, w)
	for x := 0; x < w; x++ {
		lx := (x+xOsiz)/xr - compOX
		if lx < 0 {
			lx = -1
		} else if lx >= srcW {
			lx = srcW - 1
		}
		colSrc[x] = lx
	}
	rowSrc := make([]int, h)
	for y := 0; y < h; y++ {
		ly := (y+yOsiz)/yr - compOY
		if ly < 0 {
			ly = -1
		} else if ly >= srcH {
			ly = srcH - 1
		}
		rowSrc[y] = ly
	}
	out := make([]int32, w*h)
	for y := 0; y < h; y++ {
		ly := rowSrc[y]
		di := y * w
		if ly < 0 {
			continue // leading rows before the component's first sample: zero
		}
		for x := 0; x < w; x++ {
			lx := colSrc[x]
			if lx < 0 {
				continue
			}
			out[di+x] = src[ly*srcW+lx]
		}
	}
	return out
}

// -- public API -------------------------------------------------------------

// Decode implements jp2k.Decode.
func Decode(r io.Reader) (image.Image, error) {
	return DecodeContext(context.Background(), r)
}

// DecodeContext implements jp2k.DecodeContext.
func DecodeContext(ctx context.Context, r io.Reader) (image.Image, error) {
	img, _, err := decodeWhole(ctx, r, nil, Options{})
	return img, err
}

// DecodeComponents implements jp2k.DecodeComponents.
func DecodeComponents(ctx context.Context, r io.Reader, comps []int) ([]*image.Gray16, error) {
	return DecodeComponentsRound(ctx, r, comps, Options{})
}

// Options is jp2k.Options as the decoder takes it.
type Options struct {
	Round   dsp.Rounding
	Workers int // goroutine cap, 0 or less for GOMAXPROCS
}

// DecodeComponentsRound is DecodeComponents with o.
func DecodeComponentsRound(ctx context.Context, r io.Reader, comps []int, o Options) ([]*image.Gray16, error) {
	if len(comps) == 0 {
		return nil, errNoComponents
	}
	_, planes, err := decodeWhole(ctx, r, comps, o)
	return planes, err
}

// errNoComponents is a component decode's error for an empty request.
var errNoComponents = errors.New("jp2k: no components requested")

// decodeWhole is decodeImage of the whole of r.
func decodeWhole(ctx context.Context, r io.Reader, comps []int, o Options) (img image.Image, planes []*image.Gray16, err error) {
	defer parallel.Recover(&img, &err)
	t, err := newTicket(ctx)
	if err != nil {
		return nil, nil, err
	}
	defer t.done()
	cs, stream, cdef, err := readCodestream(ctx, r, t.grow, t.growParse)
	if err != nil {
		return nil, nil, err
	}
	return decodeImage(ctx, t, cs, stream, cdef, fullRegion(cs.SIZ), nil, comps, o)
}

// readCodestream reads all of r and parses its codestream, unwrapping a
// JP2 box file first, reserving the heap for each through grow and
// growParse.
func readCodestream(ctx context.Context, r io.Reader, grow, growParse func(int64) error) (cs *codestream.Codestream, stream []byte, cdef []codestream.CdefEntry, err error) {
	data, err := readAll(ctx, r, grow, growParse)
	if err != nil {
		return nil, nil, nil, err
	}
	return parseFile(data, growParse)
}

// parseFile parses a whole JP2 file or raw codestream held in data,
// reserving the parse's heap through grow unless it is nil.
func parseFile(data []byte, grow func(int64) error) (cs *codestream.Codestream, stream []byte, cdef []codestream.CdefEntry, err error) {
	stream = data
	if bytes.HasPrefix(data, codestream.JP2Signature) {
		stream, cdef, err = codestream.JP2Codestream(data)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	cs, err = codestream.ParseBudgeted(stream, false, grow)
	if err != nil {
		return nil, nil, nil, err
	}
	return cs, stream, cdef, nil
}

// readAll is io.ReadAll, but sized up front when r can say how big it
// is: growing by doubling costs up to twice the file in garbage, which
// matters for a multi-hundred-MB image under a memory limit. grow, if not
// nil, reserves each buffer before it is made; growMem does instead for
// a reader in memory. It stops between Reads once ctx is done.
func readAll(ctx context.Context, r io.Reader, grow, growMem func(int64) error) ([]byte, error) {
	var n int64 = -1
	var raw func() bool
	switch s := r.(type) {
	case interface{ Stat() (fs.FileInfo, error) }:
		if fi, err := s.Stat(); err == nil && fi.Mode().IsRegular() {
			n = fi.Size()
			if sk, ok := r.(io.Seeker); ok {
				if off, err := sk.Seek(0, io.SeekCurrent); err == nil {
					n -= off
				}
			}
		}
	case interface{ Len() int }:
		if inMemory(r) {
			grow = growMem
		}
		n = int64(s.Len())
	case interface {
		io.ReaderAt
		Size() int64
	}:
		n = s.Size()
		off := int64(0)
		if sk, ok := r.(io.Seeker); ok {
			if o, err := sk.Seek(0, io.SeekCurrent); err == nil {
				off = o
			}
		}
		n -= off
		// A SectionReader to the end is often sized MaxInt64: only
		// trust a size whose last byte is there.
		if n > 0 {
			var last [1]byte
			if m, _ := s.ReadAt(last[:], off+n-1); m < 1 {
				n = -1
			}
			raw = func() bool {
				var head [8]byte
				m, _ := s.ReadAt(head[:], off)
				return m == len(head) && !bytes.Equal(head[:], codestream.JP2Signature)
			}
		}
	}
	if n < 0 || n >= math.MaxInt {
		return readChunks(ctx, r, grow)
	}
	if grow != nil {
		if err := checkSized(n, raw); err != nil {
			return nil, err
		}
	}
	// One spare byte so the final Read sees EOF without a regrow.
	if grow != nil {
		if err := grow(n + 1); err != nil {
			return nil, err
		}
	}
	b := make([]byte, 0, n+1)
	for {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("jp2k: reading: %w", err)
		}
		if len(b) == cap(b) {
			// r outgrew its size: double.
			if grow != nil {
				if err := grow(int64(2 * cap(b))); err != nil {
					return nil, err
				}
			}
			b = append(make([]byte, 0, 2*cap(b)), b...)
		}
		m, err := r.Read(b[len(b):cap(b)])
		b = b[:len(b)+m]
		if err == io.EOF {
			return b, nil
		}
		if err != nil {
			return b, err
		}
	}
}

// inMemory reports whether reads of r cannot stall.
func inMemory(r io.Reader) bool {
	switch r.(type) {
	case *bytes.Reader, *strings.Reader, *bytes.Buffer:
		return true
	}
	return false
}

// readChunks reads r, of unknown size, in chunks of up to 1 MB and then
// joins them: that allocates about twice the input, where doubling one
// buffer allocates up to four times it.
func readChunks(ctx context.Context, r io.Reader, grow func(int64) error) ([]byte, error) {
	var chunks [][]byte
	total, size := 0, 512
	for {
		if grow != nil {
			if err := grow(int64(size)); err != nil {
				return nil, err
			}
		}
		c := make([]byte, size)
		m := 0
		var err error
		for m < size && err == nil {
			if err = ctx.Err(); err != nil {
				return nil, fmt.Errorf("jp2k: reading: %w", err)
			}
			var k int
			k, err = r.Read(c[m:])
			m += k
		}
		chunks = append(chunks, c[:m])
		total += m
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		size = min(2*size, 1<<20)
	}
	if grow != nil {
		if err := grow(int64(total)); err != nil {
			return nil, err
		}
	}
	b := make([]byte, 0, total)
	for _, c := range chunks {
		b = append(b, c...)
	}
	return b, nil
}

// jp2cConfigWindow matches a raw SIZ window.
const jp2cConfigWindow = 65541

// readBoxHeader reads LBox(4) TBox(4) [XLBox(8)].
// hdr reused: no per-box heap growth.
// payload is -1 for to-EOF box.
func readBoxHeader(r io.Reader, hdr *[16]byte) (typ uint32, headerLen, payload int64, err error) {
	if _, err := io.ReadFull(r, hdr[:8]); err != nil {
		return 0, 0, 0, err
	}
	typ = codestream.Be32(hdr[4:8])
	length := int64(codestream.Be32(hdr[0:4]))
	headerLen = 8
	switch length {
	case 1:
		if _, err := io.ReadFull(r, hdr[8:16]); err != nil {
			return 0, 0, 0, err
		}
		length, headerLen = int64(binary.BigEndian.Uint64(hdr[8:16])), 16
	case 0:
		return typ, headerLen, -1, nil
	}
	if length < headerLen {
		return 0, 0, 0, errBoxLength
	}
	return typ, headerLen, length - headerLen, nil
}

var errBoxLength = errors.New("jp2k: box length shorter than its header")

// readErr: a short read becomes truncation.
func readErr(err error) error {
	if err == errBoxLength {
		return err
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return fmt.Errorf("jp2k: box runs past end of data")
	}
	return fmt.Errorf("jp2k: reading: %w", err)
}

// walkJP2HChildren streams jp2h children unbuffered.
func walkJP2HChildren(ctx context.Context, r io.Reader, skip func(int64) error, total int64, child *codestream.JP2HChild, hdr *[16]byte) error {
	for total > 0 {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("jp2k: reading: %w", err)
		}
		typ, headerLen, payload, err := readBoxHeader(r, hdr)
		if err != nil {
			return readErr(err)
		}
		remaining := total - headerLen
		if remaining < 0 {
			return fmt.Errorf("jp2k: box header runs past end of jp2h")
		}
		if payload < 0 {
			payload = remaining // LBox==0: runs to end of jp2h
		} else if payload > remaining {
			return fmt.Errorf("jp2k: box runs past end of jp2h")
		}
		want := min(child.Peek(typ), payload)
		buf := make([]byte, want)
		if _, err := io.ReadFull(r, buf); err != nil {
			return readErr(err)
		}
		if err := skip(payload - want); err != nil {
			return readErr(err)
		}
		if err := child.Finish(typ, buf, want < payload); err != nil {
			return err
		}
		total -= headerLen + payload
	}
	return nil
}

// jp2ConfigStream streams jp2c from JP2 file.
// jp2h children validate same as JP2Codestream.
func jp2ConfigStream(ctx context.Context, r io.Reader) ([]byte, error) {
	skip := skipper(ctx, r)
	jpPayload := make([]byte, 4) // jP's LBox is fixed at 12
	if _, err := io.ReadFull(r, jpPayload); err != nil {
		return nil, readErr(err)
	}
	if !bytes.Equal(jpPayload, codestream.JPMagic) {
		return nil, fmt.Errorf("jp2k: missing or malformed jP signature box")
	}
	var child codestream.JP2HChild
	var hdr [16]byte
	for {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("jp2k: reading: %w", err)
		}
		typ, _, payload, err := readBoxHeader(r, &hdr)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil, fmt.Errorf("jp2k: no jp2c box found") // clean EOF, not truncation
			}
			return nil, readErr(err)
		}
		switch typ {
		case codestream.BoxJP2H:
			if payload < 0 {
				return nil, fmt.Errorf("jp2k: no jp2c box found")
			}
			if err := walkJP2HChildren(ctx, r, skip, payload, &child, &hdr); err != nil {
				return nil, err
			}
		case codestream.BoxJP2C:
			if !child.SawIhdr {
				return nil, fmt.Errorf("jp2k: jp2c box appears before a jp2h box with an ihdr child")
			}
			n := payload
			if n < 0 || n > jp2cConfigWindow {
				n = jp2cConfigWindow
			}
			buf := make([]byte, n)
			m, err := io.ReadFull(r, buf)
			if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
				return nil, fmt.Errorf("jp2k: reading: %w", err)
			}
			if m == 0 {
				return nil, fmt.Errorf("jp2k: missing SOC marker at offset 0")
			}
			return buf[:m], nil
		default:
			if payload < 0 {
				return nil, fmt.Errorf("jp2k: no jp2c box found")
			}
			if err := skip(payload); err != nil {
				return nil, readErr(err)
			}
		}
	}
}

// skipBufSize keeps a discard loop interruptible.
const skipBufSize = 32 << 10

// skipper seeks when possible, else discards.
func skipper(ctx context.Context, r io.Reader) func(int64) error {
	sk, seekable := r.(io.Seeker)
	var buf []byte // one buffer: CopyN allocates per call
	return func(n int64) error {
		if n == 0 {
			return nil
		}
		if seekable {
			if _, err := sk.Seek(n, io.SeekCurrent); err == nil {
				return nil
			}
			seekable = false
		}
		for n > 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			if buf == nil {
				buf = make([]byte, skipBufSize)
			}
			m, err := io.ReadFull(r, buf[:min(n, skipBufSize)])
			n -= int64(m)
			if err != nil {
				return err
			}
		}
		return nil
	}
}

// DecodeConfig implements jp2k.DecodeConfig.
func DecodeConfig(r io.Reader) (cfg image.Config, err error) {
	return DecodeConfigContext(context.Background(), r)
}

// DecodeConfigContext is DecodeConfig, ctx-checked between reads.
// Streams boxes unbuffered; no budget needed.
// Heap bounded regardless of box count.
func DecodeConfigContext(ctx context.Context, r io.Reader) (cfg image.Config, err error) {
	defer parallel.Recover(&cfg, &err)
	if err := ctx.Err(); err != nil {
		return image.Config{}, fmt.Errorf("jp2k: reading: %w", err)
	}
	head, err := io.ReadAll(io.LimitReader(r, int64(len(codestream.JP2Signature))))
	if err != nil {
		return image.Config{}, err
	}
	isJP2 := bytes.Equal(head, codestream.JP2Signature)
	var stream []byte
	if isJP2 {
		stream, err = jp2ConfigStream(ctx, r)
	} else {
		if err := ctx.Err(); err != nil {
			return image.Config{}, fmt.Errorf("jp2k: reading: %w", err)
		}
		var rest []byte
		rest, err = io.ReadAll(io.LimitReader(r, jp2cConfigWindow-int64(len(head))))
		stream = append(head, rest...)
	}
	if err != nil {
		return image.Config{}, err
	}
	if len(stream) == 0 {
		return image.Config{}, io.ErrUnexpectedEOF
	}
	if len(stream) < 2 || codestream.Be16(stream[0:2]) != codestream.MarkerSOC {
		return image.Config{}, fmt.Errorf("jp2k: missing SOC marker at offset 0")
	}
	if len(stream) < 4 || codestream.Be16(stream[2:4]) != codestream.MarkerSIZ {
		return image.Config{}, fmt.Errorf("jp2k: expected SIZ marker immediately after SOC at offset 2")
	}
	siz, _, err := codestream.ParseSIZSegment(stream, 2)
	if err != nil {
		return image.Config{}, err
	}
	if parallel.FaultHook != nil {
		parallel.FaultHook("config")
	}
	kind, err := classifyOutput(siz)
	if err != nil {
		return image.Config{}, err
	}
	// Guards Xsiz/Ysiz overflowing int on 32-bit.
	if err := coordsFit(siz, 0); err != nil {
		return image.Config{}, err
	}
	return image.Config{
		ColorModel: kind.colorModel(),
		Width:      int(siz.Xsiz - siz.XOsiz),
		Height:     int(siz.Ysiz - siz.YOsiz),
	}, nil
}

// top-down slots tolerate varying level counts.
func byLevel[T int32 | float32](slots [][]T, top int) [][]T {
	lv := make([][]T, top+1)
	for l := range lv {
		lv[l] = slots[top-l]
	}
	return lv
}
