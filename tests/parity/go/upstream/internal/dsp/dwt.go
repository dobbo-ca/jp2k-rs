// Package dsp holds the sample-level stages of a decode: dequantization,
// the inverse wavelet transforms, the inverse colour transforms and
// output packing, with SIMD kernels on amd64 and arm64.
package dsp

import (
	"github.com/dobbo-ca/jp2k-go/internal/codestream"
	"github.com/dobbo-ca/jp2k-go/internal/parallel"
)

// T.800 Annex F: inverse discrete wavelet transform.
//
// Both the reversible 5/3 and irreversible 9/7 filters operate on an
// interleaved 1D sequence (low-band samples at positions of parity cas,
// high-band samples at parity 1-cas, F.3.1) and are applied 2D-separably,
// horizontal pass then vertical, one resolution level at a time from the
// smallest resolution up (F.3.2). cas is each resolution's own rect
// origin parity per axis (BuildTileGeom's res.rect.x0/y0 & 1) — nonzero
// whenever the tile-component origin is odd.

// SnDn splits a resolution axis of length rw into its low-band count sn
// and high-band count dn, given the axis origin's parity cas (F.3.1).
func SnDn(rw, cas int) (sn, dn int) {
	if cas == 0 {
		return (rw + 1) / 2, rw / 2
	}
	return rw / 2, (rw + 1) / 2
}

// inverse1D53 performs T.800 F.3.8.2's inverse reversible 5/3 lifting in
// place on the interleaved sequence a (a[2i] is the i'th low-band sample,
// a[2i+1] the i'th high-band sample), given the low/high counts sn, dn and
// the axis origin parity cas. Boundary samples are handled by the
// reference-decoder symmetric clamp (T.800 F.3.7 / OpenJPEG dwt.c), not by
// a padded/extended buffer.
func clampDWT(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

func inverse1D53(a []int32, sn, dn, cas int) {
	if cas == 0 {
		if dn == 0 && sn <= 1 {
			return // F.3.7: a single unmodified sample
		}
		for i := 0; i < sn; i++ {
			im1, i0 := clampDWT(i-1, dn), clampDWT(i, dn)
			a[2*i] -= (a[2*im1+1] + a[2*i0+1] + 2) >> 2
		}
		for i := 0; i < dn; i++ {
			i0, i1 := clampDWT(i, sn), clampDWT(i+1, sn)
			a[2*i+1] += (a[2*i0] + a[2*i1]) >> 1
		}
		return
	}

	if sn == 0 && dn == 1 {
		a[0] /= 2 // F.3.7: length-1 odd-origin signal
		return
	}
	for i := 0; i < sn; i++ {
		i0, i1 := clampDWT(i, dn), clampDWT(i+1, dn)
		a[2*i+1] -= (a[2*i0] + a[2*i1] + 2) >> 2
	}
	for i := 0; i < dn; i++ {
		i0, im1 := clampDWT(i, sn), clampDWT(i-1, sn)
		a[2*i] += (a[2*i0+1] + a[2*im1+1]) >> 1
	}
}

// -- 9/7 irreversible -------------------------------------------------

// T.800 F.3.8.2 lifting constants (also OpenJPEG dwt.c / pdf.js jpx.js).
const (
	dwt97Alpha = -1.586134342059924
	dwt97Beta  = -0.052980118572961
	dwt97Gamma = 0.882911075530934
	dwt97Delta = 0.443506852043971
	dwt97K     = 1.230174104914001
	// dwt97TwoInvK is OpenJPEG's decode-only high-band scale (dwt.c
	// BUG_WEIRD_TWO_INVK): its historic literal, not 2/K computed fresh.
	dwt97TwoInvK = 1.625732422
)

// dwtPad is the per-side padding around a vertical pass's column block:
// one mirrored row per side (F.3.7), plus one spare so the cas-dependent
// band offsets never index below zero.
const dwtPad = 2

// filter97Split applies T.800 F.3.8.2's inverse 9/7 lifting steps to a
// line held split into its low band l and high band h (Mallat order)
// rather than interleaved, so every step runs over contiguous memory.
// cas is the line's origin parity: the low band sits at even positions
// when cas is 0. Needs len(l)+len(h) >= 2.
// hi is false when the caller knows h is entirely zero on entry (an
// empty-high-band level): the delta step is then a no-op, so it and its
// scale are skipped.
func filter97Split(l, h []float32, cas int, hi bool, r Rounding) {
	// Low band *= K, high band *= two_invK (not 1/K): quant.go's Rb drops
	// the T.800 E-3 per-subband gain term to match OpenJPEG's decode-only
	// convention, and this factor-of-2-too-much high-band scale is what
	// cancels it back out (each 1D pass a sample spends as "high" doubles
	// its share of the dropped 2^gain, and HL/LH cross one such pass,
	// HH two, matching gain 1/1/2). two_invK is OpenJPEG's literal
	// constant (dwt.c BUG_WEIRD_TWO_INVK), not a derived value: it is not
	// exactly 2/K in float32, so keep it as its own constant rather than
	// computing 2*invK.
	scale(l, l, dwt97K)
	// A band at even positions reads neighbors i-1 and i of the other.
	lo, ho := cas-1, -cas
	if hi {
		scale(h, h, dwt97TwoInvK)
		liftSplit(l, h, lo, -dwt97Delta, r)
	}
	liftSplit(h, l, ho, -dwt97Gamma, r)
	liftSplit(l, h, lo, -dwt97Beta, r)
	liftSplit(h, l, ho, -dwt97Alpha, r)
}

// liftSplit is one lifting step, d[i] += c*(u[i+o] + u[i+o+1]) for o in
// {-1, 0}, reading u mirror-extended by one sample per side (F.3.7);
// in split form that mirror is a clamp. The edges are scalar, the rest
// goes to the lift kernel.
func liftSplit(d, u []float32, o int, c float32, r Rounding) {
	nu := len(u)
	edge := func(i int) {
		d[i] = r.mulAdd(c, u[clampDWT(i+o, nu)]+u[clampDWT(i+o+1, nu)], d[i])
	}
	i0 := -o
	i1 := max(i0, min(len(d), nu-1-o))
	for i := 0; i < min(i0, len(d)); i++ {
		edge(i)
	}
	if i1 > i0 {
		r.lift(d[i0:i1], d[i0:i1], u[i0+o:], u[i0+o+1:], c)
	}
	for i := i1; i < len(d); i++ {
		edge(i)
	}
}

// ColBlock53 and ColBlock97 are how many columns a vertical pass lifts at
// once: the stripe is gathered into a small row-major block so every
// lifting step runs along contiguous memory instead of striding a full
// plane row per sample. 9/7 is wider: fewer, longer lift calls.
const ColBlock53, ColBlock97 = 32, 64

// RowChunk is how many rows one parallel task handles.
const RowChunk = 16

// minParallelLevel is the level area below which a pass stays on one
// goroutine; spawning workers costs more than lifting a small level.
const minParallelLevel = 1 << 15

// InverseDWT53 and InverseDWT97 reconstruct one tile component in place.
// p holds it at stride in OpenJPEG's Mallat layout: before level r runs,
// resolution r-1's samples fill the top-left corner of resolution r's
// area and level r's HL, LH and HH bands sit to its right, below it and
// diagonally from it, low samples first on each axis. Every level reads
// and writes the same memory, so the whole transform needs one plane plus
// small per-worker scratch, kept in *scratch so a decode reuses it for
// every component. rects[r] is resolution r's rect; only its size and
// origin parity matter. Once done is closed no further row or column
// batch starts, and p is left unfinished.
// high[r], when high is non-nil, reports whether resolution level r has
// any HL/LH/HH code-block; nil means every level may have high-band data.
// A level without one has its whole high band zeroed by p.level, so its
// row and column lifting reduces to no-ops that can be skipped.
func InverseDWT53(p []int32, stride int, rects []codestream.SubbandRect, high []bool, scratch *[][]int32, nw int, done <-chan struct{}) {
	inverseDWT(p, stride, rects, high, scratch, nw, 0, ColBlock53, done, nil, nil,
		func(row, line []int32, sn, cas int, _ bool) {
			n := len(row)
			interleave(line[:n], row, sn, cas)
			inverse1D53(line[:n], sn, n-sn, cas)
			copy(row, line)
		},
		func(p []int32, stride, bn int, b []int32, n, cas int, _ bool) {
			sn, _ := SnDn(n, cas)
			for i := 0; i < n; i++ {
				q := cas + 2*i
				if i >= sn {
					q = 1 - cas + 2*(i-sn)
				}
				copy(b[q*bn:(q+1)*bn], p[i*stride:i*stride+bn])
			}
			inverse53Rows(b, bn, n, cas)
			for q := 0; q < n; q++ {
				copy(p[q*stride:q*stride+bn], b[q*bn:(q+1)*bn])
			}
		})
}

// emit, when non-nil, is a sink for the last zero-high level's rows,
// packed straight into their destination instead of into p; the return
// reports whether emit ran (the caller's p is then left unfinished).
// r is the lift's rounding; JJ2000 never runs emit.
func InverseDWT97(p []float32, stride int, rects []codestream.SubbandRect, high []bool, scratch *[][]float32, nw int, done <-chan struct{}, emit func(y int, row []float32), r Rounding) bool {
	if r == JJ2000 {
		inverseDWT97JJ(p, stride, rects, high, scratch, nw, done)
		return false
	}
	var band func(p []float32, stride, w, n, cas, y0, y1 int, s []float32)
	if emit != nil {
		band = func(p []float32, stride, w, n, cas, y0, y1 int, s []float32) {
			sweep97LowTo(p, stride, w, n, cas, y0, y1, s, emit, r)
		}
	}
	low := func(p []float32, stride, w, n, cas int, s []float32) { sweep97Low(p, stride, w, n, cas, s, r) }
	return inverseDWT(p, stride, rects, high, scratch, nw, dwtPad, ColBlock97, done, low, band,
		func(row, line []float32, sn, cas int, hi bool) {
			if len(row) < 2 {
				return // F.3.7: a length-1 9/7 signal is unchanged
			}
			copy(line, row)
			filter97Split(line[:sn], line[sn:len(row)], cas, hi, r)
			interleave97(row, line[:len(row)], sn, cas)
		},
		func(p []float32, stride, bn int, b []float32, n, cas int, hi bool) {
			if n > 1 { // F.3.7: a length-1 9/7 signal is unchanged
				filter97Cols(p, stride, bn, b, n, cas, hi, r)
			}
		})
}

// inverseDWT runs every level's horizontal then vertical pass. rowLift
// transforms one Mallat-ordered row with sn low samples in place, using
// line (at least as long) as scratch; colLift lifts bn columns of p at
// x0..x0+bn-1 using b (padded by pad rows per side) as scratch.
func inverseDWT[T int32 | float32](p []T, stride int, rects []codestream.SubbandRect, high []bool, scratch *[][]T, nw, pad, colBlock int, done <-chan struct{},
	lowCols func(p []T, stride, w, n, cas int, s []T), band func(p []T, stride, w, n, cas, y0, y1 int, s []T), rowLift func(row, line []T, sn, cas int, hi bool), colLift func(p []T, stride, bn int, b []T, n, cas int, hi bool)) bool {
	if len(rects) < 2 {
		return false
	}
	top := rects[len(rects)-1]
	scratchN := max(top.X1-top.X0+2*pad, colBlock*(top.Y1-top.Y0+2*pad))
	get := func(w int) []T {
		s := *scratch
		if len(s[w]) < scratchN {
			s[w] = make([]T, scratchN)
		}
		return s[w]
	}
	for r := 1; r < len(rects); r++ {
		if parallel.IsDone(done) {
			return false
		}
		rw, rh := rects[r].X1-rects[r].X0, rects[r].Y1-rects[r].Y0
		if rw <= 0 || rh <= 0 {
			continue
		}
		casX, casY := rects[r].X0&1, rects[r].Y0&1
		snX, _ := SnDn(rw, casX)
		snY, _ := SnDn(rh, casY)
		hi := high == nil || high[r]
		rowsN := rh
		if !hi { // LH/HH rows are all zero; lifting zeros gives zeros
			rowsN = snY
		}
		rowTasks, colTasks := codestream.CeilDiv(rowsN, RowChunk), codestream.CeilDiv(rw, colBlock)
		nwRows, nwCols := 1, 1
		if rw*rh >= minParallelLevel {
			nwRows, nwCols = parallel.Workers(rowTasks, nw), parallel.Workers(colTasks, nw)
		}
		// GOMAXPROCS can change mid-decode; size scratch from this level's nw.
		for len(*scratch) < max(nwRows, nwCols) {
			*scratch = append(*scratch, nil)
		}
		_ = parallel.For(nwRows, rowTasks, func(w, t int) error {
			if parallel.FaultHook != nil {
				parallel.FaultHook("dwt")
			}
			if parallel.IsDone(done) {
				return parallel.ErrStopped
			}
			line := get(w)[:rw]
			for y := t * RowChunk; y < min(rowsN, (t+1)*RowChunk); y++ {
				rowLift(p[y*stride:y*stride+rw], line, snX, casX, hi)
			}
			return nil
		})
		cb, sweep := colBlock, !hi && lowCols != nil && rh > 1
		if sweep {
			// One wide chunk per worker.
			cb = (codestream.CeilDiv(rw, nwCols) + 15) &^ 15
			if 5*cb > scratchN {
				cb, sweep = colBlock, false
			}
		}
		// Last zero-high level has no in-place hazard: emit its rows.
		if sweep && band != nil && r == len(rects)-1 && 8*rw <= scratchN {
			bh := max(64, codestream.CeilDiv(rh, nwCols))
			_ = parallel.For(nwCols, codestream.CeilDiv(rh, bh), func(w, t int) error {
				if parallel.IsDone(done) {
					return parallel.ErrStopped
				}
				band(p, stride, rw, rh, casY, t*bh, min(rh, (t+1)*bh), get(w))
				return nil
			})
			return true
		}
		_ = parallel.For(nwCols, codestream.CeilDiv(rw, cb), func(w, t int) error {
			if parallel.IsDone(done) {
				return parallel.ErrStopped
			}
			k0 := t * cb
			bn := min(cb, rw-k0)
			if sweep {
				lowCols(p[k0:], stride, bn, rh, casY, get(w))
				return nil
			}
			colLift(p[k0:], stride, bn, get(w)[:(rh+2*pad)*bn], rh, casY, hi)
			return nil
		})
	}
	return false
}

// interleave spreads a Mallat-ordered line (sn low samples, then the high
// samples) into dst in F.3.1 interleaved order for origin parity cas.
func interleave[T int32 | float32](dst, src []T, sn, cas int) {
	for i := 0; i < sn; i++ {
		dst[cas+2*i] = src[i]
	}
	for i, j := sn, 1-cas; i < len(src); i, j = i+1, j+2 {
		dst[j] = src[i]
	}
}

// interleave97 is interleave via the zip kernel.
func interleave97(dst, src []float32, sn, cas int) {
	a, b := src[:sn], src[sn:]
	if cas == 1 {
		a, b = b, a
	}
	m := len(b)
	zip(dst, a[:m], b)
	if len(a) > m {
		dst[2*m] = a[m]
	}
}

// zipGo is the reference zip kernel.
func zipGo(dst, a, b []float32) {
	b = b[:len(a)]
	dst = dst[:2*len(a)]
	for i := range a {
		dst[2*i], dst[2*i+1] = a[i], b[i]
	}
}

// inverse53Rows is inverse1D53 applied to bn columns at once: row j of b
// (bn wide) is sample j of every column. Same arithmetic per sample.
func inverse53Rows(b []int32, bn, n, cas int) {
	sn, dn := SnDn(n, cas)
	row := func(j int) []int32 { return b[j*bn : j*bn+bn] }
	if cas == 0 {
		if dn == 0 && sn <= 1 {
			return
		}
		for i := 0; i < sn; i++ {
			d, u, v := row(2*i), row(2*clampDWT(i-1, dn)+1), row(2*clampDWT(i, dn)+1)
			u, v = u[:len(d)], v[:len(d)]
			for x := range d {
				d[x] -= (u[x] + v[x] + 2) >> 2
			}
		}
		for i := 0; i < dn; i++ {
			d, u, v := row(2*i+1), row(2*clampDWT(i, sn)), row(2*clampDWT(i+1, sn))
			u, v = u[:len(d)], v[:len(d)]
			for x := range d {
				d[x] += (u[x] + v[x]) >> 1
			}
		}
		return
	}
	if sn == 0 && dn == 1 {
		d := row(0)
		for x := range d {
			d[x] /= 2
		}
		return
	}
	for i := 0; i < sn; i++ {
		d, u, v := row(2*i+1), row(2*clampDWT(i, dn)), row(2*clampDWT(i+1, dn))
		u, v = u[:len(d)], v[:len(d)]
		for x := range d {
			d[x] -= (u[x] + v[x] + 2) >> 2
		}
	}
	for i := 0; i < dn; i++ {
		d, u, v := row(2*i), row(2*clampDWT(i, sn)+1), row(2*clampDWT(i-1, sn)+1)
		u, v = u[:len(d)], v[:len(d)]
		for x := range d {
			d[x] += (u[x] + v[x]) >> 1
		}
	}
}

// filter97Cols is filter97 applied to bn Mallat-ordered columns of p at
// once, read into and written back from the padded scratch block b: the
// scale step gathers p straight into b, and the beta and alpha lifts
// scatter their results straight back into p, so no separate copy pass
// is needed. Rows of b sit at offset dwtPad; the per-sample arithmetic
// (and so the float32 result) is the same as the interleaved reference.
func filter97Cols(p []float32, stride, bn int, b []float32, n, cas int, hasHi bool, r Rounding) {
	sn, dn := SnDn(n, cas)
	lo, hi := dwtPad+cas, dwtPad+1-cas
	row := func(j int) []float32 { return b[j*bn : j*bn+bn] }
	prow := func(y int) []float32 { return p[y*stride : y*stride+bn] }
	scaleN(b[lo*bn:], p, bn, sn, 2*bn, stride, dwt97K)
	if hasHi {
		scaleN(b[hi*bn:], p[sn*stride:], bn, dn, 2*bn, stride, dwt97TwoInvK)
	} else {
		for i := 0; i < dn; i++ {
			clear(row(hi + 2*i))
		}
	}
	if hasHi {
		extendRows(b, bn, n)
		liftRows(b, bn, lo, sn, -dwt97Delta, r)
	}
	extendRows(b, bn, n)
	liftRows(b, bn, hi, dn, -dwt97Gamma, r)
	extendRows(b, bn, n)
	r.liftN(p[cas*stride:], b[lo*bn:], b[(lo-1)*bn:], b[(lo+1)*bn:], bn, sn, 2*stride, 2*bn, 2*bn, -dwt97Beta)
	// Alpha rows mirror only at the ends.
	i0, i1 := 0, dn
	alpha := func(i int) {
		y := hi + 2*i - dwtPad
		r.lift(prow(y), row(y+dwtPad), prow(mirror(y-1, n)), prow(mirror(y+1, n)), -dwt97Alpha)
	}
	if cas == 1 {
		alpha(0)
		i0 = 1
	}
	if hi+2*(dn-1)-dwtPad == n-1 {
		i1 = dn - 1
	}
	if i1 > i0 {
		y := hi + 2*i0 - dwtPad
		r.liftN(p[y*stride:], b[(y+dwtPad)*bn:], p[(y-1)*stride:], p[(y+1)*stride:], bn, i1-i0, 2*stride, 2*bn, 2*stride, -dwt97Alpha)
	}
	if i1 < dn && i1 >= i0 {
		alpha(dn - 1)
	}
}

// sweep97Low is filter97Cols(hasHi=false) for a zero-high-band level,
// done in place over the full row width instead of a padded column
// block: since the Mallat high rows are all zero and unread, walking
// output row y downward only ever touches low rows already consumed,
// so no gather/scatter buffer is needed, just w-wide scale/gamma/beta/
// alpha steps over 5 hot scratch rows (s, w wide each).
func sweep97Low(p []float32, stride, w, n, cas int, s []float32, r Rounding) {
	zero := s[4*w : 5*w]
	clear(zero)
	slot := func(k int) []float32 { return s[k*w : k*w+w] }
	aRow := func(y int) []float32 { return slot((mirror(y, n) - cas) >> 1 & 1) }
	gRow := func(y int) []float32 { return slot(2 + (mirror(y, n)-1+cas)>>1&1) }
	prow := func(y int) []float32 { return p[y*stride : y*stride+w] }
	top := n + 2 + (n+2-cas)&1
	for t := top; t >= -3; t -= 2 {
		if t >= 0 && t < n {
			scale(aRow(t), prow((t-cas)/2), dwt97K)
		}
		if y := t + 1; y >= 0 && y < n {
			r.lift(gRow(y), zero, aRow(y-1), aRow(y+1), -dwt97Gamma)
		}
		if y := t + 2; y >= 0 && y < n {
			r.lift(prow(y), aRow(y), gRow(y-1), gRow(y+1), -dwt97Beta)
		}
		if y := t + 3; y >= 0 && y < n {
			r.lift(prow(y), gRow(y), prow(mirror(y-1, n)), prow(mirror(y+1, n)), -dwt97Alpha)
		}
	}
}

// sweep97LowTo is sweep97Low for rows [y0,y1) only, emitted instead of
// written to p, so bands of a level can run in parallel and land
// straight in their destination.
func sweep97LowTo(p []float32, stride, w, n, cas, y0, y1 int, s []float32, emit func(y int, row []float32), r Rounding) {
	zero := s[4*w : 5*w]
	clear(zero)
	slot := func(k int) []float32 { return s[k*w : k*w+w] }
	aRow := func(y int) []float32 { return slot((mirror(y, n) - cas) >> 1 & 1) }
	gRow := func(y int) []float32 { return slot(2 + (mirror(y, n)-1+cas)>>1&1) }
	bRow := func(y int) []float32 { return slot(5 + (mirror(y, n)-cas)>>1&1) }
	out := slot(7)
	prow := func(y int) []float32 { return p[y*stride : y*stride+w] }
	top := min(y1+1+(y1+1-cas)&1, n+2+(n+2-cas)&1)
	for t := top; t >= y0-3; t -= 2 {
		if t >= 0 && t < n {
			scale(aRow(t), prow((t-cas)/2), dwt97K)
		}
		if y := t + 1; y >= 0 && y < n {
			r.lift(gRow(y), zero, aRow(y-1), aRow(y+1), -dwt97Gamma)
		}
		if y := t + 2; y >= 0 && y < n {
			r.lift(bRow(y), aRow(y), gRow(y-1), gRow(y+1), -dwt97Beta)
			if y >= y0 && y < y1 {
				emit(y, bRow(y))
			}
		}
		if y := t + 3; y >= y0 && y < min(y1, n) {
			r.lift(out, gRow(y), bRow(y-1), bRow(y+1), -dwt97Alpha)
			emit(y, out)
		}
	}
}

// mirror reflects y into [0, n) about the edge samples (F.3.7), the same
// reflection extendRows applies to b's mirrored rows.
func mirror(y, n int) int {
	if y < 0 {
		return -y
	}
	if y >= n {
		return 2*n - 2 - y
	}
	return y
}

// extendRows is extend97 for whole rows.
func extendRows(b []float32, bn, n int) {
	copy(b[(dwtPad-1)*bn:dwtPad*bn], b[(dwtPad+1)*bn:(dwtPad+2)*bn])
	copy(b[(dwtPad+n)*bn:(dwtPad+n+1)*bn], b[(dwtPad+n-2)*bn:(dwtPad+n-1)*bn])
}

func liftRows(b []float32, bn, first, count int, c float32, r Rounding) {
	f := b[first*bn:]
	r.liftN(f, f, b[(first-1)*bn:], b[(first+1)*bn:], bn, count, 2*bn, 2*bn, 2*bn, c)
}

// liftNGo is liftGo over rows rows; row r of each operand starts r
// strides in.
func liftNGo(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	for r := 0; r < rows; r++ {
		liftGo(dst[r*sDst:r*sDst+n], d[r*sD:], u[r*sUV:], v[r*sUV:], c)
	}
}

// scaleNGo is scaleGo over rows rows; row r of each operand starts r
// strides in.
func scaleNGo(dst, src []float32, n, rows, sDst, sSrc int, k float32) {
	for r := 0; r < rows; r++ {
		scaleGo(dst[r*sDst:r*sDst+n], src[r*sSrc:], k)
	}
}

// liftGo and scaleGo are the reference 9/7 kernels. lift and scale are
// per-arch (dwt_*.go) and must match these bit for bit, on every arch.
// dst may equal an operand for an in-place call; it must not otherwise
// overlap one.
func liftGo(dst, d, u, v []float32, c float32) {
	n := len(dst)
	if len(d) < n || len(u) < n || len(v) < n {
		panic("jp2k: lift operand shorter than its destination")
	}
	d, u, v = d[:n:n], u[:n:n], v[:n:n]
	for x := range dst {
		dst[x] = fma32(c, u[x]+v[x], d[x])
	}
}

func scaleGo(dst, src []float32, k float32) {
	src = src[:len(dst)]
	for x := range dst {
		dst[x] = src[x] * k
	}
}
