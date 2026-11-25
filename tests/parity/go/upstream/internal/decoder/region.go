package decoder

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"math"
	"os"

	"github.com/dobbo-ca/jp2k-go/internal/codestream"
	"github.com/dobbo-ca/jp2k-go/internal/parallel"
	"github.com/dobbo-ca/jp2k-go/internal/tier2"
)

// region is what decodeImage produces: the reference-grid rect
// [x0,x1)x[y0,y1), absolute (XOsiz/YOsiz included) and already at
// 1/2^reduce scale.
type region struct {
	x0, y0, x1, y1 int
	reduce         int
}

// fullRegion is the whole image at full size, what Decode produces.
func fullRegion(siz *codestream.SIZ) region {
	return region{int(siz.XOsiz), int(siz.YOsiz), int(siz.Xsiz), int(siz.Ysiz), 0}
}

// DecodeRegion implements jp2k.DecodeRegion.
func DecodeRegion(ctx context.Context, r io.Reader, rect image.Rectangle, reduce int) (image.Image, error) {
	img, _, err := decodeRegion(ctx, r, rect, reduce, nil, Options{})
	return img, err
}

// DecodeRegionComponents implements jp2k.DecodeRegionComponents.
func DecodeRegionComponents(ctx context.Context, r io.Reader, rect image.Rectangle, reduce int, comps []int) ([]*image.Gray16, error) {
	return DecodeRegionComponentsRound(ctx, r, rect, reduce, comps, Options{})
}

// DecodeRegionComponentsRound is DecodeRegionComponents with o.
func DecodeRegionComponentsRound(ctx context.Context, r io.Reader, rect image.Rectangle, reduce int, comps []int, o Options) ([]*image.Gray16, error) {
	if len(comps) == 0 {
		return nil, errNoComponents
	}
	_, planes, err := decodeRegion(ctx, r, rect, reduce, comps, o)
	return planes, err
}

// decodeRegion is decodeImage of rect at reduce of r.
func decodeRegion(ctx context.Context, r io.Reader, rect image.Rectangle, reduce int, comps []int, o Options) (img image.Image, planes []*image.Gray16, err error) {
	defer parallel.Recover(&img, &err)
	if reduce < 0 || reduce > 32 {
		return nil, nil, fmt.Errorf("jp2k: reduce %d out of range", reduce)
	}
	var (
		cs     *codestream.Codestream
		stream []byte
		cdef   []codestream.CdefEntry
	)
	t, err := newTicket(ctx)
	if err != nil {
		return nil, nil, err
	}
	defer t.done()
	grow := t.grow
	if inMemory(r) {
		grow = t.growParse
	}
	data, err := sparseFile(ctx, r, grow, func(siz *codestream.SIZ) func(uint16) bool {
		reg, err := regionOf(siz, rect, reduce)
		if err != nil {
			return nil
		}
		wins := make([]codestream.SubbandRect, len(siz.Components))
		for c := range wins {
			wins[c] = compWindow(siz, c, reg)
		}
		return func(t uint16) bool { return tileTouches(siz, t, wins, reduce) }
	})
	switch {
	case err != nil:
	case ctx.Err() != nil:
		err = fmt.Errorf("jp2k: reading: %w", ctx.Err())
	case data != nil:
		cs, stream, cdef, err = parseFile(data, t.growParse)
	default:
		// Avoid touching a shared *os.File's offset.
		rc := r
		if f, ok := r.(*os.File); ok {
			if sk, ok := r.(io.Seeker); ok {
				if base, serr := sk.Seek(0, io.SeekCurrent); serr == nil {
					// Stat always works on *os.File.
					size := int64(math.MaxInt64 - base)
					if fi, serr := f.Stat(); serr == nil {
						size = fi.Size() - base
					}
					rc = io.NewSectionReader(f, base, size)
				}
			}
		}
		cs, stream, cdef, err = readCodestream(ctx, rc, t.grow, t.growParse)
	}
	if err != nil {
		return nil, nil, err
	}
	reg, err := checkRegion(cs, rect, reduce)
	if err != nil {
		return nil, nil, err
	}
	return decodeImage(ctx, t, cs, stream, cdef, reg, nil, comps, o)
}

// checkRegion is DecodeRegion's rect at reduce as a region of cs, or
// an error if cs cannot give it.
func checkRegion(cs *codestream.Codestream, rect image.Rectangle, reduce int) (region, error) {
	reg, err := regionOf(cs.SIZ, rect, reduce)
	if err != nil {
		return region{}, err
	}
	// OpenJPEG checks every COD/COC it reads, used by a tile or not, and
	// never reads the tile-parts of tiles the region misses; skip those
	// here too so the result does not depend on whether r can seek.
	wins := make([]codestream.SubbandRect, len(cs.SIZ.Components))
	for c := range wins {
		wins[c] = compWindow(cs.SIZ, c, reg)
	}
	styles := []codestream.CodingStyle{cs.COD.CodingStyle}
	for _, c := range cs.COC {
		styles = append(styles, c.CodingStyle)
	}
	for _, tp := range cs.Parts() {
		if !tileTouches(cs.SIZ, tp.SOT.Isot, wins, reduce) {
			continue
		}
		if tp.COD != nil {
			styles = append(styles, tp.COD.CodingStyle)
		}
		for _, c := range tp.COC {
			styles = append(styles, c.CodingStyle)
		}
	}
	for _, s := range styles {
		if reduce > int(s.NumDecompLevels) {
			return region{}, fmt.Errorf("jp2k: reduce %d exceeds the %d decomposition levels of a component", reduce, s.NumDecompLevels)
		}
	}
	return reg, nil
}

// regionOf is DecodeRegion's rect at reduce as a region of the image.
func regionOf(siz *codestream.SIZ, rect image.Rectangle, reduce int) (region, error) {
	if err := coordsFit(siz, reduce); err != nil {
		return region{}, err
	}
	xo, yo := int(siz.XOsiz), int(siz.YOsiz)
	rect = rect.Intersect(image.Rect(0, 0, int(siz.Xsiz)-xo, int(siz.Ysiz)-yo))
	if rect.Empty() {
		return region{}, fmt.Errorf("jp2k: region does not intersect the image")
	}
	s := uint(reduce)
	reg := region{
		codestream.CeilDivPow2(rect.Min.X+xo, s), codestream.CeilDivPow2(rect.Min.Y+yo, s),
		codestream.CeilDivPow2(rect.Max.X+xo, s), codestream.CeilDivPow2(rect.Max.Y+yo, s),
		reduce,
	}
	if reg.x0 >= reg.x1 || reg.y0 >= reg.y1 {
		return region{}, fmt.Errorf("jp2k: region is empty at reduce %d", reduce)
	}
	return reg, nil
}

// coordsFit reports an error unless siz's coordinates, and rounding them
// up by a subsampling factor and by 2^reduce, fit in int: int() of a
// uint32 SIZ field goes negative on 32-bit platforms.
func coordsFit(siz *codestream.SIZ, reduce int) error {
	// The tile grid covers the image, so this bounds every coordinate.
	ext := max(uint64(siz.XTOsiz)+uint64(siz.TilesWide)*uint64(siz.XTsiz),
		uint64(siz.YTOsiz)+uint64(siz.TilesHigh)*uint64(siz.YTsiz))
	if reduce > 32 || ext+255+uint64(1)<<uint(reduce) > math.MaxInt {
		return fmt.Errorf("jp2k: image coordinates too large for this platform")
	}
	return nil
}

// sparseFile reads a JP2 file or raw codestream from r when r is an
// io.ReaderAt and io.Seeker (an *os.File, say), keeping only the
// tile-parts of tiles keep(SIZ) accepts: the result has the same
// structure, so a small region of a large tiled image costs I/O and
// memory for its own tiles only. It returns nil, having left r where it
// was, for any other r, a PPM codestream (whose shared header stream
// every tile-part feeds), keep returning nil, or a layout it does not
// walk (Psot 0 or 12, a stream without EOC, no tile-part to keep, read
// errors); the caller then reads everything. grow, if not nil, reserves
// what it allocates, and an error from it is returned. Reads stop once
// ctx is done.
func sparseFile(ctx context.Context, r io.Reader, grow func(int64) error, keep func(*codestream.SIZ) func(uint16) bool) (data []byte, err error) {
	ra0, ok := r.(io.ReaderAt)
	sk, ok2 := r.(io.Seeker)
	if !ok || !ok2 {
		return nil, nil
	}
	ra := ctxReaderAt{ctx, ra0}
	base, err := sk.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, nil
	}
	// reserve charges n bytes to grow; after an error every step fails
	// and err is returned.
	reserve := func(n int) bool {
		if grow != nil && err == nil {
			err = grow(int64(n))
		}
		return err == nil
	}
	read := func(off int64, n int) []byte {
		// A box length is input: check the file holds it before
		// allocating it.
		if n > 1 {
			var last [1]byte
			if m, _ := ra.ReadAt(last[:], base+off+int64(n)-1); m < 1 {
				return nil
			}
		}
		if !reserve(n) {
			return nil
		}
		b := make([]byte, n)
		if m, _ := ra.ReadAt(b, base+off); m < n {
			return nil
		}
		return b
	}
	// A JP2 file: keep the boxes JP2Codestream reads, skip the rest, and
	// let the codestream run to the end of the file.
	var prefix []byte
	off := int64(0)
	if sig := read(0, len(codestream.JP2Signature)); sig != nil && bytes.Equal(sig, codestream.JP2Signature) {
		for {
			h := read(off, 16)
			if h == nil {
				return nil, err
			}
			n, typ, hl := int64(binary.BigEndian.Uint32(h)), binary.BigEndian.Uint32(h[4:]), int64(8)
			if n == 1 {
				n, hl = int64(binary.BigEndian.Uint64(h[8:])), 16
			}
			if typ == codestream.BoxJP2C {
				off += hl
				break
			}
			if n < hl || n > math.MaxInt32 {
				return nil, err // to-end or malformed box before jp2c
			}
			if typ == codestream.BoxJP || typ == codestream.BoxJP2H {
				b := read(off, int(n))
				if b == nil {
					return nil, err
				}
				if !reserve(2 * len(b)) {
					return nil, err
				}
				prefix = append(prefix, b...)
			}
			off += n
		}
		prefix = append(prefix, 0, 0, 0, 0, 'j', 'p', '2', 'c') // runs to the end
	}
	// Main header: SOC, then marker segments up to the first SOT.
	hdr := read(off, 2)
	if hdr == nil || codestream.Be16(hdr) != codestream.MarkerSOC {
		return nil, err
	}
	pos := off + 2
	// Buffered: a header of many tiny markers cost two preads each.
	br := bufio.NewReaderSize(io.NewSectionReader(ra, base+pos, math.MaxInt64-base-pos), 1<<16)
	for {
		var m [4]byte
		if _, rerr := io.ReadFull(br, m[:]); rerr != nil || m[0] != 0xFF || codestream.Be16(m[:]) == codestream.MarkerPPM {
			return nil, err
		}
		if codestream.Be16(m[:]) == codestream.MarkerSOT {
			break
		}
		n := int(codestream.Be16(m[2:]))
		if n < 2 {
			return nil, err
		}
		if !reserve(2 * (n + 2)) {
			return nil, err
		}
		hdr = append(hdr, m[:]...)
		hdr = append(hdr, make([]byte, n-2)...)
		if _, rerr := io.ReadFull(br, hdr[len(hdr)-(n-2):]); rerr != nil {
			return nil, err
		}
		pos += 2 + int64(n)
	}
	siz, _, serr := codestream.ParseSIZSegment(hdr, 2)
	if serr != nil {
		return nil, nil
	}
	want := keep(siz)
	if want == nil {
		return nil, err
	}
	// Walk the SOTs by their Psot, noting the tile-parts to keep.
	type part struct{ off, n int64 }
	var parts []part
	total := int64(len(prefix) + len(hdr) + 2)
	// Read ahead a page: small tile-parts often sit back to back, and
	// this walk's pread count dominates small regions.
	win := make([]byte, 4096)
	wpos, wn := int64(0), 0
	for {
		if pos+12 > wpos+int64(wn) {
			wpos = pos
			wn, _ = ra.ReadAt(win, base+pos)
		}
		sot := win[pos-wpos : pos-wpos+min(12, wpos+int64(wn)-pos)]
		if len(sot) >= 2 && codestream.Be16(sot) == codestream.MarkerEOC {
			break
		}
		if len(sot) < 12 || codestream.Be16(sot) != codestream.MarkerSOT || codestream.Be16(sot[2:]) != 10 {
			return nil, err
		}
		psot := int64(codestream.Be32(sot[6:]))
		if psot < 14 {
			return nil, err
		}
		if want(codestream.Be16(sot[4:])) {
			if !reserve(32) {
				return nil, err
			}
			parts = append(parts, part{pos, psot})
			total += psot
		}
		pos += psot
	}
	if len(parts) == 0 {
		return nil, err // the parser wants a tile-part; the region's tiles have none
	}
	if total > math.MaxInt {
		return nil, err // 32-bit
	}
	if !reserve(int(total)) {
		return nil, err
	}
	data = make([]byte, 0, total)
	data = append(data, prefix...)
	data = append(data, hdr...)
	for _, p := range parts {
		b := data[len(data) : len(data)+int(p.n)]
		if m, _ := ra.ReadAt(b, base+p.off); m < len(b) {
			return nil, err
		}
		data = data[:len(data)+len(b)]
	}
	return append(data, 0xFF, 0xD9), nil
}

// compWindow is the part of component c that reg reads, in the
// component's own sample units at reg's scale: reference-grid sample X
// shows component sample floor(X/XRsiz), clipped to the samples the
// component has (upsampleComponent's rule).
func compWindow(siz *codestream.SIZ, c int, reg region) codestream.SubbandRect {
	xr, yr := int(siz.Components[c].XRsiz), int(siz.Components[c].YRsiz)
	s := uint(reg.reduce)
	cx0, cx1 := codestream.CeilDivPow2(codestream.CeilDiv(int(siz.XOsiz), xr), s), codestream.CeilDivPow2(codestream.CeilDiv(int(siz.Xsiz), xr), s)
	cy0, cy1 := codestream.CeilDivPow2(codestream.CeilDiv(int(siz.YOsiz), yr), s), codestream.CeilDivPow2(codestream.CeilDiv(int(siz.Ysiz), yr), s)
	clamp := func(v, lo, hi int) int { return min(max(v, lo), hi) }
	return codestream.SubbandRect{
		X0: clamp(reg.x0/xr, cx0, cx1), X1: clamp((reg.x1-1)/xr+1, cx0, cx1),
		Y0: clamp(reg.y0/yr, cy0, cy1), Y1: clamp((reg.y1-1)/yr+1, cy0, cy1),
	}
}

// dwtWindows returns, for resolutions 0..top of tc, the window each
// inverse DWT level computes so that win, a window of resolution top,
// comes out exact. A level computed on a window instead of the whole
// resolution is wrong only within reach samples of a window edge that is
// not a real resolution edge (each lifting step spreads an edge error by
// one sample, and edges at the resolution's own bounds extend exactly as
// the full transform does), so each window is the one above grown by
// reach, and the level below must supply its low-pass half.
func dwtWindows(tc *tier2.TileComponent, top int, win codestream.SubbandRect, reach int) []codestream.SubbandRect {
	wins := make([]codestream.SubbandRect, top+1)
	n := win
	for r := top; r > 0; r-- {
		e := codestream.SubbandRect{X0: n.X0 - reach, Y0: n.Y0 - reach, X1: n.X1 + reach, Y1: n.Y1 + reach}.Intersect(tc.Resolutions[r].Rect)
		wins[r] = e
		n = codestream.SubbandRect{X0: (e.X0 + 1) >> 1, Y0: (e.Y0 + 1) >> 1, X1: (e.X1 + 1) >> 1, Y1: (e.Y1 + 1) >> 1}
	}
	wins[0] = n
	return wins
}

// dwtLevels is dwtWindows for a component window: nil when win is the
// whole resolution top and the plain whole-tile path applies, empty when
// win is.
func dwtLevels(tc *tier2.TileComponent, top int, win codestream.SubbandRect, rev bool) []codestream.SubbandRect {
	if win.Empty() {
		return []codestream.SubbandRect{}
	}
	if win == tc.Resolutions[top].Rect {
		return nil
	}
	reach := 4 // 9/7 lifting steps
	if rev {
		reach = 2
	}
	return dwtWindows(tc, top, win, reach)
}

// partialDWT runs the inverse DWT one level at a time over the windows
// from dwtWindows. lv[r] holds level r's window in Mallat order with its
// high-pass bands already filled; each level first copies in its low-pass
// quadrant from the finished level below. Returns the top level's buffer,
// unfinished if done is closed.
func partialDWT[T int32 | float32](lv [][]T, wins []codestream.SubbandRect, high []bool, scratch *[][]T, nw int, done <-chan struct{}, idwt func([]T, int, []codestream.SubbandRect, []bool, *[][]T, int, <-chan struct{})) []T {
	for r := 1; r < len(wins) && !parallel.IsDone(done); r++ {
		e, prev := wins[r], wins[r-1]
		w, pw := e.X1-e.X0, prev.X1-prev.X0
		lx0, ly0 := (e.X0+1)>>1, (e.Y0+1)>>1
		lw, lh := (e.X1+1)>>1-lx0, (e.Y1+1)>>1-ly0
		for y := 0; y < lh; y++ {
			src := (ly0-prev.Y0+y)*pw + lx0 - prev.X0
			copy(lv[r][y*w:y*w+lw], lv[r-1][src:src+lw])
		}
		idwt(lv[r], w, []codestream.SubbandRect{{}, e}, high[r-1:r+1], scratch, nw, done)
	}
	return lv[len(wins)-1]
}

// tileTouches reports whether tile tileIdx holds any sample of the
// component windows (compWindow), without building its geometry. It is
// per component because a subsampled component's window can reach a
// sample of a tile the reference-grid region itself misses.
func tileTouches(siz *codestream.SIZ, tileIdx uint16, compWin []codestream.SubbandRect, reduce int) bool {
	p := int(tileIdx) % int(siz.TilesWide)
	q := int(tileIdx) / int(siz.TilesWide)
	tx0 := max(int(siz.XTOsiz)+p*int(siz.XTsiz), int(siz.XOsiz))
	tx1 := min(int(siz.XTOsiz)+(p+1)*int(siz.XTsiz), int(siz.Xsiz))
	ty0 := max(int(siz.YTOsiz)+q*int(siz.YTsiz), int(siz.YOsiz))
	ty1 := min(int(siz.YTOsiz)+(q+1)*int(siz.YTsiz), int(siz.Ysiz))
	s := uint(reduce)
	for c, w := range compWin {
		xr, yr := int(siz.Components[c].XRsiz), int(siz.Components[c].YRsiz)
		t := codestream.SubbandRect{
			X0: codestream.CeilDivPow2(codestream.CeilDiv(tx0, xr), s), Y0: codestream.CeilDivPow2(codestream.CeilDiv(ty0, yr), s),
			X1: codestream.CeilDivPow2(codestream.CeilDiv(tx1, xr), s), Y1: codestream.CeilDivPow2(codestream.CeilDiv(ty1, yr), s),
		}
		if !t.Intersect(w).Empty() {
			return true
		}
	}
	return false
}

// ctxReaderAt fails every ReadAt once ctx is done: a walk of many
// tile-parts on a slow r would otherwise outlive its deadline.
type ctxReaderAt struct {
	ctx context.Context
	r   io.ReaderAt
}

func (c ctxReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.ReadAt(p, off)
}
