package decoder

import (
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"runtime"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/dobbo-ca/jp2k-go/internal/codestream"
	"github.com/dobbo-ca/jp2k-go/internal/parallel"
	"github.com/dobbo-ca/jp2k-go/internal/tier2"
)

// Handle is an open file for repeated region decodes (jp2k.Open). Its
// headers are parsed at Open, and each tile's packet headers once, by
// the first decode that needs the tile, into a tileIndex kept as long as
// the Handle, up to keepMax bytes of them. Decodes then skip tier-2 and
// read only the code-block bytes they use. It is safe for concurrent use.
type Handle struct {
	ra      io.ReaderAt
	size    int64
	base    int64   // where the codestream starts in ra
	csLen   int64   // codestream bytes
	hdrLen  int64   // main header bytes, SOC to the first SOT
	partLen []int64 // each tile-part's bytes, SOT to the next, in cs.Parts order
	cs      *codestream.Codestream
	cdef    []codestream.CdefEntry
	cfg     image.Config
	levels  int
	// pkg: a tile-part header failed to parse, which fails only the
	// regions that read it; decodes go to the package DecodeRegion.
	pkg      bool
	mainSkel []byte // skeleton to the end of the main header
	readErr  bool   // a ReadAt failed during Open
	whole    bool   // the package path reads it all
	opts     Options

	*ixState
	fin *keptBytes // its finalizer gives kept back
}

// keepFloor is the index bytes a Handle keeps whatever its codestream's
// size: see keepMax.
const keepFloor = 16 << 20

// keepMax bounds the header and index bytes h keeps: past it a tile is
// indexed for each decode that needs it. Real files index to a few
// percent of their size; a hostile one can declare code-blocks its bytes
// do not hold. It counts codestream bytes: others need not be read.
// Open Images together also keep at most a quarter of the memory limit
// (see keep).
func (h *Handle) keepMax() int64 { return keepFloor + h.csLen }

// slot is a tile's index, or the build of it in progress.
type slot struct {
	ix       *tileIndex
	building *building
	big      bool // its index is past keepMax: decodes go to the package path
}

// building is a build of an index: ix and err once done is closed.
type building struct {
	done  chan struct{}
	ix    *tileIndex
	err   error
	retry bool // err is not the tile's: waiters build again
	kept  bool
}

// ixState is what a Handle keeps, apart from it so evictIndexes
// reaches it without keeping the Handle alive.
type ixState struct {
	mu     sync.Mutex
	kept   int64  // of the headers and of the indexes in tiles and all
	keptIx int64  // of it, indexes
	tiles  []slot // by tile index
	all    slot   // under PPM, whose header stream runs through every tile

	tier2Work atomic.Int64  // tier-2 packet-header work spent or held indexing tiles
	workHeld  int64         // of it, held by builds running; under mu
	workFreed chan struct{} // closed when builds give some back
}

// keptBytes gives back what s keeps from its finalizer. It is apart
// from the Handle, which a cycle through ra may hold: a cycle runs no
// finalizer.
type keptBytes struct{ s *ixState }

// kepts are the ixStates that have kept an index.
var kepts struct {
	sync.Mutex
	m map[*ixState]bool
}

// markKept lists s for evictIndexes.
func markKept(s *ixState) {
	kepts.Lock()
	if kepts.m == nil {
		kepts.m = map[*ixState]bool{}
	}
	kepts.m[s] = true
	kepts.Unlock()
}

// evictIndexes drops the kept indexes, which are optional, that no
// decode is planned on, for an Open's headers or a decode that needs
// their room. Those in use stay kept and counted.
func evictIndexes() {
	kepts.Lock()
	m := kepts.m
	kepts.m = nil
	kepts.Unlock()
	for s := range m {
		s.mu.Lock()
		var n int64
		busy := false
		drop := func(sl *slot) {
			switch {
			case sl.ix == nil:
			case sl.ix.refs.Load() > 0:
				busy = true
			default:
				s.giveWork(0, sl.ix.work) // as for an unkept index
				n += sl.ix.size
				sl.ix = nil
			}
		}
		drop(&s.all)
		for i := range s.tiles {
			drop(&s.tiles[i])
		}
		s.kept, s.keptIx = s.kept-n, s.keptIx-n
		s.mu.Unlock()
		if busy {
			markKept(s)
		}
		unkeep(n, true)
	}
}

// release gives back all s keeps, once its Handle is freed.
func (k *keptBytes) release() {
	kepts.Lock()
	delete(kepts.m, k.s)
	kepts.Unlock()
	k.s.mu.Lock()
	n := k.s.kept
	k.s.kept, k.s.keptIx = 0, 0
	k.s.mu.Unlock()
	unkeep(n, false)
}

// errBigIndex sends a decode to the package path.
var errBigIndex = errors.New("jp2k: tile index past keepMax")

// Open implements jp2k.Open.
func Open(r io.ReaderAt, size int64) (*Handle, error) {
	return OpenContext(context.Background(), r, size)
}

// OpenContextRound is OpenContext whose decodes use o.
func OpenContextRound(ctx context.Context, r io.ReaderAt, size int64, o Options) (*Handle, error) {
	h, err := OpenContext(ctx, r, size)
	if err != nil {
		return nil, err
	}
	h.opts = o
	return h, nil
}

// OpenContext implements jp2k.OpenContext.
func OpenContext(ctx context.Context, r io.ReaderAt, size int64) (h *Handle, err error) {
	defer parallel.Recover(&h, &err)
	if size < 0 {
		return nil, fmt.Errorf("jp2k: negative size %d", size)
	}
	t, err := newTicket(ctx)
	if err != nil {
		return nil, err
	}
	defer t.done()
	h = &Handle{ra: r, size: size, ixState: new(ixState)}
	skel, err := h.skeleton(ctx, size, t.growParse)
	var cs *codestream.Codestream
	var cdef []codestream.CdefEntry
	if err == nil {
		cs, _, cdef, err = parseFile(skel, t.growParse)
	}
	if err != nil {
		return h.fallback(err, t.growParse)
	}
	parts := cs.Parts()
	if len(parts) != len(h.partLen) {
		return nil, fmt.Errorf("jp2k: parsed %d tile-parts of %d", len(parts), len(h.partLen))
	}
	// Tile-part data was cut from skel, each Psot to its header: point
	// at the codestream instead. Tile-parts sit back to back.
	at := h.hdrLen
	for i := range parts {
		if at+h.partLen[i] > math.MaxInt {
			return nil, fmt.Errorf("jp2k: tile-part at offset %d too large for this platform", at)
		}
		hl := int64(parts[i].SOT.Psot)
		parts[i].DataOffset, parts[i].DataLength = int(at+hl), int(h.partLen[i]-hl)
		at += h.partLen[i]
	}
	kind, err := classifyOutput(cs.SIZ)
	if err != nil {
		return nil, err
	}
	if err := coordsFit(cs.SIZ, 0); err != nil {
		return nil, err
	}
	levels := int(cs.COD.CodingStyle.NumDecompLevels)
	for _, c := range cs.COC {
		levels = min(levels, int(c.CodingStyle.NumDecompLevels))
	}
	aliased := len(cs.PPM) > 0 // PPM and PPT data point into skel
	var merged int64           // what decodes merge of them, kept in cs
	for _, e := range cs.PPM {
		merged += int64(len(e.Data))
	}
	for _, tp := range parts {
		aliased = aliased || len(tp.PPT) > 0
		for _, e := range tp.PPT {
			merged += int64(len(e.Data))
		}
		if len(tp.PPT) > 0 {
			merged += 64 // a tile's map entry
		}
		if tp.COD != nil {
			levels = min(levels, int(tp.COD.CodingStyle.NumDecompLevels))
		}
		for _, c := range tp.COC {
			levels = min(levels, int(c.CodingStyle.NumDecompLevels))
		}
	}
	h.cs, h.cdef, h.levels = cs, cdef, levels+1
	h.cfg = image.Config{ColorModel: kind.colorModel(), Width: int(cs.SIZ.Xsiz - cs.SIZ.XOsiz), Height: int(cs.SIZ.Ysiz - cs.SIZ.YOsiz)}
	h.tiles = make([]slot, int(cs.SIZ.TilesWide)*int(cs.SIZ.TilesHigh))
	// What the Image holds of the headers.
	h.kept = int64(unsafe.Sizeof(*h)+unsafe.Sizeof(*h.ixState)+unsafe.Sizeof(*cs)) + cs.ParseCost + 8*int64(cap(h.partLen)) + int64(len(h.tiles))*int64(unsafe.Sizeof(slot{}))
	if aliased {
		h.kept += int64(cap(skel)) + merged + merged/8 // size classes
	}
	if !keep(h.kept) {
		if h.kept > keepShare() {
			// Retrying cannot help.
			return h.fallback(fmt.Errorf("jp2k: headers of %d bytes past what Images keep", h.kept), t.growParse)
		}
		if evictIndexes(); !keep(h.kept) {
			if reclaimKept(); !keep(h.kept) {
				return nil, fmt.Errorf("%w: open Images hold their share of it", ErrMemoryLimit)
			}
		}
	}
	h.mainSkel = nil // only fallback reads it
	h.fin = &keptBytes{h.ixState}
	runtime.SetFinalizer(h.fin, (*keptBytes).release)
	return h, nil
}

// fallback is Open's result when the headers after the main header fail
// with err, or are more than Images may keep: a Handle that decodes as
// the package DecodeRegion does, which reads only the tile-parts a
// region needs. A failed read, a refusal of memory, or a main header
// that fails too, is err.
func (h *Handle) fallback(err error, grow func(int64) error) (*Handle, error) {
	if h.readErr || h.mainSkel == nil || errors.Is(err, ErrMemoryLimit) {
		return nil, err
	}
	// A stub tile-part: the parser wants one.
	skel := append(h.mainSkel, 0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 14, 0, 1, 0xFF, 0x93, 0xFF, 0xD9)
	cs, _, _, perr := parseFile(skel, grow)
	if perr != nil {
		return nil, err
	}
	kind, kerr := classifyOutput(cs.SIZ)
	if kerr != nil {
		return nil, err
	}
	if err := coordsFit(cs.SIZ, 0); err != nil {
		return nil, err
	}
	levels := int(cs.COD.CodingStyle.NumDecompLevels)
	for _, c := range cs.COC {
		levels = min(levels, int(c.CodingStyle.NumDecompLevels))
	}
	return &Handle{ra: h.ra, size: h.size, pkg: true, levels: levels + 1,
		cfg: image.Config{ColorModel: kind.colorModel(), Width: int(cs.SIZ.Xsiz - cs.SIZ.XOsiz), Height: int(cs.SIZ.Ysiz - cs.SIZ.YOsiz)}}, nil
}

// DecodeRegion implements jp2k.Image.DecodeRegion.
func (h *Handle) DecodeRegion(ctx context.Context, rect image.Rectangle, reduce int) (image.Image, error) {
	img, _, err := h.decodeRegion(ctx, rect, reduce, nil)
	return img, err
}

// DecodeRegionComponents implements jp2k.Image.DecodeRegionComponents.
func (h *Handle) DecodeRegionComponents(ctx context.Context, rect image.Rectangle, reduce int, comps []int) ([]*image.Gray16, error) {
	if len(comps) == 0 {
		return nil, errNoComponents
	}
	_, planes, err := h.decodeRegion(ctx, rect, reduce, comps)
	return planes, err
}

func (h *Handle) decodeRegion(ctx context.Context, rect image.Rectangle, reduce int, comps []int) (img image.Image, planes []*image.Gray16, err error) {
	if h.pkg {
		return decodeRegion(ctx, io.NewSectionReader(h.ra, 0, h.size), rect, reduce, comps, h.opts)
	}
	if img, planes, err = h.decodeIndexed(ctx, rect, reduce, comps); errors.Is(err, errBigIndex) {
		return decodeRegion(ctx, io.NewSectionReader(h.ra, 0, h.size), rect, reduce, comps, h.opts)
	}
	return img, planes, err
}

func (h *Handle) decodeIndexed(ctx context.Context, rect image.Rectangle, reduce int, comps []int) (img image.Image, planes []*image.Gray16, err error) {
	defer parallel.Recover(&img, &err)
	if reduce < 0 || reduce > 32 {
		return nil, nil, fmt.Errorf("jp2k: reduce %d out of range", reduce)
	}
	reg, err := checkRegion(h.cs, rect, reduce)
	if err != nil {
		return nil, nil, err
	}
	t, err := newTicket(ctx)
	if err != nil {
		return nil, nil, err
	}
	defer t.done()
	return decodeImage(ctx, t, h.cs, nil, h.cdef, reg, h, comps, h.opts)
}

// Config implements jp2k.Image.Config.
func (h *Handle) Config() image.Config { return h.cfg }

// Levels implements jp2k.Image.Levels.
func (h *Handle) Levels() int { return h.levels }

// skeleton reads the headers of the file in ra[:size] into a stream
// parseFile takes: a JP2 file's jP and jp2h boxes, the main header, and
// each tile-part's header, its Psot cut to the header, so no tile data
// is read. It sets h.base, h.csLen, h.hdrLen and h.partLen. grow
// reserves what it allocates. Reads fail once ctx is done.
func (h *Handle) skeleton(ctx context.Context, size int64, grow func(int64) error) (skel []byte, err error) {
	rd := &windowReader{ra: ctxReaderAt{ctx, h.ra}, end: size, win: make([]byte, 64<<10)}
	defer func() { h.readErr = rd.failed }()
	if err := grow(int64(len(rd.win))); err != nil {
		return nil, err
	}
	cend := size
	sig, err := rd.at(0, len(codestream.JP2Signature))
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, fmt.Errorf("jp2k: reading: %w", err)
	}
	if err == nil && bytes.Equal(sig, codestream.JP2Signature) {
		// The boxes JP2Codestream reads, then a jp2c box to the end.
		off := int64(0)
		for {
			b, err := rd.at(off, 8)
			if err != nil {
				return nil, fmt.Errorf("jp2k: no jp2c box: %w", err)
			}
			n, typ, hl := int64(binary.BigEndian.Uint32(b)), binary.BigEndian.Uint32(b[4:]), int64(8)
			switch n {
			case 1:
				b, err := rd.at(off+8, 8)
				if err != nil {
					return nil, err
				}
				n, hl = int64(binary.BigEndian.Uint64(b)), 16
			case 0:
				n = size - off
			}
			if n < hl {
				return nil, fmt.Errorf("jp2k: box at offset %d has length %d shorter than its header", off, n)
			}
			if typ == codestream.BoxJP2C {
				h.base = off + hl
				if n <= size-off {
					cend = off + n
				}
				break
			}
			if n > size-off {
				return nil, fmt.Errorf("jp2k: box at offset %d (length %d) runs past end of data", off, n)
			}
			if typ == codestream.BoxJP || typ == codestream.BoxJP2H {
				if err := grow(2 * n); err != nil {
					return nil, err
				}
				b, err := rd.at(off, int(min(n, math.MaxInt32)))
				if err != nil {
					return nil, err
				}
				skel = appendx(skel, b...)
			}
			off += n
		}
		skel = appendx(skel, 0, 0, 0, 0, 'j', 'p', '2', 'c')
	}
	rd.end = cend
	h.csLen = cend - h.base
	// Main header: SOC, then marker segments up to the first SOT.
	pos := h.base
	b, err := rd.at(pos, 2)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, fmt.Errorf("jp2k: reading: %w", err)
	}
	if err != nil || codestream.Be16(b) != codestream.MarkerSOC {
		return nil, fmt.Errorf("jp2k: missing SOC marker at offset 0")
	}
	skel = appendx(skel, b...)
	if skel, pos, err = rd.segments(skel, pos+2, codestream.MarkerSOT, cend, grow); err != nil {
		return nil, fmt.Errorf("jp2k: main header: %w", err)
	}
	h.hdrLen = pos - h.base
	h.mainSkel = skel[:len(skel):len(skel)]
	// Tile-parts: each header, with Psot cut to it; the data stays in
	// the file.
	eoc := false
	for pos < cend {
		if pos+2 > cend {
			return nil, fmt.Errorf("jp2k: truncated codestream at offset %d, expected SOT or EOC", pos-h.base)
		}
		m, err := rd.at(pos, int(min(12, cend-pos)))
		if err != nil {
			return nil, err
		}
		if codestream.Be16(m) == codestream.MarkerEOC {
			skel = appendx(skel, 0xFF, 0xD9)
			eoc = true
			break
		}
		if len(m) < 12 || codestream.Be16(m) != codestream.MarkerSOT || codestream.Be16(m[2:]) != 10 {
			return nil, fmt.Errorf("jp2k: expected SOT or EOC at offset %d", pos-h.base)
		}
		psot := int64(codestream.Be32(m[6:]))
		start := len(skel)
		skel = appendx(skel, m...)
		hend := cend
		if psot != 0 && psot != 12 {
			hend = min(cend, pos+psot)
		}
		var q int64
		if skel, q, err = rd.segments(skel, pos+12, codestream.MarkerSOD, hend, grow); err != nil {
			return nil, fmt.Errorf("jp2k: tile-part header at offset %d: %w", pos-h.base, err)
		}
		skel = appendx(skel, 0xFF, 0x93)
		q += 2
		binary.BigEndian.PutUint32(skel[start+6:], uint32(len(skel)-start))
		end, truncated := pos+psot, false
		switch {
		case psot == 0 || psot == 12:
			h.whole = true
			// Data runs to the next SOT or EOC, as the parser finds it.
			if end, err = rd.markerFrom(q); err != nil {
				return nil, err
			}
			truncated = end == cend
		case end > cend:
			end, truncated = cend, true
		case end < q:
			return nil, fmt.Errorf("jp2k: SOT Psot %d puts tile-part data out of bounds at offset %d", psot, pos-h.base)
		}
		if err := grow(64); err != nil {
			return nil, err
		}
		h.partLen = appendx(h.partLen, end-pos)
		pos = end
		if truncated {
			break
		}
	}
	h.whole = h.whole || !eoc
	return skel, nil
}

// segments appends to skel the marker segments from pos up to marker
// stop, which must start before end, and returns where stop is.
func (w *windowReader) segments(skel []byte, pos int64, stop uint16, end int64, grow func(int64) error) ([]byte, int64, error) {
	for {
		m, err := w.at(pos, 2)
		if err == nil && pos+2 > end {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			return nil, 0, err
		}
		id := codestream.Be16(m)
		n := 2
		switch {
		case id == stop:
			return skel, pos, nil
		case id < 0xFF30:
			return nil, 0, fmt.Errorf("jp2k: invalid marker 0x%04X", id)
		case id <= 0xFF3F:
			// Reserved delimiters: no segment, as the parser reads them.
		case id == codestream.MarkerSOC || id == codestream.MarkerSOD || id == codestream.MarkerEOC || id == 0xFF91 || id == 0xFF92:
			return nil, 0, fmt.Errorf("jp2k: unexpected marker 0x%04X in a header", id)
		default:
			if m, err = w.at(pos, 4); err != nil {
				return nil, 0, err
			}
			n = int(codestream.Be16(m[2:])) + 2
			if n < 4 {
				return nil, 0, fmt.Errorf("jp2k: marker 0x%04X segment too short", id)
			}
		}
		if err := grow(2 * int64(n)); err != nil {
			return nil, 0, err
		}
		seg, err := w.at(pos, n)
		if err != nil {
			return nil, 0, err
		}
		skel = appendx(skel, seg...)
		pos += int64(n)
	}
}

// appendx is append, doubling: append grows a large slice by a quarter,
// which allocates five times its final size along the way.
func appendx[T any](s []T, v ...T) []T {
	if len(s)+len(v) > cap(s) {
		t := make([]T, len(s), max(2*cap(s), len(s)+len(v), 64))
		copy(t, s)
		s = t
	}
	return append(s, v...)
}

// windowReader reads ra[:end] through a window, so the many small reads
// of a header walk cost few ReadAt calls.
type windowReader struct {
	ra   io.ReaderAt
	end  int64
	win  []byte
	wpos int64
	wn   int
	// failed: a ReadAt failed or ended early.
	failed bool
}

// page is what at reads past a window: small tile-parts often sit
// back to back.
const page = 4 << 10

// at returns ra[off:off+n], valid until the next call. Past end is an
// error.
func (w *windowReader) at(off int64, n int) ([]byte, error) { return w.read(off, n, page) }

// read is at, reading ahead bytes, up to the window's size, on a miss.
func (w *windowReader) read(off int64, n, ahead int) ([]byte, error) {
	if off < 0 || n < 0 || off > w.end || int64(n) > w.end-off {
		return nil, io.ErrUnexpectedEOF
	}
	if off >= w.wpos && off+int64(n) <= w.wpos+int64(w.wn) {
		return w.win[off-w.wpos : off-w.wpos+int64(n)], nil
	}
	var b []byte
	if n > len(w.win) {
		b = make([]byte, n)
	} else {
		b = w.win[:max(n, ahead)]
	}
	m := int(min(int64(len(b)), w.end-off))
	k, err := w.ra.ReadAt(b[:m], off)
	if k < n {
		w.failed = true // short inside size: r failed
		if err == nil || err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		w.wn = 0
		return nil, err
	}
	if n > len(w.win) {
		return b[:n], nil
	}
	w.wpos, w.wn = off, k
	return w.win[:n], nil
}

// markerFrom is where the first SOT or EOC marker at or after off is,
// or end: codestream.findTileDataEnd over the file. It scans the
// window first, then reads whole windows, which hold the headers after.
func (w *windowReader) markerFrom(off int64) (int64, error) {
	for off+1 < w.end {
		if _, err := w.read(off, 2, len(w.win)); err != nil {
			return 0, err
		}
		b := w.win[off-w.wpos : w.wn]
		for i := 0; i+1 < len(b); i++ {
			if b[i] == 0xFF && (b[i+1] == 0x90 || b[i+1] == 0xD9) {
				return off + int64(i), nil
			}
		}
		off += int64(len(b) - 1)
	}
	return w.end, nil
}

// tileIndex is one tile's packet headers, parsed once: for every
// code-block its zero bit-planes and, per layer and segment, where its
// bytes lie and how many passes they hold. It costs cellBytes per
// code-block plus segBytes per segment each layer adds, besides tg and
// bands: size counts it all.
type tileIndex struct {
	tg     *tier2.TileGeom // geometry only: DropState has run
	spans  []ixSpan        // the tile's data, tile-part by tile-part
	bands  []ixBand        // in tg's component, resolution, band order
	cells  []ixCell        // each band's code-block grid, row-major
	segs   []ixSeg
	charge tier2.Charge // what parsing the tile draws from a decode's budgets
	// err, when not nil, is the tier-2 error of decodes that read a
	// packet of tail, or of every decode when tail is empty (see
	// tier2.IndexTile).
	err  error
	tail []tier2.PacketIndex
	all  []*tileIndex // under PPM, every tile's, by tile index
	size int64
	work int64        // tier-2 work its build was charged
	cut  bool         // its build had less work than a decode may
	refs atomic.Int32 // decodes planned on it: not evicted
}

// tileErr is ix's error for a decode of n tiles. Past the tile's work
// budget alone, the package path, whose budget may be larger, decides.
func tileErr(ix *tileIndex, n int) error {
	if errors.Is(ix.err, tier2.ErrWork) && (n > 1 || ix.cut) {
		return errBigIndex
	}
	return ix.err
}

// bytes is about the heap ix holds.
func (ix *tileIndex) bytes() int64 {
	n := int64(unsafe.Sizeof(*ix)) + int64(len(ix.spans))*int64(unsafe.Sizeof(ixSpan{})) +
		int64(cap(ix.bands))*int64(unsafe.Sizeof(ixBand{})) + int64(len(ix.cells))*cellBytes +
		int64(len(ix.segs))*segBytes + int64(len(ix.tail))*int64(unsafe.Sizeof(tier2.PacketIndex{}))
	if ix.tg != nil {
		n += ix.tg.Size()
	}
	for _, t := range ix.all {
		if t != nil {
			n += t.size
		}
	}
	return n
}

// ixSpan is tile data [td, td+n) at codestream offset off.
type ixSpan struct{ td, off, n int64 }

// ixBand is one band's code-block grid: gw x gh cells of 2^cbw x 2^cbh
// band samples from cell (gx0, gy0), the first at cells[cell0].
type ixBand struct {
	comp, res, sub   int
	rect             codestream.SubbandRect
	cbw, cbh         uint
	gx0, gy0, gw, gh int
	cell0            int
}

// ixCell is a code-block: segs[seg0:seg0+nseg] in layer order, none if
// it is never included.
type ixCell struct {
	seg0 uint32
	nseg uint8
	zbp  uint8
}

// ixSeg is one layer's bytes of one of a code-block's segments: tile
// data [off, off+n).
type ixSeg struct {
	off    int64
	n      uint32
	seg    uint8 // tier-2 SegIndex, under 109
	passes uint8
}

const (
	cellBytes = 8
	segBytes  = 16
)

// indexer builds a tileIndex from tier-2's packets as they are parsed:
// the segments first in parse order, then sorted by code-block.
type indexer struct {
	ix     *tileIndex
	first  [][]int // each resolution's first band
	counts []int32 // segments per cell
	recs   []ixRec
}

// ixRec is a segment of cell, from a packet of layer.
type ixRec struct {
	cell  uint32
	layer uint16
	s     ixSeg
}

// geom sets up the band grids of tg.
func (b *indexer) geom(tg *tier2.TileGeom) error {
	ix := b.ix
	ix.tg = tg
	b.first = make([][]int, len(tg.Comps))
	cells := 0
	for c := range tg.Comps {
		b.first[c] = make([]int, len(tg.Comps[c].Resolutions))
		for r, res := range tg.Comps[c].Resolutions {
			b.first[c][r] = len(ix.bands)
			for _, band := range res.Bands {
				bd := ixBand{comp: c, res: r, sub: band.Subband, rect: band.Rect, cbw: res.CbW, cbh: res.CbH, cell0: cells}
				if !band.Rect.Empty() {
					bd.gx0, bd.gy0 = codestream.FloorDivPow2(band.Rect.X0, res.CbW), codestream.FloorDivPow2(band.Rect.Y0, res.CbH)
					bd.gw = codestream.CeilDivPow2(band.Rect.X1, res.CbW) - bd.gx0
					bd.gh = codestream.CeilDivPow2(band.Rect.Y1, res.CbH) - bd.gy0
				}
				cells += bd.gw * bd.gh
				ix.bands = append(ix.bands, bd)
			}
		}
	}
	if int64(cells) > math.MaxUint32 {
		return fmt.Errorf("jp2k: too many code-blocks to index")
	}
	ix.cells = make([]ixCell, cells)
	b.counts = make([]int32, cells)
	return nil
}

// cellOf is the cell of code-block cb of packet p, or -1.
func (b *indexer) cellOf(p *tier2.PacketRef, cb *tier2.CodeBlockRef) int {
	if p.Component < 0 || p.Component >= len(b.first) || p.Resolution < 0 || p.Resolution >= len(b.first[p.Component]) {
		return -1
	}
	bi := b.first[p.Component][p.Resolution] + max(0, cb.Subband-1)
	if bi >= len(b.ix.bands) {
		return -1
	}
	bd := &b.ix.bands[bi]
	if bd.res != p.Resolution || bd.sub != cb.Subband {
		return -1
	}
	x := codestream.FloorDivPow2(cb.X0, bd.cbw) - bd.gx0
	y := codestream.FloorDivPow2(cb.Y0, bd.cbh) - bd.gy0
	if x < 0 || y < 0 || x >= bd.gw || y >= bd.gh {
		return -1
	}
	return bd.cell0 + y*bd.gw + x
}

// packet takes the segments of p, as groupCodeBlocks gathers them.
func (b *indexer) packet(p *tier2.PacketRef) error {
	for j := range p.Blocks {
		cb := &p.Blocks[j]
		if cb.X1-cb.X0 <= 0 || cb.Y1-cb.Y0 <= 0 {
			continue
		}
		k := b.cellOf(p, cb)
		if k < 0 {
			return fmt.Errorf("jp2k: code-block outside its band grid")
		}
		if b.counts[k] == 0 {
			b.ix.cells[k].zbp = uint8(cb.ZeroBitPlanes)
		}
		b.counts[k] += int32(len(cb.Segments))
		if b.counts[k] > math.MaxUint8 || int64(len(b.recs)) >= math.MaxUint32 {
			return fmt.Errorf("jp2k: code-block has too many segments to index")
		}
		for _, s := range cb.Segments {
			b.recs = append(b.recs, ixRec{uint32(k), uint16(p.Layer), ixSeg{off: int64(s.Start), n: uint32(s.End - s.Start), seg: uint8(s.SegIndex), passes: uint8(s.Passes)}})
		}
	}
	return nil
}

// finish sorts the segments by cell, in layer order within one, stably
// as groupCodeBlocks does: packet order need not be (POC).
func (b *indexer) finish() *tileIndex {
	ix := b.ix
	n := uint32(0)
	for k, c := range b.counts {
		ix.cells[k].seg0 = n
		n += uint32(c)
	}
	b.counts = nil
	ix.segs = make([]ixSeg, len(b.recs))
	layers := make([]uint16, len(b.recs))
	for _, r := range b.recs {
		c := &ix.cells[r.cell]
		at := c.seg0 + uint32(c.nseg)
		ix.segs[at], layers[at] = r.s, r.layer
		c.nseg++
	}
	b.recs = nil
	for k := range ix.cells {
		c := &ix.cells[k]
		l := layers[c.seg0 : c.seg0+uint32(c.nseg)]
		if slices.IsSorted(l) {
			continue
		}
		type ls struct {
			l uint16
			s ixSeg
		}
		tmp := make([]ls, len(l))
		for i := range tmp {
			tmp[i] = ls{l[i], ix.segs[int(c.seg0)+i]}
		}
		slices.SortStableFunc(tmp, func(a, b ls) int { return int(a.l) - int(b.l) })
		for i := range tmp {
			ix.segs[int(c.seg0)+i] = tmp[i].s
		}
	}
	return ix
}

// index returns tile t's index, building it if it has none, and
// whether it did. Under PPM, t is the index of every tile, in all. held
// is the bytes of indexes the decode built and holds unreserved.
func (h *Handle) index(ctx context.Context, t uint16, held int64) (*tileIndex, bool, error) {
	if len(h.cs.PPM) == 0 {
		return h.once(ctx, &h.tiles[t], func(ctx context.Context) (*tileIndex, error) {
			// Budgets of the whole codestream: no decode's are looser.
			// Work: what a decode of the tile alone may spend.
			n := h.hdrLen + 2
			for _, k := range codestream.TilePartIndex(h.cs, t) {
				n += h.partLen[k]
			}
			ix, err := h.build(ctx, t, new(tier2.Cursor), int(min(h.csLen, math.MaxInt)), held, tier2.WorkBudget(int(min(n, math.MaxInt))))
			if ix != nil && h.whole {
				ix.cut = true // past this work, the package decides
			}
			return ix, err
		})
	}
	return h.once(ctx, &h.all, func(ctx context.Context) (*tileIndex, error) { return h.buildAll(ctx, held) })
}

// spend reserves want of the tier-2 work total the file allows its
// index builds for cur, and reports whether it got less; its func
// refunds what cur has left, cancelled work staying spent, and returns
// what cur spent.
func (h *Handle) spend(ctx context.Context, cur *tier2.Cursor, want, total int64) (func() int64, bool, error) {
	grant, err := h.takeWork(ctx, want, total)
	if err != nil {
		return nil, false, err
	}
	cur.WorkSpent = total - grant
	return func() int64 {
		left := grant
		if cur.Started() {
			left = max(0, cur.Work())
		}
		h.mu.Lock()
		h.giveWork(grant, left)
		h.mu.Unlock()
		return grant - left
	}, grant < want, nil
}

// takeWork reserves want of the tier-2 work total the file allows its
// index builds, or what is left if less, waiting while other builds
// hold the difference.
func (h *Handle) takeWork(ctx context.Context, want, total int64) (int64, error) {
	for {
		h.mu.Lock()
		used := h.tier2Work.Load()
		n := min(want, max(0, total-(used-h.workHeld)))
		if n <= total-used {
			h.tier2Work.Add(n)
			h.workHeld += n
			h.mu.Unlock()
			return n, nil
		}
		if h.workFreed == nil {
			h.workFreed = make(chan struct{})
		}
		ch := h.workFreed
		h.mu.Unlock()
		select {
		case <-ch:
		case <-ctx.Done():
			return 0, fmt.Errorf("jp2k: waiting to index a tile: %w", ctx.Err())
		}
	}
}

// giveWork gives back left of n work takeWork reserved; s.mu held.
func (s *ixState) giveWork(n, left int64) {
	s.tier2Work.Add(-left)
	s.workHeld -= n
	if s.workFreed != nil {
		close(s.workFreed)
		s.workFreed = nil
	}
}

// once returns s's index, and whether the decode holds it unkept: it
// builds it unless another decode is building it, then it takes that
// one's result, or builds it itself if that build was cancelled,
// panicked or refused memory for now. A failed build is not kept, so
// the next decode tries again; tier-2 errors are kept, in the index. An index is not kept,
// nor its work charged, past keepMax or the cap keep sets; one past
// keepMax alone is errBigIndex from then on.
func (h *Handle) once(ctx context.Context, s *slot, build func(context.Context) (*tileIndex, error)) (*tileIndex, bool, error) {
	for {
		h.mu.Lock()
		if s.ix != nil || s.big {
			ix := s.ix
			h.mu.Unlock()
			if ix == nil {
				return nil, false, errBigIndex
			}
			return ix, false, nil
		}
		if b := s.building; b != nil {
			h.mu.Unlock()
			select {
			case <-b.done:
				if b.retry {
					continue
				}
				return b.ix, b.err == nil && !b.kept, b.err
			case <-ctx.Done():
				return nil, false, fmt.Errorf("jp2k: waiting for a tile index: %w", ctx.Err())
			}
		}
		b := &building{done: make(chan struct{})}
		s.building = b
		h.mu.Unlock()
		panicked := true
		ix, err := func() (ix *tileIndex, err error) {
			defer parallel.Recover(&ix, &err)
			ix, err = build(ctx)
			panicked = false
			return ix, err
		}()
		h.mu.Lock()
		s.building = nil
		kept := false
		if err == nil {
			kept = h.kept <= h.keepMax()-ix.size && keep(ix.size)
			if kept {
				s.ix = ix
				h.kept += ix.size
				h.keptIx += ix.size
				markKept(h.ixState)
			} else {
				s.big = ix.size > h.keepMax()
				h.giveWork(0, ix.work)
			}
		}
		b.ix, b.err, b.kept = ix, err, kept
		b.retry = err != nil && (ctx.Err() != nil || panicked || errors.Is(err, ErrMemoryLimit))
		h.mu.Unlock()
		close(b.done)
		return ix, err == nil && !kept, err
	}
}

// build parses tile t's packet headers into an index, reading all its
// data once, under the memory budget, which also holds held more bytes.
// cur is the tier-2 cursor, dataLen what its budgets draw on; unless
// work is 0, it draws work of the file's, once the data is read.
func (h *Handle) build(ctx context.Context, t uint16, cur *tier2.Cursor, dataLen int, held, work int64) (ix *tileIndex, err error) {
	defer func() {
		if ix != nil {
			ix.size = ix.bytes()
		}
	}()
	spans, err := tier2.TileSpans(h.cs, t)
	if err != nil {
		return &tileIndex{err: err}, nil
	}
	var total int64
	ss := make([]ixSpan, len(spans))
	for i, sp := range spans {
		ss[i] = ixSpan{total, int64(sp.DataOffset), int64(sp.DataLength)}
		total += int64(sp.DataLength)
	}
	if total > math.MaxInt {
		return nil, fmt.Errorf("jp2k: tile %d too large for this platform", t)
	}
	tk, err := newTicket(ctx)
	if err != nil {
		return nil, err
	}
	defer tk.done()
	est, floor := estimateIndex(h.cs, t, total)
	need, fl := est+held, floor+held
	if collectFor(need) {
		est, floor = estimateIndex(h.cs, t, total)
		need, fl = est+held, floor+held
	}
	// A hung read holds only its bytes.
	if err := tk.grow(total + held); err != nil {
		return nil, err
	}
	data := make([]byte, total)
	for _, s := range ss {
		if err := h.readFull(ctx, data[s.td:s.td+s.n], s.off); err != nil {
			return nil, err
		}
	}
	if err := tk.run(need, fl); err != nil {
		return nil, err
	}
	if work > 0 {
		// Taken after reading: a hung read holds none.
		refund, cut, err := h.spend(ctx, cur, work, tier2.WorkBudget(dataLen))
		if err != nil {
			return nil, err
		}
		defer func() {
			if w := refund(); ix != nil {
				ix.work, ix.cut = w, cut
			}
		}()
	}
	if parallel.FaultHook != nil {
		parallel.FaultHook("index")
	}
	cur.Done = ctx.Done()
	b := &indexer{ix: &tileIndex{spans: ss}}
	tail, charge, err := tier2.IndexTile(h.cs, data, dataLen, t, cur, b.geom, b.packet)
	if errors.Is(err, tier2.ErrStopped) {
		return nil, fmt.Errorf("jp2k: indexing tile %d stopped: %w", t, ctx.Err())
	}
	if b.ix.tg == nil || b.counts == nil {
		return &tileIndex{err: err}, nil // failed before any packet
	}
	b.ix.tg.DropState(len(tail) > 0)
	ix = b.finish()
	ix.err, ix.tail, ix.charge = err, tail, charge
	return ix, nil
}

// buildAll indexes every tile, in the order a decode reads a PPM
// codestream's tiles, with one cursor: PPM is one stream for them all,
// and a decode reads all of it, so draws on these budgets. A tier-2
// error in any tile fails every decode, as it does without an index.
func (h *Handle) buildAll(ctx context.Context, held int64) (ix *tileIndex, err error) {
	order := decodeOrder(h.cs, true, nil, 0)
	dataLen := int(min(h.csLen, math.MaxInt))
	cur := new(tier2.Cursor)
	total := tier2.WorkBudget(dataLen)
	refund, cut, err := h.spend(ctx, cur, total, total)
	if err != nil {
		return nil, err
	}
	defer func() {
		if w := refund(); ix != nil {
			ix.work, ix.cut = w, cut
		}
	}()
	ix = &tileIndex{all: make([]*tileIndex, len(h.tiles))}
	for _, t := range order {
		tix, err := h.build(ctx, t, cur, dataLen, held, 0)
		if err != nil {
			return nil, err
		}
		if tix.err != nil {
			return &tileIndex{err: tix.err}, nil
		}
		ix.all[t] = tix
		held += tix.size
	}
	ix.size = ix.bytes()
	return ix, nil
}

// readFull reads len(b) bytes at codestream offset off, in pieces so ctx
// is checked between them.
func (h *Handle) readFull(ctx context.Context, b []byte, off int64) error {
	if off < 0 || int64(len(b)) > h.csLen-off {
		return fmt.Errorf("jp2k: read of %d bytes at offset %d past the end of the codestream", len(b), off)
	}
	const piece = 8 << 20
	for len(b) > 0 {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("jp2k: reading: %w", err)
		}
		n := min(len(b), piece)
		k, err := h.ra.ReadAt(b[:n], h.base+off)
		if k < n {
			if err == nil || err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return fmt.Errorf("jp2k: reading at offset %d: %w", off, err)
		}
		b, off = b[n:], off+int64(n)
	}
	return nil
}

// tilePlan is what a decode reads of one indexed tile: per band, the
// cells [cx0,cx1)x[cy0,cy1) it wants. Its counts size the reservation.
type tilePlan struct {
	ix     *tileIndex
	tops   []int
	wins   []codestream.SubbandRect
	picks  []pick
	groups int       // code-blocks read
	pieces int       // bound on the byte ranges they are in
	bytes  int64     // their bytes
	data   []byte    // those bytes, once read
	rs     []ixRange // the codestream ranges data holds
}

// ixRange is n codestream bytes at off, at data[at:].
type ixRange struct{ off, n, at int64 }

// pick is the cells of band b a decode reads.
type pick struct{ b, cx0, cy0, cx1, cy1 int }

// cost is the heap loading p holds besides what read holds.
func (p *tilePlan) cost() float64 {
	return float64(costTile + int64(p.groups)*costGroup + int64(p.pieces)*costSegment)
}

// readCost is the heap read of p holds.
func (p *tilePlan) readCost() int64 {
	return p.bytes + int64(p.pieces)*int64(unsafe.Sizeof(ixRange{}))
}

// plan works out what a decode of reg, whose component windows are
// compWin, reads of each tile, building indexes that are missing; the
// codestream bytes that decode would read without an index, for the
// tier-1 work budget; and the bytes of the indexes it built, which the
// decode holds. It stops at the first tile that fails, as a decode
// without an index does, and draws on the tier-2 budgets as that would.
func (h *Handle) plan(ctx context.Context, tiles []uint16, compWin []codestream.SubbandRect, reg region, full bool) (plans []*tilePlan, dataLen int, built int64, release func(), err error) {
	var used []*tileIndex
	unhold := func() {
		for _, ix := range used {
			ix.refs.Add(-1)
		}
	}
	hold := func(ix *tileIndex) {
		ix.refs.Add(1)
		used = append(used, ix)
	}
	defer func() {
		if err != nil {
			unhold()
		}
	}()
	plans = make([]*tilePlan, len(tiles))
	n := h.csLen
	var all *tileIndex
	if len(h.cs.PPM) > 0 {
		ix, fresh, err := h.index(ctx, 0, 0)
		if err != nil {
			return nil, 0, 0, nil, err
		}
		hold(ix)
		if ix.err != nil {
			return nil, 0, 0, nil, tileErr(ix, 1)
		}
		if fresh {
			built = ix.size
		}
		all = ix
	} else if !h.whole {
		n = h.hdrLen + 2
		for _, t := range tiles {
			for _, k := range codestream.TilePartIndex(h.cs, t) {
				n += h.partLen[k]
			}
		}
	}
	dataLen = int(min(n, math.MaxInt))
	cur := new(tier2.Cursor)
	numComps := len(h.cs.SIZ.Components)
	for i, t := range tiles {
		var ix *tileIndex
		if all != nil {
			ix = all.all[t]
		} else {
			var fresh bool
			if ix, fresh, err = h.index(ctx, t, built); err != nil {
				return nil, 0, 0, nil, err
			}
			hold(ix)
			if fresh {
				built += ix.size
			}
		}
		if ix.err != nil && len(ix.tail) == 0 {
			return nil, 0, 0, nil, tileErr(ix, len(tiles))
		}
		var werr error
		if all == nil {
			if err := cur.Replay(h.cs, dataLen, t, ix.charge); errors.Is(err, tier2.ErrWork) {
				werr = errBigIndex // the packets it reads may cost less
			} else if err != nil {
				return nil, 0, 0, nil, err
			}
		}
		tileCOD, tileCOC, _ := codestream.TileOverrides(h.cs, t)
		styles := make([]codestream.CodingStyle, numComps)
		for c := range styles {
			styles[c] = codestream.EffectiveCodingStyle(h.cs, tileCOD, tileCOC, c)
		}
		tops, wins, want, err := tileWindows(ix.tg, t, styles, compWin, reg.reduce, full)
		if err != nil {
			return nil, 0, 0, nil, err
		}
		if ix.err != nil && want.AnyPacket(ix.tg, ix.tail) {
			return nil, 0, 0, nil, tileErr(ix, len(tiles))
		}
		if werr != nil {
			return nil, 0, 0, nil, werr
		}
		p := &tilePlan{ix: ix, tops: tops, wins: wins, pieces: len(ix.spans)}
		for bi := range ix.bands {
			b := &ix.bands[bi]
			comp := b.comp
			var bw codestream.SubbandRect
			switch {
			case b.gw == 0 || b.gh == 0:
				continue
			case want == nil:
				bw = b.rect
			case b.res > want.Top[comp] || want.Wins[comp] != nil && len(want.Wins[comp]) == 0:
				continue
			case want.Wins[comp] == nil:
				bw = b.rect
			default:
				bw, _, _ = tier2.BandWindow(want.Wins[comp], b.res, b.sub)
				bw = bw.Intersect(b.rect)
			}
			if bw.Empty() {
				continue
			}
			pk := pick{bi,
				codestream.FloorDivPow2(bw.X0, b.cbw) - b.gx0, codestream.FloorDivPow2(bw.Y0, b.cbh) - b.gy0,
				codestream.CeilDivPow2(bw.X1, b.cbw) - b.gx0, codestream.CeilDivPow2(bw.Y1, b.cbh) - b.gy0}
			for y := pk.cy0; y < pk.cy1; y++ {
				for x := pk.cx0; x < pk.cx1; x++ {
					c := ix.cells[b.cell0+y*b.gw+x]
					if c.nseg == 0 {
						continue
					}
					p.groups++
					p.pieces += int(c.nseg)
					for _, s := range ix.segs[c.seg0 : c.seg0+uint32(c.nseg)] {
						p.bytes += int64(s.n)
					}
				}
			}
			p.picks = append(p.picks, pk)
		}
		plans[i] = p
	}
	return plans, dataLen, built, unhold, nil
}

// split calls f with the codestream offset and length of each piece of
// tile data [off, off+n), split where tile-parts meet.
func (ix *tileIndex) split(off, n int64, f func(fo, n int64)) {
	i := sort.Search(len(ix.spans), func(i int) bool { return ix.spans[i].td+ix.spans[i].n > off })
	for ; n > 0 && i < len(ix.spans); i++ {
		s := ix.spans[i]
		k := min(n, s.td+s.n-off)
		f(s.off+off-s.td, k)
		off, n = off+k, n-k
	}
}

// read reads plan p's code-block bytes into p.data, coalescing adjacent
// ranges.
func (h *Handle) read(ctx context.Context, p *tilePlan) error {
	ix := p.ix
	// Codestream ranges of every wanted segment, sorted and joined.
	rs := make([]ixRange, 0, p.pieces)
	p.each(func(_ *ixBand, _, _ int, c ixCell) {
		for _, s := range ix.segs[c.seg0 : c.seg0+uint32(c.nseg)] {
			ix.split(s.off, int64(s.n), func(fo, n int64) { rs = append(rs, ixRange{fo, n, 0}) })
		}
	})
	slices.SortFunc(rs, func(a, b ixRange) int { return cmp.Compare(a.off, b.off) })
	j := 0
	var total int64
	for _, r := range rs {
		if j > 0 && rs[j-1].off+rs[j-1].n >= r.off {
			rs[j-1].n = max(rs[j-1].n, r.off+r.n-rs[j-1].off)
			continue
		}
		rs[j] = r
		j++
	}
	rs = rs[:j]
	for i := range rs {
		rs[i].at = total
		total += rs[i].n
	}
	if total > math.MaxInt {
		return fmt.Errorf("jp2k: tile too large for this platform")
	}
	data := make([]byte, total)
	for _, r := range rs {
		if err := h.readFull(ctx, data[r.at:r.at+r.n], r.off); err != nil {
			return err
		}
	}
	p.data, p.rs = data, rs
	return nil
}

// load returns the bytes read of p with each component's code-blocks,
// ready for decodeComponent, and lets p drop them.
func (p *tilePlan) load() ([]byte, [][]*cbGroup) {
	ix, data, rs := p.ix, p.data, p.rs
	p.data, p.rs = nil, nil
	at := func(fo int64) int {
		i := sort.Search(len(rs), func(i int) bool { return rs[i].off+rs[i].n > fo })
		return int(rs[i].at + fo - rs[i].off)
	}
	groups := make([][]*cbGroup, len(ix.tg.Comps))
	gs := make([]cbGroup, 0, p.groups)
	segs := make([]cbSeg, 0, p.pieces)
	p.each(func(b *ixBand, x, y int, c ixCell) {
		x0, y0 := (b.gx0+x)<<b.cbw, (b.gy0+y)<<b.cbh
		g := cbGroup{res: b.res, cb: tier2.CodeBlockRef{
			Subband: b.sub, ZeroBitPlanes: int32(c.zbp),
			X0: max(x0, b.rect.X0), Y0: max(y0, b.rect.Y0),
			X1: min(x0+1<<b.cbw, b.rect.X1), Y1: min(y0+1<<b.cbh, b.rect.Y1),
		}}
		s0 := len(segs)
		for _, s := range ix.segs[c.seg0 : c.seg0+uint32(c.nseg)] {
			if s.n == 0 {
				segs = append(segs, cbSeg{segIndex: int(s.seg), passes: int(s.passes)})
				continue
			}
			passes := int(s.passes)
			// A piece past a tile-part boundary joins the segment's first.
			ix.split(s.off, int64(s.n), func(fo, n int64) {
				a := at(fo)
				segs = append(segs, cbSeg{segIndex: int(s.seg), start: a, end: a + int(n), passes: passes})
				passes = 0
			})
		}
		g.segs = segs[s0:len(segs):len(segs)]
		gs = append(gs, g)
		groups[b.comp] = append(groups[b.comp], &gs[len(gs)-1])
	})
	return data, groups
}

// each calls f for every wanted cell of p that holds bytes, band by
// band: cell (x, y) of band b.
func (p *tilePlan) each(f func(b *ixBand, x, y int, c ixCell)) {
	ix := p.ix
	for _, pk := range p.picks {
		b := &ix.bands[pk.b]
		for y := pk.cy0; y < pk.cy1; y++ {
			for x := pk.cx0; x < pk.cx1; x++ {
				if c := ix.cells[b.cell0+y*b.gw+x]; c.nseg > 0 {
					f(b, x, y, c)
				}
			}
		}
	}
}
