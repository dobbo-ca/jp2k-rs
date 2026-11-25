package tier1

import "math/bits"

// MQ arithmetic decoder, T.800 Annex C.

// qeRow is one row of T.800 Table C-2: the probability estimate for a
// context state plus the state transitions on MPS/LPS exchange.
type qeRow struct {
	qe         uint32
	nmps, nlps uint8
	sw         uint8 // 1 if MPS should flip on an NLPS transition
}

// qeTable is T.800 Table C-2, all 47 states, verified against both the
// pdf.js and OpenJPEG reference implementations.
var qeTable = [47]qeRow{
	{0x5601, 1, 1, 1},
	{0x3401, 2, 6, 0},
	{0x1801, 3, 9, 0},
	{0x0ac1, 4, 12, 0},
	{0x0521, 5, 29, 0},
	{0x0221, 38, 33, 0},
	{0x5601, 7, 6, 1},
	{0x5401, 8, 14, 0},
	{0x4801, 9, 14, 0},
	{0x3801, 10, 14, 0},
	{0x3001, 11, 17, 0},
	{0x2401, 12, 18, 0},
	{0x1c01, 13, 20, 0},
	{0x1601, 29, 21, 0},
	{0x5601, 15, 14, 1},
	{0x5401, 16, 14, 0},
	{0x5101, 17, 15, 0},
	{0x4801, 18, 16, 0},
	{0x3801, 19, 17, 0},
	{0x3401, 20, 18, 0},
	{0x3001, 21, 19, 0},
	{0x2801, 22, 19, 0},
	{0x2401, 23, 20, 0},
	{0x2201, 24, 21, 0},
	{0x1c01, 25, 22, 0},
	{0x1801, 26, 23, 0},
	{0x1601, 27, 24, 0},
	{0x1401, 28, 25, 0},
	{0x1201, 29, 26, 0},
	{0x1101, 30, 27, 0},
	{0x0ac1, 31, 28, 0},
	{0x09c1, 32, 29, 0},
	{0x08a1, 33, 30, 0},
	{0x0521, 34, 31, 0},
	{0x0441, 35, 32, 0},
	{0x02a1, 36, 33, 0},
	{0x0221, 37, 34, 0},
	{0x0141, 38, 35, 0},
	{0x0111, 39, 36, 0},
	{0x0085, 40, 37, 0},
	{0x0049, 41, 38, 0},
	{0x0025, 42, 39, 0},
	{0x0015, 43, 40, 0},
	{0x0009, 44, 41, 0},
	{0x0005, 45, 42, 0},
	{0x0001, 45, 43, 0},
	{0x5601, 46, 46, 0},
}

// mqTab is qeTable indexed by a packed context byte (stateIndex << 1 |
// mps): its Qe and the packed successors after an MPS (n[0]) or an LPS
// (n[1], with the MPS switch applied), indexed by the decision's LPS
// flag so the passes need no branch to pick one. 128 entries so indexing
// by a byte masked to 7 bits needs no bounds check.
type mqEntry struct {
	qe uint32
	n  [2]uint8
}

var mqTab = func() (t [128]mqEntry) {
	for i, r := range qeTable {
		for mps := uint8(0); mps < 2; mps++ {
			t[i<<1|int(mps)] = mqEntry{r.qe, [2]uint8{r.nmps<<1 | mps, r.nlps<<1 | mps ^ r.sw}}
		}
	}
	return
}()

// mqW is mqTab repacked one word per state: qe in bits 0..31, n[0] in
// 32..39, n[1] in 40..47, the state byte itself in 48..55. A decision
// then needs one load (ctxs[cx]) instead of two (ctxs[cx], mqTab[st&127]).
var mqW = func() (t [128]uint64) {
	for s, e := range mqTab {
		t[s] = uint64(e.qe) | uint64(e.n[0])<<32 | uint64(e.n[1])<<40 | uint64(s)<<48
	}
	return
}()

// mqDecoder is the MQ decoder over one code-block's concatenated segment
// bytes. T.800's 32-bit C register (chigh in bits 16..31, clow below) is
// the top half of c; the bottom half buffers bytes read ahead, ct bits
// of them counting down from where BYTEIN would put the next byte. Read
// ahead, a renormalization only has to shift: BYTEIN's additions commute
// with the shifts, and bits carried out the top are the ones a 32-bit C
// would drop.
type mqDecoder struct {
	data []byte
	bp   int // last byte read
	c    uint64
	a    uint32
	ct   uint
}

// at reads data[i], returning 0xFF past the end of the segment. T.800
// C.3.4 requires the decoder to keep feeding 1-bits once the compressed
// data is exhausted; a bounds-checked accessor gets that behaviour for
// free instead of writing a sentinel into (and later restoring) the
// caller's buffer.
func (d *mqDecoder) at(i int) uint64 {
	if i < len(d.data) {
		return uint64(d.data[i])
	}
	return 0xFF
}

// newMQDecoder runs INITDEC (T.800 C.3.5).
func newMQDecoder(data []byte) *mqDecoder {
	d := &mqDecoder{data: data}
	d.c = d.at(0) << 48
	d.byteIn()
	d.c <<= 7
	d.ct -= 7
	d.c, d.ct = d.fill(d.c, d.ct)
	d.a = 0x8000
	return d
}

// byteIn is BYTEIN, T.800 C.3.4, including the 0xFF > 0x8F stuffing
// rule, appending below the ct bits already buffered.
func (d *mqDecoder) byteIn() {
	if d.at(d.bp) == 0xFF {
		if d.at(d.bp+1) > 0x8F {
			d.c += 0xFF << (40 - d.ct)
			d.ct += 8
		} else {
			d.bp++
			d.c += d.at(d.bp) << (41 - d.ct)
			d.ct += 7
		}
	} else {
		d.bp++
		d.c += d.at(d.bp) << (40 - d.ct)
		d.ct += 8
	}
}

// decodeBit is DECODE, T.800 C.3.2, with the C.3.3 renormalization.
// The tier-1 passes open-code this same sequence on register copies of
// a, c and ct (a call would spill every live register); this copy serves
// the cold sites and tests.
func (d *mqDecoder) decodeBit(contexts []uint64, cx int) int {
	st := uint8(contexts[cx] >> 48)
	e := &mqTab[st&127]
	bit := int(st & 1)
	d.a -= e.qe
	if hi := uint64(e.qe) << 48; d.c < hi || d.a&0x8000 == 0 {
		// Conditional exchange: which sub-interval is the LPS depends on
		// their sizes (C.3.2 Figures C.4, C.5).
		lps := d.a < e.qe
		if d.c < hi {
			lps = !lps
			d.a = e.qe
		} else {
			d.c -= hi
		}
		if lps {
			bit ^= 1
			contexts[cx] = mqW[e.n[1]]
		} else {
			contexts[cx] = mqW[e.n[0]]
		}
		n := uint(bits.LeadingZeros32(d.a)-16) & 15
		if n > d.ct {
			d.c, d.ct = d.fill(d.c, d.ct)
		}
		d.a, d.c, d.ct = d.a<<n, d.c<<n, d.ct-n
	} else {
		d.c -= hi
	}
	return bit
}

// rawDecoder is the raw (selective-bypass) bit reader used by SPP/MRP
// coding passes once a code-block's style enables bypass and the current
// bit-plane is deep enough (T.800 Annex C.13 / D.4). It applies the same
// FF-stuffing rule as the MQ decoder, but with no arithmetic coding: each
// bit is read straight from the byte stream.
type rawDecoder struct {
	data []byte
	bp   int
	c    uint32
	ct   int
}

func newRawDecoder(data []byte) *rawDecoder {
	return &rawDecoder{data: data}
}

// at mirrors mqDecoder.at: 0xFF past the segment end, so a raw segment that
// runs out of real bytes keeps feeding the artificial stuffed 0xFF sentinel
// instead of needing a bounds check at every call site.
func (d *rawDecoder) at(i int) uint32 {
	if i < len(d.data) {
		return uint32(d.data[i])
	}
	return 0xFF
}

func (d *rawDecoder) readBit() int {
	if d.ct == 0 {
		if d.c == 0xFF {
			if d.at(d.bp) > 0x8F {
				d.c = 0xFF
				d.ct = 8
			} else {
				d.c = d.at(d.bp)
				d.bp++
				d.ct = 7
			}
		} else {
			d.c = d.at(d.bp)
			d.bp++
			d.ct = 8
		}
	}
	d.ct--
	return int((d.c >> uint(d.ct)) & 1)
}
