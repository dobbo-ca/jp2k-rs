// Package tier1 decodes JPEG 2000 code-blocks: the MQ arithmetic decoder
// and the EBCOT coding passes of ISO/IEC 15444-1 Annexes C and D.
package tier1

import "fmt"

// EBCOT Tier-1 decode, T.800 Annex D, default code-block style only.

// Context slots, T.800 Table D.7 / Annex C.2: 9 zero-coding contexts, 5
// sign-coding, 3 magnitude-refinement, plus run-length and uniform.
const (
	ctxRunLength = 17 // "AGG" in the reference: initial state I=3
	ctxUniform   = 18 // "UNI": initial state I=46
)

// MaxCodeBlockSamples is T.800's implicit cap on code-block area (64x64);
// OpenJPEG asserts it explicitly (t1.c) and so do we, rather than let a
// corrupt COD drive an unbounded allocation below.
const MaxCodeBlockSamples = 4096

// Code-block style bits, T.800 Table A.19 (SPcod/SPcoc byte 4). CbBypass is
// named "LAZY" in some references (OpenJPEG's j2k.h); it is the selective
// arithmetic-coding-bypass bit.
// 0x10 (predictable termination) has no named constant: it changes
// no decode output (it only gates an optional
// diagnostic OpenJPEG itself treats as informational), so this package
// intentionally leaves it unhandled rather than carrying an unused bit.
const (
	CbBypass  = 0x01 // selective arithmetic coding bypass (raw SPP/MRP)
	cbReset   = 0x02 // reset context probabilities at every pass boundary
	CbTermAll = 0x04 // terminate the codeword on every coding pass
	cbCausal  = 0x08 // vertically causal context formation
	CbSegSym  = 0x20 // segmentation symbols at the end of each cleanup pass
	CbHTMask  = 0xC0 // HT block coding, not supported
)

// initialContexts builds the 19-context state array with the T.800
// Table D.7 initial states: ZC context 0 starts at state 4, AGG at 3,
// UNI at 46; every other context (including the other 8 ZC contexts and
// all SC/MR contexts) starts at state 0 with MPS 0.
func initialContexts() []uint64 {
	c := initCtx
	return c[:19]
}

var initCtx = func() (c [32]uint64) {
	for i := range c {
		c[i] = mqW[0]
	}
	c[0], c[ctxRunLength], c[ctxUniform] = mqW[4<<1], mqW[3<<1], mqW[46<<1]
	return
}()

// zcLLLH is the zero-coding context rule for LL/LH subbands (T.800 Table
// D.1), h/v/d the count of significant horizontal/vertical/diagonal
// neighbours.
func zcLLLH(h, v, d int) int {
	switch {
	case h == 2:
		return 8
	case h == 1 && v >= 1:
		return 7
	case h == 1 && d >= 1:
		return 6
	case h == 1:
		return 5
	case v == 2:
		return 4
	case v == 1:
		return 3
	case d >= 2:
		return 2
	case d == 1:
		return 1
	default:
		return 0
	}
}

// zcHH is the zero-coding context rule for the HH subband (T.800 Table
// D.1), keyed by diagonal count and h+v.
func zcHH(d, hv int) int {
	switch {
	case d >= 3:
		return 8
	case d == 2 && hv >= 1:
		return 7
	case d == 2:
		return 6
	case d == 1 && hv >= 2:
		return 5
	case d == 1 && hv == 1:
		return 4
	case d == 1:
		return 3
	case hv >= 2:
		return 2
	case hv == 1:
		return 1
	default:
		return 0
	}
}

// zcContext picks the zero-coding context for a sample given its band
// orientation (0=LL,1=HL,2=LH,3=HH, matching CodeBlockRef.Subband) and
// its neighbour significance counts. HL reuses the LL/LH rule with h and
// v swapped (T.800 Table D.1's HL table is LL/LH transposed).
func zcContext(orientation, h, v, d int) int {
	switch orientation {
	case 1:
		return zcLLLH(v, h, d)
	case 3:
		return zcHH(d, h+v)
	default:
		return zcLLLH(h, v, d)
	}
}

// Tier-1 state is one uint32 per column of a 4-row stripe, OpenJPEG
// 2.x's t1.h layout: one load tells whether a column has anything to do
// in a pass, and a sample's contexts are shifts of that word. Bits 0..17
// are the significance of the column's 3-wide, 6-tall neighbourhood
// (rows -1..4, columns x-1..x+1, row-major), so row ci's 3x3 is bits
// 3ci..3ci+8. The rest hold each row's sign (chi, also for rows -1 and
// 4), mu (refined before) and pi (visited by this bit-plane's SPP).
// Vertically causal mode (0x08) is applied on write: a stripe's row 0
// never marks the stripe above, so that stripe's row 3 never sees it.
const (
	t1Sig    = 1 << 4  // row 0's own significance
	t1SigNb  = 0x1EF   // row 0's 8 neighbours' significance
	t1Chi    = 19      // bit of row 0's sign; row ci's is t1Chi+3ci
	t1Mu     = 1 << 20 // row 0 refined before
	t1Pi     = 1 << 21 // row 0 visited this bit-plane
	t1SigCol = t1Sig | t1Sig<<3 | t1Sig<<6 | t1Sig<<9
	t1PiCol  = t1Pi | t1Pi<<3 | t1Pi<<6 | t1Pi<<9
)

// zcLUT maps orientation and a row's 3x3 significance bits (flags >>
// 3ci & 0x1FF) to the zero-coding context; scLUT maps scIndex to the
// sign-coding context and XOR bit packed as ctx<<1|xor. Both are built
// from the T.800 rules above.
var (
	zcLUT [4][512]uint8
	scLUT [256]uint8
)

func init() {
	for nb := 0; nb < 512; nb++ {
		h := nb>>3&1 + nb>>5&1 // W, E
		v := nb>>1&1 + nb>>7&1 // N, S
		d := nb&1 + nb>>2&1 + nb>>6&1 + nb>>8&1
		for or := 0; or < 4; or++ {
			zcLUT[or][nb] = uint8(zcContext(or, h, v, d))
		}
	}
	// contribOf mirrors the T.800 Table D.3 per-axis contribution of a
	// neighbour pair: both significant and agreeing = +/-1, disagreeing
	// = 0, one significant = its sign, none = 0.
	contribOf := func(sa, na, sb, nb int) int {
		switch {
		case sa != 0 && sb != 0:
			return 1 - na - nb
		case sa != 0:
			return 1 - 2*na
		case sb != 0:
			return 1 - 2*nb
		default:
			return 0
		}
	}
	for lu := 0; lu < 256; lu++ {
		// scIndex bits: 0 W sign, 1 N sig, 2 E sign, 3 W sig, 4 N sign,
		// 5 E sig, 6 S sign, 7 S sig.
		horiz := contribOf(lu>>5&1, lu>>2&1, lu>>3&1, lu&1)
		vert := contribOf(lu>>1&1, lu>>4&1, lu>>7&1, lu>>6&1)
		total := 3*horiz + vert
		ctx, xor := 9+total, 0
		if total < 0 {
			ctx, xor = 9-total, 1
		}
		scLUT[lu] = uint8(ctx<<1 | xor)
	}
}

// scIndex gathers row ci's four direct neighbours' significance and sign
// into scLUT's index; wf and ef are the west and east columns' words.
func scIndex(f, wf, ef uint32, ci uint) uint32 {
	s := 3 * ci & 15
	lu := f>>s&(1<<1|1<<3|1<<5|1<<7) | wf>>((t1Chi+s)&31)&1 | ef>>((t1Chi-2+s)&31)&4 | f>>(t1Chi-3+s)&64
	if ci == 0 {
		lu |= f >> (t1Chi - 5) & 16
	} else {
		lu |= f >> (t1Chi - 7 + s) & 16
	}
	return lu & 0xFF
}

// State is the working flag grid for one code-block: one word per
// stripe column, padded by one column each side and one stripe above and
// below so neighbour updates never need bounds checks.
type State struct {
	w, h  int
	fs    int // flags stride: w + 2
	flags []uint32
	ctxs  [32]uint64 // 19 used; 32 lets a masked index skip bounds checks
}

// col is the flags index of column x of the stripe holding row y.
func (t *State) col(x, y int) int { return (y/4+1)*t.fs + x + 1 }

// reset sizes t for a w x h code-block, reusing its backing array when
// it is already big enough. Each tier-1 worker owns one State.
func (t *State) reset(w, h int) {
	fs := w + 2
	n := fs * ((h+3)/4 + 2)
	if cap(t.flags) < n {
		t.flags = make([]uint32, n)
	} else {
		t.flags = t.flags[:n]
		clear(t.flags)
	}
	t.w, t.h, t.fs = w, h, fs
}

// sig marks row ci of column i significant with sign s (1 = negative):
// in f, the column's word the caller holds, and in the words of the
// columns and stripes around it. vsc is set only for row 0 under
// vertically causal mode.
func (t *State) sig(f uint32, i int, ci uint, s uint32, vsc bool) uint32 {
	fl, fs := t.flags, t.fs
	sh := 3 * ci & 15 // < 32: no shift-overflow check
	fl[i-1] |= 1 << 5 << sh
	fl[i+1] |= 1 << 3 << sh
	if ci == 0 && !vsc {
		n := i - fs
		fl[n] |= s<<31 | 1<<16
		fl[n-1] |= 1 << 17
		fl[n+1] |= 1 << 15
	}
	if ci == 3 {
		n := i + fs
		fl[n] |= s<<18 | 1<<1
		fl[n-1] |= 1 << 2
		fl[n+1] |= 1
	}
	return f | (s<<t1Chi|t1Sig)<<sh
}

// The MQ passes (significance propagation, magnitude refinement and
// cleanup) are generated into tier1_passes.go.
//
//go:generate go run gen_tier1.go

// significancePropagationPassRaw is the selective-bypass (0x01) counterpart
// of significancePropagationPass: the ZC decision is a plain bit and, on a
// hit, the sign is a second plain bit with no sign-prediction XOR (T.800
// D.4 / JJ2000 rawSigProgPass).
func (t *State) significancePropagationPassRaw(raw *rawDecoder, out []int32, plane int, vsc bool) {
	fl, w := t.flags, t.w
	one := int32(1) << uint(plane)
	for y0 := 0; y0 < t.h; y0 += 4 {
		rows := uint(min(4, t.h-y0))
		i := t.col(0, y0)
		for x := 0; x < w; x, i = x+1, i+1 {
			f := fl[i]
			if f == 0 {
				continue
			}
			for ci := uint(0); ci < rows; ci++ {
				sh := 3 * ci & 15 // < 32: no shift-overflow check
				if f>>sh&(t1Sig|t1Pi) != 0 || f>>sh&t1SigNb == 0 {
					continue
				}
				if raw.readBit() == 1 {
					f = t.sig(f, i, ci, uint32(raw.readBit()), vsc && ci == 0)
					out[(y0+int(ci))*w+x] |= one
				}
				f |= t1Pi << sh
			}
			fl[i] = f
		}
	}
}

// magnitudeRefinementPassRaw is the selective-bypass counterpart: one raw
// bit per refined sample, no context. It still sets mu, matching
// OpenJPEG's opj_t1_dec_refpass_step_raw: if a later MQ MR pass ever
// refines a sample that already went through a raw MR pass, it must pick
// context 16, not mistake it for that sample's first refinement.
func (t *State) magnitudeRefinementPassRaw(raw *rawDecoder, out []int32, plane int) {
	fl, w := t.flags, t.w
	for y0 := 0; y0 < t.h; y0 += 4 {
		rows := uint(min(4, t.h-y0))
		i := t.col(0, y0)
		for x := 0; x < w; x, i = x+1, i+1 {
			f := fl[i]
			if f&t1SigCol == 0 {
				continue
			}
			for ci := uint(0); ci < rows; ci++ {
				sh := 3 * ci & 15 // < 32: no shift-overflow check
				if f>>sh&(t1Sig|t1Pi) != t1Sig {
					continue
				}
				out[(y0+int(ci))*w+x] |= int32(raw.readBit()) << (uint(plane) & 31)
				f |= t1Mu << sh
			}
			fl[i] = f
		}
	}
}

// Segment is one termination segment's coded bytes and the number of
// coding passes it carries, in the order tier-2 reported them (layer
// order, then segment order within a layer). Style 0 (and TERMALL/LAZY off)
// always yields exactly one segment covering the whole code-block; the
// other style bits can split it into several, one per MQ/raw run.
type Segment struct {
	Data   []byte
	Passes int
}

// DecodeCodeBlock EBCOT-decodes one code-block's coding-pass segments into
// sign-magnitude int32 coefficients: bit 31 is the sign (1 = negative),
// bits 0..30 are the magnitude, already left-aligned to bit-plane 0 (no
// further shift needed by the caller). orientation is the subband's band
// index (0=LL,1=HL,2=LH,3=HH, matching CodeBlockRef.Subband). mb is the
// subband's Mb = guard bits + quantization exponent - 1 (T.800 E.1);
// zeroBitPlanes comes from the Tier-2 packet header, segs from tier-2's
// segmentation of the code-block's coding passes across every including
// layer. style is the COD/COC code-block style byte (T.800 Table A.19);
// HT block coding (0x40/0x80) is not supported.
//
// The second return value, q, is one undecoded-low-bit-plane count per
// coefficient (parallel to the first return value; 0 for a coefficient
// that was never decoded significant or whose passes ran all the way to
// bit-plane 0): quant.go's Dequantize* need it to reproduce T.800's r=0.5
// midpoint reconstruction for truncated code-blocks. This must be tracked
// per coefficient, not once for the whole block: a code-block whose pass
// budget ends right after a significance-propagation pass leaves samples
// that were already significant one bit-plane short of samples that just
// became significant in that pass (see the comment above the q
// computation below).
func DecodeCodeBlock(segs []Segment, width, height, orientation int, zeroBitPlanes int32, mb int, style uint8) ([]int32, []int32, error) {
	if err := ValidateCodeBlockDims(width, height, style); err != nil {
		return nil, nil, err
	}
	out := make([]int32, width*height)
	qs := make([]int32, width*height)
	t := new(State)
	if err := decodeCodeBlockInto(t, out, qs, segs, width, height, orientation, zeroBitPlanes, mb, style); err != nil {
		return nil, nil, err
	}
	return out, qs, nil
}

// ValidateCodeBlockDims is DecodeCodeBlock's dimension/style guard, split
// out so decodeComponent can run it before slicing its worker buffers sized
// MaxCodeBlockSamples (which would slice out of bounds on an oversized
// w*h otherwise, e.g. from a fuzzed/corrupt COD/COC).
func ValidateCodeBlockDims(width, height int, style uint8) error {
	if style&CbHTMask != 0 {
		return fmt.Errorf("jp2k: code-block uses unsupported HT block coding (style 0x%02x)", style)
	}
	if width <= 0 || height <= 0 {
		return fmt.Errorf("jp2k: code-block has non-positive dimensions %dx%d", width, height)
	}
	if width*height > MaxCodeBlockSamples {
		return fmt.Errorf("jp2k: code-block %dx%d exceeds the T.800 maximum of %d samples", width, height, MaxCodeBlockSamples)
	}
	return nil
}

// decodeCodeBlockInto is DecodeCodeBlock's allocation-free core: out and qs
// must already be sized width*height (validated by the caller, matching
// DecodeCodeBlock's guards above), and are overwritten in place rather than
// freshly allocated. decodeComponent's workers call this directly with
// their own t, out and qs.
func decodeCodeBlockInto(t *State, out, qs []int32, segs []Segment, width, height, orientation int, zeroBitPlanes int32, mb int, style uint8) error {
	return t.Finish(out, qs, nil, 0, segs, width, height, orientation, zeroBitPlanes, mb, style)
}

// Finish decodes like decodeCodeBlockInto, then writes each coefficient
// one of three ways: with outF nil and delta 0, sign-magnitude into out
// and q into qs; with qs nil and outF nil, the reversible coefficient
// (dequantizeReversibleInto's) into out; with qs nil, the irreversible
// one (dequantizeIrreversibleInto's with step delta) into outF, out then
// holding scratch. Fusing the dequantization into the walk over
// significant samples saves a pass over the whole block.
func (t *State) Finish(out, qs []int32, outF []float32, delta float64, segs []Segment, width, height, orientation int, zeroBitPlanes int32, mb int, style uint8) error {
	passType, plane, empty, err := t.decode(out, segs, width, height, orientation, zeroBitPlanes, mb, style)
	if err != nil {
		return err
	}
	if qs != nil {
		clear(qs)
	}
	if empty {
		clear(out) // no coded bit-planes: an all-zero block, no bias
		clear(outF)
		return nil
	}
	// plane holds the last bit-plane index whose passes never fully ran
	// (-1 if every plane down to 0 was fully decoded): the count of
	// undecoded low bit-planes is normally that index plus one. Two
	// exceptions need a per-sample count instead of one shared by the
	// whole block:
	//
	//   - passType==2 (cleanup pending): significance-propagation and
	//     magnitude-refinement both ran for `plane`, so every currently
	//     significant sample already has its `plane` bit decided, whether
	//     it was significant from an earlier plane and refined here, or
	//     newly significant this plane via SPP. Only cleanup-only
	//     candidates are still unresolved, and those stay at magnitude 0
	//     (DequantizeReversible/Irreversible's mag==0 guard).
	//   - passType==1 (magnitude-refinement pending): SPP ran for
	//     `plane` but MR did not. A sample that just became significant
	//     in that SPP has its `plane` bit decided (that is what
	//     significance means), so only bits below `plane` are undecided:
	//     q=plane. A sample that was already significant before `plane`
	//     missed its MR visit for `plane`, so that bit is still
	//     undecided too: q=plane+1. pi distinguishes the two: the last
	//     cleanup cleared it and that SPP set it on every sample it
	//     visited.
	//
	// passType==0 (SPP pending) needs no per-sample split: cleanup for
	// `plane+1` was the last pass to run, so every significant sample's
	// last decided bit is `plane+1` uniformly.
	// q of a sample whose pi is clear, and of one whose pi is set.
	qNo, qPi := plane+1, plane+1
	switch passType {
	case 2:
		qNo, qPi = plane, plane
	case 1:
		qPi = plane
	}
	// dequantizeReversibleInto: 0.5 truncates to 0.
	revBias := func(q int) int32 {
		if q > 0 && q < 32 {
			return 1 << (q - 1)
		}
		return 0
	}
	// dequantizeIrreversibleInto's arithmetic exactly.
	irrBias := func(q int) float32 {
		if q > 0 {
			return float32(uint64(1)<<(uint(q)&63)) * 0.5
		}
		return 0.5
	}
	rb := [2]int32{revBias(qNo), revBias(qPi)}
	fb := [2]float32{irrBias(qNo), irrBias(qPi)}
	d := float32(delta)
	fl := t.flags
	if qs != nil {
		for y0 := 0; y0 < height; y0 += 4 {
			rows := uint(min(4, height-y0))
			i := t.col(0, y0)
			for x := 0; x < width; x, i = x+1, i+1 {
				f := fl[i]
				for ci := uint(0); ci < rows; ci++ {
					sh := 3 * ci & 15 // < 32: no shift-overflow check
					if f>>sh&t1Sig == 0 {
						continue
					}
					o := (y0+int(ci))*width + x
					out[o] = int32(uint32(out[o]) | f>>(t1Chi+sh)&1<<31)
					q := qNo
					if f>>sh&t1Pi != 0 {
						q = qPi
					}
					qs[o] = int32(max(q, 0))
				}
			}
		}
		return nil
	}
	// Row by row and branch-free: whether a sample is significant is as
	// unpredictable as its sign, and insignificant ones come out 0. Bit
	// 4 of a row's shifted flags is t1Sig, bit 21 t1Pi.
	for y := 0; y < height; y++ {
		sh := uint(3 * (y & 3))
		frow := fl[t.col(0, y):][:width]
		orow := out[y*width:][:width]
		if outF == nil {
			for x, f := range frow {
				f >>= sh
				m := -int32(f >> t1Chi & 1)
				orow[x] = ((orow[x] + rb[f>>21&1]) ^ m - m) & -int32(f>>4&1)
			}
			continue
		}
		frow2 := outF[y*width:][:width]
		dequantRow(frow2, orow, frow, sh, fb, d)
	}
	return nil
}

// decode runs the coding passes for one code-block, ORing magnitude bits
// into out and leaving significance, sign and pi in t.flags. It returns
// the pass type and bit-plane it stopped at, or empty for a block with no
// coded bit-planes (out and t.flags then untouched).
func (t *State) decode(out []int32, segs []Segment, width, height, orientation int, zeroBitPlanes int32, mb int, style uint8) (passType, plane int, empty bool, err error) {
	numbps := mb - int(zeroBitPlanes)
	totalPasses := 0
	for _, s := range segs {
		totalPasses += s.Passes
	}
	if numbps <= 0 || totalPasses <= 0 {
		return 0, 0, true, nil
	}
	if numbps > 31 {
		return 0, 0, false, fmt.Errorf("jp2k: code-block bit-plane count %d exceeds int32 sign-magnitude capacity", numbps)
	}
	clear(out) // the passes OR magnitude bits in

	t.reset(width, height)
	t.ctxs = initCtx

	bypass := style&CbBypass != 0
	reset := style&cbReset != 0
	vsc := style&cbCausal != 0
	segsym := style&CbSegSym != 0

	plane = numbps - 1
	passType = 2 // T.800 D.2: the first bit-plane starts with a cleanup pass
segLoop:
	for _, seg := range segs {
		if plane < 0 {
			break
		}
		// T.800 D.4 / OpenJPEG t1.c: bypass only ever applies to SPP/MRP
		// (passType < 2) once the bit-plane is at least the 5th coded one;
		// the cleanup pass always stays MQ. Tier-2's segmentation (initSeg)
		// only ever splits segments at these exact transitions, so this
		// holds for every pass in the segment, not just its first.
		isRaw := bypass && plane+1 <= numbps-4 && passType < 2
		var mq *mqDecoder
		var raw *rawDecoder
		if isRaw {
			raw = newRawDecoder(seg.Data)
		} else {
			mq = newMQDecoder(seg.Data)
		}
		for p := 0; p < seg.Passes; p++ {
			if plane < 0 {
				break segLoop
			}
			if isRaw && passType == 2 {
				// Raw segments have no mq here.
				return 0, 0, false, fmt.Errorf("jp2k: raw segment reached a cleanup pass")
			}
			switch passType {
			case 0:
				if isRaw {
					t.significancePropagationPassRaw(raw, out, plane, vsc)
				} else {
					t.significancePropagationPass(mq, out, orientation, plane, vsc)
				}
			case 1:
				if isRaw {
					t.magnitudeRefinementPassRaw(raw, out, plane)
				} else {
					t.magnitudeRefinementPass(mq, out, plane)
				}
			case 2:
				// Segsym mismatch: decode anyway, don't fail.
				t.cleanupPass(mq, out, orientation, plane, vsc, segsym)
			}
			if reset && !isRaw {
				// T.800 Table A.19 0x02: reset context probabilities after
				// every pass (raw passes have no contexts to reset).
				t.ctxs = initCtx
			}
			passType++
			if passType == 3 {
				passType = 0
				plane--
			}
		}
	}
	return passType, plane, false, nil
}
