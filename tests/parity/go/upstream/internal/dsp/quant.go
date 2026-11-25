package dsp

import (
	"fmt"
	"math"

	"github.com/dobbo-ca/jp2k-go/internal/codestream"
)

// T.800 Annex E: dequantization.

// SubbandType identifies a subband's orientation for gain/dequant
// purposes, matching CodeBlockRef.Subband's 0=LL,1=HL,2=LH,3=HH convention.
type SubbandType int

const (
	SubbandLL SubbandType = 0
	SubbandHL SubbandType = 1
	SubbandLH SubbandType = 2
	SubbandHH SubbandType = 3
)

// bandIndex returns T.800 Annex E's flat band index: 0 for the LL-only
// resolution (r=0), otherwise resolution-major HL,LH,HH per level above it.
// r=0 only ever holds an LL band and r>0 never does; either mismatch is a
// caller bug, not a value bandIndex can silently paper over.
func bandIndex(r int, orientation SubbandType) (int, error) {
	if r == 0 {
		if orientation != SubbandLL {
			return 0, fmt.Errorf("jp2k: resolution 0 has no orientation %d band (LL only)", orientation)
		}
		return 0, nil
	}
	if orientation == SubbandLL {
		return 0, fmt.Errorf("jp2k: resolution %d has no LL band (LL only exists at resolution 0)", r)
	}
	return 1 + 3*(r-1) + (int(orientation) - 1), nil
}

// StepSize returns Annex E's dequantization step size delta_b (E.1.1.1)
// and Mb, the total bit-plane count (E.2), for the subband at resolution r
// (0 = the LL-only resolution) and orientation. q is the component's
// QCD/QCC quantization segment; precision is the component's bit depth
// (image header Ssiz & 0x7f, +1). reversible is the COD/COC qmfbid (true
// for the 5/3 transform): like OpenJPEG (t1.c:1798), StepSize keys the
// "no step size" decision on the transform, not on q.QuantizationStyle —
// a conforming reversible stream always pairs qmfbid=1 with style 0, but
// nothing stops a malformed stream from combining qmfbid=0 (9/7) with
// style 0, and that combination still needs a real delta rather than
// silently decoding to a flat field. For a reversible transform delta
// carries no meaning and is returned as 0 — use DequantizeReversible
// instead of DequantizeIrreversible there.
func StepSize(q codestream.Quantization, r int, orientation SubbandType, precision int, reversible bool) (delta float64, mb int, err error) {
	b, err := bandIndex(r, orientation)
	if err != nil {
		return 0, 0, err
	}
	var expn uint8
	var mant uint16
	switch q.QuantizationStyle {
	case 1: // scalar derived, T.800 E-5: every band's exponent derives from band 0's
		if len(q.Exponents) < 1 {
			return 0, 0, fmt.Errorf("jp2k: scalar-derived quantization has no exponent for band 0")
		}
		e0 := q.Exponents[0]
		if b == 0 {
			expn = e0
		} else {
			d := int(e0) - (b-1)/3
			if d < 0 {
				d = 0
			}
			expn = uint8(d)
		}
		if len(q.Mantissas) < 1 {
			return 0, 0, fmt.Errorf("jp2k: scalar-derived quantization has no mantissa for band 0")
		}
		mant = q.Mantissas[0]
	default: // 0 (reversible) and 2 (expounded): one exponent per band already
		if b < 0 || b >= len(q.Exponents) {
			return 0, 0, fmt.Errorf("jp2k: quantization segment has no band %d (resolution %d, orientation %d) among its %d bands", b, r, orientation, len(q.Exponents))
		}
		expn = q.Exponents[b]
		if q.QuantizationStyle == 2 {
			mant = q.Mantissas[b]
		}
	}
	mb = int(expn) + int(q.GuardBits) - 1
	if reversible {
		return 0, mb, nil
	}
	// Rb = precision, with no per-subband gain term: this deliberately
	// diverges from T.800 E-3's Rb = precision + gain(band) to match
	// OpenJPEG's decode-only convention (opj_tcd_init_tile,
	// BUG_WEIRD_TWO_INVK), which drops the gain here and folds it into
	// the inverse DWT's high-band scale (dwt.go's twoInvK) instead. The
	// two diverge individually but cancel identically to the standard
	// split; matching OpenJPEG's choice makes this package's decode
	// output bit-comparable with opj_decompress's.
	rb := precision
	// math.Exp2 rather than math.Pow(2, x): rb-expn can be negative and
	// this stays exact where Pow would round.
	delta = (1 + float64(float64(mant)/2048)) * math.Exp2(float64(rb-int(expn)))
	return delta, mb, nil
}

// midpointBias returns T.800's r=0.5 reconstruction bias, 0.5*2^q, for a
// coefficient whose q lowest bit-planes were never decoded (q=0 for a
// coefficient whose passes ran all the way to bit-plane 0). This is a
// per-coefficient quantity, not a per-code-block one: a code-block whose
// pass budget ends right after a significance-propagation pass leaves
// already-significant samples one bit-plane short of samples that just
// became significant in that pass, so DecodeCodeBlock reports q per
// coefficient (see its doc comment).
func midpointBias(q int) float64 {
	if q <= 0 {
		return 0.5
	}
	return 0.5 * math.Exp2(float64(q))
}

// DequantizeReversible unpacks DecodeCodeBlock's sign-magnitude int32
// coefficients into signed integers, applying the midpoint bias (see
// midpointBias) to every coefficient that was ever decoded significant.
// T.800 Annex E style 0 applies no step-size scaling: the magnitude,
// already bit-plane-0 aligned by tier-1, is the coefficient value itself
// up to that bias. q is DecodeCodeBlock's per-coefficient
// undecoded-low-bit-plane count, parallel to coeffs.
func DequantizeReversible(coeffs []int32, q []int32) []int32 {
	return dequantizeReversibleInto(make([]int32, len(coeffs)), coeffs, q)
}

// dequantizeReversibleInto is DequantizeReversible writing into out, which
// it fully overwrites.
func dequantizeReversibleInto(out, coeffs []int32, q []int32) []int32 {
	out = out[:len(coeffs)]
	for i, c := range coeffs {
		u := uint32(c)
		mag := int32(u &^ (1 << 31))
		if mag == 0 { // never significant: no bias, coefficient stays 0
			out[i] = 0
			continue
		}
		// midpointBias in integers: 0.5 truncates to 0, and q never
		// exceeds tier-1's 31 bit-planes.
		if b := q[i]; b > 0 && b < 32 {
			mag += 1 << (b - 1)
		}
		if u&(1<<31) != 0 {
			mag = -mag
		}
		out[i] = mag
	}
	return out
}

// DequantizeIrreversible unpacks sign-magnitude coefficients and applies
// Annex E's scalar step size with the T.800 r=0.5 midpoint reconstruction:
// value = (magnitude + 0.5*2^q) * delta (see midpointBias). A coefficient
// that was never decoded significant (magnitude 0) gets no bias: it stays
// exactly 0, matching OpenJPEG t1.c:1859's zero-datap multiply and pdf.js's
// `if (n !== 0)` guard. q is DecodeCodeBlock's per-coefficient
// undecoded-low-bit-plane count, parallel to coeffs.
func DequantizeIrreversible(coeffs []int32, delta float64, q []int32) []float32 {
	return dequantizeIrreversibleInto(make([]float32, len(coeffs)), coeffs, delta, q)
}

// dequantizeIrreversibleInto is DequantizeIrreversible with a caller-owned
// destination (out[:len(coeffs)] is fully overwritten, including the mag==0
// case, so a pool-reused out carries none of its previous contents), used
// by decodeComponent to write straight into its plane.
func dequantizeIrreversibleInto(out []float32, coeffs []int32, delta float64, q []int32) []float32 {
	out = out[:len(coeffs)]
	d := float32(delta)
	for i, c := range coeffs {
		u := uint32(c)
		mag := float32(u &^ (1 << 31))
		if mag == 0 {
			out[i] = 0
			continue
		}
		v := (mag + float32(midpointBias(int(q[i])))) * d
		if u&(1<<31) != 0 {
			v = -v
		}
		out[i] = v
	}
	return out
}

// dequantFGo is tier1.State.Finish's irreversible per-sample dequant,
// flags already shifted by sh: bit 21 selects bNo/bPi, bit 19 is the
// sign, bit 4 gates significance (0 for insignificant samples).
func dequantFGo(dst []float32, mag []int32, flags []uint32, sh uint, bNo, bPi, d float32) {
	fb := [2]float32{bNo, bPi}
	for x, f := range flags {
		f >>= sh
		v := (float32(mag[x]) + fb[f>>21&1]) * d
		dst[x] = math.Float32frombits((math.Float32bits(v) ^ f>>19&1<<31) & -(f >> 4 & 1))
	}
}
