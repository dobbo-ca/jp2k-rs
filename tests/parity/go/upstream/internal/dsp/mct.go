package dsp

import "math"

// T.800 Annex G: multiple component transform, DC level shift, and the
// final clamp to the component's declared precision. These run, in this
// order, after the inverse DWT (Annex F) for each component.

// InverseRCT applies T.800 G.2's reversible (5/3) inverse color transform
// to components 0, 1, 2 in place, undoing the encoder's forward RCT.
func InverseRCT(c0, c1, c2 []int32) {
	for i := range c0 {
		y, u, v := c0[i], c1[i], c2[i]
		g := y - ((u + v) >> 2)
		c0[i] = v + g
		c1[i] = g
		c2[i] = u + g
	}
}

// ICT coefficients: OpenJPEG's and pdf.js's five-significant-figure
// literals, not T.800 G.3's longer constants, so results stay within a
// float32 ULP of opj_decompress. Operand order (left to right) matters in
// float32 and is kept as both reference implementations have it.
const (
	ictCrR = 1.402
	ictCbG = 0.34413
	ictCrG = 0.71414
	ictCbB = 1.772
)

// InverseICT applies T.800 G.3's irreversible (9/7) inverse color
// transform to components 0, 1, 2 in place, rounding as r says.
func InverseICT(c0, c1, c2 []float32, r Rounding) { ict(c0, c1, c2, r) }

// ictGo is the reference InverseICT; ict (per arch) must match it bit
// for bit.
func ictGo(c0, c1, c2 []float32, r Rounding) {
	for i := range c0 {
		y, u, v := c0[i], c1[i], c2[i]
		c0[i] = r.mulAdd(v, ictCrR, y)
		c1[i] = r.mulAdd(-v, ictCrG, r.mulAdd(-u, ictCbG, y))
		c2[i] = r.mulAdd(u, ictCbB, y)
	}
}

// ClampToPrecision clamps v to the range a component of the given
// precision and signedness can hold: [0, 2^precision - 1] unsigned,
// [-2^(precision-1), 2^(precision-1) - 1] signed.
func ClampToPrecision(v, precision int32, signed bool) int32 {
	var lo, hi int32
	if signed {
		lo = -(int32(1) << uint(precision-1))
		hi = (int32(1) << uint(precision-1)) - 1
	} else {
		lo = 0
		hi = (int32(1) << uint(precision)) - 1
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// DCLevelShift is T.800's decoder-side DC shift, 2^(precision-1) for
// unsigned components and 0 for signed ones (the encoder shifted unsigned
// samples down before coding; nothing to undo for signed samples).
func DCLevelShift(precision int32, signed bool) int32 {
	if signed {
		return 0
	}
	return int32(1) << uint(precision-1)
}

// SamplesR DC-shifts and clamps a row of reversible (5/3) samples into
// dst. A signed component's true range is then biased up by
// 2^(precision-1) so every component lands in [0, 2^precision), the range
// outImage expects.
func SamplesR(dst, src []int32, precision int32, signed bool) {
	shift := DCLevelShift(precision, signed)
	var bias int32
	if signed {
		bias = int32(1) << uint(precision-1)
	}
	for i, v := range src {
		dst[i] = ClampToPrecision(v+shift, precision, signed) + bias
	}
}

// SamplesF is SamplesR's irreversible (9/7) counterpart: it rounds each
// float32 sample to the nearest integer (round-half-to-even, matching
// lrintf as used by opj_decompress) before shifting and clamping. The
// out-of-int32-range guard mirrors OpenJPEG's tcd.c dc_level_shift_decode.
// Signedness drops out: a signed sample clamped to its signed range and
// then biased by 2^(precision-1) equals the unsigned shift-and-clamp, since
// this works in int64 where neither can overflow.
func SamplesF(dst []int32, src []float32, precision int32) {
	for i, v := range src {
		dst[i] = sampleF(v, precision)
	}
}

func sampleF(v float32, precision int32) int32 {
	shift := int64(1) << uint(precision-1)
	hi := (int64(1) << uint(precision)) - 1
	f := float64(v)
	var iv int64
	switch {
	case f > math.MaxInt32:
		iv = math.MaxInt32
	case f < math.MinInt32 || f != f: // NaN as amd64 converts it
		iv = math.MinInt32
	default:
		iv = int64(math.RoundToEven(f))
	}
	iv += shift
	if iv < 0 {
		iv = 0
	} else if iv > hi {
		iv = hi
	}
	return int32(iv)
}

// packF16Go is SamplesF then outImage's 16-bit store, fused: sample i
// goes big-endian to dst[i*bpp:]. PackF16 (per arch) must match it bit
// for bit. For 9..16-bit precision.
func packF16Go(dst []byte, bpp int, src []float32, precision int32) {
	for i, v := range src {
		Put16BE(dst[i*bpp:], Expand16(uint32(sampleF(v, precision)), uint8(precision)))
	}
}

// packP16Go is SamplesF then a 16-bit store with no Expand16, for
// precision below 16 (and 16, where it is the same). PackP16 (per arch)
// must match it bit for bit.
func packP16Go(dst []byte, src []float32, precision int32) {
	for i, v := range src {
		Put16BE(dst[2*i:], uint16(sampleF(v, precision)))
	}
}

// Put16BE writes v big-endian into p[0:2], the byte order image.Pix uses
// for every 16-bit-per-channel image type.
func Put16BE(p []byte, v uint16) {
	p[0] = byte(v >> 8)
	p[1] = byte(v)
}

// Expand16 maps a precision-bit sample to the full 16-bit range by bit
// replication (v<<shift | v>>(2*precision-16)) rather than left-shift
// alone, so the component's maximum value lands on 0xffff instead of
// falling shift bits short of it. precision can be anywhere in [1,16]
// (classifyOutput routes any non-8-bit precision <= 16 here); for
// precision < 8, 2*precision-16 would be negative, so the low bits are
// replicated a full v<<shift at a time until the register is filled.
func Expand16(v uint32, precision uint8) uint16 {
	if precision == 16 {
		return uint16(v)
	}
	out := v
	filled := uint(precision)
	for filled < 16 {
		out = out<<uint(precision) | v
		filled += uint(precision)
	}
	return uint16(out >> (filled - 16))
}
