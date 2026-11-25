package dsp

import "math"

// The 9/7 lift and the ICT must round once per multiply-add, as
// opj_decompress does where its compiler contracts them (arm64), so the
// output is the same bytes on every arch. fma32 is per arch
// (fma_*.go); fma32Soft is the portable, correctly rounded fallback.

// fma32Soft returns a*b+c rounded once to float32. The product is exact
// in float64; float64() only stops arm64 fusing it. The sum is
// rounded to odd in float64 so that the final float32 conversion cannot
// double-round (53 >= 24+2 bits).
func fma32Soft(a, b, c float32) float32 {
	xy, z := float64(float64(a)*float64(b)), float64(c)
	s := xy + z
	// TwoSum: e is the exact rounding error of s.
	t := s - xy
	e := (xy - (s - t)) + (z - t)
	if e > 0 || e < 0 { // false for NaN, so Inf and NaN pass through
		// Truncate toward zero, then set the sticky bit.
		u := math.Float64bits(s)
		s = math.Float64frombits((u - (u^math.Float64bits(e))>>63) | 1)
	}
	return float32(s)
}
