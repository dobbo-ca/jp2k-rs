//go:build !arm64

package dsp

import "math"

// fma32 inlines the common case: one float64 rounding is final unless s
// may be a float32 midpoint, the only place double rounding can differ.
func fma32(a, b, c float32) float32 {
	s := float64(a)*float64(b) + float64(c)
	f := float32(s)
	if math.Float64bits(s)&(1<<28-1) == 0 && float64(f) != s {
		return fma32Soft(a, b, c)
	}
	return f
}
