//go:build (!arm64 && !amd64) || purego

package tier1

import "math"

// dequantRow is the Go loop where no SIMD kernel exists.
func dequantRow(dst []float32, mag []int32, flags []uint32, sh uint, fb [2]float32, d float32) {
	for x, f := range flags {
		f >>= sh
		v := (float32(mag[x]) + fb[f>>21&1]) * d
		dst[x] = math.Float32frombits((math.Float32bits(v) ^ f>>t1Chi&1<<31) & -(f >> 4 & 1))
	}
}
