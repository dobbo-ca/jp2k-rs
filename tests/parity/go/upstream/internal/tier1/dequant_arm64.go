//go:build arm64 && !purego

package tier1

import "github.com/dobbo-ca/jp2k-go/internal/dsp"

// dequantRow dispatches to the NEON kernel; purego stays on the plain Go
// loop, see dequant_other.go.
func dequantRow(dst []float32, mag []int32, flags []uint32, sh uint, fb [2]float32, d float32) {
	dsp.DequantF(dst, mag, flags, sh, fb[0], fb[1], d)
}
