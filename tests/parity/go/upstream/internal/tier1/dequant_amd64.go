//go:build amd64 && !purego

package tier1

import "github.com/dobbo-ca/jp2k-go/internal/dsp"

// dequantRow calls dsp.DequantF directly: it is a plain function on
// amd64, not a func variable, which cost 1-3% as an indirect call.
func dequantRow(dst []float32, mag []int32, flags []uint32, sh uint, fb [2]float32, d float32) {
	dsp.DequantF(dst, mag, flags, sh, fb[0], fb[1], d)
}
