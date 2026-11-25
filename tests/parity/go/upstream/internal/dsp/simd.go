//go:build (amd64 || arm64) && !purego

package dsp

import "os"

// noSIMD makes every kernel use its Go version, for hosts where the
// assembly is suspect.
var noSIMD = os.Getenv("JP2K_NOSIMD") != ""

func pickKernel[F any](simd, goKernel F) F {
	if noSIMD {
		return goKernel
	}
	return simd
}

// liftNCheck bounds-checks a liftN call up front, so the SIMD kernels
// below can index through raw pointers safely.
func liftNCheck(dst, d, u, v []float32, n, rows, sDst, sD, sUV int) {
	if rows <= 0 || n <= 0 {
		return
	}
	_, _ = dst[(rows-1)*sDst+n-1], d[(rows-1)*sD+n-1]
	_, _ = u[(rows-1)*sUV+n-1], v[(rows-1)*sUV+n-1]
}
