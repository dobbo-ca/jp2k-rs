//go:build (!arm64 && !amd64) || purego

package dsp

func lift(dst, d, u, v []float32, c float32) { liftGo(dst, d, u, v, c) }

func scale(dst, src []float32, k float32) { scaleGo(dst, src, k) }

func PackF16(dst []byte, bpp int, src []float32, precision int32) {
	packF16Go(dst, bpp, src, precision)
}

func PackJJ16(dst []byte, src []float32, precision int32) { packJJ16Go(dst, src, precision) }

func zip(dst, a, b []float32) { zipGo(dst, a, b) }

func liftN(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	liftNGo(dst, d, u, v, n, rows, sDst, sD, sUV, c)
}

func scaleN(dst, src []float32, n, rows, sDst, sSrc int, k float32) {
	scaleNGo(dst, src, n, rows, sDst, sSrc, k)
}

func DequantF(dst []float32, mag []int32, flags []uint32, sh uint, bNo, bPi, d float32) {
	dequantFGo(dst, mag, flags, sh, bNo, bPi, d)
}

func jjLpf(dst, l, h0, h1 []float32) { jjLpfGo(dst, l, h0, h1) }

func jjHpf(dst, h, u, v []float32) { jjHpfGo(dst, h, u, v) }

func jjSub(dst, s, u, v []float32, c float32) { jjSubGo(dst, s, u, v, c) }

func jjDiv(x []float32, k float32) { jjDivGo(x, k) }

func liftSep(dst, d, u, v []float32, c float32) { liftSepGo(dst, d, u, v, c) }

func liftNSep(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	liftNSepGo(dst, d, u, v, n, rows, sDst, sD, sUV, c)
}

func ict(c0, c1, c2 []float32, r Rounding) { ictGo(c0, c1, c2, r) }

func PackP16(dst []byte, src []float32, precision int32) { packP16Go(dst, src, precision) }
