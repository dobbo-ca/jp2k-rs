//go:build !purego

package dsp

// Variables so JP2K_NOSIMD can select the Go kernels.
var (
	lift     = pickKernel(liftSIMD, liftGo)
	scale    = pickKernel(scaleSIMD, scaleGo)
	PackF16  = pickKernel(packF16SIMD, packF16Go)
	PackJJ16 = pickKernel(packJJ16SIMD, packJJ16Go)
	jjLpf    = pickKernel(jjLpfSIMD, jjLpfGo)
	jjHpf    = pickKernel(jjHpfSIMD, jjHpfGo)
	jjSub    = pickKernel(jjSubSIMD, jjSubGo)
	jjDiv    = pickKernel(jjDivSIMD, jjDivGo)
	zip      = pickKernel(zipSIMD, zipGo)
	liftN    = pickKernel(liftNSIMD, liftNGo)
	scaleN   = pickKernel(scaleNSIMD, scaleNGo)
	liftSep  = pickKernel(liftSepSIMD, liftSepGo)
	liftNSep = pickKernel(liftNSepSIMD, liftNSepGo)
	ict      = pickKernel(ictSIMD, ictGo)
	PackP16  = pickKernel(packP16SIMD, packP16Go)
	DequantF = pickKernel(dequantFSIMD, dequantFGo)
)

// liftSIMD and scaleSIMD run NEON over whole 4-lane groups and leave the
// tail to the Go kernels.
func liftSIMD(dst, d, u, v []float32, c float32) {
	n := len(dst)
	if len(d) < n || len(u) < n || len(v) < n {
		panic("jp2k: lift operand shorter than its destination")
	}
	d, u, v = d[:n:n], u[:n:n], v[:n:n]
	m := n &^ 3
	if m > 0 {
		liftNEON(&dst[0], &d[0], &u[0], &v[0], m, c)
	}
	liftGo(dst[m:], d[m:], u[m:], v[m:], c)
}

func scaleSIMD(dst, src []float32, k float32) {
	src = src[:len(dst):len(dst)]
	m := len(dst) &^ 3
	if m > 0 {
		scaleNEON(&dst[0], &src[0], m, k)
	}
	scaleGo(dst[m:], src[m:], k)
}

//go:noescape
func liftNEON(dst, d, u, v *float32, n int, c float32)

//go:noescape
func scaleNEON(dst, src *float32, n int, k float32)

// packF16SIMD runs NEON over whole 4-sample groups and leaves the tail
// to packF16Go.
func packF16SIMD(dst []byte, bpp int, src []float32, precision int32) {
	if precision < 9 || precision > 16 {
		// The asm's shifts assume this range.
		packF16Go(dst, bpp, src, precision)
		return
	}
	if bpp == 2 && precision == 16 && len(src) >= 16 {
		// prec 16 needs no Expand16 shifts: SQXTN/EOR replace them.
		m := len(src) &^ 15
		_ = dst[2*m-1]
		pack16NEON(&dst[0], &src[0], m)
		dst, src = dst[2*m:], src[m:]
	}
	n := len(src)
	m := n &^ 3
	if m > 0 {
		_ = dst[(m-1)*bpp+1]
		shift := int32(1) << (precision - 1)
		packF16NEON(&dst[0], bpp, &src[0], m, -shift, shift-1, shift, 16-precision, 16-2*precision)
	}
	packF16Go(dst[min(m*bpp, len(dst)):], bpp, src[m:], precision)
}

//go:noescape
func pack16NEON(dst *byte, src *float32, n int)

// packF16NEON clamps the rounded samples to [lo, hi] before adding
// shift, then shifts left by sl and right by -sr (Expand16).
//
//go:noescape
func packF16NEON(dst *byte, bpp int, src *float32, n int, lo, hi, shift, sl, sr int32)

// packJJ16SIMD: NEON body, Go tail.
func packJJ16SIMD(dst []byte, src []float32, precision int32) {
	m := len(src) &^ 7
	if precision < 1 || precision > 16 || m == 0 {
		packJJ16Go(dst, src, precision)
		return
	}
	_ = dst[2*m-1]
	packJJ16NEON(&dst[0], &src[0], m, int32(1)<<(precision-1), int32(1)<<precision-1)
	packJJ16Go(dst[2*m:], src[m:], precision)
}

// packJJ16NEON: FCVTZS saturates like Go.
//
//go:noescape
func packJJ16NEON(dst *byte, src *float32, n int, shift, hi int32)

func zipSIMD(dst, a, b []float32) {
	n := len(a)
	b = b[:n:n]
	dst = dst[: 2*n : 2*n]
	m := n &^ 3
	if m > 0 {
		zipNEON(&dst[0], &a[0], &b[0], m)
	}
	zipGo(dst[2*m:], a[m:], b[m:])
}

//go:noescape
func zipNEON(dst, a, b *float32, n int)

// liftNSIMD runs NEON over whole 4-lane groups of each row and leaves
// the tail columns to liftNGo.
func liftNSIMD(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	liftNCheck(dst, d, u, v, n, rows, sDst, sD, sUV)
	m := n &^ 3
	if m > 0 && rows > 0 {
		liftNNEON(&dst[0], &d[0], &u[0], &v[0], m, rows, 4*sDst, 4*sD, 4*sUV, c)
	}
	if m < n {
		liftNGo(dst[m:], d[m:], u[m:], v[m:], n-m, rows, sDst, sD, sUV, c)
	}
}

//go:noescape
func liftNNEON(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)

// scaleNSIMD runs NEON over whole 4-lane groups of each row and leaves
// the tail columns to scaleNGo.
func scaleNSIMD(dst, src []float32, n, rows, sDst, sSrc int, k float32) {
	if rows <= 0 || n <= 0 {
		return
	}
	_, _ = dst[(rows-1)*sDst+n-1], src[(rows-1)*sSrc+n-1]
	m := n &^ 3
	if m > 0 {
		scaleNNEON(&dst[0], &src[0], m, rows, 4*sDst, 4*sSrc, k)
	}
	if m < n {
		scaleNGo(dst[m:], src[m:], n-m, rows, sDst, sSrc, k)
	}
}

//go:noescape
func scaleNNEON(dst, src *float32, n, rows, sDst, sSrc int, k float32)

// dequantFSIMD runs NEON over whole 4-sample groups and leaves the tail
// to dequantFGo.
func dequantFSIMD(dst []float32, mag []int32, flags []uint32, sh uint, bNo, bPi, d float32) {
	n := len(dst)
	if len(mag) < n || len(flags) < n {
		panic("jp2k: dequant operand shorter than its destination")
	}
	mag, flags = mag[:n:n], flags[:n:n]
	m := n &^ 3
	if m > 0 {
		dequantFNEON(&dst[0], &mag[0], &flags[0], m, uint32(sh), bNo, bPi, d)
	}
	dequantFGo(dst[m:], mag[m:], flags[m:], sh, bNo, bPi, d)
}

//go:noescape
func dequantFNEON(dst *float32, mag *int32, flags *uint32, n int, sh uint32, bNo, bPi, d float32)

// JJ2000 lifts: NEON body, Go tail.
func jjLpfSIMD(dst, l, h0, h1 []float32) {
	m := len(dst) &^ 3
	if m == 0 {
		jjLpfGo(dst, l, h0, h1)
		return
	}
	_, _, _ = l[m-1], h0[m-1], h1[m-1]
	jjLpfNEON(&dst[0], &l[0], &h0[0], &h1[0], m, jjKL, jjKH, dwt97Delta)
	jjLpfGo(dst[m:], l[m:], h0[m:], h1[m:])
}

func jjHpfSIMD(dst, h, u, v []float32) {
	m := len(dst) &^ 3
	if m == 0 {
		jjHpfGo(dst, h, u, v)
		return
	}
	_, _, _ = h[m-1], u[m-1], v[m-1]
	jjHpfNEON(&dst[0], &h[0], &u[0], &v[0], m, jjKH, dwt97Gamma)
	jjHpfGo(dst[m:], h[m:], u[m:], v[m:])
}

func jjSubSIMD(dst, s, u, v []float32, c float32) {
	m := len(dst) &^ 3
	if m == 0 {
		jjSubGo(dst, s, u, v, c)
		return
	}
	_, _, _ = s[m-1], u[m-1], v[m-1]
	jjSubNEON(&dst[0], &s[0], &u[0], &v[0], m, c)
	jjSubGo(dst[m:], s[m:], u[m:], v[m:], c)
}

func jjDivSIMD(x []float32, k float32) {
	m := len(x) &^ 3
	if m == 0 {
		jjDivGo(x, k)
		return
	}
	jjDivNEON(&x[0], m, k)
	jjDivGo(x[m:], k)
}

//go:noescape
func jjLpfNEON(dst, l, h0, h1 *float32, n int, kl, kh, d float32)

//go:noescape
func jjHpfNEON(dst, h, u, v *float32, n int, kh, c float32)

//go:noescape
func jjSubNEON(dst, s, u, v *float32, n int, c float32)

//go:noescape
func jjDivNEON(x *float32, n int, k float32)

// liftSepSIMD is liftSIMD with a separate multiply and add, as
// Separate rounding needs.
func liftSepSIMD(dst, d, u, v []float32, c float32) {
	n := len(dst)
	if len(d) < n || len(u) < n || len(v) < n {
		panic("jp2k: lift operand shorter than its destination")
	}
	d, u, v = d[:n:n], u[:n:n], v[:n:n]
	m := n &^ 3
	if m > 0 {
		liftSepNEON(&dst[0], &d[0], &u[0], &v[0], m, c)
	}
	liftSepGo(dst[m:], d[m:], u[m:], v[m:], c)
}

func liftNSepSIMD(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	liftNCheck(dst, d, u, v, n, rows, sDst, sD, sUV)
	m := n &^ 3
	if m > 0 && rows > 0 {
		liftNSepNEON(&dst[0], &d[0], &u[0], &v[0], m, rows, 4*sDst, 4*sD, 4*sUV, c)
	}
	if m < n {
		liftNSepGo(dst[m:], d[m:], u[m:], v[m:], n-m, rows, sDst, sD, sUV, c)
	}
}

//go:noescape
func liftSepNEON(dst, d, u, v *float32, n int, c float32)

//go:noescape
func liftNSepNEON(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)

// ictSIMD runs NEON over whole 4-lane groups and leaves the tail to ictGo.
func ictSIMD(c0, c1, c2 []float32, r Rounding) {
	n := len(c0)
	if len(c1) < n || len(c2) < n {
		panic("jp2k: ICT plane shorter than c0")
	}
	c1, c2 = c1[:n:n], c2[:n:n]
	m := n &^ 3
	if m > 0 {
		if r == Fused {
			ictFusedNEON(&c0[0], &c1[0], &c2[0], m, ictCrR, ictCbG, ictCrG, ictCbB)
		} else {
			ictSepNEON(&c0[0], &c1[0], &c2[0], m, ictCrR, ictCbG, ictCrG, ictCbB)
		}
	}
	ictGo(c0[m:], c1[m:], c2[m:], r)
}

//go:noescape
func ictFusedNEON(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)

//go:noescape
func ictSepNEON(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)

// packP16SIMD reuses packF16NEON with no Expand16 shifts; FCVTNS and
// the clamps are exact for every precision 1..16.
func packP16SIMD(dst []byte, src []float32, precision int32) {
	if precision < 1 || precision > 16 {
		packP16Go(dst, src, precision)
		return
	}
	m := len(src) &^ 3
	if m > 0 {
		_ = dst[2*m-1]
		shift := int32(1) << (precision - 1)
		packF16NEON(&dst[0], 2, &src[0], m, -shift, shift-1, shift, 0, 0)
	}
	packP16Go(dst[min(2*m, len(dst)):], src[m:], precision)
}
