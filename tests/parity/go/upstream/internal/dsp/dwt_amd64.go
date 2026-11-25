//go:build !purego

package dsp

// useAVX2 picks the 8-lane AVX2+FMA kernels over the SSE2 and Go
// baseline; tests clear it to cover that path.
var useAVX2 = !noSIMD && hasAVX2FMA()

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
)

// DequantF runs AVX2 over whole 8-lane groups and leaves the tail to
// dequantFGo. A plain function, not a pickKernel variable: tier1 calls
// it directly to avoid an indirect call.
func DequantF(dst []float32, mag []int32, flags []uint32, sh uint, bNo, bPi, d float32) {
	n := len(dst)
	if len(mag) < n || len(flags) < n {
		panic("jp2k: dequant operand shorter than its destination")
	}
	mag, flags = mag[:n:n], flags[:n:n]
	m := 0
	if useAVX2 {
		m = n &^ 7
	}
	if m > 0 {
		dequantFAVX2(&dst[0], &mag[0], &flags[0], m, uint32(sh), bNo, bPi, d)
	}
	dequantFGo(dst[m:], mag[m:], flags[m:], sh, bNo, bPi, d)
}

//go:noescape
func dequantFAVX2(dst *float32, mag *int32, flags *uint32, n int, sh uint32, bNo, bPi, d float32)

func hasAVX2FMA() bool {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 7 {
		return false
	}
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&(1<<12) == 0 || ecx1&(1<<27) == 0 || ecx1&(1<<28) == 0 { // FMA, OSXSAVE, AVX
		return false
	}
	if xcr0, _ := xgetbv(); xcr0&6 != 6 { // OS saves XMM and YMM state
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&(1<<5) != 0
}

// liftSIMD runs AVX2 for all n (scalar FMA tail), scaleSIMD leaves its tail to
// the Go kernels. Without FMA, lift is all Go: SSE2 has no fused multiply-add.
// ponytail: that path is ~4x slower per decode under Rosetta; an SSE2
// round-to-odd kernel would help if FMA-less amd64 hosts matter.
func liftSIMD(dst, d, u, v []float32, c float32) {
	n := len(dst)
	if len(d) < n || len(u) < n || len(v) < n {
		panic("jp2k: lift operand shorter than its destination")
	}
	d, u, v = d[:n:n], u[:n:n], v[:n:n]
	if useAVX2 {
		if n > 0 {
			liftAVX2(&dst[0], &d[0], &u[0], &v[0], n, c)
		}
		return
	}
	liftGo(dst, d, u, v, c)
}

func scaleSIMD(dst, src []float32, k float32) {
	src = src[:len(dst):len(dst)]
	if useAVX2 {
		m := len(dst) &^ 7
		if m > 0 {
			scaleAVX2(&dst[0], &src[0], m, k)
		}
		scaleGo(dst[m:], src[m:], k)
		return
	}
	m := len(dst) &^ 3
	if m > 0 {
		scaleSSE2(&dst[0], &src[0], m, k)
	}
	scaleGo(dst[m:], src[m:], k)
}

// liftNSIMD runs AVX2+FMA over each row, scalar FMA for the n%8 tail
// columns. Without FMA the whole call goes to liftNGo, same as lift:
// SSE2 has no fused multiply-add.
func liftNSIMD(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	liftNCheck(dst, d, u, v, n, rows, sDst, sD, sUV)
	if !useAVX2 {
		liftNGo(dst, d, u, v, n, rows, sDst, sD, sUV, c)
		return
	}
	if n > 0 && rows > 0 {
		liftNAVX2(&dst[0], &d[0], &u[0], &v[0], n, rows, 4*sDst, 4*sD, 4*sUV, c)
	}
}

//go:noescape
func liftNAVX2(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)

// scaleNSIMD is scaleSIMD over rows rows; amd64 has no packed column
// kernel, so this is the pre-scaleN cost.
func scaleNSIMD(dst, src []float32, n, rows, sDst, sSrc int, k float32) {
	for r := 0; r < rows; r++ {
		scaleSIMD(dst[r*sDst:r*sDst+n], src[r*sSrc:r*sSrc+n], k)
	}
}

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

func xgetbv() (eax, edx uint32)

//go:noescape
func liftAVX2(dst, d, u, v *float32, n int, c float32)

//go:noescape
func scaleSSE2(dst, src *float32, n int, k float32)

//go:noescape
func scaleAVX2(dst, src *float32, n int, k float32)

// packF16SIMD runs SSE2 over whole 4-sample groups and leaves the tail
// to packF16Go. No AVX2 variant: the per-sample 2-byte stores dominate.
func packF16SIMD(dst []byte, bpp int, src []float32, precision int32) {
	if precision < 9 || precision > 16 {
		// The asm's shifts assume this range.
		packF16Go(dst, bpp, src, precision)
		return
	}
	n := len(src)
	m := n &^ 3
	if m > 0 {
		_ = dst[(m-1)*bpp+1]
		shift := int32(1) << (precision - 1)
		packF16SSE2(&dst[0], bpp, &src[0], m, float32(-shift), float32(shift-1), shift, uint32(16-precision), uint32(2*precision-16))
	}
	packF16Go(dst[min(m*bpp, len(dst)):], bpp, src[m:], precision)
}

// packF16SSE2 clamps to [lo, hi] before rounding (the same result for
// integer bounds, and CVTPS2DQ cannot saturate), adds shift, then
// shifts left by sl and right by sr (Expand16).
//
//go:noescape
func packF16SSE2(dst *byte, bpp int, src *float32, n int, lo, hi float32, shift int32, sl, sr uint32)

// packJJ16SIMD: AVX2 body, Go tail.
func packJJ16SIMD(dst []byte, src []float32, precision int32) {
	m := len(src) &^ 15
	if !useAVX2 || precision < 1 || precision > 16 || m == 0 {
		// PACKUSDW saturates; uint16() truncates.
		packJJ16Go(dst, src, precision)
		return
	}
	_ = dst[2*m-1]
	packJJ16AVX2(&dst[0], &src[0], m, int32(1)<<(precision-1), int32(1)<<precision-1)
	packJJ16Go(dst[2*m:], src[m:], precision)
}

// packJJ16AVX2: NaN masked, overflow to 0.
//
//go:noescape
func packJJ16AVX2(dst *byte, src *float32, n int, shift, hi int32)

// zipSIMD is SSE2 only: in-lane AVX2 unpack would need a cross-lane
// permute, and this store-bound kernel gains nothing from it.
func zipSIMD(dst, a, b []float32) {
	n := len(a)
	b = b[:n:n]
	dst = dst[: 2*n : 2*n]
	m := n &^ 3
	if m > 0 {
		zipSSE2(&dst[0], &a[0], &b[0], m)
	}
	zipGo(dst[2*m:], a[m:], b[m:])
}

//go:noescape
func zipSSE2(dst, a, b *float32, n int)

// JJ2000 lifts: AVX2 body, Go tail.
func jjLpfSIMD(dst, l, h0, h1 []float32) {
	m := len(dst) &^ 7
	if !useAVX2 || m == 0 {
		jjLpfGo(dst, l, h0, h1)
		return
	}
	_, _, _ = l[m-1], h0[m-1], h1[m-1]
	jjLpfAVX2(&dst[0], &l[0], &h0[0], &h1[0], m, jjKL, jjKH, dwt97Delta)
	jjLpfGo(dst[m:], l[m:], h0[m:], h1[m:])
}

func jjHpfSIMD(dst, h, u, v []float32) {
	m := len(dst) &^ 7
	if !useAVX2 || m == 0 {
		jjHpfGo(dst, h, u, v)
		return
	}
	_, _, _ = h[m-1], u[m-1], v[m-1]
	jjHpfAVX2(&dst[0], &h[0], &u[0], &v[0], m, jjKH, dwt97Gamma)
	jjHpfGo(dst[m:], h[m:], u[m:], v[m:])
}

func jjSubSIMD(dst, s, u, v []float32, c float32) {
	m := len(dst) &^ 7
	if !useAVX2 || m == 0 {
		jjSubGo(dst, s, u, v, c)
		return
	}
	_, _, _ = s[m-1], u[m-1], v[m-1]
	jjSubAVX2(&dst[0], &s[0], &u[0], &v[0], m, c)
	jjSubGo(dst[m:], s[m:], u[m:], v[m:], c)
}

func jjDivSIMD(x []float32, k float32) {
	m := len(x) &^ 7
	if !useAVX2 || m == 0 {
		jjDivGo(x, k)
		return
	}
	jjDivAVX2(&x[0], m, k)
	jjDivGo(x[m:], k)
}

//go:noescape
func jjLpfAVX2(dst, l, h0, h1 *float32, n int, kl, kh, d float32)

//go:noescape
func jjHpfAVX2(dst, h, u, v *float32, n int, kh, c float32)

//go:noescape
func jjSubAVX2(dst, s, u, v *float32, n int, c float32)

//go:noescape
func jjDivAVX2(x *float32, n int, k float32)

// liftSepSIMD is liftSIMD with a separate multiply and add, as
// Separate rounding needs; AVX2 only, else Go.
func liftSepSIMD(dst, d, u, v []float32, c float32) {
	n := len(dst)
	if len(d) < n || len(u) < n || len(v) < n {
		panic("jp2k: lift operand shorter than its destination")
	}
	d, u, v = d[:n:n], u[:n:n], v[:n:n]
	if useAVX2 {
		if n > 0 {
			liftSepAVX2(&dst[0], &d[0], &u[0], &v[0], n, c)
		}
		return
	}
	liftSepGo(dst, d, u, v, c)
}

func liftNSepSIMD(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	liftNCheck(dst, d, u, v, n, rows, sDst, sD, sUV)
	if !useAVX2 {
		liftNSepGo(dst, d, u, v, n, rows, sDst, sD, sUV, c)
		return
	}
	if n > 0 && rows > 0 {
		liftNSepAVX2(&dst[0], &d[0], &u[0], &v[0], n, rows, 4*sDst, 4*sD, 4*sUV, c)
	}
}

//go:noescape
func liftSepAVX2(dst, d, u, v *float32, n int, c float32)

//go:noescape
func liftNSepAVX2(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)

// ictSIMD runs AVX2 over whole 8-lane groups and leaves the tail to
// ictGo. Without FMA everything is Go: Fused needs the soft fma32.
func ictSIMD(c0, c1, c2 []float32, r Rounding) {
	n := len(c0)
	if len(c1) < n || len(c2) < n {
		panic("jp2k: ICT plane shorter than c0")
	}
	c1, c2 = c1[:n:n], c2[:n:n]
	m := n &^ 7
	if !useAVX2 || m == 0 {
		ictGo(c0, c1, c2, r)
		return
	}
	if r == Fused {
		ictFusedAVX2(&c0[0], &c1[0], &c2[0], m, ictCrR, ictCbG, ictCrG, ictCbB)
	} else {
		ictSepAVX2(&c0[0], &c1[0], &c2[0], m, ictCrR, ictCbG, ictCrG, ictCbB)
	}
	ictGo(c0[m:], c1[m:], c2[m:], r)
}

//go:noescape
func ictFusedAVX2(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)

//go:noescape
func ictSepAVX2(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)

// packP16SIMD reuses packF16SSE2 with no Expand16 shifts; it is exact
// for every precision 1..16.
func packP16SIMD(dst []byte, src []float32, precision int32) {
	if precision < 1 || precision > 16 {
		packP16Go(dst, src, precision)
		return
	}
	m := len(src) &^ 3
	if m > 0 {
		_ = dst[2*m-1]
		shift := int32(1) << (precision - 1)
		packF16SSE2(&dst[0], 2, &src[0], m, float32(-shift), float32(shift-1), shift, 0, 0)
	}
	packP16Go(dst[min(2*m, len(dst)):], src[m:], precision)
}
