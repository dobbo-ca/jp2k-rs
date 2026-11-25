package dsp

// Rounding is how the 9/7 lift and the ICT round each c*x + d.
type Rounding uint8

const (
	// Fused rounds once, as opj_decompress does where its compiler
	// contracts the multiply-add (arm64, or x86-64 built with FMA).
	Fused Rounding = iota
	// Separate rounds the product and then the sum, as OpenJPEG built
	// for x86-64 without contraction does (Ubuntu's libopenjp2, and so
	// libvips). Its SIMD kernels never fuse; amd64 needs AVX2+FMA for
	// them, else Go.
	Separate
	// JJ2000 is Separate plus JJ2000's own 9/7, dequantization gain and
	// sample rounding (jj2000.go). Its lifts use the jj kernels and
	// its ICT the Separate one.
	JJ2000
)

// mulAdd is a*b + c rounded as r says.
func (r Rounding) mulAdd(a, b, c float32) float32 {
	if r != Fused {
		return float32(a*b) + c // the conversion forbids fusion
	}
	return fma32(a, b, c)
}

func (r Rounding) lift(dst, d, u, v []float32, c float32) {
	if r != Fused {
		liftSep(dst, d, u, v, c)
		return
	}
	lift(dst, d, u, v, c)
}

func (r Rounding) liftN(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	if r != Fused {
		liftNSep(dst, d, u, v, n, rows, sDst, sD, sUV, c)
		return
	}
	liftN(dst, d, u, v, n, rows, sDst, sD, sUV, c)
}

// liftNSepGo is liftNGo with Separate rounding.
func liftNSepGo(dst, d, u, v []float32, n, rows, sDst, sD, sUV int, c float32) {
	for k := 0; k < rows; k++ {
		liftSepGo(dst[k*sDst:k*sDst+n], d[k*sD:], u[k*sUV:], v[k*sUV:], c)
	}
}

// liftSepGo is liftGo with Separate rounding.
func liftSepGo(dst, d, u, v []float32, c float32) {
	n := len(dst)
	if len(d) < n || len(u) < n || len(v) < n {
		panic("jp2k: lift operand shorter than its destination")
	}
	d, u, v = d[:n:n], u[:n:n], v[:n:n]
	for x := range dst {
		dst[x] = float32(c*(u[x]+v[x])) + d[x]
	}
}
