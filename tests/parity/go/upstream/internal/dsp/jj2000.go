package dsp

import (
	"math"

	"github.com/dobbo-ca/jp2k-go/internal/codestream"
)

// JJ2000 rounding reproduces the 9/7 decode of JJ2000 (jj2000.j2k in
// jai-imageio, ome-jai 0.1.5), which Bio-Formats decodes JPEG 2000 with:
// SynWTFilterFloatLift9x7's synthesis, InvWTFull's row-then-column order
// and ImgDataConverter's rounding. Java float has no FMA, so every
// product here goes through an explicit float32 conversion, which the Go
// spec forbids fusing. The subband gain is in the dequantization step
// (the decoder multiplies it in), so the high band is divided by K here,
// not multiplied by OpenJPEG's two_invK.

// jjKL and jjKH are SynWTFilterFloatLift9x7's KL and KH (1/K and K).
const (
	jjKL float32 = 0.8128930655
	jjKH float32 = 1.230174106
)

// synth97JJ writes the synthesis of one Mallat-ordered line (low band
// l, high band h) to o[0], o[s], o[2s], ...: synthetize_lpf when the
// line starts on a low sample (cas 0), synthetize_hpf otherwise. hpf
// scales l and h in place, as JJ2000 does.
func synth97JJ(o []float32, s int, l, h []float32, cas int) {
	const a, b, g, d float32 = dwt97Alpha, dwt97Beta, dwt97Gamma, dwt97Delta
	n := len(l) + len(h)
	at := func(i int) *float32 { return &o[i*s] }
	if cas == 0 {
		if n > 1 {
			*at(0) = l[0]/jjKL - float32(2*d*h[0])/jjKH
		} else {
			*at(0) = l[0]
		}
		for i := 2; i < n-1; i += 2 {
			k := i / 2
			*at(i) = l[k]/jjKL - float32(d*(h[k-1]+h[k]))/jjKH
		}
		if n%2 == 1 && n > 2 {
			m := (n - 1) / 2
			*at(n - 1) = l[m]/jjKL - float32(2*d*h[m-1])/jjKH
		}
		for i := 1; i < n-1; i += 2 {
			*at(i) = h[(i-1)/2]/jjKH - float32(g*(*at(i - 1)+*at(i + 1)))
		}
		if n%2 == 0 {
			*at(n - 1) = h[n/2-1]/jjKH - float32(2*g*(*at(n - 2)))
		}
		if n > 1 {
			*at(0) -= float32(2 * b * *at(1))
		}
		for i := 2; i < n-1; i += 2 {
			*at(i) -= float32(b * (*at(i - 1) + *at(i + 1)))
		}
		if n%2 == 1 && n > 2 {
			*at(n - 1) -= float32(2 * b * *at(n - 2))
		}
		for i := 1; i < n-1; i += 2 {
			*at(i) -= float32(a * (*at(i - 1) + *at(i + 1)))
		}
		if n%2 == 0 {
			*at(n - 1) -= float32(2 * a * *at(n - 2))
		}
		return
	}
	if n != 1 {
		for i := range l {
			l[i] /= jjKL
		}
		for i := range h {
			h[i] /= jjKH
		}
	} else {
		h[0] /= 2
	}
	for i := 1; i < n-1; i += 2 {
		k := (i - 1) / 2
		*at(i) = l[k] - float32(d*(h[k]+h[k+1]))
	}
	if n%2 == 0 && n > 1 {
		*at(n - 1) = l[n/2-1] - float32(2*d*h[n/2-1])
	}
	if n > 1 {
		*at(0) = h[0] - float32(2*g*(*at(1)))
	} else {
		*at(0) = h[0]
	}
	for i := 2; i < n-1; i += 2 {
		*at(i) = h[i/2] - float32(g*(*at(i - 1)+*at(i + 1)))
	}
	if n%2 == 1 && n > 1 {
		*at(n - 1) = h[(n-1)/2] - float32(2*g*(*at(n - 2)))
	}
	for i := 1; i < n-1; i += 2 {
		*at(i) -= float32(b * (*at(i - 1) + *at(i + 1)))
	}
	if n%2 == 0 && n > 1 {
		*at(n - 1) -= float32(2 * b * *at(n - 2))
	}
	if n > 1 {
		*at(0) -= float32(2 * a * *at(1))
	}
	for i := 2; i < n-1; i += 2 {
		*at(i) -= float32(a * (*at(i - 1) + *at(i + 1)))
	}
	if n%2 == 1 && n > 1 {
		*at(n - 1) -= float32(2 * a * *at(n - 2))
	}
}

// synth97JJSplit is synth97JJ lifting l and h in place with the jj
// kernels. cas 0 leaves even outputs in l and odd in h; cas 1 the
// reverse. interleave97 then zips them.
func synth97JJSplit(l, h []float32, cas int) {
	const a, b, g, d float32 = dwt97Alpha, dwt97Beta, dwt97Gamma, dwt97Delta
	n := len(l) + len(h)
	if cas == 0 {
		if n == 1 {
			return
		}
		m, c := len(h), len(l)-1
		l[0] = l[0]/jjKL - float32(2*d*h[0])/jjKH
		jjLpf(l[1:m], l[1:m], h[:m-1], h[1:m])
		if n%2 == 1 {
			l[m] = l[m]/jjKL - float32(2*d*h[m-1])/jjKH
		}
		jjHpf(h[:c], h[:c], l[:c], l[1:c+1])
		if n%2 == 0 {
			h[m-1] = h[m-1]/jjKH - float32(2*g*l[m-1])
		}
		l[0] -= float32(2 * b * h[0])
		jjSub(l[1:m], l[1:m], h[:m-1], h[1:m], b)
		if n%2 == 1 {
			l[m] -= float32(2 * b * h[m-1])
		}
		jjSub(h[:c], h[:c], l[:c], l[1:c+1], a)
		if n%2 == 0 {
			h[m-1] -= float32(2 * a * l[m-1])
		}
		return
	}
	if n == 1 {
		h[0] /= 2
		return
	}
	s, c := len(l), len(h)-1
	jjDiv(l, jjKL)
	jjDiv(h, jjKH)
	jjSub(l[:c], l[:c], h[:c], h[1:c+1], d)
	if n%2 == 0 {
		l[s-1] = l[s-1] - float32(2*d*h[s-1])
	}
	h[0] = h[0] - float32(2*g*l[0])
	jjSub(h[1:s], h[1:s], l[:s-1], l[1:s], g)
	if n%2 == 1 {
		h[s] = h[s] - float32(2*g*l[s-1])
	}
	jjSub(l[:c], l[:c], h[:c], h[1:c+1], b)
	if n%2 == 0 {
		l[s-1] -= float32(2 * b * h[s-1])
	}
	h[0] -= float32(2 * a * l[0])
	jjSub(h[1:s], h[1:s], l[:s-1], l[1:s], a)
	if n%2 == 1 {
		h[s] -= float32(2 * a * l[s-1])
	}
}

// synth97JJBlock: synth97JJ over bn columns.
func synth97JJBlock(o []float32, s int, l, h []float32, bn, cas int) {
	const a, b, g, d float32 = dwt97Alpha, dwt97Beta, dwt97Gamma, dwt97Delta
	n := (len(l) + len(h)) / bn
	at := func(i int) []float32 { return o[i*s : i*s+bn] }
	lo := func(k int) []float32 { return l[k*bn : k*bn+bn] }
	hi := func(k int) []float32 { return h[k*bn : k*bn+bn] }
	if cas == 0 {
		if n > 1 {
			r, lk, hk := at(0), lo(0), hi(0)
			for x := range r {
				r[x] = lk[x]/jjKL - float32(2*d*hk[x])/jjKH
			}
		} else {
			copy(at(0), lo(0))
		}
		for i := 2; i < n-1; i += 2 {
			k := i / 2
			r, lk, h0, h1 := at(i), lo(k), hi(k-1), hi(k)
			jjLpf(r, lk, h0, h1)
		}
		if n%2 == 1 && n > 2 {
			m := (n - 1) / 2
			r, lm, hm := at(n-1), lo(m), hi(m-1)
			for x := range r {
				r[x] = lm[x]/jjKL - float32(2*d*hm[x])/jjKH
			}
		}
		for i := 1; i < n-1; i += 2 {
			r, hk, u, v := at(i), hi((i-1)/2), at(i-1), at(i+1)
			jjHpf(r, hk, u, v)
		}
		if n%2 == 0 {
			r, hk, u := at(n-1), hi(n/2-1), at(n-2)
			for x := range r {
				r[x] = hk[x]/jjKH - float32(2*g*u[x])
			}
		}
		if n > 1 {
			r, u := at(0), at(1)
			for x := range r {
				r[x] -= float32(2 * b * u[x])
			}
		}
		for i := 2; i < n-1; i += 2 {
			r, u, v := at(i), at(i-1), at(i+1)
			jjSub(r, r, u, v, b)
		}
		if n%2 == 1 && n > 2 {
			r, u := at(n-1), at(n-2)
			for x := range r {
				r[x] -= float32(2 * b * u[x])
			}
		}
		for i := 1; i < n-1; i += 2 {
			r, u, v := at(i), at(i-1), at(i+1)
			jjSub(r, r, u, v, a)
		}
		if n%2 == 0 {
			r, u := at(n-1), at(n-2)
			for x := range r {
				r[x] -= float32(2 * a * u[x])
			}
		}
		return
	}
	if n != 1 {
		jjDiv(l, jjKL)
		jjDiv(h, jjKH)
	} else {
		for i := range h {
			h[i] /= 2
		}
	}
	for i := 1; i < n-1; i += 2 {
		k := (i - 1) / 2
		r, lk, h0, h1 := at(i), lo(k), hi(k), hi(k+1)
		jjSub(r, lk, h0, h1, d)
	}
	if n%2 == 0 && n > 1 {
		r, lk, hk := at(n-1), lo(n/2-1), hi(n/2-1)
		for x := range r {
			r[x] = lk[x] - float32(2*d*hk[x])
		}
	}
	if n > 1 {
		r, hk, u := at(0), hi(0), at(1)
		for x := range r {
			r[x] = hk[x] - float32(2*g*u[x])
		}
	} else {
		copy(at(0), hi(0))
	}
	for i := 2; i < n-1; i += 2 {
		r, hk, u, v := at(i), hi(i/2), at(i-1), at(i+1)
		jjSub(r, hk, u, v, g)
	}
	if n%2 == 1 && n > 1 {
		r, hk, u := at(n-1), hi((n-1)/2), at(n-2)
		for x := range r {
			r[x] = hk[x] - float32(2*g*u[x])
		}
	}
	for i := 1; i < n-1; i += 2 {
		r, u, v := at(i), at(i-1), at(i+1)
		jjSub(r, r, u, v, b)
	}
	if n%2 == 0 && n > 1 {
		r, u := at(n-1), at(n-2)
		for x := range r {
			r[x] -= float32(2 * b * u[x])
		}
	}
	if n > 1 {
		r, u := at(0), at(1)
		for x := range r {
			r[x] -= float32(2 * a * u[x])
		}
	}
	for i := 2; i < n-1; i += 2 {
		r, u, v := at(i), at(i-1), at(i+1)
		jjSub(r, r, u, v, a)
	}
	if n%2 == 1 && n > 1 {
		r, u := at(n-1), at(n-2)
		for x := range r {
			r[x] -= float32(2 * a * u[x])
		}
	}
}

// JJ2000 lifts; SIMD matches these bitwise.
func jjLpfGo(dst, l, h0, h1 []float32) {
	const d float32 = dwt97Delta
	l, h0, h1 = l[:len(dst)], h0[:len(dst)], h1[:len(dst)]
	for x := range dst {
		dst[x] = l[x]/jjKL - float32(d*(h0[x]+h1[x]))/jjKH
	}
}

func jjHpfGo(dst, h, u, v []float32) {
	const g float32 = dwt97Gamma
	h, u, v = h[:len(dst)], u[:len(dst)], v[:len(dst)]
	for x := range dst {
		dst[x] = h[x]/jjKH - float32(g*(u[x]+v[x]))
	}
}

func jjSubGo(dst, s, u, v []float32, c float32) {
	s, u, v = s[:len(dst)], u[:len(dst)], v[:len(dst)]
	for x := range dst {
		dst[x] = s[x] - float32(c*(u[x]+v[x]))
	}
}

func jjDivGo(x []float32, k float32) {
	for i := range x {
		x[i] /= k
	}
}

// inverseDWT97JJ is InverseDWT97 with JJ2000 rounding.
func inverseDWT97JJ(p []float32, stride int, rects []codestream.SubbandRect, high []bool, scratch *[][]float32, nw int, done <-chan struct{}) {
	inverseDWT(p, stride, rects, high, scratch, nw, dwtPad, ColBlock97, done, nil, nil,
		func(row, line []float32, sn, cas int, _ bool) {
			n := len(row)
			copy(line[:n], row)
			synth97JJSplit(line[:sn], line[sn:n], cas)
			interleave97(row, line[:n], sn, cas)
		},
		func(p []float32, stride, bn int, b []float32, n, cas int, _ bool) {
			sn, _ := SnDn(n, cas)
			b = b[:n*bn]
			for i := 0; i < n; i++ {
				copy(b[i*bn:(i+1)*bn], p[i*stride:i*stride+bn])
			}
			synth97JJBlock(p, stride, b[:sn*bn], b[sn*bn:], bn, cas)
		})
}

// SamplesJJ is SamplesF with JJ2000's rounding: ImgDataConverter adds
// 0.5 away from zero in float32 and truncates as Java's int cast does,
// then J2KReadState adds the level shift (wrapping) and clamps.
func SamplesJJ(dst []int32, src []float32, precision int32) {
	for i, v := range src {
		dst[i] = sampleJJ(v, precision)
	}
}

// packJJ16Go: SamplesJJ plus big-endian store.
func packJJ16Go(dst []byte, src []float32, precision int32) {
	for i, v := range src {
		Put16BE(dst[2*i:], uint16(sampleJJ(v, precision)))
	}
}

func sampleJJ(v float32, precision int32) int32 {
	if v > 0 {
		v += 0.5
	} else {
		v -= 0.5
	}
	var i int32
	switch {
	case v != v: // NaN casts to 0
	case v >= 1<<31:
		i = math.MaxInt32
	case v <= -1<<31:
		i = math.MinInt32
	default:
		i = int32(v)
	}
	i += int32(1) << (precision - 1)
	return min(max(i, 0), int32(1)<<precision-1)
}
