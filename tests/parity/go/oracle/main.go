// Command oracle is the Go side of the parity check (see docs/parity.md).
//
//	oracle -gen <fixtures> <inputs> [<seeds>]   write the test inputs (and each file of <seeds>)
//	oracle <inputs> <outputs>         decode every input
//	oracle [-threads P] -bench N <fixtures>   time the decode of every fixture (P = GOMAXPROCS, 0 default)
//	oracle -ppm <tiles> <outputs>     decode every .j2k as a PGM or PPM (ppm.go)
//
// Decode mode writes, per input and operation, <name>.<op>.out (a description of the result: the
// pixel format, the bounds and the FNV-1a 64 hash of the samples) or <name>.<op>.err (the error class:
// the message, or "limit" or "internal"). examples/oracle.rs does the same with the Rust crate.
// Go returns nothing with an error, so there are no partial outputs.
package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"testing/iotest"
	"time"

	jp2k "github.com/dobbo-ca/jp2k-go"
)

// An image with more pixels than this is only configured, not decoded, so that Go's and Rust's
// memory limits cannot disagree about it and the oracle stays small. Inputs named hostile-* (the
// crafted ones, which are cheap) may be larger.
const (
	maxPixels        = 1 << 22
	maxHostilePixels = 1 << 24
)

func main() {
	gen := flag.Bool("gen", false, "write the inputs: -gen <fixtures> <inputs>")
	bench := flag.Int("bench", 0, "decode every fixture N times: -bench N <fixtures>")
	threads := flag.Int("threads", 0, "with -bench, GOMAXPROCS (0: the default, as a Go caller has it)")
	ppm := flag.Bool("ppm", false, "decode every .j2k as a PGM or PPM: -ppm <tiles> <outputs>")
	flag.Parse()
	a := flag.Args()
	// Decode and -gen modes run one worker: Go then reports the error of the first failing tile,
	// as Rust does with any number of threads. The benchmark does not force it: the default
	// run compares Go at its default GOMAXPROCS with Rust at its default max_threads, and
	// -threads 1 compares one thread with one thread.
	if *bench == 0 {
		runtime.GOMAXPROCS(1)
	} else if *threads > 0 {
		runtime.GOMAXPROCS(*threads)
	}
	jp2k.SetMemoryLimit(2 << 30)
	debug.SetMemoryLimit(3 << 30)
	var err error
	switch {
	case *gen && (len(a) == 2 || len(a) == 3):
		seeds := ""
		if len(a) == 3 {
			seeds = a[2]
		}
		err = genInputs(a[0], a[1], seeds)
	case *ppm && !*gen && *bench == 0 && len(a) == 2:
		err = ppmDir(a[0], a[1])
	case *bench > 0 && len(a) == 1:
		err = benchDir(a[0], *bench)
	case !*gen && !*ppm && *bench == 0 && len(a) == 2:
		err = runDir(a[0], a[1])
	default:
		fmt.Fprintln(os.Stderr, "usage: oracle -gen <fixtures> <inputs> [<seeds>] | oracle <inputs> <outputs> | oracle [-threads P] -bench N <fixtures> | oracle -ppm <tiles> <outputs>")
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "oracle:", err)
		os.Exit(1)
	}
}

// class is the error class of err: both sides print the same token for the same failure. Go has
// no per-call limits, so every message about a memory limit is "limit".
func class(err error) string {
	switch {
	case errors.Is(err, jp2k.ErrInternal):
		return "internal"
	case errors.Is(err, jp2k.ErrMemoryLimit):
		return "limit"
	}
	s := err.Error()
	for _, k := range []string{"working memory", "memory limit", "input limit", "out of memory", "more memory than"} {
		if strings.Contains(s, k) {
			return "limit"
		}
	}
	return strings.ReplaceAll(s, "\n", " ")
}

func modelName(m color.Model) string {
	switch m {
	case color.GrayModel:
		return "gray"
	case color.Gray16Model:
		return "gray16"
	case color.RGBAModel:
		return "rgba"
	case color.RGBA64Model:
		return "rgba64"
	case color.NRGBAModel:
		return "nrgba"
	case color.NRGBA64Model:
		return "nrgba64"
	}
	return fmt.Sprintf("other:%v", m)
}

func describeCfg(c image.Config) string {
	return fmt.Sprintf("%s %d %d\n", modelName(c.ColorModel), c.Width, c.Height)
}

func fnv64(b []byte) uint64 {
	h := fnv.New64a()
	h.Write(b)
	return h.Sum64()
}

// describe is the pixel format, bounds, stride and FNV-1a 64 hash of an image's samples.
func describe(img image.Image) string {
	var name string
	var pix []byte
	var stride int
	switch m := img.(type) {
	case *image.Gray:
		name, pix, stride = "gray", m.Pix, m.Stride
	case *image.Gray16:
		name, pix, stride = "gray16", m.Pix, m.Stride
	case *image.RGBA:
		name, pix, stride = "rgba", m.Pix, m.Stride
	case *image.RGBA64:
		name, pix, stride = "rgba64", m.Pix, m.Stride
	case *image.NRGBA:
		name, pix, stride = "nrgba", m.Pix, m.Stride
	case *image.NRGBA64:
		name, pix, stride = "nrgba64", m.Pix, m.Stride
	default:
		name = fmt.Sprintf("other:%T", img)
	}
	r := img.Bounds()
	return fmt.Sprintf("%s %d %d %d %d stride=%d len=%d fnv=%016x\n", name, r.Min.X, r.Min.Y, r.Max.X, r.Max.Y, stride, len(pix), fnv64(pix))
}

func describePlanes(ps []*image.Gray16) string {
	var b strings.Builder
	fmt.Fprintf(&b, "planes %d\n", len(ps))
	for _, p := range ps {
		r := p.Bounds()
		fmt.Fprintf(&b, "plane %d %d %d %d len=%d fnv=%016x\n", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y, len(p.Pix), fnv64(p.Pix))
	}
	return b.String()
}

// rng is xorshift64, the same as examples/oracle.rs.
type rng uint64

func newRng(name string) *rng {
	h := fnv.New64a()
	h.Write([]byte(name))
	r := rng(h.Sum64() | 1)
	return &r
}

func (r *rng) next() uint64 {
	x := uint64(*r)
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*r = rng(x)
	return x
}

func (r *rng) below(n int) int { return int(r.next() % uint64(n)) }

type region struct {
	rect   image.Rectangle
	reduce int
}

// regions are the whole image at reduce 0 and 1, then four random ones (regionCases of
// handle_test.go: a random corner, a random extent from it, reduce 0 to 3).
func regions(name string, w, h int) []region {
	if w < 1 || h < 1 {
		w, h = 64, 64
	}
	rs := []region{{image.Rect(0, 0, w, h), 0}, {image.Rect(0, 0, w, h), 1}}
	r := newRng(name)
	for i := 0; i < 4; i++ {
		x0, y0 := r.below(w), r.below(h)
		rs = append(rs, region{image.Rect(x0, y0, x0+1+r.below(w-x0), y0+1+r.below(h-y0)), r.below(4)})
	}
	return rs
}

func writeResult(dir, name, op, out string, err error) error {
	p := filepath.Join(dir, name+"."+op)
	if err != nil {
		return os.WriteFile(p+".err", []byte(class(err)+"\n"), 0o644)
	}
	return os.WriteFile(p+".out", []byte(out), 0o644)
}

func runDir(in, out string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	ents, err := os.ReadDir(in)
	if err != nil {
		return err
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(in, e.Name()))
		if err != nil {
			return err
		}
		if err := runInput(out, e.Name(), data); err != nil {
			return err
		}
	}
	return nil
}

func runInput(dir, name string, data []byte) error {
	ctx := context.Background()
	rd := func() *bytes.Reader { return bytes.NewReader(data) }
	var werr error
	w := func(op, s string, err error) {
		if e := writeResult(dir, name, op, s, err); e != nil && werr == nil {
			werr = e
		}
	}
	cfg, cfgErr := jp2k.DecodeConfig(rd())
	w("cfg", describeCfg(cfg), cfgErr)
	// DecodeConfig on a reader that returns one byte at a time.
	cfg1, cfg1Err := jp2k.DecodeConfig(iotest.OneByteReader(rd()))
	w("cfg1", describeCfg(cfg1), cfg1Err)
	cw, ch := 64, 64
	if cfgErr == nil {
		limit := int64(maxPixels)
		if strings.HasPrefix(name, "hostile-") {
			limit = maxHostilePixels
		}
		if int64(cfg.Width)*int64(cfg.Height) > limit && !strings.HasPrefix(name, "bigtiles") {
			w("big", fmt.Sprintf("%d %d\n", cfg.Width, cfg.Height), nil)
			return werr
		}
		cw, ch = cfg.Width, cfg.Height
	}
	img, decErr := jp2k.Decode(rd())
	s := ""
	if decErr == nil {
		s = describe(img)
	}
	w("dec", s, decErr)
	// bigtiles (33 Mpixel, 1024x1024 tiles, for the tile lanes): the whole image and component 0.
	if strings.HasPrefix(name, "bigtiles") {
		ps, err := jp2k.DecodeComponentsOptions(ctx, rd(), []int{0}, jp2k.Options{})
		s := ""
		if err == nil {
			s = describePlanes(ps)
		}
		w("c0", s, err)
		return werr
	}
	// Decode on a reader of unknown size that returns one byte at a time (Go's readAll path).
	img1, dec1Err := jp2k.Decode(iotest.OneByteReader(rd()))
	s1 := ""
	if dec1Err == nil {
		s1 = describe(img1)
	}
	w("dec1", s1, dec1Err)

	// The crafted inputs are cheap to fail, not to run a dozen times: they get fewer operations.
	hostile := strings.HasPrefix(name, "hostile-")
	rs := regions(name, cw, ch)
	// The region operations run whatever Decode returned: a malformed tile-part fails only the
	// regions that read it.
	for i, r := range rs {
		if hostile && i > 1 {
			break
		}
		img, err := jp2k.DecodeRegion(ctx, rd(), r.rect, r.reduce)
		s := ""
		if err == nil {
			s = describe(img)
		}
		w(fmt.Sprintf("r%d", i), s, err)
	}

	planes := func(op string, comps []int, o jp2k.Options) {
		ps, err := jp2k.DecodeComponentsOptions(ctx, rd(), comps, o)
		s := ""
		if err == nil {
			s = describePlanes(ps)
		}
		w(op, s, err)
	}
	all := []int{0, 1, 2}
	planes("c0", []int{0}, jp2k.Options{})
	planes("cA", all, jp2k.Options{})
	if !hostile {
		planes("cS", all, jp2k.Options{SeparateRounding: true})
		planes("cJ", all, jp2k.Options{JJ2000: true})
		planes("cS0", []int{0}, jp2k.Options{SeparateRounding: true})
		planes("cJ0", []int{0}, jp2k.Options{JJ2000: true})
	}
	regionPlanes := func(op string, r region, comps []int, o jp2k.Options) {
		ps, err := jp2k.DecodeRegionComponentsOptions(ctx, rd(), r.rect, r.reduce, comps, o)
		s := ""
		if err == nil {
			s = describePlanes(ps)
		}
		w(op, s, err)
	}
	if !hostile {
		regionPlanes("rc0", rs[2], []int{0}, jp2k.Options{})
		regionPlanes("rcA", rs[3], all, jp2k.Options{SeparateRounding: true})
		regionPlanes("rcJ", rs[3], all, jp2k.Options{JJ2000: true})
	}

	// An Image opened once and used for several regions, plain and with each rounding mode.
	openOps(ctx, data, rs, "open", jp2k.Options{}, w)
	if !hostile {
		openOps(ctx, data, rs, "openS", jp2k.Options{SeparateRounding: true}, w)
		openOps(ctx, data, rs, "openJ", jp2k.Options{JJ2000: true}, w)
	}
	return werr
}

// openOps opens data once with o and writes one result per call on the Image: `tag` for Open
// itself (Config and Levels), `tag`0..3 for DecodeRegion of regions 2, 3, 2 again (the second
// call runs on what the first built) and 0 (the whole image last, so a bad tile fails it and not
// the calls before), `tag`c for DecodeRegionComponents of region 4. A failed call does not stop
// the ones after it. Go's Open words its header errors differently from Decode's (and from
// Rust's), so only whether it fails is compared for Open, besides limit and internal errors. The
// Image's own calls compare the error class, which is the message, in full.
func openOps(ctx context.Context, data []byte, rs []region, tag string, o jp2k.Options, w func(op, s string, err error)) {
	h, err := jp2k.OpenOptions(bytes.NewReader(data), int64(len(data)), o)
	if err != nil {
		if class(err) != "limit" && class(err) != "internal" {
			err = errors.New("error")
		}
		w(tag, "", err)
		return
	}
	w(tag, fmt.Sprintf("%slevels %d\n", describeCfg(h.Config()), h.Levels()), nil)
	for k, i := range []int{2, 3, 2, 0} {
		img, err := h.DecodeRegion(ctx, rs[i].rect, rs[i].reduce)
		s := ""
		if err == nil {
			s = describe(img)
		}
		w(fmt.Sprintf("%s%d", tag, k), s, err)
	}
	ps, err := h.DecodeRegionComponents(ctx, rs[4].rect, rs[4].reduce, []int{0})
	s := ""
	if err == nil {
		s = describePlanes(ps)
	}
	w(tag+"c", s, err)
}

func isCodestream(name string) bool {
	switch filepath.Ext(name) {
	case ".j2k", ".j2c", ".jp2":
		return true
	}
	return false
}

// benchDir prints, per fixture that decodes, "<name>:<op>\t<output bytes>\t<ns per decode>" for
// Decode ("dec") and DecodeComponents of component 0 ("c0"). Each is run at least n times and for
// at least 200 ms; examples/bench.rs does the same.
func benchDir(dir string, n int) error {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var names []string
	for _, e := range ents {
		if !e.IsDir() && isCodestream(e.Name()) {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	ctx := context.Background()
	for _, name := range names {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		time1 := func(f func() error) (int64, bool) {
			if f() != nil {
				return 0, false
			}
			t := time.Now()
			i := 0
			for ; i < n || time.Since(t) < 200*time.Millisecond; i++ {
				if f() != nil {
					return 0, false
				}
			}
			return time.Since(t).Nanoseconds() / int64(i), true
		}
		img, err := jp2k.Decode(bytes.NewReader(data))
		if err != nil {
			continue
		}
		if ns, ok := time1(func() error { _, err := jp2k.Decode(bytes.NewReader(data)); return err }); ok {
			r := img.Bounds()
			fmt.Printf("%s:dec\t%d\t%d\n", name, r.Dx()*r.Dy(), ns)
		}
		if ns, ok := time1(func() error { _, err := jp2k.DecodeComponents(ctx, bytes.NewReader(data), []int{0}); return err }); ok {
			r := img.Bounds()
			fmt.Printf("%s:c0\t%d\t%d\n", name, r.Dx()*r.Dy(), ns)
		}
	}
	return nil
}
