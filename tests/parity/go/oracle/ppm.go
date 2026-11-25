package main

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"

	jp2k "github.com/dobbo-ca/jp2k-go"
)

// ppmDir is the Go side of the real-file comparison: it decodes every .j2k in dir with the
// default options (as the parity decode does) and writes <stem>.pgm (8-bit gray) or <stem>.ppm
// (8-bit RGB, alpha dropped) to out. examples/ppm.rs is the Rust side. Real-file tiles must
// decode, so any error, and any other pixel format, stops the run.
func ppmDir(dir, out string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	names, err := filepath.Glob(filepath.Join(dir, "*.j2k"))
	if err != nil {
		return err
	}
	sort.Strings(names)
	for _, p := range names {
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		img, err := jp2k.Decode(bytes.NewReader(src))
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		var buf []byte
		ext := ".ppm"
		switch m := img.(type) {
		case *image.Gray:
			ext = ".pgm"
			w, h := m.Rect.Dx(), m.Rect.Dy()
			buf = fmt.Appendf(make([]byte, 0, 32+w*h), "P5\n%d %d\n255\n", w, h)
			for y := range h {
				buf = append(buf, m.Pix[y*m.Stride:y*m.Stride+w]...)
			}
		case *image.RGBA:
			w, h := m.Rect.Dx(), m.Rect.Dy()
			buf = fmt.Appendf(make([]byte, 0, 32+3*w*h), "P6\n%d %d\n255\n", w, h)
			for y := range h {
				row := m.Pix[y*m.Stride:]
				for x := range w {
					buf = append(buf, row[4*x:4*x+3]...)
				}
			}
		default:
			return fmt.Errorf("%s: decoded %T, want *image.Gray or *image.RGBA", p, img)
		}
		name := strings.TrimSuffix(filepath.Base(p), ".j2k") + ext
		if err := os.WriteFile(filepath.Join(out, name), buf, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("%d tiles decoded\n", len(names))
	return nil
}
