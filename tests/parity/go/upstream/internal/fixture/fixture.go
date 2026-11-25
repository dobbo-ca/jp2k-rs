// Package fixture holds what tests in several packages share: paths to
// the committed test files and the conformance corpus, and builders of
// small synthetic codestreams and JP2 files.
package fixture

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// Dir is the repository's testdata directory. Tests run in their package
// directory, so walk up to go.mod; runtime.Caller breaks under -trimpath.
var Dir = func() string {
	dir, err := os.Getwd()
	if err != nil {
		return "internal/jp2k/testdata"
	}
	for d := dir; ; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, "go.mod")); err == nil {
			return filepath.Join(d, "internal/jp2k/testdata")
		}
		if filepath.Dir(d) == d {
			return filepath.Join(dir, "internal/jp2k/testdata")
		}
	}
}()

// Path is the path of the named file in Dir.
func Path(name string) string { return filepath.Join(Dir, name) }

// BandTilesSize and BandTilesTile are gray16_bandtiles.j2k's image and
// tile size.
const BandTilesSize, BandTilesTile = 128, 64

// BandTilesValue is the flat sample value tile quadrant (qx, qy) holds.
func BandTilesValue(qx, qy int) uint16 { return uint16(1000 + 20000*qx + 40000*qy) }

// RGBABandTilesValue is rgba16_bandtiles.j2k's flat value for
// component c of tile quadrant (qx, qy).
func RGBABandTilesValue(qx, qy, c int) uint16 {
	return uint16(1000 + 12000*qx + 24000*qy + 7000*c)
}

// Read returns the named file in Dir, failing t if it cannot.
func Read(t testing.TB, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(Path(name))
	if err != nil {
		t.Fatalf("reading %s: %v", name, err)
	}
	return data
}

// ConformanceDir is the openjpeg-data checkout the conformance tests
// read: $JP2K_CONFORMANCE_DIR, else /tmp/openjpeg-data. Tests that need
// it skip when it is absent.
func ConformanceDir() string {
	if d := os.Getenv("JP2K_CONFORMANCE_DIR"); d != "" {
		return d
	}
	return "/tmp/openjpeg-data"
}

// HostileJ2K hand-builds a one-tile, one-component 8-bit codestream: the
// audit's crafted inputs are all headers, so bytes beat checked-in files.
// poc, extra (raw main-header markers), tpPOC (a tile-part POC body) and
// tileData may be nil.
func HostileJ2K(size, prog, layers, nl int, precincts []byte, poc, extra, tpPOC, tileData []byte) []byte {
	var b []byte
	u8 := func(v int) { b = append(b, byte(v)) }
	u16 := func(v int) { b = binary.BigEndian.AppendUint16(b, uint16(v)) }
	u32 := func(v int) { b = binary.BigEndian.AppendUint32(b, uint32(v)) }
	seg := func(m int, body func()) {
		u16(m)
		at := len(b)
		u16(0)
		body()
		binary.BigEndian.PutUint16(b[at:], uint16(len(b)-at))
	}
	u16(0xFF4F)
	seg(0xFF51, func() {
		u16(0)
		u32(size)
		u32(size)
		u32(0)
		u32(0)
		u32(size)
		u32(size)
		u32(0)
		u32(0)
		u16(1)
		u8(7)
		u8(1)
		u8(1)
	})
	seg(0xFF52, func() {
		scod := 0
		if precincts != nil {
			scod = 1
		}
		u8(scod)
		u8(prog)
		u16(layers)
		u8(0)
		u8(nl)
		u8(4)
		u8(4)
		u8(0)
		u8(1)
		b = append(b, precincts...)
	})
	seg(0xFF5C, func() {
		u8(0x40)
		for i := 0; i < 3*nl+1; i++ {
			u8(8 << 3)
		}
	})
	if poc != nil {
		seg(0xFF5F, func() { b = append(b, poc...) })
	}
	b = append(b, extra...)
	u16(0xFF90)
	u16(10)
	u16(0)
	psot := len(b)
	u32(0)
	u8(0)
	u8(1)
	if tpPOC != nil {
		seg(0xFF5F, func() { b = append(b, tpPOC...) })
	}
	u16(0xFF93)
	b = append(b, tileData...)
	binary.BigEndian.PutUint32(b[psot:], uint32(len(b)-psot+6))
	u16(0xFFD9)
	return b
}

// TilePartFlood is n bare 14-byte tile-parts spread over a 65535-tile
// grid of one-pixel tiles.
func TilePartFlood(n int) []byte {
	return HeaderFlood(nil, nil, n)
}

// HeaderFlood is TilePartFlood with mainExtra at the end of the main
// header and tpHdr in every tile-part header.
func HeaderFlood(mainExtra, tpHdr []byte, n int) []byte {
	b := []byte{0xff, 0x4f, 0xff, 0x51, 0, 41, 0, 0}
	for _, v := range []uint32{65535, 1, 0, 0, 1, 1, 0, 0} {
		b = binary.BigEndian.AppendUint32(b, v)
	}
	b = append(b, 0, 1, 7, 1, 1)
	b = append(b, 0xff, 0x52, 0, 12, 0, 0, 0, 1, 0, 0, 4, 4, 0, 1)
	b = append(b, 0xff, 0x5c, 0, 4, 0x40, 8<<3)
	b = append(b, mainExtra...)
	for i := 0; i < n; i++ {
		b = append(b, 0xff, 0x90, 0, 10)
		b = binary.BigEndian.AppendUint16(b, uint16(i%65535))
		b = binary.BigEndian.AppendUint32(b, uint32(14+len(tpHdr)))
		b = append(b, byte(i/65535), 0)
		b = append(b, tpHdr...)
		b = append(b, 0xff, 0x93)
	}
	return append(b, 0xff, 0xd9)
}
