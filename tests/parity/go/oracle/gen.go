package main

// The inputs of the parity check (oracle -gen): the fixtures, deterministic mutations of them,
// JP2 files built around two of them, and the crafted codestreams of jp2k-go's security tests.
// The names say what an input is; a "~" in a name marks a mutation.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

type gen struct {
	out    string
	err    error
	counts map[string]int
}

func (g *gen) put(cat, name string, data []byte) {
	if g.err != nil {
		return
	}
	g.counts[cat]++
	g.err = os.WriteFile(filepath.Join(g.out, name), data, 0o644)
}

func genInputs(fixtures, out, seeds string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	g := &gen{out: out, counts: map[string]int{}}
	ents, err := os.ReadDir(fixtures)
	if err != nil {
		return err
	}
	files := map[string][]byte{}
	for _, e := range ents {
		if e.IsDir() || !isCodestream(e.Name()) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(fixtures, e.Name()))
		if err != nil {
			return err
		}
		files[e.Name()] = data
		g.put("fixture", "fx-"+e.Name(), data)
		g.mutations("fx-"+e.Name(), data, false)
		g.psot("fx-"+e.Name(), data)
	}
	g.jp2(files["rgb8_rct_m5.j2k"], files["g64_53.j2k"])
	g.hostile()
	// The decoded jp2k-go fuzz-corpus seeds, as they are.
	if seeds != "" {
		ents, err := os.ReadDir(seeds)
		if err != nil {
			return err
		}
		for _, e := range ents {
			if e.IsDir() {
				continue
			}
			data, err := os.ReadFile(filepath.Join(seeds, e.Name()))
			if err != nil {
				return err
			}
			g.put("fuzzseed", "seed-"+e.Name(), data)
		}
	}
	if g.err != nil {
		return g.err
	}
	var cats []string
	for c := range g.counts {
		cats = append(cats, c)
	}
	sort.Strings(cats)
	total := 0
	for _, c := range cats {
		fmt.Printf("%-10s %6d\n", c, g.counts[c])
		total += g.counts[c]
	}
	fmt.Printf("%-10s %6d\n", "total", total)
	return nil
}

// mainSegments are the offsets of the marker segments of the main header, from the one after SOC
// to the one before the first SOT.
func mainSegments(data []byte) []int {
	soc := bytes.Index(data, []byte{0xff, 0x4f, 0xff, 0x51})
	if soc < 0 {
		return nil
	}
	var offs []int
	for pos := soc + 2; pos+4 <= len(data) && data[pos] == 0xff && data[pos+1] != 0x90; {
		offs = append(offs, pos)
		pos += 2 + int(binary.BigEndian.Uint16(data[pos+2:]))
	}
	return offs
}

// mutations writes the truncations, bit flips, byte sets, length-field edits, deleted marker
// segments, appended bytes and zeroed ranges of one fixture. Large fixtures get a few; the
// crafted inputs (truncOnly) get only the truncations.
func (g *gen) mutations(name string, data []byte, truncOnly bool) {
	L := len(data)
	if L < 16 {
		return
	}
	r := newRng(name)
	put := func(tag string, b []byte) { g.put("mutation", fmt.Sprintf("%s~%s", name, tag), b) }
	cuts := []int{0, 1, 2, 4, 8, 16, 41, 50, 64, L / 8, L / 4, L / 2, 3 * L / 4, L - 2, L - 1}
	nflip, nset, nzero := 24, 8, 3
	if truncOnly {
		nflip, nset, nzero = 0, 0, 0
	}
	if L > 100<<10 {
		cuts = []int{50, L / 2, L - 1}
		nflip, nset, nzero = 6, 0, 1
	}
	seen := map[int]bool{}
	for _, c := range cuts {
		if c >= 0 && c < L && !seen[c] {
			seen[c] = true
			put(fmt.Sprintf("t%d", c), data[:c])
		}
	}
	head := min(L, 160)
	for i := 0; i < nflip; i++ {
		bit := r.below(L * 8)
		if i%2 == 0 {
			bit = r.below(head * 8)
		}
		b := append([]byte(nil), data...)
		b[bit/8] ^= 1 << (bit % 8)
		put(fmt.Sprintf("f%d", bit), b)
	}
	for i := 0; i < nset; i++ {
		pos := r.below(min(L, 200))
		v := [3]byte{0x00, 0xff, byte(r.next())}[i%3]
		b := append([]byte(nil), data...)
		b[pos] = v
		put(fmt.Sprintf("s%d_%02x", pos, v), b)
	}
	if L <= 100<<10 && !truncOnly {
		for k, off := range mainSegments(data) {
			if k >= 12 {
				break
			}
			l := int(binary.BigEndian.Uint16(data[off+2:]))
			for _, nl := range []int{l - 1, l + 1, 0xffff, 0} {
				b := append([]byte(nil), data...)
				binary.BigEndian.PutUint16(b[off+2:], uint16(nl))
				put(fmt.Sprintf("l%d_%d", off, nl), b)
			}
			if end := off + 2 + l; end <= L {
				put(fmt.Sprintf("d%d", off), append(append([]byte(nil), data[:off]...), data[end:]...))
			}
		}
		put("a1", append(append([]byte(nil), data...), 0x55))
		extra := make([]byte, 16)
		for i := range extra {
			extra[i] = byte(r.next())
		}
		put("a16", append(append([]byte(nil), data...), extra...))
	}
	for i := 0; i < nzero; i++ {
		pos := r.below(L)
		b := append([]byte(nil), data...)
		for j := pos; j < min(L, pos+16); j++ {
			b[j] = 0
		}
		put(fmt.Sprintf("z%d", pos), b)
	}
}

// psot writes, for a multi-tile fixture, TestHandleHostile's lying Psot of the first tile-part
// (`~psot<value>`) and its lying TLM marker ahead of the first SOT (`~tlm`).
func (g *gen) psot(name string, data []byte) {
	sot := bytes.Index(data, []byte{0xff, 0x90, 0, 10})
	if sot < 0 || len(data) > 100<<10 || bytes.Index(data[sot+4:], []byte{0xff, 0x90, 0, 10}) < 0 {
		return
	}
	for _, psot := range []uint32{0, 12, 13, 14, 100, 0x7fffffff, 0xffffffff} {
		b := append([]byte(nil), data...)
		binary.BigEndian.PutUint32(b[sot+6:], psot)
		g.put("mutation", fmt.Sprintf("%s~psot%d", name, psot), b)
	}
	tlm := []byte{0xff, 0x55, 0, 16, 0, 0x60, 0, 0, 0, 0, 0, 5, 0, 7, 0xff, 0xff, 0xff, 0xff}
	b := append(append(append([]byte(nil), data[:sot]...), tlm...), data[sot:]...)
	g.put("mutation", name+"~tlm", b)
}

// The builders of internal/fixture.

func box(typ string, payload []byte) []byte {
	b := []byte{0, 0, 0, 0}
	b = append(b, typ...)
	b = append(b, payload...)
	binary.BigEndian.PutUint32(b, uint32(len(b)))
	return b
}

func ihdrBox() []byte {
	p := make([]byte, 14)
	binary.BigEndian.PutUint32(p[0:], 1)
	binary.BigEndian.PutUint32(p[4:], 1)
	binary.BigEndian.PutUint16(p[8:], 1)
	p[10], p[11] = 7, 7
	return box("ihdr", p)
}

func colrBox(enumcs uint32) []byte {
	p := make([]byte, 7)
	p[0] = 1
	binary.BigEndian.PutUint32(p[3:], enumcs)
	return box("colr", p)
}

func cdefBox(entries [][3]uint16) []byte {
	p := make([]byte, 2+6*len(entries))
	binary.BigEndian.PutUint16(p, uint16(len(entries)))
	for i, e := range entries {
		for j, v := range e {
			binary.BigEndian.PutUint16(p[2+6*i+2*j:], v)
		}
	}
	return box("cdef", p)
}

var (
	jp2Sig = []byte{0, 0, 0, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a}
	ftyp   = box("ftyp", []byte("jp2 \x00\x00\x00\x00jp2 "))
)

func cat(parts ...[]byte) []byte {
	var b []byte
	for _, p := range parts {
		b = append(b, p...)
	}
	return b
}

func jp2File(jp2h, stream []byte) []byte {
	return cat(jp2Sig, ftyp, box("jp2h", jp2h), box("jp2c", stream))
}

// jp2 writes JP2 files around an RGB and a grey codestream, after the tests of decoder/jp2_test.go
// and codestream/jp2_test.go: colour specifications, channel definitions, box structure, the
// codestream box in each of its length forms, and every truncation of the file header.
func (g *gen) jp2(rgb, grey []byte) {
	if rgb == nil || grey == nil {
		g.err = fmt.Errorf("fixtures rgb8_rct_m5.j2k and g64_53.j2k are needed")
		return
	}
	put := func(name string, b []byte) { g.put("jp2", "jp2-"+name, b) }
	ihdr := ihdrBox()
	put("plain", jp2File(ihdr, rgb))
	put("grey", jp2File(ihdr, grey))
	for _, cs := range []uint32{0, 1, 12, 14, 16, 17, 18, 20, 24} {
		put(fmt.Sprintf("colr%d", cs), jp2File(cat(ihdr, colrBox(cs)), rgb))
		put(fmt.Sprintf("gcolr%d", cs), jp2File(cat(ihdr, colrBox(cs)), grey))
	}
	put("colr-icc", jp2File(cat(ihdr, box("colr", []byte{2, 0, 0, 1, 2, 3, 4})), rgb))
	put("colr-short", jp2File(cat(ihdr, box("colr", []byte{1, 0})), rgb))
	perms := [][3]uint16{{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}}
	for _, p := range perms {
		put(fmt.Sprintf("cdef%d%d%d", p[0], p[1], p[2]), jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, p[0]}, {1, 0, p[1]}, {2, 0, p[2]}})), rgb))
	}
	put("cdef-incomplete", jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, 1}, {1, 0, 2}})), rgb))
	put("cdef-dup", jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, 1}, {0, 0, 2}, {2, 0, 3}})), rgb))
	put("cdef-asoc0", jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, 0}, {1, 0, 2}, {2, 0, 3}})), rgb))
	put("cdef-asoc9", jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, 1}, {1, 0, 2}, {2, 0, 9}})), rgb))
	for _, typ := range []uint16{1, 2, 65535} {
		put(fmt.Sprintf("cdef-typ%d", typ), jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, 1}, {1, 0, 2}, {2, typ, 3}})), rgb))
	}
	put("cdef-cn9", jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, 1}, {1, 0, 2}, {9, 0, 3}})), rgb))
	put("cdef-empty", jp2File(cat(ihdr, cdefBox(nil)), rgb))
	put("cdef-grey", jp2File(cat(ihdr, cdefBox([][3]uint16{{0, 0, 1}})), grey))
	dup := make([][3]uint16, 700)
	for i := range dup {
		dup[i] = [3]uint16{0, 0, 1}
	}
	put("cdef-700dup", jp2File(cat(ihdr, cdefBox(dup)), rgb))
	trail := append(cdefBox([][3]uint16{{0, 0, 1}}), make([]byte, 5000)...)
	binary.BigEndian.PutUint32(trail, uint32(len(trail)))
	put("cdef-trailing", jp2File(cat(ihdr, trail), rgb))
	// Box structure.
	free := box("free", nil)
	for _, n := range []int{0, 1, 100, 5000} {
		put(fmt.Sprintf("free%d", n), cat(jp2Sig, bytes.Repeat(free, n), box("jp2h", ihdr), box("jp2c", grey)))
	}
	put("xml-before", cat(jp2Sig, ftyp, box("xml ", make([]byte, 300000)), box("jp2h", ihdr), box("jp2c", grey)))
	put("xml-after", cat(jp2Sig, ftyp, box("jp2h", ihdr), box("xml ", make([]byte, 300000)), box("jp2c", grey)))
	put("c-first", cat(jp2Sig, ftyp, box("jp2c", grey), box("jp2h", ihdr)))
	put("no-jp2h", cat(jp2Sig, ftyp, box("jp2c", grey)))
	put("no-ihdr", jp2File(nil, grey))
	put("no-ftyp", cat(jp2Sig, box("jp2h", ihdr), box("jp2c", grey)))
	put("no-jp2c", cat(jp2Sig, ftyp, box("jp2h", ihdr)))
	put("two-ihdr", jp2File(cat(ihdr, ihdr), grey))
	put("two-jp2c", cat(jp2Sig, ftyp, box("jp2h", ihdr), box("jp2c", grey), box("jp2c", rgb)))
	put("pclr", jp2File(cat(ihdr, box("pclr", []byte{0, 1, 1, 7, 0, 0})), grey))
	put("cmap", jp2File(cat(ihdr, box("cmap", []byte{0, 0, 1, 0})), grey))
	put("unknown-box", jp2File(cat(ihdr, box("abcd", []byte{1, 2, 3})), grey))
	put("jp2h-nested", jp2File(cat(ihdr, box("jp2h", ihdr)), grey))
	// The codestream box in each length form.
	half := box("jp2c", grey)
	put("jp2c-half", cat(jp2Sig, ftyp, box("jp2h", ihdr), half[:len(half)/2]))
	long := box("jp2c", grey)
	binary.BigEndian.PutUint32(long, uint32(len(long)+100))
	put("jp2c-long", cat(jp2Sig, ftyp, box("jp2h", ihdr), long))
	zero := box("jp2c", grey)
	binary.BigEndian.PutUint32(zero, 0)
	put("jp2c-lbox0", cat(jp2Sig, ftyp, box("jp2h", ihdr), zero))
	xl := func(size uint64, payload []byte) []byte {
		b := []byte{0, 0, 0, 1, 'j', 'p', '2', 'c', 0, 0, 0, 0, 0, 0, 0, 0}
		binary.BigEndian.PutUint64(b[8:], size)
		return append(b, payload...)
	}
	put("jp2c-xlbox", cat(jp2Sig, ftyp, box("jp2h", ihdr), xl(uint64(16+len(grey)), grey)))
	put("jp2c-xlbox-half", cat(jp2Sig, ftyp, box("jp2h", ihdr), xl(uint64(16+len(grey)), grey[:len(grey)/2])))
	put("jp2c-xlbox-huge", cat(jp2Sig, ftyp, box("jp2h", ihdr), xl(1<<62, grey)))
	put("jp2c-xlbox-max", cat(jp2Sig, ftyp, box("jp2h", ihdr), xl(^uint64(0), grey)))
	put("jp2c-xlbox-small", cat(jp2Sig, ftyp, box("jp2h", ihdr), xl(8, grey)))
	small := box("jp2c", grey)
	binary.BigEndian.PutUint32(small, 4)
	put("jp2c-lbox4", cat(jp2Sig, ftyp, box("jp2h", ihdr), small))
	huge := box("xml ", []byte{1})
	binary.BigEndian.PutUint32(huge, 0xfffffff0)
	put("box-huge", cat(jp2Sig, ftyp, huge, box("jp2h", ihdr), box("jp2c", grey)))
	// Signature.
	bad := jp2File(ihdr, grey)
	bad[9] ^= 1
	put("bad-sig", bad)
	bad = jp2File(ihdr, grey)
	bad[5] = 'X'
	put("bad-sigbox", bad)
	// Every truncation of the part before the codestream, and one in each box header after it.
	file := jp2File(cat(ihdr, colrBox(16)), rgb)
	for n := 0; n < 100 && n < len(file); n++ {
		put(fmt.Sprintf("trunc%d", n), file[:n])
	}
	for n := 0; n < len(file); n += 97 {
		put(fmt.Sprintf("trunc-step%d", n), file[:n])
	}
}

// hostileJ2K is fixture.HostileJ2K: a one-tile, one-component 8-bit codestream with a crafted
// header. poc, extra (raw main-header markers), tpPOC (a tile-part POC body) and tileData may be
// nil.
func hostileJ2K(size, prog, layers, nl int, precincts, poc, extra, tpPOC, tileData []byte) []byte {
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

// headerFlood is fixture.HeaderFlood: n bare tile-parts over a 65535-tile grid of one-pixel
// tiles, with mainExtra at the end of the main header and tpHdr in every tile-part header.
func headerFlood(mainExtra, tpHdr []byte, n int) []byte {
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

// multiTileBomb is nTiles tier-2 bomb tiles; filler sizes the file's work budget to cover one
// tile alone, not all of them together.
func multiTileBomb(nTiles, size, layers, filler int) []byte {
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
		u32(nTiles * size)
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
		u8(0)
		u8(0)
		u16(layers)
		u8(0)
		u8(0)
		u8(0)
		u8(0)
		u8(0)
		u8(1)
	})
	seg(0xFF5C, func() {
		u8(0x40)
		u8(8 << 3)
	})
	for filler > 0 {
		n := min(filler, 65000)
		seg(0xFF64, func() { u16(0); b = append(b, make([]byte, n)...) })
		filler -= n
	}
	tile := bytes.Repeat([]byte{0x80}, layers)
	for i := 0; i < nTiles; i++ {
		u16(0xFF90)
		u16(10)
		u16(i)
		psot := len(b)
		u32(0)
		u8(0)
		u8(1)
		u16(0xFF93)
		b = append(b, tile...)
		binary.BigEndian.PutUint32(b[psot:], uint32(len(b)-psot+6))
	}
	u16(0xFFD9)
	return b
}

func pocRecords(n int) []byte {
	var b []byte
	for i := 0; i < n; i++ {
		b = append(b, 5, 0, 0xFF, 0xFF, 5, 1, 0)
	}
	return b
}

// tier2Bomb is one precinct of about a million 4x4 code-blocks over `layers` layers of "packet
// present, nothing included"; expensive adds a real root-inclusion bit, which defeats the
// exclusion shortcut.
func tier2Bomb(layers int, expensive bool) []byte {
	hdr := bytes.Repeat([]byte{0x80}, layers)
	if expensive {
		hdr[0] = 0xC0
	}
	data := hostileJ2K(4096, 0, layers, 0, nil, nil, nil, nil, append(hdr, make([]byte, 70000)...))
	cod := bytes.Index(data, []byte{0xFF, 0x52})
	data[cod+10], data[cod+11] = 0, 0
	return data
}

// hostile writes the crafted inputs of the security tests (decoder/security_test.go,
// codestream/security_test.go, tier2/security_test.go), each at a few sizes, and every one of the
// deep-level sweeps. The small ones (not the sweeps) also get the mutations of a fixture.
func (g *gen) hostile() {
	put := func(name string, b []byte) {
		g.put("hostile", "hostile-"+name, b)
		if len(b) < 2048 {
			g.mutations("hostile-"+name, b, true)
		}
	}
	for prog, n := range []string{"lrcp", "rlcp", "rpcl", "pcrl", "cprl"} {
		put("layerbomb-"+n, hostileJ2K(32, prog, 65535, 0, []byte{0}, nil, nil, nil, []byte{0}))
		put("layers999-"+n, hostileJ2K(32, prog, 999, 0, []byte{0}, nil, nil, nil, make([]byte, 100)))
		put("packets3-"+n, hostileJ2K(4, prog, 3, 0, []byte{0}, nil, nil, nil, []byte{0}))
	}
	put("poc-window", hostileJ2K(1, 0, 65535, 15, nil, append(pocRecords(30), 0, 0, 0, 1, 16, 1, 0), nil, nil, make([]byte, 16)))
	put("poc-cap", hostileJ2K(1, 0, 65535, 15, nil, pocRecords(31), nil, pocRecords(1), nil))
	put("seglen", hostileJ2K(1, 0, 2, 0, nil, nil, nil, nil, []byte{0xEF, 0xFF, 0x7F, 0xFF, 0x68, 0, 0, 0, 0}))
	for _, size := range []int{1, 16} {
		for _, prog := range []int{2, 3, 4} {
			for _, nl := range []int{17, 20, 30, 31, 32} {
				g.put("hostile", fmt.Sprintf("hostile-deep-%d-%d-%d", size, prog, nl), hostileJ2K(size, prog, 1, nl, nil, nil, nil, nil, make([]byte, 40)))
			}
		}
	}
	ppm := []byte{0xFF, 0x60, 0, 0x0F, 0, 0xFF, 0xFF, 0xFF, 0xFF, 1, 2, 3, 4, 5, 6, 7, 8}
	put("ppm-huge", hostileJ2K(1, 0, 1, 0, nil, nil, ppm, nil, nil))
	put("precinct-geom", hostileJ2K(1024, 0, 1, 0, []byte{0}, nil, nil, nil, []byte{0}))
	cb := hostileJ2K(4096, 0, 1, 0, []byte{0xBB}, nil, nil, nil, []byte{0})
	cod := bytes.Index(cb, []byte{0xFF, 0x52})
	cb[cod+10], cb[cod+11] = 0, 0
	put("cb-budget", cb)
	for _, l := range []int{10, 100, 999} {
		put(fmt.Sprintf("t2bomb-%d", l), tier2Bomb(l, false))
	}
	put("t2bomb-exp-10", tier2Bomb(10, true))
	put("t2bomb-exp-50", tier2Bomb(50, true))
	put("multitile-bomb", multiTileBomb(4, 512, 999, 18000))
	put("header-exhaustion", hostileJ2K(1, 0, 2, 0, nil, nil, nil, nil, []byte{0}))
	lye0 := []byte{0, 0, 0, 0, 33, 1, 0}
	put("uncovered-precincts", hostileJ2K(512, 0, 1, 0, []byte{0}, lye0, nil, nil, nil))
	ppm2 := append([]byte{0xFF, 0x60, 0, 0, 0, 0, 0, 2, 0x58}, make([]byte, 600)...)
	binary.BigEndian.PutUint16(ppm2[2:], uint16(len(ppm2)-2))
	data := hostileJ2K(256, 0, 1, 0, []byte{0}, bytes.Repeat([]byte{0, 0, 0, 0, 33, 1, 2}, 31), ppm2, nil, nil)
	binary.BigEndian.PutUint32(data[24:], 32)
	binary.BigEndian.PutUint32(data[28:], 32)
	data = data[:len(data)-2]
	for i := 1; i < 64; i++ {
		data = append(data, 0xFF, 0x90, 0, 10, 0, byte(i), 0, 0, 0, 14, 0, 1, 0xFF, 0x93)
	}
	put("ppm-shared", append(data, 0xFF, 0xD9))
	base := hostileJ2K(1, 0, 1, 0, nil, nil, nil, nil, []byte{0})
	for _, last := range []int{255, 256} {
		b := append([]byte(nil), base[:len(base)-2]...)
		for n := 1; n <= last; n++ {
			b = append(b, 0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 14, byte(n), 0, 0xFF, 0x93)
		}
		put(fmt.Sprintf("tileparts-%d", last), append(b, 0xFF, 0xD9))
	}
	amp := []byte{}
	for _, h := range []string{"ff4fff510029000000001000000010000000000000000000000010000000100000000000000000000001070101",
		"ff52000d0100000100000404000166ff5c000420f0ff90000a00000000400e0001ff93"} {
		for i := 0; i+1 < len(h); i += 2 {
			var v byte
			fmt.Sscanf(h[i:i+2], "%02x", &v)
			amp = append(amp, v)
		}
	}
	amp = append(amp, bytes.Repeat([]byte{0xff, 0x7b, 0x30, 0x00}, 4096)...)
	put("tier1-amplifier", append(amp, 0xff, 0xd9))
	qcc := []byte{0xff, 0x5d, 0, 4, 0, 0x40}
	coc := []byte{0xff, 0x53, 0, 9, 0, 0, 0, 4, 4, 0, 1}
	com := []byte{0xff, 0x64, 0, 4, 0, 1}
	const size = 4 << 20
	put("flood-main-qcc", headerFlood(bytes.Repeat(qcc, size/len(qcc)), nil, 1))
	put("flood-main-coc", headerFlood(bytes.Repeat(coc, size/len(coc)), nil, 1))
	put("flood-main-com", headerFlood(bytes.Repeat(com, size/len(com)), nil, 1))
	put("flood-tp-qcc", headerFlood(nil, qcc, size/20))
	put("flood-tp-com", headerFlood(nil, com, size/20))
	put("flood-tileparts-1000", headerFlood(nil, nil, 1000))
	put("flood-tileparts-70000", headerFlood(nil, nil, 70000))
}
