package codestream

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// JP2Signature is the JP2 signature box (ISO/IEC 15444-1 I.5.1): a file
// starting with it is a JP2 box file rather than a raw codestream, and its
// jp2c box's payload (found by JP2Codestream) is parsed the same way a raw
// codestream is.
var JP2Signature = []byte{0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20}

// JP2 box types this package looks for (ISO/IEC 15444-1 Annex I). Only
// enough of the box tree is walked to find jp2c: component depth/colour
// info is redundant with SIZ (which classifyOutput already reads) and
// OpenJPEG treats ihdr/bpcc/colr as advisory, not authoritative (jp2.c
// opj_jp2_read_ihdr/read_colr), so this package doesn't store them.
const (
	BoxJP2H uint32 = 0x6a703268 // "jp2h"
	boxIHDR uint32 = 0x69686472 // "ihdr"
	BoxJP2C uint32 = 0x6a703263 // "jp2c"
	boxPCLR uint32 = 0x70636c72 // "pclr"
	boxCMAP uint32 = 0x636d6170 // "cmap"
	boxCOLR uint32 = 0x636f6c72 // "colr"
	boxCDEF uint32 = 0x63646566 // "cdef"
	BoxJP   uint32 = 0x6a502020 // "jP  " (signature box)
)

// JPMagic is the jP signature box's required 4-byte payload (ISO/IEC
// 15444-1 I.5.1). Decode's own prefix sniff (JP2Signature in decode.go)
// only matches the box's 8-byte header; a corrupt payload would otherwise
// go undetected and this package would attempt a box walk on a file that
// isn't really JP2 (OpenJPEG jp2.c:2565 makes the same check a hard error).
var JPMagic = []byte{0x0d, 0x0a, 0x87, 0x0a}

// CdefEntry is one (Cn, Typ, Asoc) record from a cdef box (ISO/IEC 15444-1
// I.5.3.6): component Cn plays channel role Typ (0=colour, 1=opacity,
// 2=premultiplied opacity), associated with output channel Asoc (1-based
// for Typ==0; ignored for Typ==1/2, which always mean "the alpha channel").
type CdefEntry struct {
	Cn, Typ, Asoc uint16
}

// parseCdef reads a cdef box payload: N(2), then N * (Cn(2) Typ(2) Asoc(2)).
func parseCdef(payload []byte) ([]CdefEntry, error) {
	if len(payload) < 2 {
		return nil, fmt.Errorf("jp2k: malformed cdef box")
	}
	n := int(binary.BigEndian.Uint16(payload[0:2]))
	if len(payload) != 2+6*n {
		return nil, fmt.Errorf("jp2k: malformed cdef box")
	}
	entries := make([]CdefEntry, n)
	seen := make(map[uint16]bool, n)
	for i := 0; i < n; i++ {
		p := payload[2+6*i:]
		entries[i] = CdefEntry{
			Cn:   binary.BigEndian.Uint16(p[0:2]),
			Typ:  binary.BigEndian.Uint16(p[2:4]),
			Asoc: binary.BigEndian.Uint16(p[4:6]),
		}
		// Repeated Cn silently drops channel role.
		if seen[entries[i].Cn] {
			return nil, fmt.Errorf("jp2k: cdef declares component %d more than once", entries[i].Cn)
		}
		seen[entries[i].Cn] = true
	}
	return entries, nil
}

// CdefPermutation turns a cdef box's entries into a permutation mapping
// output channel index -> source component index (SIZ order), so the
// channel that opj_decompress calls "R" is whichever component the file's
// cdef box says is Asoc==1, not always component 0 (ISO/IEC 15444-1
// I.5.3.6; OpenJPEG applies this in opj_jp2_apply_cdef). A cdef box this
// package can't fully resolve (multiple alpha channels, an alpha channel on
// an output type without one, or a channel left unmapped) is a clean error
// rather than a guess: guessing here is exactly the "reorder silently
// ignored" bug this function exists to close.
func CdefPermutation(cdef []CdefEntry, numComps int) ([]int, error) {
	if cdef == nil {
		return nil, nil
	}
	nAlpha := 0
	for _, e := range cdef {
		if e.Typ == 1 || e.Typ == 2 {
			nAlpha++
		}
	}
	if nAlpha > 1 {
		return nil, fmt.Errorf("jp2k: cdef declares more than one alpha channel, not supported")
	}
	if nAlpha == 1 && numComps != 2 && numComps != 4 {
		return nil, fmt.Errorf("jp2k: cdef declares an alpha channel for %d components, not supported", numComps)
	}
	colourChannels := numComps - nAlpha
	perm := make([]int, numComps)
	filled := make([]bool, numComps)
	for _, e := range cdef {
		if int(e.Cn) >= numComps {
			return nil, fmt.Errorf("jp2k: cdef component index %d out of range for %d components", e.Cn, numComps)
		}
		var pos int
		switch e.Typ {
		case 0:
			if e.Asoc == 0 || int(e.Asoc) > colourChannels {
				return nil, fmt.Errorf("jp2k: cdef entry has unsupported Asoc %d for a colour channel", e.Asoc)
			}
			pos = int(e.Asoc) - 1
		case 2:
			// Premultiplied opacity: NRGBA/NRGBA64 output is defined
			// non-premultiplied, so passing it through renders wrong.
			return nil, fmt.Errorf("jp2k: cdef premultiplied opacity (Typ 2) is not supported")
		case 1:
			pos = numComps - 1
		default:
			return nil, fmt.Errorf("jp2k: cdef entry has unsupported Typ %d", e.Typ)
		}
		if filled[pos] {
			return nil, fmt.Errorf("jp2k: cdef maps two components to output channel %d", pos)
		}
		filled[pos] = true
		perm[pos] = int(e.Cn)
	}
	for i, f := range filled {
		if !f {
			return nil, fmt.Errorf("jp2k: cdef does not cover output channel %d", i)
		}
	}
	return perm, nil
}

// walkBoxes streams JP2 boxes (ISO/IEC 15444-1 I.4).
// LBox(4) TBox(4) [XLBox(8)] payload; LBox 0 runs to EOF.
// visit error or false stops walking.
func walkBoxes(data []byte, visit func(typ uint32, payload []byte) (keepGoing bool, err error)) error {
	pos := 0
	for pos < len(data) {
		if pos+8 > len(data) {
			return fmt.Errorf("jp2k: truncated box header at offset %d", pos)
		}
		lbox := binary.BigEndian.Uint32(data[pos : pos+4])
		tbox := binary.BigEndian.Uint32(data[pos+4 : pos+8])
		header := 8
		var length int64
		switch lbox {
		case 1:
			if pos+16 > len(data) {
				return fmt.Errorf("jp2k: truncated XLBox at offset %d", pos)
			}
			length = int64(binary.BigEndian.Uint64(data[pos+8 : pos+16]))
			header = 16
		case 0:
			length = int64(len(data) - pos)
		default:
			length = int64(lbox)
		}
		if length < int64(header) {
			return fmt.Errorf("jp2k: box at offset %d has length %d shorter than its header", pos, length)
		}
		// Compare against remaining bytes, not int64(pos)+length: an XLBox
		// can hold values up to 2^63-1, and pos+length would wrap negative
		// and pass both checks below (silent slice-bounds panic).
		remaining := int64(len(data) - pos)
		// The truncation tolerance applies to a jp2c box's declared length
		// regardless of whether it's a 32-bit LBox or an 8-byte XLBox: both
		// forms are equally "the box runs past what DecodeConfig's bounded
		// read fetched", not a corrupt file.
		if tbox == BoxJP2C && length > remaining {
			length = remaining
		}
		if length > remaining {
			return fmt.Errorf("jp2k: box at offset %d (length %d) runs past end of data", pos, length)
		}
		keepGoing, err := visit(tbox, data[pos+header:pos+int(length)])
		if err != nil {
			return err
		}
		pos += int(length)
		if !keepGoing {
			return nil
		}
	}
	return nil
}

// cdefPeekBytes covers any real component count.
const cdefPeekBytes = 4096

// JP2HChild validates jp2h children box by box.
// Shared across boxes; first colr/cdef wins.
type JP2HChild struct {
	SawIhdr bool
	Cdef    []CdefEntry
	sawColr bool
}

// Peek reports bytes Finish needs.
func (c *JP2HChild) Peek(typ uint32) int64 {
	switch typ {
	case boxCOLR:
		if c.sawColr {
			return 0
		}
		return 7 // checkColr only looks at METH and EnumCS
	case boxCDEF:
		if c.Cdef != nil {
			return 0
		}
		return cdefPeekBytes
	default:
		return 0
	}
}

// Finish validates up to Peek(typ) bytes.
// truncated marks bytes left unread.
func (c *JP2HChild) Finish(typ uint32, partial []byte, truncated bool) error {
	switch typ {
	case boxIHDR:
		c.SawIhdr = true
	case boxPCLR, boxCMAP:
		// Unapplied palette would silently give wrong pixels.
		return fmt.Errorf("jp2k: palette (pclr/cmap) JP2 files are not supported")
	case boxCOLR:
		// Only the first colr box is used (I.5.3.3); later ones are
		// ignored, not validated.
		if !c.sawColr {
			c.sawColr = true
			if err := checkColr(partial); err != nil {
				return err
			}
		}
	case boxCDEF:
		if c.Cdef != nil {
			return nil // a later cdef box is ignored
		}
		if truncated {
			return fmt.Errorf("jp2k: cdef box too large to validate")
		}
		cdef, err := parseCdef(partial)
		if err != nil {
			return err
		}
		c.Cdef = cdef
	}
	return nil
}

// JP2Codestream walks a JP2 box file's top-level boxes and returns the raw
// codestream payload from the jp2c box. Validation is limited to what
// Decode/DecodeConfig need: jp2h must appear (with an ihdr child) before
// jp2c, per the required-box list in ISO/IEC 15444-1 I.5.2/Annex I
// (OpenJPEG jp2.c opj_jp2_read_header enforces the same two boxes; it only
// warns, not errors, on a missing colr). Everything else SIZ already
// carries is read from the codestream itself once Parse runs.
func JP2Codestream(data []byte) ([]byte, []CdefEntry, error) {
	first := true
	var child JP2HChild
	var csData []byte
	found := false
	err := walkBoxes(data, func(typ uint32, payload []byte) (bool, error) {
		if first {
			first = false
			if typ != BoxJP || !bytes.Equal(payload, JPMagic) {
				return false, fmt.Errorf("jp2k: missing or malformed jP signature box")
			}
		}
		if typ == BoxJP2H {
			err := walkBoxes(payload, func(ctyp uint32, cpayload []byte) (bool, error) {
				// Already in memory; nothing to cap.
				if err := child.Finish(ctyp, cpayload, false); err != nil {
					return false, err
				}
				return true, nil
			})
			if err != nil {
				return false, err
			}
		}
		if typ == BoxJP2C {
			if !child.SawIhdr {
				return false, fmt.Errorf("jp2k: jp2c box appears before a jp2h box with an ihdr child")
			}
			csData, found = payload, true
			return false, nil // what follows is not read, as by DecodeRegion
		}
		return true, nil
	})
	if err != nil {
		return nil, nil, err
	}
	if first {
		return nil, nil, fmt.Errorf("jp2k: missing or malformed jP signature box")
	}
	if !found {
		return nil, nil, fmt.Errorf("jp2k: no jp2c box found")
	}
	return csData, child.Cdef, nil
}

// checkColr rejects a colr box this package can't render faithfully. This
// package has no colour management step: it hands the codestream's decoded
// samples straight to image.Image assuming they're already sRGB or
// greyscale (which a codestream's own RCT/ICT, if any, produces). An ICC
// profile (METH==2) or any enumerated colourspace other than sRGB/
// greyscale (e.g. EnumCS==18 sYCC, common in the wild) needs an actual
// colour conversion this package doesn't implement; decoding anyway would
// silently emit wrong-but-plausible colours instead of an error.
func checkColr(payload []byte) error {
	if len(payload) < 3 {
		return fmt.Errorf("jp2k: malformed colr box")
	}
	meth := payload[0]
	if meth != 1 {
		return fmt.Errorf("jp2k: colr specifies an ICC profile colourspace, not supported")
	}
	if len(payload) < 7 {
		return fmt.Errorf("jp2k: malformed colr box")
	}
	enumcs := binary.BigEndian.Uint32(payload[3:7])
	if enumcs != 16 && enumcs != 17 {
		return fmt.Errorf("jp2k: colr enumerated colourspace %d not supported (only sRGB=16, greyscale=17)", enumcs)
	}
	return nil
}
