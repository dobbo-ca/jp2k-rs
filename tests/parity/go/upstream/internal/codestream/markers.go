// Package codestream parses JPEG 2000 codestreams (the marker segments
// of ISO/IEC 15444-1 Annex A) and JP2 files, and resolves the per-tile
// coding parameters and canvas geometry that decoding needs.
package codestream

import (
	"fmt"
	"sort"
	"sync"
	"unsafe"
)

// Marker codes, ISO/IEC 15444-1 Annex A.
const (
	MarkerSOC = 0xFF4F
	MarkerSIZ = 0xFF51
	markerCOD = 0xFF52
	markerCOC = 0xFF53
	markerTLM = 0xFF55
	MarkerPLT = 0xFF58
	markerQCD = 0xFF5C
	markerQCC = 0xFF5D
	markerRGN = 0xFF5E
	markerPOC = 0xFF5F
	markerCOM = 0xFF64
	MarkerPPM = 0xFF60
	markerPPT = 0xFF61
	MarkerSOT = 0xFF90
	markerSOP = 0xFF91
	markerEPH = 0xFF92
	MarkerSOD = 0xFF93
	MarkerEOC = 0xFFD9
)

func markerName(id uint16) string {
	switch id {
	case MarkerSOC:
		return "SOC"
	case MarkerSIZ:
		return "SIZ"
	case markerCOD:
		return "COD"
	case markerCOC:
		return "COC"
	case markerTLM:
		return "TLM"
	case MarkerPLT:
		return "PLT"
	case markerQCD:
		return "QCD"
	case markerQCC:
		return "QCC"
	case markerRGN:
		return "RGN"
	case markerPOC:
		return "POC"
	case markerCOM:
		return "COM"
	case MarkerPPM:
		return "PPM"
	case markerPPT:
		return "PPT"
	case MarkerSOT:
		return "SOT"
	case markerSOP:
		return "SOP"
	case markerEPH:
		return "EPH"
	case MarkerSOD:
		return "SOD"
	case MarkerEOC:
		return "EOC"
	default:
		return fmt.Sprintf("0x%04X", id)
	}
}

// SIZComponent is one entry of SIZ's per-component array.
type SIZComponent struct {
	Precision uint8 // bit depth, 1..38 (we cap at 31, see SIZ parsing)
	Signed    bool
	XRsiz     uint8
	YRsiz     uint8
}

// SIZ is the decoded Image and tile size marker segment.
type SIZ struct {
	Rsiz       uint16
	Xsiz       uint32
	Ysiz       uint32
	XOsiz      uint32
	YOsiz      uint32
	XTsiz      uint32
	YTsiz      uint32
	XTOsiz     uint32
	YTOsiz     uint32
	Csiz       uint16
	Components []SIZComponent

	// Derived, needed by later markers (SOT tile index bound, TLM).
	TilesWide uint32
	TilesHigh uint32
}

// PrecinctSize is one entry of a COD/COC custom precinct-size list.
type PrecinctSize struct {
	PPx uint8 // precinct width exponent, precinct width = 1<<PPx
	PPy uint8
}

// CodingStyle is the SPcod/SPcoc body shared by COD and COC.
type CodingStyle struct {
	NumDecompLevels    uint8 // raw value; resolutions = this + 1
	CodeBlockWidthExp  uint8 // log2 of code-block width, i.e. (raw&0xF)+2
	CodeBlockHeightExp uint8
	CodeBlockStyle     uint8
	Transformation     uint8          // 0 = 9/7 irreversible, 1 = 5/3 reversible
	PrecinctSizes      []PrecinctSize // nil unless custom precincts present
}

// COD is the decoded Coding style default marker segment.
type COD struct {
	Scod                       uint8
	UsesCustomPrecincts        bool
	UsesSOP                    bool
	UsesEPH                    bool
	ProgressionOrder           uint8
	NumberOfLayers             uint16
	MultipleComponentTransform uint8
	CodingStyle                CodingStyle
}

// COC is the decoded Coding style component marker segment.
type COC struct {
	ComponentIndex      uint16
	Scoc                uint8
	UsesCustomPrecincts bool
	CodingStyle         CodingStyle
}

// Quantization is the SQcd/SQcc body shared by QCD and QCC.
type Quantization struct {
	QuantizationStyle uint8 // 0 none, 1 scalar derived, 2 scalar expounded
	GuardBits         uint8
	Exponents         []uint8
	Mantissas         []uint16 // 0 for the no-quantization style
}

// QCD is the decoded Quantization default marker segment.
type QCD struct {
	Sqcd         uint8
	Quantization Quantization
}

// QCC is the decoded Quantization component marker segment.
type QCC struct {
	ComponentIndex uint16
	Sqcc           uint8
	Quantization   Quantization
}

// RGN is the decoded Region-of-interest marker segment.
type RGN struct {
	ComponentIndex uint16
	Srgn           uint8
	SPrgn          uint8
}

// POCChange is one progression-change entry inside a POC marker segment.
type POCChange struct {
	RSpoc  uint8
	CSpoc  uint16
	LYEpoc uint16
	REpoc  uint8
	CEpoc  uint16
	Ppoc   uint8
}

// POC is the decoded Progression order change marker segment.
type POC struct {
	Changes []POCChange
}

// COM is the decoded Comment marker segment.
type COM struct {
	Rcom uint16 // 0 = binary, 1 = ISO 8859-15 text
	Data []byte
}

// TLMEntry is one tile-part length entry inside a TLM marker segment.
type TLMEntry struct {
	Ttlm uint32
	Ptlm uint32
}

// TLM is the decoded Tile-part lengths marker segment. Per the reference
// spec, any internal inconsistency invalidates the index rather than
// failing the parse: TLM is an optional optimization hint.
type TLM struct {
	Ztlm    uint8
	Stlm    uint8
	Entries []TLMEntry
	Invalid bool
}

// PLT is the decoded Packet length, tile-part header marker segment.
type PLT struct {
	Zplt          uint8
	PacketLengths []uint64
}

// SOT is the decoded Start of tile-part marker segment.
type SOT struct {
	Isot  uint16
	Psot  uint32
	TPsot uint8
	TNsot uint8
}

// PPMEntry is one PPM marker's payload, keyed by Zppm (T.800 A.7.4): PPM
// markers can arrive out of Z order and are merged by mergePPM.
type PPMEntry struct {
	Z    uint8
	Data []byte
}

// PPTEntry is one PPT marker's payload, keyed by Zppt (T.800 A.7.5).
type PPTEntry struct {
	Z    uint8
	Data []byte
}

// TilePart holds one tile-part's header markers plus the location of its
// (unparsed) packet data within the codestream.
type TilePart struct {
	SOT SOT
	COD *COD
	COC []COC
	QCD *QCD
	QCC []QCC
	RGN []RGN
	POC []POC
	COM []COM
	PLT []PLT
	PPT []PPTEntry

	DataOffset int // offset of first data byte after SOD
	DataLength int
}

// tpRec is a tile-part as decoding keeps it: 40 bytes, not TilePart's
// 216, since a flood of 14-byte tile-parts is otherwise 15 times its
// size in heap. Its markers are shared, so treat them as read-only.
type tpRec struct {
	SOT SOT
	*headerMarkers
	DataOffset int
	DataLength int
}

// noMarkers is the header of every tpRec without markers.
var noMarkers = &headerMarkers{}

func newTPRec(sot SOT, hm headerMarkers, off, n int) tpRec {
	h := noMarkers
	if hm.COD != nil || hm.QCD != nil || len(hm.COC)+len(hm.QCC)+len(hm.RGN)+len(hm.POC)+len(hm.COM)+len(hm.PLT)+len(hm.PPT) > 0 {
		h = new(headerMarkers)
		*h = hm
	}
	return tpRec{SOT: sot, headerMarkers: h, DataOffset: off, DataLength: n}
}

func (r *tpRec) tilePart() TilePart {
	return TilePart{SOT: r.SOT, COD: r.COD, COC: r.COC, QCD: r.QCD, QCC: r.QCC, RGN: r.RGN, POC: r.POC, COM: r.COM, PLT: r.PLT, PPT: r.PPT, DataOffset: r.DataOffset, DataLength: r.DataLength}
}

// Codestream is the result of parsing a full JPEG 2000 codestream's marker
// segments (main header plus every tile-part header).
type Codestream struct {
	SIZ *SIZ
	COD *COD
	COC []COC
	QCD *QCD
	QCC []QCC
	RGN []RGN
	POC []POC
	COM []COM
	TLM []TLM
	PPM []PPMEntry

	TileParts []TilePart
	EOC       bool

	// Truncated is set when the stream ends mid tile-part with no EOC
	// (OpenJPEG's NEOC state); the parsed prefix is still returned.
	Truncated bool

	Warnings []string

	// ppmOnce/pptOnce cache the merged PPM stream and per-tile merged PPT
	// streams (see PPMHeaders and PPTHeaders): without this, decoding
	// every tile of a codestream re-sorts and re-copies the whole PPM/PPT
	// marker set once per tile, an O(tiles*markers) re-scan for no reason,
	// since a Codestream's markers never change after Parse
	// returns. sync.Once rather than a plain bool since tier2.Decode is
	// documented as safe to call per-tile, which callers may do
	// concurrently on the same *Codestream.
	ppmOnce   sync.Once
	ppmData   []byte
	ppmErr    error
	pptOnce   sync.Once
	pptByTile map[uint16][]byte
	pptErr    error
	tpOnce    sync.Once
	parts     []tpRec // see Parts
	ParseCost int64   // heap the parse held, roughly
	tpStart   []int32 // see TilePartIndex
	tpIdx     []int32
}

// maxWarnings caps Codestream.Warnings: a stream of 4-byte unknown
// markers would otherwise cost ~30 bytes of warnings per input byte.
const maxWarnings = 100

// parseState is what one parse shares across its headers.
type parseState struct {
	// warnings past maxWarnings are only counted: formatting them cost
	// more than parsing their markers.
	warnings []string
	dropped  int
	// keep keeps what decoding never reads or only reads the first of:
	// PLT, COM, a repeated COC, QCC or RGN in one header. Without it,
	// the parse is a decode's and its heap is capped and reserved.
	keep bool
	// cost is roughly the heap the parse holds; grow, when not nil,
	// reserves twice it from the budget ahead of reserved, for the
	// garbage its slices leave as they grow (decodeImage gives that back
	// as garbage). charged is what the
	// cap counts: the same, less the tile-part records made ahead of the
	// tile-parts, plus the records parsed so far.
	cost, reserved, charged int64
	grow                    func(int64) error
	// seen[k*csiz+c] == hdr when this header already has marker k for
	// component c; hdr counts headers.
	seen []uint32
	hdr  uint32
}

// parseBytesPer and parseSlack cap the heap a decode's parse holds at
// any point to parseBytesPer per input byte up to it, plus parseSlack.
// A bare tile-part costs 40 bytes per 14; one whose header repeats
// markers for 6 bytes each costs over 400 per 20, and fails early.
const parseBytesPer, parseSlack = 8, 1 << 20

// spend notes n more bytes of parse heap, held from offset pos on.
func (ps *parseState) spend(n int64, pos int) error {
	return ps.account(n, n, pos)
}

// account adds n to cost and charge to charged.
func (ps *parseState) account(n, charge int64, pos int) error {
	ps.cost += n
	ps.charged += charge
	if ps.keep {
		return nil
	}
	if ps.charged > parseBytesPer*int64(pos)+parseSlack {
		return fmt.Errorf("jp2k: the headers before offset %d need more memory than their size allows", pos)
	}
	if ps.grow != nil && 2*ps.cost > ps.reserved {
		n := max(2*ps.cost-ps.reserved, 1<<20)
		if err := ps.grow(n); err != nil {
			return err
		}
		ps.reserved += n
	}
	return nil
}

// repeat reports whether this header already had marker kind k (0 COC,
// 1 QCC, 2 RGN) for component c, when the parse drops repeats: the
// decoder reads the first only, and a header of repeats is a cheap way to
// make a big parse.
func (ps *parseState) repeat(k int, c, csiz uint16, pos int) (bool, error) {
	if ps.keep {
		return false, nil
	}
	if ps.seen == nil {
		ps.seen = make([]uint32, 3*int(csiz))
		if err := ps.spend(int64(len(ps.seen))*4, pos); err != nil {
			return false, err
		}
	}
	i := k*int(csiz) + int(c)
	if ps.seen[i] == ps.hdr {
		return true, nil
	}
	ps.seen[i] = ps.hdr
	return false, nil
}

// warn takes a func so a dropped warning costs no formatting.
func (ps *parseState) warn(msg func() string) {
	if len(ps.warnings) >= maxWarnings {
		ps.dropped++
		return
	}
	ps.warnings = append(ps.warnings, msg())
}

// Parse parses a raw JPEG 2000 codestream (no JP2 box wrapper):
// SOC, then the main header up to the first SOT, then every tile-part
// (header markers decoded, packet data located but not decoded) up to EOC.
func Parse(data []byte) (*Codestream, error) {
	return parseCodestream(data, true)
}

func parseCodestream(data []byte, keep bool) (*Codestream, error) {
	return ParseBudgeted(data, keep, nil)
}

// ParseBudgeted is parseCodestream, reserving its heap through grow.
func ParseBudgeted(data []byte, keep bool, grow func(int64) error) (*Codestream, error) {
	ps := &parseState{keep: keep, grow: grow}
	if len(data) < 2 || Be16(data[0:2]) != MarkerSOC {
		return nil, fmt.Errorf("jp2k: missing SOC marker at offset 0")
	}
	pos := 2

	if pos+2 > len(data) || Be16(data[pos:pos+2]) != MarkerSIZ {
		return nil, fmt.Errorf("jp2k: expected SIZ marker immediately after SOC at offset %d", pos)
	}
	siz, next, err := ParseSIZSegment(data, pos)
	if err != nil {
		return nil, err
	}
	pos = next

	cs := &Codestream{SIZ: siz}

	hm, next, err := parseMarkers(data, pos, MarkerSOT, siz.Csiz, true, false, siz.TilesWide, siz.TilesHigh, ps)
	if err != nil {
		return nil, err
	}
	pos = next
	cs.COD = hm.COD
	cs.COC = hm.COC
	cs.QCD = hm.QCD
	cs.QCC = hm.QCC
	cs.RGN = hm.RGN
	cs.POC = hm.POC
	cs.COM = hm.COM
	cs.TLM = hm.TLM
	cs.PPM = hm.PPM

	if cs.COD == nil {
		return nil, fmt.Errorf("jp2k: missing mandatory COD marker in main header")
	}
	if cs.QCD == nil {
		return nil, fmt.Errorf("jp2k: missing mandatory QCD marker in main header")
	}

	// pos now points at the first SOT. Walk tile-parts until EOC.
	// tpNext tracks, per tile, the TPsot expected on its next tile-part;
	// ISO 15444-1 A.4.2 requires strictly increasing TPsot from 0. An int,
	// so a 257th tile-part cannot wrap back to TPsot 0.
	tpNext := make(map[uint16]int)
	// pptSeenZ dedups Zppt across every tile-part of the SAME tile: PPT is
	// scoped per tile (T.800 A.7.5), but parseMarkers only sees one
	// tile-part header at a time, so a Zppt repeated in a later tile-part
	// of the same tile would otherwise slip past its own per-header check.
	pptSeenZ := make(map[uint16]map[uint8]bool)
	// Sized once: regrowth garbage dominated the heap for a flood of tiny
	// tile-parts. The 4 is its entry in the index Parts builds.
	recSize := int64(unsafe.Sizeof(tpRec{})) + 4
	n, end := countTileParts(data, pos)
	if err := ps.account(int64(n)*recSize, 0, end); err != nil {
		return nil, err
	}
	parts := make([]tpRec, 0, n)
	for pos < len(data) {
		if pos+2 > len(data) {
			return nil, fmt.Errorf("jp2k: truncated codestream at offset %d, expected SOT or EOC", pos)
		}
		id := Be16(data[pos : pos+2])
		if id == MarkerEOC {
			cs.EOC = true
			break
		}
		if id != MarkerSOT {
			return nil, fmt.Errorf("jp2k: unexpected marker %s at offset %d, expected SOT or EOC", markerName(id), pos)
		}

		tp, next, truncated, err := parseTilePart(data, pos, siz.Csiz, ps)
		if err != nil {
			return nil, err
		}
		if uint32(tp.SOT.Isot) >= siz.TilesWide*siz.TilesHigh {
			return nil, fmt.Errorf("jp2k: SOT Isot %d out of range at offset %d (tile grid is %dx%d)", tp.SOT.Isot, pos, siz.TilesWide, siz.TilesHigh)
		}
		wantTPsot, ok := tpNext[tp.SOT.Isot]
		cost := int64(0)
		if !ok {
			cost += 48 // a map entry
		}
		if tp.headerMarkers != noMarkers {
			cost += int64(unsafe.Sizeof(headerMarkers{}))
		}
		if len(parts) == cap(parts) {
			cost += 2 * recSize
		}
		if wantTPsot > 255 {
			return nil, fmt.Errorf("jp2k: tile %d has more than 256 tile-parts at offset %d", tp.SOT.Isot, pos)
		}
		if int(tp.SOT.TPsot) != wantTPsot {
			return nil, fmt.Errorf("jp2k: SOT Isot %d TPsot %d out of order at offset %d (expected %d)", tp.SOT.Isot, tp.SOT.TPsot, pos, wantTPsot)
		}
		if tp.SOT.TNsot != 0 && tp.SOT.TPsot >= tp.SOT.TNsot {
			return nil, fmt.Errorf("jp2k: SOT Isot %d TPsot %d >= TNsot %d at offset %d", tp.SOT.Isot, tp.SOT.TPsot, tp.SOT.TNsot, pos)
		}
		tpNext[tp.SOT.Isot] = wantTPsot + 1
		seenZ := pptSeenZ[tp.SOT.Isot]
		if seenZ == nil && len(tp.PPT) > 0 {
			seenZ = make(map[uint8]bool)
			pptSeenZ[tp.SOT.Isot] = seenZ
			cost += 256
		}
		if err := ps.account(cost, cost+recSize, next); err != nil {
			return nil, err
		}
		for _, e := range tp.PPT {
			if seenZ[e.Z] {
				return nil, fmt.Errorf("jp2k: tile %d PPT Zppt %d already read at offset %d", tp.SOT.Isot, e.Z, pos)
			}
			seenZ[e.Z] = true
		}
		parts = append(parts, tp)
		pos = next
		if truncated {
			// No EOC follows; tolerate per the reference NEOC state and
			// return the already-parsed prefix instead of failing outright.
			cs.Truncated = true
			break
		}
	}

	if len(cs.PPM) > 0 {
		// T.800 A.7.5: PPT is illegal anywhere in a codestream that also
		// carries PPM (OpenJPEG j2k.c opj_j2k_read_ppt). PPM is main-header
		// only, so by the time we get here every tile-part is already
		// collected and this one pass over them is enough.
		for _, tp := range parts {
			if len(tp.PPT) > 0 {
				return nil, fmt.Errorf("jp2k: tile %d has PPT markers in a codestream that also uses PPM, which T.800 forbids", tp.SOT.Isot)
			}
		}
	}
	// Decoding reads parts; a Parse caller gets TileParts,
	// which decoding then reads (see Parts).
	if keep {
		cs.TileParts = make([]TilePart, len(parts))
		for i := range parts {
			cs.TileParts[i] = parts[i].tilePart()
		}
	} else {
		cs.parts = parts
	}
	cs.ParseCost = ps.cost

	cs.Warnings = ps.warnings
	if ps.dropped > 0 {
		cs.Warnings = append(cs.Warnings, fmt.Sprintf("%d more warnings not kept", ps.dropped))
	}
	return cs, nil
}

// countTileParts counts the tile-parts from pos by hopping Psot, up to
// the first one of unstated length, and returns where the last starts.
func countTileParts(data []byte, pos int) (n, last int) {
	last = pos
	for pos+12 <= len(data) && Be16(data[pos:]) == MarkerSOT {
		n++
		last = pos
		psot := uint64(Be32(data[pos+6:]))
		if psot < 14 || uint64(pos)+psot > uint64(len(data)) {
			break
		}
		pos += int(psot)
	}
	return n, last
}

// PPMHeaders returns the merged PPM packet-header stream, merged once
// per Codestream.
func (cs *Codestream) PPMHeaders() ([]byte, error) {
	cs.ppmOnce.Do(func() { cs.ppmData, cs.ppmErr = mergePPM(cs.PPM) })
	return cs.ppmData, cs.ppmErr
}

// PPTHeaders returns tile's merged PPT packet-header stream, and false
// when the tile has no PPT markers. PPT is scoped to one tile (T.800
// A.7.5); every tile's stream is merged on the first call.
func (cs *Codestream) PPTHeaders(tile uint16) ([]byte, bool, error) {
	cs.pptOnce.Do(func() {
		byTile := make(map[uint16][]PPTEntry)
		for _, tp := range cs.Parts() {
			if len(tp.PPT) > 0 {
				byTile[tp.SOT.Isot] = append(byTile[tp.SOT.Isot], tp.PPT...)
			}
		}
		merged := make(map[uint16][]byte, len(byTile))
		for ti, entries := range byTile {
			m, err := mergePPT(entries)
			if err != nil {
				cs.pptErr = err
				return
			}
			merged[ti] = m
		}
		cs.pptByTile = merged
	})
	if cs.pptErr != nil {
		return nil, false, cs.pptErr
	}
	hdr, ok := cs.pptByTile[tile]
	return hdr, ok, nil
}

// mergePPM concatenates PPM marker payloads, ascending by Zppm, applying
// the Nppm length framing that spans marker boundaries (T.800 A.7.4 /
// OpenJPEG j2k.c opj_j2k_merge_ppm): the Nppm fields themselves are framing
// only and are stripped from the output, not part of the packet-header bit
// stream mergePPM produces.
func mergePPM(entries []PPMEntry) ([]byte, error) {
	sorted := append([]PPMEntry(nil), entries...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Z < sorted[j].Z })

	size := 0
	for _, e := range sorted {
		size += len(e.Data)
	}
	// Exact size: append growth left 4x garbage.
	out := make([]byte, 0, size)
	var remaining uint64 // bytes of the current Nppm run still owed; uint64 so a 32-bit int cannot wrap
	for _, e := range sorted {
		data := e.Data
		if remaining >= uint64(len(data)) {
			out = append(out, data...)
			remaining -= uint64(len(data))
			continue
		}
		out = append(out, data[:remaining]...)
		data = data[remaining:]
		remaining = 0
		for len(data) > 0 {
			if len(data) < 4 {
				return nil, fmt.Errorf("jp2k: PPM Zppm %d: not enough bytes to read Nppm", e.Z)
			}
			n := Be32(data[:4])
			data = data[4:]
			if uint64(len(data)) >= uint64(n) {
				out = append(out, data[:n]...)
				data = data[n:]
			} else {
				out = append(out, data...)
				remaining = uint64(n) - uint64(len(data))
				data = nil
			}
		}
	}
	if remaining != 0 {
		return nil, fmt.Errorf("jp2k: corrupted PPM markers, %d bytes of the final Nppm run never arrived", remaining)
	}
	return out, nil
}

// mergePPT concatenates PPT marker payloads for one tile, ascending by
// Zppt (T.800 A.7.5 / OpenJPEG j2k.c opj_j2k_merge_ppt): unlike PPM there
// is no length framing, so this is a plain sorted concatenation.
func mergePPT(entries []PPTEntry) ([]byte, error) {
	sorted := append([]PPTEntry(nil), entries...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Z < sorted[j].Z })
	n := 0
	for _, e := range sorted {
		n += len(e.Data)
	}
	out := make([]byte, 0, n)
	for _, e := range sorted {
		out = append(out, e.Data...)
	}
	return out, nil
}

// Be16 and Be32 read big-endian integers; every codestream field is big-endian.
func Be16(b []byte) uint16 { return uint16(b[0])<<8 | uint16(b[1]) }

func Be32(b []byte) uint32 {
	return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
}

// componentIndexWidth is the shared "1 byte if Csiz<=256 else 2 bytes" rule
// used by COC, QCC, RGN and POC component index fields.
func componentIndexWidth(csiz uint16) int {
	if csiz <= 256 {
		return 1
	}
	return 2
}

func readComponentIndex(b []byte, width int) uint16 {
	if width == 1 {
		return uint16(b[0])
	}
	return Be16(b)
}

// readSegment reads the Lxxx length field at data[pos:] (marker id already
// consumed) and returns the payload (Lxxx-2 bytes) plus the offset of the
// next marker.
func readSegment(data []byte, pos int, name string, markerOffset int) (payload []byte, next int, err error) {
	if pos+2 > len(data) {
		return nil, 0, fmt.Errorf("jp2k: truncated %s length field at offset %d", name, markerOffset)
	}
	l := Be16(data[pos : pos+2])
	if l < 2 {
		return nil, 0, fmt.Errorf("jp2k: invalid %s length %d at offset %d", name, l, markerOffset)
	}
	segEnd := pos + int(l)
	if segEnd > len(data) {
		return nil, 0, fmt.Errorf("jp2k: truncated %s segment at offset %d (length %d exceeds remaining data)", name, markerOffset, l)
	}
	return data[pos+2 : segEnd], segEnd, nil
}

// ParseSIZSegment parses SIZ. pos points at the marker id (0xFF51).
func ParseSIZSegment(data []byte, pos int) (*SIZ, int, error) {
	markerOffset := pos
	payload, next, err := readSegment(data, pos+2, "SIZ", markerOffset)
	if err != nil {
		return nil, 0, err
	}
	if len(payload) < 36 {
		return nil, 0, fmt.Errorf("jp2k: SIZ segment too short at offset %d", markerOffset)
	}
	if (len(payload)-36)%3 != 0 {
		return nil, 0, fmt.Errorf("jp2k: SIZ component array length not a multiple of 3 at offset %d", markerOffset)
	}

	s := &SIZ{
		Rsiz:   Be16(payload[0:2]),
		Xsiz:   Be32(payload[2:6]),
		Ysiz:   Be32(payload[6:10]),
		XOsiz:  Be32(payload[10:14]),
		YOsiz:  Be32(payload[14:18]),
		XTsiz:  Be32(payload[18:22]),
		YTsiz:  Be32(payload[22:26]),
		XTOsiz: Be32(payload[26:30]),
		YTOsiz: Be32(payload[30:34]),
		Csiz:   Be16(payload[34:36]),
	}

	if s.Csiz < 1 || s.Csiz > 16384 {
		return nil, 0, fmt.Errorf("jp2k: SIZ Csiz %d out of range [1,16384] at offset %d", s.Csiz, markerOffset)
	}
	if int(s.Csiz) != (len(payload)-36)/3 {
		return nil, 0, fmt.Errorf("jp2k: SIZ Csiz %d does not match component array length at offset %d", s.Csiz, markerOffset)
	}
	if s.XOsiz >= s.Xsiz || s.YOsiz >= s.Ysiz {
		return nil, 0, fmt.Errorf("jp2k: SIZ has zero or negative image size at offset %d", markerOffset)
	}
	if s.XTsiz == 0 || s.YTsiz == 0 {
		return nil, 0, fmt.Errorf("jp2k: SIZ tile size is zero at offset %d", markerOffset)
	}
	tx1 := satAdd32(s.XTOsiz, s.XTsiz)
	ty1 := satAdd32(s.YTOsiz, s.YTsiz)
	if s.XTOsiz > s.XOsiz || s.YTOsiz > s.YOsiz || tx1 <= s.XOsiz || ty1 <= s.YOsiz {
		return nil, 0, fmt.Errorf("jp2k: SIZ tile offset is illegal at offset %d", markerOffset)
	}

	comps := payload[36:]
	s.Components = make([]SIZComponent, s.Csiz)
	for i := 0; i < int(s.Csiz); i++ {
		b := comps[i*3 : i*3+3]
		ssiz := b[0]
		prec := (ssiz & 0x7F) + 1
		if prec > 31 {
			return nil, 0, fmt.Errorf("jp2k: SIZ component %d precision %d exceeds 31 at offset %d", i, prec, markerOffset)
		}
		if b[1] == 0 || b[2] == 0 {
			return nil, 0, fmt.Errorf("jp2k: SIZ component %d has zero subsampling factor at offset %d", i, markerOffset)
		}
		s.Components[i] = SIZComponent{
			Precision: prec,
			Signed:    ssiz&0x80 != 0,
			XRsiz:     b[1],
			YRsiz:     b[2],
		}
	}

	// Promote to 64-bit before the ceil-div: Xsiz-XTOsiz+XTsiz can exceed
	// 2^32 for a legal single-tile image, and uint32 arithmetic wraps.
	tw := uint32((uint64(s.Xsiz-s.XTOsiz) + uint64(s.XTsiz) - 1) / uint64(s.XTsiz))
	th := uint32((uint64(s.Ysiz-s.YTOsiz) + uint64(s.YTsiz) - 1) / uint64(s.YTsiz))
	if tw == 0 || th == 0 || tw > 65535/th {
		return nil, 0, fmt.Errorf("jp2k: SIZ tile grid %dx%d is illegal at offset %d", tw, th, markerOffset)
	}
	s.TilesWide, s.TilesHigh = tw, th

	return s, next, nil
}

func satAdd32(a, b uint32) uint32 {
	sum := uint64(a) + uint64(b)
	if sum > 0xFFFFFFFF {
		return 0xFFFFFFFF
	}
	return uint32(sum)
}

// parseCodingStyle parses the SPcod/SPcoc body shared by COD and COC.
// Returns the decoded style and the number of payload bytes consumed.
func parseCodingStyle(payload []byte, customPrecincts bool, name string, markerOffset int) (CodingStyle, int, error) {
	if len(payload) < 5 {
		return CodingStyle{}, 0, fmt.Errorf("jp2k: %s coding-style body too short at offset %d", name, markerOffset)
	}
	numDecomp := payload[0]
	numResolutions := int(numDecomp) + 1
	if numDecomp > 32 {
		return CodingStyle{}, 0, fmt.Errorf("jp2k: %s decomposition level count %d exceeds 32 at offset %d", name, numDecomp, markerOffset)
	}
	// why: mask reserved bits 4-7 (jpx.js's reading), tolerating them rather
	// than rejecting per OpenJPEG's whole-byte read;
	// no encoder in the wild is known to set them.
	xcb := (payload[1] & 0x0F) + 2
	ycb := (payload[2] & 0x0F) + 2
	if xcb > 10 || ycb > 10 || xcb+ycb > 12 {
		return CodingStyle{}, 0, fmt.Errorf("jp2k: %s code-block size %dx%d is illegal at offset %d", name, 1<<xcb, 1<<ycb, markerOffset)
	}
	cblkStyle := payload[3]
	transform := payload[4]
	if transform > 1 {
		return CodingStyle{}, 0, fmt.Errorf("jp2k: %s transformation %d is invalid at offset %d", name, transform, markerOffset)
	}

	cs := CodingStyle{
		NumDecompLevels:    numDecomp,
		CodeBlockWidthExp:  xcb,
		CodeBlockHeightExp: ycb,
		CodeBlockStyle:     cblkStyle,
		Transformation:     transform,
	}
	consumed := 5
	if customPrecincts {
		if len(payload)-5 < numResolutions {
			return CodingStyle{}, 0, fmt.Errorf("jp2k: %s precinct size list too short at offset %d", name, markerOffset)
		}
		cs.PrecinctSizes = make([]PrecinctSize, numResolutions)
		for i := 0; i < numResolutions; i++ {
			b := payload[5+i]
			ppx, ppy := b&0x0F, b>>4
			if i != 0 && (ppx == 0 || ppy == 0) {
				return CodingStyle{}, 0, fmt.Errorf("jp2k: %s precinct exponent 0 at resolution %d is illegal at offset %d", name, i, markerOffset)
			}
			cs.PrecinctSizes[i] = PrecinctSize{PPx: ppx, PPy: ppy}
		}
		consumed += numResolutions
	}
	return cs, consumed, nil
}

func ParseCOD(payload []byte, markerOffset int) (*COD, error) {
	if len(payload) < 5 {
		return nil, fmt.Errorf("jp2k: COD segment too short at offset %d", markerOffset)
	}
	scod := payload[0]
	if scod&^0x07 != 0 {
		return nil, fmt.Errorf("jp2k: COD Scod value 0x%02X has reserved bits set at offset %d", scod, markerOffset)
	}
	numLayers := Be16(payload[2:4])
	if numLayers < 1 {
		return nil, fmt.Errorf("jp2k: COD number of layers must be >= 1 at offset %d", markerOffset)
	}
	mct := payload[4]
	if mct > 1 {
		return nil, fmt.Errorf("jp2k: COD MCT value %d is invalid at offset %d", mct, markerOffset)
	}
	custom := scod&0x01 != 0
	style, consumed, err := parseCodingStyle(payload[5:], custom, "COD", markerOffset)
	if err != nil {
		return nil, err
	}
	if consumed != len(payload)-5 {
		return nil, fmt.Errorf("jp2k: COD segment has %d trailing bytes at offset %d", len(payload)-5-consumed, markerOffset)
	}
	return &COD{
		Scod:                       scod,
		UsesCustomPrecincts:        custom,
		UsesSOP:                    scod&0x02 != 0,
		UsesEPH:                    scod&0x04 != 0,
		ProgressionOrder:           payload[1],
		NumberOfLayers:             numLayers,
		MultipleComponentTransform: mct,
		CodingStyle:                style,
	}, nil
}

func parseCOC(payload []byte, csiz uint16, markerOffset int) (*COC, error) {
	room := componentIndexWidth(csiz)
	if len(payload) < room+1 {
		return nil, fmt.Errorf("jp2k: COC segment too short at offset %d", markerOffset)
	}
	ccoc := readComponentIndex(payload[:room], room)
	if ccoc >= csiz {
		return nil, fmt.Errorf("jp2k: COC component index %d >= Csiz %d at offset %d", ccoc, csiz, markerOffset)
	}
	scoc := payload[room]
	custom := scoc&0x01 != 0
	style, consumed, err := parseCodingStyle(payload[room+1:], custom, "COC", markerOffset)
	if err != nil {
		return nil, err
	}
	if consumed != len(payload)-room-1 {
		return nil, fmt.Errorf("jp2k: COC segment has trailing bytes at offset %d", markerOffset)
	}
	return &COC{ComponentIndex: ccoc, Scoc: scoc, UsesCustomPrecincts: custom, CodingStyle: style}, nil
}

// parseQuantization parses the SQcd/SQcc body shared by QCD and QCC. It
// consumes the entire remaining payload: band count is derived from length,
// not declared, per the reference spec.
func parseQuantization(payload []byte, name string, markerOffset int) (Quantization, error) {
	if len(payload) < 1 {
		return Quantization{}, fmt.Errorf("jp2k: %s segment too short at offset %d", name, markerOffset)
	}
	sqcx := payload[0]
	style := sqcx & 0x1F
	guard := sqcx >> 5
	body := payload[1:]

	var exponents []uint8
	var mantissas []uint16
	switch style {
	case 1: // scalar derived: exactly one band, 2 bytes
		if len(body) != 2 {
			return Quantization{}, fmt.Errorf("jp2k: %s scalar-derived body must be 2 bytes at offset %d", name, markerOffset)
		}
		expn := body[0] >> 3
		mant := (uint16(body[0]&0x07) << 8) | uint16(body[1])
		exponents = []uint8{expn}
		mantissas = []uint16{mant}
	case 0: // no quantization: 1 byte per band
		exponents = make([]uint8, len(body))
		mantissas = make([]uint16, len(body))
		for i, b := range body {
			exponents[i] = b >> 3
		}
	case 2: // scalar expounded: 2 bytes per band
		if len(body)%2 != 0 {
			return Quantization{}, fmt.Errorf("jp2k: %s scalar-expounded body has a trailing byte at offset %d", name, markerOffset)
		}
		n := len(body) / 2
		exponents = make([]uint8, n)
		mantissas = make([]uint16, n)
		for i := 0; i < n; i++ {
			b0, b1 := body[2*i], body[2*i+1]
			exponents[i] = b0 >> 3
			mantissas[i] = (uint16(b0&0x07) << 8) | uint16(b1)
		}
	default: // 3..31 are reserved by T.800 Table A.28
		return Quantization{}, fmt.Errorf("jp2k: %s has reserved quantization style %d at offset %d", name, style, markerOffset)
	}
	return Quantization{QuantizationStyle: style, GuardBits: guard, Exponents: exponents, Mantissas: mantissas}, nil
}

func parseRGN(payload []byte, csiz uint16, markerOffset int) (*RGN, error) {
	room := componentIndexWidth(csiz)
	if len(payload) != 2+room {
		return nil, fmt.Errorf("jp2k: RGN segment length %d != %d at offset %d", len(payload), 2+room, markerOffset)
	}
	crgn := readComponentIndex(payload[:room], room)
	if crgn >= csiz {
		return nil, fmt.Errorf("jp2k: RGN component index %d >= Csiz %d at offset %d", crgn, csiz, markerOffset)
	}
	return &RGN{ComponentIndex: crgn, Srgn: payload[room], SPrgn: payload[room+1]}, nil
}

func parsePOC(payload []byte, csiz uint16, markerOffset int) (*POC, error) {
	room := componentIndexWidth(csiz)
	chunkSize := 5 + 2*room
	if len(payload) == 0 || len(payload)%chunkSize != 0 {
		return nil, fmt.Errorf("jp2k: POC segment length %d is not a multiple of %d at offset %d", len(payload), chunkSize, markerOffset)
	}
	n := len(payload) / chunkSize
	changes := make([]POCChange, n)
	for i := 0; i < n; i++ {
		b := payload[i*chunkSize : (i+1)*chunkSize]
		off := 1
		cspoc := readComponentIndex(b[off:off+room], room)
		off += room
		lyepoc := Be16(b[off : off+2])
		off += 2
		repoc := b[off]
		off++
		cepoc := readComponentIndex(b[off:off+room], room)
		off += room
		ppoc := b[off]
		changes[i] = POCChange{RSpoc: b[0], CSpoc: cspoc, LYEpoc: lyepoc, REpoc: repoc, CEpoc: cepoc, Ppoc: ppoc}
	}
	return &POC{Changes: changes}, nil
}

func parseCOM(payload []byte, markerOffset int) (*COM, error) {
	if len(payload) < 2 {
		return nil, fmt.Errorf("jp2k: COM segment too short at offset %d", markerOffset)
	}
	return &COM{Rcom: Be16(payload[0:2]), Data: payload[2:]}, nil
}

func parseTLM(payload []byte, markerOffset int, tilesWide, tilesHigh uint32, ps *parseState) (*TLM, error) {
	if len(payload) < 2 {
		return nil, fmt.Errorf("jp2k: TLM segment too short at offset %d", markerOffset)
	}
	ztlm, stlm := payload[0], payload[1]
	st := (stlm >> 4) & 0x03
	sp := (stlm >> 6) & 0x01
	body := payload[2:]

	tlm := &TLM{Ztlm: ztlm, Stlm: stlm}
	if st == 3 {
		tlm.Invalid = true
		ps.warn(func() string {
			return fmt.Sprintf("TLM at offset %d: ST field is invalid (3), index discarded", markerOffset)
		})
		return tlm, nil
	}
	ptlmSize := (int(sp) + 1) * 2
	entrySize := ptlmSize + int(st)
	if len(body)%entrySize != 0 {
		tlm.Invalid = true
		ps.warn(func() string {
			return fmt.Sprintf("TLM at offset %d: entry data is not a multiple of entry size, index discarded", markerOffset)
		})
		return tlm, nil
	}
	n := len(body) / entrySize
	tileCount := tilesWide * tilesHigh
	tlm.Entries = make([]TLMEntry, n)
	for i := 0; i < n; i++ {
		off := i * entrySize
		var ttlm uint32
		switch st {
		case 0:
			ttlm = uint32(i)
		case 1:
			ttlm = uint32(body[off])
		case 2:
			ttlm = uint32(Be16(body[off : off+2]))
		}
		if ttlm >= tileCount {
			tlm.Invalid = true
			ps.warn(func() string {
				return fmt.Sprintf("TLM at offset %d: tile index %d out of range (tile grid is %dx%d), index discarded", markerOffset, ttlm, tilesWide, tilesHigh)
			})
			return tlm, nil
		}
		poff := off + int(st)
		var ptlm uint32
		if sp == 0 {
			ptlm = uint32(Be16(body[poff : poff+2]))
		} else {
			ptlm = Be32(body[poff : poff+4])
		}
		tlm.Entries[i] = TLMEntry{Ttlm: ttlm, Ptlm: ptlm}
	}
	return tlm, nil
}

// ParsePLT decodes Iplt's 7-bit continuation-encoded packet lengths.
// Caps the continuation run at 5 bytes so the uint64 accumulator never
// loses bits to a shift-past-64; the exceeds-1<<32-1 check below then
// runs on a value that was never silently truncated. Without keep the
// lengths are checked but not stored.
func ParsePLT(payload []byte, markerOffset int, keep bool) (*PLT, error) {
	if len(payload) < 1 {
		return nil, fmt.Errorf("jp2k: PLT segment too short at offset %d", markerOffset)
	}
	zplt := payload[0]
	body := payload[1:]

	var lengths []uint64
	var cur uint64
	contBytes := 0
	for _, b := range body {
		cur |= uint64(b & 0x7F)
		if b&0x80 != 0 {
			contBytes++
			if contBytes > 5 {
				return nil, fmt.Errorf("jp2k: PLT packet length continuation run exceeds 5 bytes at offset %d", markerOffset)
			}
			cur <<= 7
		} else {
			if cur > 0xFFFFFFFF {
				return nil, fmt.Errorf("jp2k: PLT packet length exceeds 1<<32-1 at offset %d", markerOffset)
			}
			if keep {
				lengths = append(lengths, cur)
			}
			cur, contBytes = 0, 0
		}
	}
	if contBytes > 0 {
		return nil, fmt.Errorf("jp2k: PLT truncated, trailing continuation byte at offset %d", markerOffset)
	}
	return &PLT{Zplt: zplt, PacketLengths: lengths}, nil
}

func parseSOTSegment(payload []byte, markerOffset int) (SOT, error) {
	if len(payload) != 8 {
		return SOT{}, fmt.Errorf("jp2k: SOT segment length %d != 8 at offset %d", len(payload), markerOffset)
	}
	sot := SOT{
		Isot:  Be16(payload[0:2]),
		Psot:  Be32(payload[2:6]),
		TPsot: payload[6],
		TNsot: payload[7],
	}
	if sot.Psot != 0 && sot.Psot < 14 && sot.Psot != 12 {
		return SOT{}, fmt.Errorf("jp2k: SOT Psot %d is illegal (must be 0 or >= 14) at offset %d", sot.Psot, markerOffset)
	}
	return sot, nil
}

// headerMarkers accumulates the markers parsed by parseMarkers, shared by
// the main header and every tile-part header.
type headerMarkers struct {
	COD *COD
	COC []COC
	QCD *QCD
	QCC []QCC
	RGN []RGN
	POC []POC
	COM []COM
	TLM []TLM
	PLT []PLT
	PPM []PPMEntry
	PPT []PPTEntry
}

// parseMarkers reads marker segments starting at pos until it sees
// terminator (SOT for the main header, SOD for a tile-part header),
// returning the offset of that terminator's marker id. allowTLM/allowPLT
// gate the two markers that are legal in only one of the two contexts;
// an out-of-place TLM/PLT falls through to a skip-with-warning path rather
// than failing the parse. allowTLM doubles as "this is the main header":
// the two call sites are main-header (allowTLM) and tile-part-header
// (allowPLT), never both or neither. tilesWide/tilesHigh are only used to
// validate TLM tile indices and are ignored when allowTLM is false.
func parseMarkers(data []byte, pos int, terminator uint16, csiz uint16, allowTLM, allowPLT bool, tilesWide, tilesHigh uint32, ps *parseState) (headerMarkers, int, error) {
	var hm headerMarkers
	pocChanges := 0 // POC accumulates across markers; J2K_MAX_POCS is 32
	ps.hdr++
	for {
		if pos+2 > len(data) {
			return hm, 0, fmt.Errorf("jp2k: truncated codestream at offset %d, expected marker", pos)
		}
		id := Be16(data[pos : pos+2])
		if id == terminator {
			return hm, pos, nil
		}
		if id < 0xFF00 {
			return hm, 0, fmt.Errorf("jp2k: invalid marker 0x%04X at offset %d", id, pos)
		}
		markerOffset := pos

		// T.800 Table A.2: 0xFF30-0xFF3F are reserved delimiting markers
		// with no marker segment (no length field), unlike every other
		// marker >= 0xFF30 this parser knows about; reading a length field
		// after one would consume unrelated codestream bytes as if it were
		// the segment length.
		if id >= 0xFF30 && id <= 0xFF3F {
			pos += 2
			continue
		}

		switch id {
		case MarkerSOC, MarkerSOD, MarkerEOC, markerSOP, markerEPH:
			return hm, 0, fmt.Errorf("jp2k: unexpected %s marker (no length field) at offset %d", markerName(id), markerOffset)

		case MarkerSIZ:
			// SIZ is legal only once, immediately after SOC; Parse
			// consumes that one before calling parseMarkers, so any SIZ seen
			// here is a second one or one that is out of position.
			return hm, 0, fmt.Errorf("jp2k: misplaced SIZ marker at offset %d (SIZ is only legal once, immediately after SOC)", markerOffset)

		case MarkerPPM:
			payload, next, err := readSegment(data, pos+2, "PPM", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			if !allowTLM {
				// T.800 A.7.4 / OpenJPEG j2k.c: PPM is main-header only.
				// Unlike TLM/PLT (optional hints, safe to skip), silently
				// dropping a misplaced PPM would leave cs.PPM empty and
				// packetHeaderSource would fall through to reading packet
				// headers straight out of the tile data - decoding garbage
				// instead of stopping.
				return hm, 0, fmt.Errorf("jp2k: PPM marker outside main header at offset %d", markerOffset)
			}
			// OpenJPEG rejects a Z byte with an empty payload; match it.
			if len(payload) < 2 {
				return hm, 0, fmt.Errorf("jp2k: PPM segment too short at offset %d", markerOffset)
			}
			z := payload[0]
			for _, e := range hm.PPM {
				if e.Z == z {
					return hm, 0, fmt.Errorf("jp2k: PPM Zppm %d already read at offset %d", z, markerOffset)
				}
			}
			hm.PPM = append(hm.PPM, PPMEntry{Z: z, Data: payload[1:]})
			if err := ps.spend(2*int64(unsafe.Sizeof(PPMEntry{})), next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerPPT:
			payload, next, err := readSegment(data, pos+2, "PPT", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			if !allowPLT {
				// T.800 A.7.5 / OpenJPEG j2k.c: PPT is tile-part-header only.
				return hm, 0, fmt.Errorf("jp2k: PPT marker outside tile-part header at offset %d", markerOffset)
			}
			// OpenJPEG rejects a Z byte with an empty payload; match it.
			if len(payload) < 2 {
				return hm, 0, fmt.Errorf("jp2k: PPT segment too short at offset %d", markerOffset)
			}
			z := payload[0]
			for _, e := range hm.PPT {
				if e.Z == z {
					return hm, 0, fmt.Errorf("jp2k: PPT Zppt %d already read at offset %d", z, markerOffset)
				}
			}
			hm.PPT = append(hm.PPT, PPTEntry{Z: z, Data: payload[1:]})
			if err := ps.spend(2*int64(unsafe.Sizeof(PPTEntry{})), next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerCOD:
			payload, next, err := readSegment(data, pos+2, "COD", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			cod, err := ParseCOD(payload, markerOffset)
			if err != nil {
				return hm, 0, err
			}
			hm.COD = cod
			if err := ps.spend(int64(unsafe.Sizeof(COD{}))+2*int64(len(payload)), next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerCOC:
			payload, next, err := readSegment(data, pos+2, "COC", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			coc, err := parseCOC(payload, csiz, markerOffset)
			if err != nil {
				return hm, 0, err
			}
			cost := int64(unsafe.Sizeof(COC{})) + 2*int64(len(payload))
			dup, err := ps.repeat(0, coc.ComponentIndex, csiz, next)
			if err != nil {
				return hm, 0, err
			}
			if !dup {
				hm.COC = append(hm.COC, *coc)
				cost += 2 * int64(unsafe.Sizeof(COC{}))
			}
			if err := ps.spend(cost, next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerQCD:
			payload, next, err := readSegment(data, pos+2, "QCD", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			if len(payload) < 1 {
				return hm, 0, fmt.Errorf("jp2k: QCD segment too short at offset %d", markerOffset)
			}
			q, err := parseQuantization(payload, "QCD", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			hm.QCD = &QCD{Sqcd: payload[0], Quantization: q}
			if err := ps.spend(int64(unsafe.Sizeof(QCD{}))+3*int64(len(payload))+32, next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerQCC:
			payload, next, err := readSegment(data, pos+2, "QCC", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			room := componentIndexWidth(csiz)
			if len(payload) < room+1 {
				return hm, 0, fmt.Errorf("jp2k: QCC segment too short at offset %d", markerOffset)
			}
			cqcc := readComponentIndex(payload[:room], room)
			if cqcc >= csiz {
				return hm, 0, fmt.Errorf("jp2k: QCC component index %d >= Csiz %d at offset %d", cqcc, csiz, markerOffset)
			}
			q, err := parseQuantization(payload[room:], "QCC", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			cost := 3 * int64(len(payload))
			if len(q.Exponents) > 0 {
				cost += 32
			}
			dup, err := ps.repeat(1, cqcc, csiz, next)
			if err != nil {
				return hm, 0, err
			}
			if !dup {
				hm.QCC = append(hm.QCC, QCC{ComponentIndex: cqcc, Sqcc: payload[room], Quantization: q})
				cost += 2 * int64(unsafe.Sizeof(QCC{}))
			}
			if err := ps.spend(cost, next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerRGN:
			payload, next, err := readSegment(data, pos+2, "RGN", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			rgn, err := parseRGN(payload, csiz, markerOffset)
			if err != nil {
				return hm, 0, err
			}
			cost := int64(unsafe.Sizeof(RGN{}))
			dup, err := ps.repeat(2, rgn.ComponentIndex, csiz, next)
			if err != nil {
				return hm, 0, err
			}
			if !dup {
				hm.RGN = append(hm.RGN, *rgn)
				cost += 2 * int64(unsafe.Sizeof(RGN{}))
			}
			if err := ps.spend(cost, next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerPOC:
			payload, next, err := readSegment(data, pos+2, "POC", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			poc, err := parsePOC(payload, csiz, markerOffset)
			if err != nil {
				return hm, 0, err
			}
			pocChanges += len(poc.Changes)
			if pocChanges >= 32 {
				return hm, 0, fmt.Errorf("jp2k: too many POC changes (%d >= 32) at offset %d", pocChanges, markerOffset)
			}
			hm.POC = append(hm.POC, *poc)
			if err := ps.spend(2*int64(unsafe.Sizeof(POC{}))+2*int64(len(payload)), next); err != nil {
				return hm, 0, err
			}
			pos = next

		case markerCOM:
			payload, next, err := readSegment(data, pos+2, "COM", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			com, err := parseCOM(payload, markerOffset)
			if err != nil {
				return hm, 0, err
			}
			if ps.keep {
				hm.COM = append(hm.COM, *com)
				ps.cost += 2 * int64(unsafe.Sizeof(COM{}))
			}
			pos = next

		case markerTLM:
			payload, next, err := readSegment(data, pos+2, "TLM", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			if !allowTLM {
				ps.warn(func() string {
					return fmt.Sprintf("TLM marker outside main header at offset %d, skipped", markerOffset)
				})
				pos = next
				continue
			}
			tlm, err := parseTLM(payload, markerOffset, tilesWide, tilesHigh, ps)
			if err != nil {
				return hm, 0, err
			}
			hm.TLM = append(hm.TLM, *tlm)
			if err := ps.spend(2*int64(unsafe.Sizeof(TLM{}))+4*int64(len(payload)), next); err != nil {
				return hm, 0, err
			}
			pos = next

		case MarkerPLT:
			payload, next, err := readSegment(data, pos+2, "PLT", markerOffset)
			if err != nil {
				return hm, 0, err
			}
			if !allowPLT {
				ps.warn(func() string {
					return fmt.Sprintf("PLT marker outside tile-part header at offset %d, skipped", markerOffset)
				})
				pos = next
				continue
			}
			plt, err := ParsePLT(payload, markerOffset, ps.keep)
			if err != nil {
				return hm, 0, err
			}
			if ps.keep {
				hm.PLT = append(hm.PLT, *plt)
				ps.cost += 2*int64(unsafe.Sizeof(PLT{})) + 8*int64(len(payload))
			}
			pos = next

		default:
			payload, next, err := readSegment(data, pos+2, "", markerOffset)
			if err != nil {
				// Named only now: markerName formats, per marker.
				_, _, err = readSegment(data, pos+2, markerName(id), markerOffset)
				return hm, 0, err
			}
			// Unknown markers are tolerated only in the main header; a
			// tile-part header must recognize every marker it contains.
			if !allowTLM {
				return hm, 0, fmt.Errorf("jp2k: unknown marker %s not allowed in tile-part header at offset %d", markerName(id), markerOffset)
			}
			ps.warn(func() string {
				return fmt.Sprintf("unknown marker %s at offset %d (length %d), skipped", markerName(id), markerOffset, len(payload)+2)
			})
			pos = next
		}
	}
}

// findTileDataEnd is used when Psot==0 ("runs to next SOT or EOC") or
// Psot==12 (Kakadu's empty-SOT quirk, whose real length Psot doesn't give).
// Scanning for a raw 0xFF90/0xFFD9 byte pair is safe here: JPEG 2000 bit
// stuffing guarantees a genuine 0xFF in packet data is never followed by a
// byte > 0x8F, so any such pair in the data is a real marker, not stuffed
// bitstream content.
func findTileDataEnd(data []byte, dataStart int) int {
	for i := dataStart; i+1 < len(data); i++ {
		if data[i] == 0xFF && (data[i+1] == 0x90 || data[i+1] == 0xD9) {
			return i
		}
	}
	return len(data)
}

// parseTilePart parses one tile-part starting at the SOT marker id (pos)
// and returns it plus the offset of the following marker. The bool result
// reports a truncated tile-part (Psot points past end of data): the caller
// should stop walking tile-parts but treat the parse as successful, per the
// reference spec's NEOC tolerance for a stream missing its EOC.
func parseTilePart(data []byte, pos int, csiz uint16, ps *parseState) (tpRec, int, bool, error) {
	segStart := pos
	sotPayload, next, err := readSegment(data, pos+2, "SOT", pos)
	if err != nil {
		return tpRec{}, 0, false, err
	}
	sot, err := parseSOTSegment(sotPayload, pos)
	if err != nil {
		return tpRec{}, 0, false, err
	}
	pos = next

	if sot.Psot == 12 {
		// Kakadu's "empty SOT" quirk: Psot==12 is a warning, not an error.
		// The reference (OpenJPEG j2k.c:4544-4552, 4631-4632) sets its
		// remaining-length budget to Psot-12==0 and keeps parsing the
		// tile-part header and SOD normally, so we do the same and fall
		// back to scanning for the data end like the Psot==0 case below.
		ps.warn(func() string { return fmt.Sprintf("SOT at offset %d: empty SOT marker detected (Psot==12)", segStart) })
	}

	hm, next, err := parseMarkers(data, pos, MarkerSOD, csiz, false, true, 0, 0, ps)
	if err != nil {
		return tpRec{}, 0, false, err
	}
	pos = next // at SOD marker id
	if pos+2 > len(data) {
		return tpRec{}, 0, false, fmt.Errorf("jp2k: truncated SOD marker at offset %d", pos)
	}
	pos += 2 // past SOD, start of tile-part data
	dataStart := pos

	var dataEnd int
	var truncated bool
	if sot.Psot == 0 || sot.Psot == 12 {
		dataEnd = findTileDataEnd(data, dataStart)
		if dataEnd == len(data) {
			// Scanned to the end of the stream without finding a
			// following SOT or EOC: mid tile-part with no EOC.
			truncated = true
		}
	} else {
		// uint64: segStart+Psot overflows a 32-bit int.
		if end := uint64(segStart) + uint64(sot.Psot); end > uint64(len(data)) {
			dataEnd = len(data)
			truncated = true
		} else {
			dataEnd = int(end)
		}
		if dataEnd < dataStart {
			return tpRec{}, 0, false, fmt.Errorf("jp2k: SOT Psot %d puts tile-part data out of bounds at offset %d", sot.Psot, segStart)
		}
	}

	return newTPRec(sot, hm, dataStart, dataEnd-dataStart), dataEnd, truncated, nil
}
