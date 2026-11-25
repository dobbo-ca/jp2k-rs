package codestream

// QuantFor resolves the effective QCD/QCC for a component, honoring T.800
// A.6.4/A.6.5: a tile-part QCC/QCD overrides the main header's, same
// precedence order as EffectiveCodingStyle's COC/COD resolution.
func QuantFor(cs *Codestream, tileIdx uint16, comp int) Quantization {
	var tileQCD *QCD
	var tileQCC []QCC
	parts := cs.Parts()
	for _, i := range TilePartIndex(cs, tileIdx) {
		tp := &parts[i]
		if tp.QCD != nil {
			tileQCD = tp.QCD
		}
		tileQCC = append(tileQCC, tp.QCC...)
	}
	for _, qcc := range tileQCC {
		if int(qcc.ComponentIndex) == comp {
			return qcc.Quantization
		}
	}
	if tileQCD != nil {
		return tileQCD.Quantization
	}
	for _, qcc := range cs.QCC {
		if int(qcc.ComponentIndex) == comp {
			return qcc.Quantization
		}
	}
	return cs.QCD.Quantization
}

// TileOverrides collects the COD/COC/POC found in tileIdx's tile-part
// headers. T.800 A.6.1/A.6.2 only allow COD/COC to appear in a tile's first
// tile-part, but nothing here depends on that; it just resolves precedence
// among however many tile-parts declared them.
func TileOverrides(cs *Codestream, tileIdx uint16) (tileCOD *COD, tileCOC []COC, hasPOC bool) {
	parts := cs.Parts()
	for _, i := range TilePartIndex(cs, tileIdx) {
		tp := &parts[i]
		if tp.COD != nil {
			tileCOD = tp.COD
		}
		tileCOC = append(tileCOC, tp.COC...)
		if len(tp.POC) > 0 {
			hasPOC = true
		}
	}
	return tileCOD, tileCOC, hasPOC
}

// TilePartsOf returns a fresh slice of tileIdx's tile-parts in codestream
// order.
func TilePartsOf(cs *Codestream, tileIdx uint16) []tpRec {
	all := cs.Parts()
	idx := TilePartIndex(cs, tileIdx)
	parts := make([]tpRec, len(idx))
	for i, j := range idx {
		parts[i] = all[j]
	}
	return parts
}

// Parts is the tile-parts decoding reads: those a decode's parse
// kept, else TileParts as they are the first time it is called, for a
// Codestream from Parse or built by hand.
func (cs *Codestream) Parts() []tpRec {
	cs.tpOnce.Do(cs.indexTileParts)
	return cs.parts
}

// TilePartIndex is tileIdx's indices into cs.Parts(), in codestream
// order, shared: callers must not modify it. The index is built once, as
// one flat array, since a flood of tiny tile-parts makes it large.
func TilePartIndex(cs *Codestream, tileIdx uint16) []int32 {
	cs.tpOnce.Do(cs.indexTileParts)
	if int(tileIdx)+1 >= len(cs.tpStart) {
		return nil
	}
	return cs.tpIdx[cs.tpStart[tileIdx]:cs.tpStart[int(tileIdx)+1]]
}

func (cs *Codestream) indexTileParts() {
	if cs.parts == nil {
		cs.parts = make([]tpRec, len(cs.TileParts))
		for i := range cs.TileParts {
			tp := &cs.TileParts[i]
			cs.parts[i] = newTPRec(tp.SOT, headerMarkers{COD: tp.COD, COC: tp.COC, QCD: tp.QCD, QCC: tp.QCC, RGN: tp.RGN, POC: tp.POC, COM: tp.COM, PLT: tp.PLT, PPT: tp.PPT}, tp.DataOffset, tp.DataLength)
		}
	}
	n := 0
	for _, tp := range cs.parts {
		n = max(n, int(tp.SOT.Isot)+1)
	}
	// A counting sort: tile t's parts end up at idx[start[t]:start[t+1]].
	start := make([]int32, n+2)
	for _, tp := range cs.parts {
		start[int(tp.SOT.Isot)+2]++
	}
	for t := 2; t < len(start); t++ {
		start[t] += start[t-1]
	}
	idx := make([]int32, len(cs.parts))
	for i, tp := range cs.parts {
		idx[start[int(tp.SOT.Isot)+1]] = int32(i)
		start[int(tp.SOT.Isot)+1]++
	}
	cs.tpStart, cs.tpIdx = start[:n+1], idx
}

// EffectiveCodingStyle resolves T.800 A.6's precedence: a per-component COC
// beats a per-tile/main COD, and a tile-part override beats the main header,
// independently for each axis (tile COC > tile COD > main COC > main COD).
func EffectiveCodingStyle(cs *Codestream, tileCOD *COD, tileCOC []COC, comp int) CodingStyle {
	for _, c := range tileCOC {
		if int(c.ComponentIndex) == comp {
			return c.CodingStyle
		}
	}
	if tileCOD != nil {
		return tileCOD.CodingStyle
	}
	for _, c := range cs.COC {
		if int(c.ComponentIndex) == comp {
			return c.CodingStyle
		}
	}
	return cs.COD.CodingStyle
}

// EffectiveMCT resolves T.800 A.6.1's precedence for the MCT bit: a COD in
// the tile's first tile-part overrides the main header's COD wholesale, and
// MCT is part of SGcod, so it must follow the same tile-COD-beats-main-COD
// rule EffectiveCodingStyle applies to the rest of SGcod.
func EffectiveMCT(cs *Codestream, tileCOD *COD) uint8 {
	if tileCOD != nil {
		return tileCOD.MultipleComponentTransform
	}
	return cs.COD.MultipleComponentTransform
}
