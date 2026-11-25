package fixture

import "encoding/binary"

// Box hand-builds one length-prefixed JP2 box (LBox TBox payload).
func Box(typ string, payload []byte) []byte {
	var buf []byte
	buf = append(buf, 0, 0, 0, 0) // LBox placeholder
	buf = append(buf, []byte(typ)...)
	buf = append(buf, payload...)
	binary.BigEndian.PutUint32(buf, uint32(len(buf)))
	return buf
}

// IhdrBox builds a minimal ihdr box (HEIGHT WIDTH NC BPC C UnkC IPR payload).
func IhdrBox() []byte {
	p := make([]byte, 14)
	binary.BigEndian.PutUint32(p[0:4], 1)  // HEIGHT
	binary.BigEndian.PutUint32(p[4:8], 1)  // WIDTH
	binary.BigEndian.PutUint16(p[8:10], 1) // NC
	p[10] = 7                              // BPC (8-bit)
	p[11] = 7                              // C
	return Box("ihdr", p)
}

func ColrBox(enumcs uint32) []byte {
	p := make([]byte, 7)
	p[0] = 1 // METH: enumerated
	binary.BigEndian.PutUint32(p[3:7], enumcs)
	return Box("colr", p)
}

// JP2File assembles a minimal signature+ftyp+jp2h(+extra)+jp2c file.
func JP2File(jp2hChildren []byte, stream []byte) []byte {
	var data []byte
	data = append(data, 0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20) // jP box header
	data = append(data, 0x0d, 0x0a, 0x87, 0x0a)                         // jP box payload
	data = append(data, Box("ftyp", []byte("jp2 \x00\x00\x00\x00jp2 "))...)
	data = append(data, Box("jp2h", jp2hChildren)...)
	data = append(data, Box("jp2c", stream)...)
	return data
}

// CdefBox builds a cdef box from (Cn, Typ, Asoc) triples.
func CdefBox(entries [][3]uint16) []byte {
	p := make([]byte, 2+6*len(entries))
	binary.BigEndian.PutUint16(p[0:2], uint16(len(entries)))
	for i, e := range entries {
		off := 2 + 6*i
		binary.BigEndian.PutUint16(p[off:off+2], e[0])
		binary.BigEndian.PutUint16(p[off+2:off+4], e[1])
		binary.BigEndian.PutUint16(p[off+4:off+6], e[2])
	}
	return Box("cdef", p)
}
