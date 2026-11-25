//go:build arm64

package tier1

import "encoding/binary"

// fill reads ahead until more than 32 bits are buffered, enough for two
// renormalizations. It takes and returns c and ct by value so callers
// keep them in registers.
func (d *mqDecoder) fill(c uint64, ct uint) (uint64, uint) {
	// k plain bytes in one load; slower on c7i.
	if ct <= 32 && d.bp+8 <= len(d.data) {
		w := binary.BigEndian.Uint64(d.data[d.bp:])
		k := (40 - ct) >> 3
		low := uint64(1)<<(64-8*k) - 1
		if t := ^w | low; (t-0x0101010101010101)&^t&0x8080808080808080 == 0 {
			d.bp += int(k)
			d.c, d.ct = c+(w<<8&^low)>>(16+ct), ct+8*k
			return d.c, d.ct
		}
	}
	d.c, d.ct = c, ct
	for d.ct <= 32 {
		d.byteIn()
	}
	return d.c, d.ct
}
