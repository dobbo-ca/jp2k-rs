//go:build !arm64

package tier1

// fill reads ahead until more than 32 bits are buffered, enough for two
// renormalizations. It takes and returns c and ct by value so callers
// keep them in registers.
func (d *mqDecoder) fill(c uint64, ct uint) (uint64, uint) {
	d.c, d.ct = c, ct
	for d.ct <= 32 {
		d.byteIn()
	}
	return d.c, d.ct
}
