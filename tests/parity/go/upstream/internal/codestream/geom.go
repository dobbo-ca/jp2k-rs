package codestream

// -- integer helpers (T.800 Annex B geometry) --------------------------

// FloorDivPow2 and CeilDivPow2 use an arithmetic right shift, so they are
// correct for negative a (subband origins go negative near the image
// origin at deep decomposition levels) without special-casing the sign.
// int64 shift avoids 386 32-bit wrap.
func FloorDivPow2(a int, b uint) int { return int(int64(a) >> b) }

func CeilDivPow2(a int, b uint) int { return int((int64(a) + (1 << b) - 1) >> b) }

// CeilDiv is plain integer ceiling division; callers only ever pass
// non-negative operands (tile/reference-grid coordinates), where Go's
// truncating "/" is safe with the "+b-1" bias.
func CeilDiv(a, b int) int { return (a + b - 1) / b }

// -- Annex B partitioning geometry --------------------------------------

// SubbandRect is a rectangle in the coordinate system of whatever it
// belongs to (resolution, subband, or precinct); the type is shared
// because the math is identical, only the origin's meaning differs.
type SubbandRect struct{ X0, Y0, X1, Y1 int }

func (r SubbandRect) Empty() bool { return r.X1 <= r.X0 || r.Y1 <= r.Y0 }

// Intersect is r clipped to o; the result may be empty.
func (r SubbandRect) Intersect(o SubbandRect) SubbandRect {
	return SubbandRect{max(r.X0, o.X0), max(r.Y0, o.Y0), min(r.X1, o.X1), min(r.Y1, o.Y1)}
}
