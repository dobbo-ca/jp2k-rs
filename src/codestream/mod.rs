//! Parses JPEG 2000 codestreams (the marker segments of ISO/IEC 15444-1 Annex A) and JP2 files,
//! and resolves the per-tile coding parameters and canvas geometry that decoding needs.

pub mod jp2;
mod markers;
mod tile;

pub use markers::*;
pub use tile::*;

// -- integer helpers (T.800 Annex B geometry) --

/// `floor(a / 2^b)`; an arithmetic shift, so correct for negative `a`.
pub fn floor_div_pow2(a: i64, b: u32) -> i64 {
    a >> b
}

/// `ceil(a / 2^b)`.
pub fn ceil_div_pow2(a: i64, b: u32) -> i64 {
    (a + (1i64 << b) - 1) >> b
}

/// Ceiling division of non-negative operands.
pub fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// A rectangle `[x0, x1) x [y0, y1)` in the coordinate system of whatever it belongs to: the
/// region argument of `decode_region` and `Image::decode_region` (pixels of the image, origin at
/// its top left, as `Raster::bounds` reports), and inside the decoder a resolution, subband or
/// precinct. The math is the same, only the origin's meaning differs. It enforces no limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    /// Left edge, inclusive.
    pub x0: i64,
    /// Top edge, inclusive.
    pub y0: i64,
    /// Right edge, exclusive.
    pub x1: i64,
    /// Bottom edge, exclusive.
    pub y1: i64,
}

impl Rect {
    /// Whether the rectangle holds no pixel (`x1 <= x0` or `y1 <= y0`).
    pub fn empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    /// `self` clipped to `o`; the result may be empty.
    pub fn intersect(&self, o: &Rect) -> Rect {
        Rect {
            x0: self.x0.max(o.x0),
            y0: self.y0.max(o.y0),
            x1: self.x1.min(o.x1),
            y1: self.y1.min(o.y1),
        }
    }
}
