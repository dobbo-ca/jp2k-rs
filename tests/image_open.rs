//! `Image::open` on a header flood: Go's `Open` falls back to decoding each region as
//! `DecodeRegion` does when the headers after the main header are larger than their size allows,
//! where this crate used to return a limit error.
mod common;

use common::*;
use jp2k::{Image, Options, Rect};
use std::sync::atomic::AtomicBool;

#[test]
fn open_falls_back_on_a_tile_part_header_flood() {
    const SIZE: usize = 4 << 20;
    let qcc = [0xff, 0x5d, 0, 4, 0, 0x40];
    let data = header_flood(&[], &qcc, SIZE / 20);
    let mut o = Options::default();
    o.limits.max_alloc = 8 * SIZE;
    let nc = AtomicBool::new(false);
    let img = Image::open(&data, &o, &nc).expect("open falls back");
    assert_eq!(img.config().width, 65535);
    // The fallback decodes as decode_region does, so a call fails the same way.
    let rect = Rect {
        x0: 0,
        y0: 0,
        x1: 10,
        y1: 1,
    };
    assert_eq!(
        img.decode_region(rect, 0, &nc).unwrap_err(),
        jp2k::decode_region(&data, rect, 0, &o, &nc).unwrap_err()
    );
}
