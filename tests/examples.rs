//! Ports of `example_test.go`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::{Image, Rect};

fn rect(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
    Rect { x0, y0, x1, y1 }
}

#[test]
fn example_decode() {
    let img = jp2k::decode(&read("g64_53.j2k"), &opts(), &nc()).unwrap();
    assert_eq!(img.bounds, rect(0, 0, 64, 64));
}

/// Decode one tile's worth of a tiled image at half size.
#[test]
fn example_decode_region() {
    let data = read("g64_multitile.j2k"); // 64x64, 32x32 tiles
    let img = jp2k::decode_region(&data, rect(32, 0, 64, 32), 1, &opts(), &nc()).unwrap();
    assert_eq!(img.bounds, rect(16, 0, 32, 16)); // in half-size coordinates
}

/// Decode one component of a 3-component 16-bit image: the other two cost no tier-1 or wavelet
/// work. Samples are the component's own, not scaled.
#[test]
fn example_decode_components() {
    let data = read("gray16x3_97_nomct.j2k"); // 160x120, no MCT
    let planes = jp2k::decode_components(&data, &[2], &opts(), &nc()).unwrap();
    assert_eq!(planes.len(), 1);
    assert_eq!(planes[0].bounds, rect(0, 0, 160, 120));
}

/// Serve tiles of a file: open it once, then decode each tile.
#[test]
fn example_open() {
    let data = read("g64_multitile.j2k"); // 64x64, 32x32 tiles
    let m = Image::open(&data, &opts(), &nc()).unwrap();
    assert_eq!(
        (m.config().width, m.config().height, m.levels()),
        (64, 64, 3)
    );
    for (tile, want) in [
        (rect(0, 0, 32, 32), rect(0, 0, 32, 32)),
        (rect(32, 0, 64, 32), rect(32, 0, 64, 32)),
    ] {
        assert_eq!(m.decode_region(tile, 0, &nc()).unwrap().bounds, want);
    }
}

/// `Limits` cap what one decode may hold, and tell a refusal from a bad image.
#[test]
fn example_limits() {
    let data = read("g64_53.j2k");
    let mut o = opts();
    o.limits.max_alloc = 512 << 20;
    jp2k::decode(&data, &o, &nc()).unwrap();
    o.limits.max_pixels = 64 * 64 - 1;
    assert!(matches!(
        jp2k::decode(&data, &o, &nc()),
        Err(jp2k::Error::LimitExceeded(_))
    ));
    o.limits.max_pixels = 64 * 64;
    o.limits.max_alloc = 1000;
    assert!(matches!(
        jp2k::decode(&data, &o, &nc()),
        Err(jp2k::Error::LimitExceeded(_))
    ));
}
