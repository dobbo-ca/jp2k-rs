//! Ports of `internal/codestream/tile_test.go` and `security_test.go`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::fuzz::{self, Cod};

fn segment(id: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = id.to_be_bytes().to_vec();
    out.extend(((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// SOC + SIZ (2 components) + a COD with `mct` + a QCD with `guard` guard bits.
fn head(mct: u8, guard: u8) -> Vec<u8> {
    let mut siz = vec![0, 0];
    for v in [64u32, 64, 0, 0, 64, 64, 0, 0] {
        siz.extend(v.to_be_bytes());
    }
    siz.extend(2u16.to_be_bytes());
    siz.extend([7, 1, 1, 7, 1, 1]);
    let mut b = vec![0xFF, 0x4F];
    b.extend(segment(0xFF51, &siz));
    b.extend(segment(0xFF52, &[0, 0, 0, 1, mct, 1, 2, 2, 0, 0]));
    b.extend(segment(0xFF5C, &[guard << 5, 8, 16, 24, 32]));
    b
}

fn sot() -> Vec<u8> {
    segment(0xFF90, &[0, 0, 0, 0, 0, 0, 0, 1])
}

/// A tile's COD overrides the main header's COD wholesale, and MCT is part of SGcod, so the MCT
/// gate follows a tile-part COD's MCT bit rather than the main header's.
#[test]
fn effective_mct_prefers_tile_cod() {
    let mut b = head(1, 1);
    b.extend(sot());
    b.extend(segment(0xFF52, &[0, 0, 0, 1, 0, 1, 2, 2, 0, 0]));
    b.extend([0xFF, 0x93, 1, 2, 0xFF, 0xD9]);
    let cs = fuzz::parse(&b, &limits(), &nc()).unwrap();
    assert_eq!(fuzz::effective_mct(&cs, None), 1, "main header's");
    let tile_cod: Option<&Cod> = fuzz::tile_overrides(&cs, 0).0;
    assert!(tile_cod.is_some());
    assert_eq!(fuzz::effective_mct(&cs, tile_cod), 0, "tile COD's");
}

/// A tile-part QCD/QCC overrides the main header's: conformance/p1_04.j2k's tiles carry their
/// own QCD.
#[test]
fn quant_for_prefers_tile_qcd() {
    let mut b = head(0, 1);
    b.extend(sot());
    b.extend(segment(0xFF5C, &[2 << 5, 8, 16, 24, 32])); // tile QCD, 2 guard bits
    b.extend(segment(0xFF5D, &[1, 3 << 5, 8])); // tile QCC for component 1, 3 guard bits
    b.extend([0xFF, 0x93, 1, 2, 0xFF, 0xD9]);
    let cs = fuzz::parse(&b, &limits(), &nc()).unwrap();
    assert_eq!(fuzz::quant_for(&cs, 0, 0).guard_bits, 2, "tile QCD");
    assert_eq!(fuzz::quant_for(&cs, 0, 1).guard_bits, 3, "tile QCC");
    assert_eq!(
        fuzz::quant_for(&cs, 1, 0).guard_bits,
        1,
        "main QCD: no tile-part for tile 1"
    );
}

/// Decoding never reads PLT lengths, so its parse keeps none, while `parse` still returns them.
#[test]
fn decode_parse_drops_plt() {
    let data = read("plt_rpcl_97.j2k");
    let kept = fuzz::parse(&data, &limits(), &nc()).unwrap();
    let dropped = fuzz::parse_codestream(&data, false, &limits(), &nc()).unwrap();
    let mut n = 0;
    for (i, tp) in kept.tile_parts.iter().enumerate() {
        n += tp.plt().len();
        assert!(
            dropped.tile_parts[i].plt().is_empty(),
            "decode parse kept PLT lengths"
        );
    }
    assert!(n > 0, "fixture has no PLT");
}

/// The tile-part index keeps codestream order per tile, up to the last tile index.
#[test]
fn tile_part_index_keeps_order() {
    let cs =
        fuzz::parse_codestream(&tile_part_flood(3 * 65535 + 2), false, &limits(), &nc()).unwrap();
    for tile in [0u16, 1, 2, 65534] {
        let idx = cs.tile_part_index(tile);
        let want = if tile < 2 { 4 } else { 3 };
        assert_eq!(idx.len(), want, "tile {tile}");
        for (k, &i) in idx.iter().enumerate() {
            let tp = cs.tile_parts[i as usize].sot;
            assert!(
                tp.isot == tile && tp.tpsot as usize == k,
                "tile {tile} part {k}: {tp:?}"
            );
        }
    }
    assert!(cs.tile_part_index(65535).is_empty());
}

/// `parse` keeps everything but still stays within `max_alloc`.
#[test]
fn parse_checks_max_alloc() {
    let data = read("g64_multitile.j2k");
    let l = jp2k::Limits {
        max_alloc: 0,
        ..limits()
    };
    assert!(matches!(
        fuzz::parse(&data, &l, &nc()),
        Err(jp2k::Error::LimitExceeded(_))
    ));
}
