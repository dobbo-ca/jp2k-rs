//! Ports of `internal/tier2/tier2_test.go` (geometry, fixtures, bombs).
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::fuzz::{self, Codestream, Cursor, build_tile_geom};
use jp2k::{Budget, Error};

/// A one-tile, one-component 8-bit codestream: enough to build geometry. `cb_exp` is the raw
/// code-block exponent field (exponent - 2), `scod_flags` the SOP/EPH bits of Scod, `qcd` the
/// QCD payload (default: no quantization, 2 guard bits, 8-bit exponents) and `tile` the tile data.
#[allow(clippy::too_many_arguments)]
fn stream_full(
    xsiz: u32,
    ysiz: u32,
    nl: u8,
    cb_exp: u8,
    style: u8,
    precincts: Option<&[u8]>,
    scod_flags: u8,
    qcd: Option<&[u8]>,
    tile: &[u8],
) -> Vec<u8> {
    let mut b = vec![0xFF, 0x4F, 0xFF, 0x51];
    let mut siz = Vec::new();
    siz.extend_from_slice(&0u16.to_be_bytes());
    for v in [xsiz, ysiz, 0, 0, xsiz, ysiz, 0, 0] {
        siz.extend_from_slice(&v.to_be_bytes());
    }
    siz.extend_from_slice(&1u16.to_be_bytes());
    siz.extend_from_slice(&[7, 1, 1]);
    b.extend_from_slice(&((siz.len() + 2) as u16).to_be_bytes());
    b.extend_from_slice(&siz);
    let scod = scod_flags | if precincts.is_some() { 1 } else { 0 };
    let mut cod = vec![scod, 0, 0, 1, 0, nl, cb_exp, cb_exp, style, 1];
    if let Some(p) = precincts {
        cod.extend_from_slice(p);
    }
    b.extend_from_slice(&[0xFF, 0x52]);
    b.extend_from_slice(&((cod.len() + 2) as u16).to_be_bytes());
    b.extend_from_slice(&cod);
    let qcd = match qcd {
        Some(q) => q.to_vec(),
        None => {
            let mut q = vec![0x40];
            q.extend((0..3 * nl as usize + 1).map(|_| 8u8 << 3));
            q
        }
    };
    b.extend_from_slice(&[0xFF, 0x5C]);
    b.extend_from_slice(&((qcd.len() + 2) as u16).to_be_bytes());
    b.extend_from_slice(&qcd);
    b.extend_from_slice(&[0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 0, 0, 1, 0xFF, 0x93]);
    b.extend_from_slice(tile);
    b.extend_from_slice(&[0xFF, 0xD9]);
    b
}

fn stream(
    xsiz: u32,
    ysiz: u32,
    nl: u8,
    cb_exp: u8,
    style: u8,
    precincts: Option<&[u8]>,
) -> Vec<u8> {
    stream_full(xsiz, ysiz, nl, cb_exp, style, precincts, 0, None, &[])
}

fn geom(data: &[u8]) -> jp2k::Result<fuzz::TileGeom> {
    let cs = fuzz::parse(data, &limits(), &nc())?;
    build_tile_geom(&cs, 0, &mut Budget::new(&limits()))
}

/// Resolution/subband/code-block rectangles for a 129x67 image, 5 decomposition levels, 64x64
/// code-blocks, default (15,15) precincts, single tile. Expected values were computed
/// independently from the same Annex B formulas (ceildivpow2 anchored bounds).
#[test]
fn build_tile_geom_odd_size() {
    let tg = geom(&stream(129, 67, 5, 4, 0, None)).unwrap();
    let comp = &tg.comps[0];
    assert_eq!(comp.num_resolutions, 6);
    // (x0, y0, x1, y1, cw, ch) per band.
    let expect: [&[(i64, i64, i64, i64, usize, usize)]; 6] = [
        &[(0, 0, 5, 3, 1, 1)],
        &[(0, 0, 4, 3, 1, 1), (0, 0, 5, 2, 1, 1), (0, 0, 4, 2, 1, 1)],
        &[(0, 0, 8, 5, 1, 1), (0, 0, 9, 4, 1, 1), (0, 0, 8, 4, 1, 1)],
        &[
            (0, 0, 16, 9, 1, 1),
            (0, 0, 17, 8, 1, 1),
            (0, 0, 16, 8, 1, 1),
        ],
        &[
            (0, 0, 32, 17, 1, 1),
            (0, 0, 33, 17, 1, 1),
            (0, 0, 32, 17, 1, 1),
        ],
        &[
            (0, 0, 64, 34, 1, 1),
            (0, 0, 65, 33, 2, 1),
            (0, 0, 64, 33, 1, 1),
        ], // r=5: LH is wider than one block
    ];
    for (r, bands) in expect.iter().enumerate() {
        let res = &comp.resolutions[r];
        assert_eq!(res.bands.len(), bands.len(), "r={r}");
        for (b, w) in bands.iter().enumerate() {
            let got = res.bands[b].rect;
            assert_eq!(
                (got.x0, got.y0, got.x1, got.y1),
                (w.0, w.1, w.2, w.3),
                "r={r} band={b}"
            );
            assert_eq!(res.bands[b].num_precincts(), 1, "r={r} band={b}");
            assert_eq!(
                res.bands[b].code_block_grid(0),
                (w.4, w.5),
                "r={r} band={b}"
            );
        }
    }
}

/// HT block coding (0x40/0x80) is the one code-block style not decoded; 0x01-0x20 must NOT be
/// rejected.
#[test]
fn build_tile_geom_rejects_ht_code_block_style() {
    assert!(matches!(
        geom(&stream(64, 64, 1, 4, 0x40, None)),
        Err(Error::Unsupported(_))
    ));
    assert!(geom(&stream(64, 64, 1, 4, 0x3F, None)).is_ok());
}

fn load(name: &str) -> (Codestream, Vec<u8>) {
    let data = read(name);
    (fuzz::parse(&data, &limits(), &nc()).unwrap(), data)
}

fn decode(
    cs: &Codestream,
    data: &[u8],
) -> jp2k::Result<(Vec<fuzz::PacketRef>, fuzz::TileGeom, Vec<u8>)> {
    let mut b = Budget::new(&limits());
    let (p, g, t) = fuzz::decode_tile(cs, data, 0, &mut Cursor::new(), None, &nc(), &mut b)?;
    Ok((p, g, t.into_owned()))
}

fn test_fixture(name: &str) {
    let (cs, data) = load(name);
    assert!(
        cs.cod.uses_sop && cs.cod.uses_eph,
        "{name}: expected SOP and EPH to be in use"
    );
    let tg = build_tile_geom(&cs, 0, &mut Budget::new(&limits())).unwrap();
    let total: usize = tg
        .comps
        .iter()
        .flat_map(|c| c.resolutions.iter())
        .map(|r| r.precinct_count())
        .sum();
    // LRCP/RLCP/RPCL all visit the same packet set, just in different order.
    let want = cs.cod.number_of_layers as usize * cs.siz.csiz as usize * total;
    let (pkts, _, tile_data) = decode(&cs, &data).unwrap();
    assert_eq!(pkts.len(), want, "{name}");
    for (i, p) in pkts.iter().enumerate() {
        assert_eq!(p.nsop, i as i64, "{name}: packet {i} Nsop");
        assert!(
            p.header_pos <= tile_data.len(),
            "{name}: packet {i} header_pos"
        );
        for cb in &p.blocks {
            for s in &cb.segments {
                assert!(
                    s.end <= tile_data.len() && s.start <= s.end,
                    "{name}: packet {i}"
                );
            }
        }
    }
}

#[test]
fn fixture_progressions() {
    for n in ["a_lrcp.j2k", "b_rlcp.j2k", "c_rpcl.j2k", "d_96x80.j2k"] {
        test_fixture(n);
    }
}

/// Segment byte ranges within a packet are non-empty and contiguous across its code-blocks.
#[test]
fn code_block_offsets_are_contiguous() {
    for name in ["a_lrcp.j2k", "b_rlcp.j2k", "c_rpcl.j2k", "d_96x80.j2k"] {
        let (cs, data) = load(name);
        let (pkts, _, _) = decode(&cs, &data).unwrap();
        let mut saw_multi = false;
        for (pi, p) in pkts.iter().enumerate() {
            saw_multi |= p.blocks.len() > 1;
            for (bi, cb) in p.blocks.iter().enumerate() {
                assert_eq!(cb.segments.len(), 1, "{name} packet {pi} block {bi}");
                let seg = cb.segments[0];
                assert!(seg.start < seg.end, "{name} packet {pi} block {bi}: empty");
                if bi > 0 {
                    let prev = p.blocks[bi - 1].segments[0];
                    assert_eq!(seg.start, prev.end, "{name} packet {pi} block {bi}");
                }
            }
        }
        assert!(saw_multi, "{name}: no packet had more than one code-block");
    }
}

/// A crafted SIZ (huge Xsiz/Ysiz, PPx=PPy=0 legal at r=0) drives pw*ph past the range: an error,
/// not a panic.
#[test]
fn rejects_precinct_overflow_and_bomb() {
    let e = geom(&stream(0xFFFF_FFFF, 0xFFFF_FFFF, 0, 4, 0, Some(&[0x00]))).unwrap_err();
    assert!(!matches!(e, Error::Internal(_)), "{e}");
    // A grid that does not overflow but is still absurd (millions of empty precincts).
    let e = geom(&stream(0x0800_0000, 0x0800_0000, 0, 4, 0, None)).unwrap_err();
    assert!(matches!(e, Error::LimitExceeded(_)), "{e}");
}

/// A 1024x1024 tile of 1x1 precincts, and a 4100x4100 tile of 4x4 code-blocks: bounded.
#[test]
fn rejects_precinct_and_code_block_grid_bombs() {
    let mut o = limits();
    o.max_alloc = 64 << 20;
    let cs = fuzz::parse(&stream(1024, 1024, 0, 4, 0, Some(&[0x00])), &o, &nc()).unwrap();
    assert!(build_tile_geom(&cs, 0, &mut Budget::new(&o)).is_err());
    let cs = fuzz::parse(&stream(4100, 4100, 0, 0, 0, None), &o, &nc()).unwrap();
    assert!(build_tile_geom(&cs, 0, &mut Budget::new(&o)).is_err());
}

/// `passes_read`: 3*Mb - 2 with the tile's largest Mb, between 1 and 109.
#[test]
fn passes_read_range() {
    let (cs, _) = load("g64_53.j2k");
    let n = fuzz::passes_read(&cs, 0);
    assert!((1..=109).contains(&n), "{n}");
}

/// A tile-part COD overrides the main header's coding style for that tile (T.800 A.6.1).
#[test]
fn tile_part_cod_overrides_main() {
    let (cs, _) = load("region_rgb_53.j2k");
    let (cod, _, _) = fuzz::tile_overrides(&cs, 0);
    let style = fuzz::effective_coding_style(&cs, cod, &[], 0);
    assert_eq!(
        style.num_decomp_levels,
        cs.cod.coding_style.num_decomp_levels
    );
}

/// SOP is optional per packet even when Scod signals its use (T.800 A.8.1; OpenJPEG warns
/// rather than fails when one is missing): a packet without it decodes, with `nsop` -1.
#[test]
fn sop_is_optional() {
    // Packet header bits, MSB first: present(1), inclusion tag tree (1), zero-bitplane tag tree
    // (1), coding passes codeword 0 (one pass), comma code 0 (Lblock stays 3), length 001.
    let header = 0b1110_0001u8;
    let data = stream_full(8, 8, 0, 4, 0, None, 0x02, None, &[header, 0xAB]);
    let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
    assert!(cs.cod.uses_sop);
    let (pkts, _, tile) = decode(&cs, &data).unwrap();
    assert_eq!(tile.len(), 2);
    assert_eq!(pkts.len(), 1);
    assert_eq!(pkts[0].nsop, -1, "no SOP was present");
    assert_eq!(pkts[0].blocks.len(), 1);
    let segs = &pkts[0].blocks[0].segments;
    assert!(
        segs.len() == 1 && segs[0].start == 1 && segs[0].end == 2,
        "{segs:?}"
    );
}

/// A POC marker drives the packet iterator with its own windows; two records that both cover the
/// tile's only packet dedup to a single emission, and a full-range record decodes like no POC.
#[test]
fn poc_windows_and_dedup() {
    let one = |poc: Option<&[u8]>, tp_poc: Option<&[u8]>| -> usize {
        let data = hostile_j2k(8, 0, 1, 0, None, poc, None, tp_poc, Some(&[0]));
        let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
        decode(&cs, &data).unwrap().0.len()
    };
    let full = [0u8, 0, 0, 1, 1, 1, 0]; // RS CS LYE(2) RE CE P: the whole tile
    assert_eq!(one(None, None), 1, "baseline");
    assert_eq!(one(Some(&full), None), 1, "full-range main-header POC");
    assert_eq!(one(Some(&full), Some(&full)), 1, "duplicate windows dedup");
}

/// `passes_read` is 3*Mb - 2 for the tile's largest Mb, at least 1 and at most 109.
#[test]
fn passes_read_cases() {
    for (guard, exps, want) in [
        (2u8, &[8u8, 9, 10][..], 31),
        (1, &[1], 1),    // Mb 1
        (0, &[0], 1),    // Mb -1, still one pass
        (7, &[31], 109), // Mb 37
    ] {
        let mut qcd = vec![guard << 5];
        qcd.extend(exps.iter().map(|e| e << 3));
        let data = stream_full(8, 8, 0, 4, 0, None, 0, Some(&qcd), &[]);
        let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
        assert_eq!(
            fuzz::passes_read(&cs, 0),
            want,
            "guard {guard} exponents {exps:?}"
        );
    }
}

/// A tile whose only component's sample grid collapses to nothing legitimately has no
/// tile-part (b2_mono.j2c: XRsiz=5 with a 1-pixel-wide tile grid).
#[test]
fn assemble_tile_tolerates_empty_component() {
    // 6x4 image in 1x4 tiles, component XRsiz=5: tile 1 covers x in [1,2), an empty component.
    let mut siz = Vec::new();
    siz.extend_from_slice(&0u16.to_be_bytes());
    for v in [6u32, 4, 0, 0, 1, 4, 0, 0] {
        siz.extend_from_slice(&v.to_be_bytes());
    }
    siz.extend_from_slice(&1u16.to_be_bytes());
    siz.extend_from_slice(&[7, 5, 1]);
    let mut b = vec![0xFF, 0x4F, 0xFF, 0x51];
    b.extend_from_slice(&((siz.len() + 2) as u16).to_be_bytes());
    b.extend_from_slice(&siz);
    b.extend_from_slice(&[0xFF, 0x52, 0, 12, 0, 0, 0, 1, 0, 0, 4, 4, 0, 1]);
    b.extend_from_slice(&[0xFF, 0x5C, 0, 4, 0x40, 8 << 3]);
    // Tile 0 has a tile-part; tile 1 (empty component grid) has none.
    b.extend_from_slice(&[0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 14, 0, 1, 0xFF, 0x93]);
    b.extend_from_slice(&[0xFF, 0xD9]);
    let cs = fuzz::parse(&b, &limits(), &nc()).unwrap();
    let mut budget = jp2k::Budget::new(&limits());
    assert!(
        fuzz::assemble_tile(&cs, &b, 1, &mut budget)
            .unwrap()
            .is_empty()
    );
    assert!(fuzz::assemble_tile(&cs, &b, 2, &mut budget).is_ok());
}

/// Every code-block record and segment tier-2 keeps is charged to the budget.
#[test]
fn code_blocks_are_charged() {
    let (cs, data) = load("g64_53.j2k");
    let mut b = Budget::new(&limits());
    let (pkts, _, _) =
        fuzz::decode_tile(&cs, &data, 0, &mut Cursor::new(), None, &nc(), &mut b).unwrap();
    let blocks: usize = pkts.iter().map(|p| p.blocks.len()).sum();
    let segs: usize = pkts
        .iter()
        .flat_map(|p| p.blocks.iter())
        .map(|c| c.segments.len())
        .sum();
    assert!(blocks > 0 && segs > 0);
    assert!(
        b.held()
            >= blocks * size_of::<fuzz::CodeBlockRef>()
                + segs * size_of::<fuzz::CodeBlockSegment>(),
        "charged {} for {blocks} blocks and {segs} segments",
        b.held()
    );
    // And a budget too small for them fails rather than allocating.
    let tiny = jp2k::Limits {
        max_alloc: b.held() / 2,
        ..limits()
    };
    let r = fuzz::decode_tile(
        &cs,
        &data,
        0,
        &mut Cursor::new(),
        None,
        &nc(),
        &mut Budget::new(&tiny),
    );
    assert!(matches!(r, Err(Error::LimitExceeded(_))));
}
