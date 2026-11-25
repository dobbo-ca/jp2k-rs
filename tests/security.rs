//! Ports of `internal/decoder/security_test.go`: hostile inputs must fail (or succeed) cheaply
//! and never panic. Go measured total allocation; here the working-memory limit is set to 64 MB
//! and a decode that needs more fails with `LimitExceeded`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::{Error, Options};
use std::time::{Duration, Instant};

fn small_opts() -> Options {
    let mut o = Options::default();
    o.limits.max_alloc = 64 << 20;
    o
}

/// Decodes `data` and requires an error within a generous time bound and no internal error.
fn must_fail_cheaply(data: &[u8]) {
    decode_cheaply(data, true);
}

fn decode_cheaply(data: &[u8], want_err: bool) {
    assert!(
        data.len() < 2048,
        "fixture is {} bytes, want < 2 KB",
        data.len()
    );
    let start = Instant::now();
    let r = jp2k::decode(data, &small_opts(), &nc());
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "took {:?}",
        start.elapsed()
    );
    match r {
        Ok(_) => assert!(!want_err, "decode succeeded, want an error"),
        Err(e) => {
            assert!(want_err, "{e}");
            assert!(!matches!(e, Error::Internal(_)), "{e}");
        }
    }
}

/// The audit's 83-byte input: 65535 layers over 1x1 precincts with one byte of tile data, in
/// every progression order.
#[test]
fn layer_bomb_every_order() {
    for prog in 0..5 {
        must_fail_cheaply(&hostile_j2k(
            32,
            prog,
            65535,
            0,
            Some(&[0]),
            None,
            None,
            None,
            Some(&[0]),
        ));
    }
}

/// `n` POC changes (Csiz < 257) with an empty resolution window: RSpoc = REpoc = 5, all 65535
/// layers, LRCP.
fn poc_records(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| [5u8, 0, 0xFF, 0xFF, 5, 1, 0]).collect()
}

/// 31 empty-window records over 65535 layers and 16 resolutions: each record used to generate
/// its full ~1M-packet progression before filtering to nothing.
#[test]
fn poc_window_is_cheap() {
    let mut poc = poc_records(30);
    poc.extend_from_slice(&[0, 0, 0, 1, 16, 1, 0]);
    decode_cheaply(
        &hostile_j2k(
            1,
            0,
            65535,
            15,
            None,
            Some(&poc),
            None,
            None,
            Some(&[0u8; 16]),
        ),
        false,
    );
}

/// 31 records in the main header plus more in a tile-part header: each header was capped at 31
/// on its own, so the tile's total grew with its tile-part count.
#[test]
fn poc_cap_is_per_tile() {
    must_fail_cheaply(&hostile_j2k(
        1,
        0,
        65535,
        15,
        None,
        Some(&poc_records(31)),
        None,
        Some(&poc_records(1)),
        None,
    ));
}

/// A 32-bit code-block length of 0x80000000, which went negative in an int on 386/arm.
#[test]
fn segment_length_wrap() {
    let tile = [0xEF, 0xFF, 0x7F, 0xFF, 0x68, 0x00, 0x00, 0x00, 0x00];
    must_fail_cheaply(&hostile_j2k(
        1,
        0,
        2,
        0,
        None,
        None,
        None,
        None,
        Some(&tile),
    ));
}

/// Malformed deep-NL headers must not panic.
#[test]
fn precinct_at_deep_levels() {
    for size in [1usize, 16] {
        for prog in [2usize, 3, 4] {
            for nl in [17usize, 20, 30, 31, 32] {
                let data = hostile_j2k(size, prog, 1, nl, None, None, None, None, Some(&[0u8; 40]));
                let r = jp2k::decode(&data, &small_opts(), &nc());
                assert!(
                    !matches!(r, Err(Error::Internal(_))),
                    "size={size} prog={prog} nl={nl}: {r:?}"
                );
            }
        }
    }
}

/// Nppm = 0xFFFFFFFF, which went negative in an int on 386/arm and then panicked slicing.
#[test]
fn ppm_huge_nppm() {
    let ppm = [
        0xFF, 0x60, 0x00, 0x0F, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 1, 2, 3, 4, 5, 6, 7, 8,
    ];
    must_fail_cheaply(&hostile_j2k(1, 0, 1, 0, None, None, Some(&ppm), None, None));
}

/// A 1024x1024 tile of 1x1 precincts with one byte of data: geometry for 1M precincts was built
/// before the packet budget ran.
#[test]
fn precinct_geometry_budget() {
    must_fail_cheaply(&hostile_j2k(
        1024,
        0,
        1,
        0,
        Some(&[0]),
        None,
        None,
        None,
        Some(&[0]),
    ));
}

fn cod_offset(data: &[u8]) -> usize {
    data.windows(2).position(|w| w == [0xFF, 0x52]).unwrap()
}

/// 4x4 code-blocks over a 4100x4100 tile (~1.05M code-blocks) from a few bytes.
#[test]
fn code_block_budget() {
    let mut data = hostile_j2k(4100, 0, 1, 0, Some(&[0xBB]), None, None, None, Some(&[0]));
    let cod = cod_offset(&data);
    data[cod + 10] = 0;
    data[cod + 11] = 0; // 4x4 code-blocks
    must_fail_cheaply(&data);
}

fn tier2_bomb_from(layers: usize, tile: &[u8]) -> Vec<u8> {
    let mut data = hostile_j2k(4096, 0, layers, 0, None, None, None, None, Some(tile));
    let cod = cod_offset(&data);
    data[cod + 10] = 0;
    data[cod + 11] = 0; // 4x4 code-blocks
    data
}

/// One precinct of ~1M 4x4 code-blocks over `layers` layers of "packet present, nothing
/// included" (0x80): every leaf excludes on layer 0, so later layers take the O(1) shortcut.
fn tier2_bomb(layers: usize) -> Vec<u8> {
    let mut tile = vec![0x80u8; layers];
    tile.extend(vec![0u8; 70000]);
    tier2_bomb_from(layers, &tile)
}

/// `tier2_bomb` but a real root-inclusion bit defeats the exclusion shortcut, so every layer
/// pays a full O(code-blocks) walk.
fn tier2_expensive_bomb(layers: usize) -> Vec<u8> {
    let mut hdr = vec![0x80u8; layers];
    hdr[0] = 0xC0; // packet present, root inclusion bit
    hdr.extend(vec![0u8; 70000]);
    tier2_bomb_from(layers, &hdr)
}

/// The tier-2 work budget must reject this fast with no cancellation involved.
#[test]
fn tier2_work_budget_rejects_bomb() {
    let start = Instant::now();
    let err = jp2k::decode(&tier2_bomb(999), &Options::default(), &nc()).unwrap_err();
    // Go's `tier2.ErrWork` is `Invalid`, whatever the limits and thread count.
    assert!(matches!(err, jp2k::Error::Invalid(_)), "{err:?}");
    assert!(err.to_string().contains("work budget"), "{err}");
    assert!(
        start.elapsed() < Duration::from_secs(60),
        "took {:?}",
        start.elapsed()
    );
}

/// The shortcut-defeating shape needs the cancel flag, not just the work budget, to return
/// promptly.
#[test]
fn tier2_cancel_honored_promptly() {
    let data = tier2_expensive_bomb(999);
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let start = Instant::now();
    let r = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(200));
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        jp2k::decode(&data, &Options::default(), &cancel)
    });
    assert!(r.is_err(), "decode succeeded, want it stopped");
    assert!(
        start.elapsed() < Duration::from_secs(30),
        "took {:?}",
        start.elapsed()
    );
}

#[test]
fn header_exhaustion_fails() {
    must_fail_cheaply(&hostile_j2k(1, 0, 2, 0, None, None, None, None, Some(&[0])));
}

/// One LRCP record over everything with LYEpoc = 0: it emits no packets, so no header bytes are
/// ever spent.
const LYE0_POC: [u8; 7] = [0, 0, 0, 0, 33, 1, 0];

/// A POC that emits nothing used to exempt a 512x512 tile's 262144 1x1 precincts from the byte
/// budget, though they were all still allocated.
#[test]
fn uncovered_precincts_count() {
    must_fail_cheaply(&hostile_j2k(
        512,
        0,
        1,
        0,
        Some(&[0]),
        Some(&LYE0_POC),
        None,
        None,
        None,
    ));
}

/// 64 empty 32x32 tiles share 600 PPM bytes that no packet consumes. Each tile was credited
/// with all of them, so the budget never shrank.
#[test]
fn ppm_budget_is_shared() {
    let mut ppm = vec![0xFF, 0x60, 0, 0, 0, 0, 0, 2, 0x58];
    ppm.extend(vec![0u8; 600]);
    let l = (ppm.len() - 2) as u16;
    ppm[2..4].copy_from_slice(&l.to_be_bytes());
    let poc: Vec<u8> = (0..31).flat_map(|_| [0u8, 0, 0, 0, 33, 1, 2]).collect(); // RPCL, LYEpoc = 0
    let mut data = hostile_j2k(256, 0, 1, 0, Some(&[0]), Some(&poc), Some(&ppm), None, None);
    data[24..28].copy_from_slice(&32u32.to_be_bytes()); // XTsiz
    data[28..32].copy_from_slice(&32u32.to_be_bytes()); // YTsiz
    data.truncate(data.len() - 2); // drop EOC
    for i in 1..64u8 {
        data.extend_from_slice(&[0xFF, 0x90, 0, 10, 0, i, 0, 0, 0, 14, 0, 1, 0xFF, 0x93]);
    }
    data.extend_from_slice(&[0xFF, 0xD9]);
    must_fail_cheaply(&data);
}

/// TPsot is 8 bits, so a 257th tile-part of one tile must fail at parse instead of wrapping to
/// TPsot 0.
#[test]
fn tile_part_cap() {
    let base = hostile_j2k(1, 0, 1, 0, None, None, None, None, Some(&[0]));
    let with_parts = |tpsot: &[usize]| -> Vec<u8> {
        let mut b = base[..base.len() - 2].to_vec(); // drop EOC
        for &n in tpsot {
            b.extend_from_slice(&[0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 14, n as u8, 0, 0xFF, 0x93]);
        }
        b.extend_from_slice(&[0xFF, 0xD9]);
        b
    };
    let mut parts: Vec<usize> = (1..256).collect();
    let cs = jp2k::fuzz::parse(
        &with_parts(&parts),
        &Default::default(),
        &std::sync::atomic::AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(cs.tile_parts.len(), 256);
    parts.push(0);
    let e = jp2k::fuzz::parse(
        &with_parts(&parts),
        &Default::default(),
        &std::sync::atomic::AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(e.to_string().contains("more than 256 tile-parts"), "{e}");
}

/// 4096 packets of 4 bytes, each signalling dozens of coding passes over an empty 64x64
/// code-block, about 90000 sample-passes per input byte. That is under the tier-1 work budget,
/// which must leave room for legitimate files, so it has to stay cheap instead.
#[test]
fn tier1_amplifier() {
    let head: Vec<u8> = (0..)
        .step_by(2)
        .take_while(|&i| i < HEAD_HEX.len())
        .map(|i| u8::from_str_radix(&HEAD_HEX[i..i + 2], 16).unwrap())
        .collect();
    let mut data = head;
    for _ in 0..4096 {
        data.extend_from_slice(&[0xff, 0x7b, 0x30, 0x00]);
    }
    data.extend_from_slice(&[0xff, 0xd9]);
    let start = Instant::now();
    jp2k::decode(&data, &Options::default(), &nc()).unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(60),
        "took {:?}",
        start.elapsed()
    );
}

const HEAD_HEX: &str = "ff4fff510029000000001000000010000000000000000000000010000000100000000000000000000001070101ff52000d0100000100000404000166ff5c000420f0ff90000a00000000400e0001ff93";

/// A flood of bare tile-parts must parse within a bounded heap: 16 MB of them used to cost
/// 350 MB.
#[test]
fn tile_part_flood_is_bounded() {
    let data = tile_part_flood((16 << 20) / 14);
    let mut o = Options::default();
    o.limits.max_alloc = 96 << 20;
    let start = Instant::now();
    let _ = jp2k::decode(&data, &o, &nc());
    assert!(
        start.elapsed() < Duration::from_secs(60),
        "took {:?}",
        start.elapsed()
    );
}

/// Main-header repeats of a marker cost next to nothing to parse for a decode, which reads only
/// the first; a marker in every tile-part header fails the parse early, at eight bytes of heap
/// per byte read.
#[test]
fn header_flood_is_bounded() {
    const SIZE: usize = 4 << 20;
    let qcc = [0xff, 0x5d, 0, 4, 0, 0x40];
    let coc = [0xff, 0x53, 0, 9, 0, 0, 0, 4, 4, 0, 1];
    let com = [0xff, 0x64, 0, 4, 0, 1];
    let rep = |m: &[u8]| m.repeat(SIZE / m.len());
    let cases: Vec<(&str, Vec<u8>, bool)> = vec![
        ("main QCC", header_flood(&rep(&qcc), &[], 1), false),
        ("main COC", header_flood(&rep(&coc), &[], 1), false),
        ("main COM", header_flood(&rep(&com), &[], 1), false),
        ("tile-part QCC", header_flood(&[], &qcc, SIZE / 20), true),
        ("tile-part COM", header_flood(&[], &com, SIZE / 20), false),
    ];
    let mut o = Options::default();
    o.limits.max_alloc = 8 * SIZE;
    for (name, data, parse_err) in cases {
        let r = jp2k::decode(&data, &o, &nc());
        let got = matches!(&r, Err(e) if e.to_string().contains("need more memory than their size allows"));
        assert_eq!(got, parse_err, "{name}: {r:?}");
    }
}

/// A reader's input counts against the decode's `max_alloc`, on top of what the decode itself
/// allocates: a limit that fits either alone but not both fails.
#[test]
fn reader_input_is_charged_to_max_alloc() {
    let data = std::fs::read(format!(
        "{}/tests/data/g64_53.j2k",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let nc = std::sync::atomic::AtomicBool::new(false);
    let o = jp2k::Options::default();
    assert!(jp2k::decode(&data, &o, &nc).is_ok());
    // Find the least max_alloc a slice decode needs, then give a reader exactly that.
    let (mut lo, mut hi) = (0usize, 1usize << 30);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        let o = jp2k::Options {
            limits: jp2k::Limits {
                max_alloc: mid,
                ..jp2k::Limits::default()
            },
            ..o.clone()
        };
        if jp2k::decode(&data, &o, &nc).is_ok() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let at = |n: usize| jp2k::Options {
        limits: jp2k::Limits {
            max_alloc: n,
            ..jp2k::Limits::default()
        },
        ..o.clone()
    };
    assert!(jp2k::decode_reader(&data[..], &at(hi), &nc).is_err());
    assert!(jp2k::decode_reader(&data[..], &at(hi + data.len()), &nc).is_ok());
}
