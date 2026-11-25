//! Ports of `internal/codestream/jp2_test.go` and `internal/decoder/jp2_test.go`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::ColorModel;
use jp2k::fuzz::{JP2_SIGNATURE, jp2_codestream};

fn with_ihdr(extra: &[u8]) -> Vec<u8> {
    let mut v = ihdr_box();
    v.extend_from_slice(extra);
    v
}

fn icc_colr_box() -> Vec<u8> {
    jp2_box("colr", &[2, 0, 0, 0xde, 0xad]) // METH=2 (ICC), a fake profile
}

#[test]
fn finds_box() {
    let stream = [0xff, 0x4f, 1, 2, 3];
    let data = jp2_file(&with_ihdr(&colr_box(17)), &stream);
    let (got, cdef) = jp2_codestream(&data, &limits()).unwrap();
    assert_eq!(got, stream);
    assert!(cdef.is_none());
}

#[test]
fn rejects_duplicate_cdef_cn() {
    let cdef = cdef_box(&[[0, 0, 1], [0, 0, 2]]);
    let data = jp2_file(&with_ihdr(&cdef), &[0xff, 0x4f]);
    assert!(jp2_codestream(&data, &limits()).is_err());
}

/// A cdef box bigger than the streaming peek cap is still fully validated in memory.
#[test]
fn validates_oversized_cdef() {
    let entries = vec![[0u16, 0, 1]; 700];
    let data = jp2_file(&with_ihdr(&cdef_box(&entries)), &[0xff, 0x4f]);
    assert!(jp2_codestream(&data, &limits()).is_err());
}

#[test]
fn rejects_cdef_trailing_bytes() {
    let mut payload = vec![0u8; 2 + 6 + 5000];
    payload[1] = 1;
    payload[5] = 1; // Asoc 1
    let data = jp2_file(&with_ihdr(&jp2_box("cdef", &payload)), &[0xff, 0x4f]);
    assert!(jp2_codestream(&data, &limits()).is_err());
}

#[test]
fn requires_ihdr() {
    let data = jp2_file(&colr_box(17), &[0xff, 0x4f]);
    assert!(jp2_codestream(&data, &limits()).is_err());
}

#[test]
fn rejects_palette() {
    for typ in ["pclr", "cmap"] {
        let data = jp2_file(&with_ihdr(&jp2_box(typ, &[0, 0])), &[0xff, 0x4f]);
        assert!(jp2_codestream(&data, &limits()).is_err(), "{typ}");
    }
}

#[test]
fn rejects_unsupported_colourspace() {
    for (name, colr) in [("ICC profile", icc_colr_box()), ("sYCC", colr_box(18))] {
        let data = jp2_file(&with_ihdr(&colr), &[0xff, 0x4f]);
        assert!(jp2_codestream(&data, &limits()).is_err(), "{name}");
    }
}

#[test]
fn accepts_srgb_and_greyscale() {
    for enumcs in [16, 17] {
        let data = jp2_file(&with_ihdr(&colr_box(enumcs)), &[0xff, 0x4f]);
        jp2_codestream(&data, &limits()).unwrap();
    }
}

/// jp2c's declared length running past the bytes actually read is "runs to EOF" for jp2c
/// specifically, matching a raw codestream's truncated-after-SIZ tolerance.
#[test]
fn truncated_jp2c_tolerated() {
    let full = jp2_file(
        &with_ihdr(&colr_box(17)),
        &[0xff, 0x4f, 1, 2, 3, 4, 5, 6, 7, 8],
    );
    let (got, _) = jp2_codestream(&full[..full.len() - 4], &limits()).unwrap();
    assert_eq!(got, [0xff, 0x4f, 1, 2, 3, 4]);
}

/// A jp2c box using the 8-byte XLBox length form (LBox == 1).
fn xl_box_jp2c(stream: &[u8]) -> Vec<u8> {
    let mut buf = vec![0, 0, 0, 1];
    buf.extend_from_slice(b"jp2c");
    buf.extend_from_slice(&((16 + stream.len()) as u64).to_be_bytes());
    buf.extend_from_slice(stream);
    buf
}

fn jp_head() -> Vec<u8> {
    let mut d = JP2_SIGNATURE.to_vec();
    d.extend_from_slice(&[0x0d, 0x0a, 0x87, 0x0a]);
    d
}

#[test]
fn xl_box_jp2c_truncated_tolerated() {
    let stream = [0xff, 0x4f, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut full = jp_head();
    full.extend(jp2_box("ftyp", b"jp2 \x00\x00\x00\x00jp2 "));
    full.extend(jp2_box("jp2h", &with_ihdr(&colr_box(17))));
    full.extend(xl_box_jp2c(&stream));
    let (got, _) = jp2_codestream(&full[..full.len() - 4], &limits()).unwrap();
    assert_eq!(got, [0xff, 0x4f, 1, 2, 3, 4]);
}

#[test]
fn rejects_bad_signature() {
    let mut data = JP2_SIGNATURE.to_vec();
    data.extend_from_slice(&[0, 0, 0, 0]); // bad jP payload, should be 0x0D0A870A
    data.extend(jp2_box("ftyp", b"jp2 \x00\x00\x00\x00jp2 "));
    data.extend(jp2_box("jp2h", &with_ihdr(&colr_box(17))));
    data.extend(jp2_box("jp2c", &[0xff, 0x4f]));
    assert!(jp2_codestream(&data, &limits()).is_err());
}

/// A box whose XLBox is near i64::MAX is a clean error, not a panic.
#[test]
fn huge_xlbox_does_not_panic() {
    let mut data = jp_head();
    data.extend_from_slice(&[0, 0, 0, 1]);
    data.extend_from_slice(b"ftyp");
    data.extend_from_slice(&0x7FFF_FFFF_FFFF_FFFFu64.to_be_bytes());
    data.extend_from_slice(&[0, 0, 0, 0]);
    assert!(jp2_codestream(&data, &limits()).is_err());
}

#[test]
fn no_jp2c() {
    let mut data = jp_head();
    data.extend(jp2_box("jp2h", &with_ihdr(&colr_box(17))));
    assert!(jp2_codestream(&data, &limits()).is_err());
}

// -- decoder-level (internal/decoder/jp2_test.go) --

/// A cdef box reorders output channels instead of decoding in raw SIZ order. Cn 0/1/2 map to
/// Asoc 3/2/1, i.e. R and B swapped, G unchanged.
#[test]
fn cdef_reorders_channels() {
    let stream = read("rgb8_rct_m5.j2k");
    let plain = jp2k::decode(&stream, &opts(), &nc()).unwrap();
    assert_eq!(plain.color_model, ColorModel::Rgba);
    let cdef = cdef_box(&[[0, 0, 3], [1, 0, 2], [2, 0, 1]]);
    let data = jp2_file(&with_ihdr(&cdef), &stream);
    let got = jp2k::decode(&data, &opts(), &nc()).unwrap();
    assert_eq!(got.color_model, ColorModel::Rgba);
    assert_eq!(got.bounds, plain.bounds);
    for i in 0..plain.pix.len() / 4 {
        let (gp, pp) = (&got.pix[4 * i..4 * i + 4], &plain.pix[4 * i..4 * i + 4]);
        assert!(
            gp[0] == pp[2] && gp[1] == pp[1] && gp[2] == pp[0] && gp[3] == pp[3],
            "pixel {i}"
        );
    }
}

#[test]
fn cdef_rejects_incomplete_mapping() {
    let stream = read("rgb8_rct_m5.j2k");
    let cdef = cdef_box(&[[0, 0, 1], [1, 0, 2]]); // component 2 unmapped
    let data = jp2_file(&with_ihdr(&cdef), &stream);
    assert!(jp2k::decode(&data, &opts(), &nc()).is_err());
}

#[test]
fn cdef_large_stays_fully_validated() {
    let stream = read("rgb8_rct_m5.j2k");
    let entries = vec![[0u16, 0, 1]; 700]; // all Cn 0: duplicate, must be rejected
    let data = jp2_file(&with_ihdr(&cdef_box(&entries)), &stream);
    assert!(jp2k::decode(&data, &opts(), &nc()).is_err());

    let mut cdef = cdef_box(&[[0, 0, 1]]);
    cdef.extend(vec![0u8; 5000]); // trailing bytes past the declared entries
    let l = cdef.len() as u32;
    cdef[0..4].copy_from_slice(&l.to_be_bytes());
    let data = jp2_file(&with_ihdr(&cdef), &stream);
    assert!(jp2k::decode(&data, &opts(), &nc()).is_err());
}

/// `decode_config` succeeds on a JP2 file with a large box (e.g. an embedded XML packet) ahead
/// of jp2c.
#[test]
fn config_large_box_before_jp2c() {
    let stream = read("g64_53.j2k");
    let mut data = jp_head();
    data.extend(jp2_box("ftyp", b"jp2 \x00\x00\x00\x00jp2 "));
    data.extend(jp2_box("xml ", &vec![0u8; 2 << 20]));
    data.extend(jp2_box("jp2h", &with_ihdr(&colr_box(17))));
    data.extend(jp2_box("jp2c", &stream));
    let cfg = jp2k::decode_config(&data, &limits()).unwrap();
    assert_eq!((cfg.width, cfg.height), (64, 64));
}

fn many_boxes_jp2(n: usize, jp2h_children: &[u8], stream: &[u8]) -> Vec<u8> {
    let mut data = jp_head();
    let free = jp2_box("free", &[]);
    for _ in 0..n {
        data.extend_from_slice(&free);
    }
    data.extend(jp2_box("jp2h", jp2h_children));
    data.extend(jp2_box("jp2c", stream));
    data
}

#[test]
fn many_empty_boxes_decode() {
    let stream = read("t81_r40_97.j2k");
    let small = many_boxes_jp2(100, &ihdr_box(), &stream);
    let large = many_boxes_jp2(100_000, &ihdr_box(), &stream);
    let a = jp2k::decode(&small, &opts(), &nc()).unwrap();
    let b = jp2k::decode(&large, &opts(), &nc()).unwrap();
    assert_eq!(a, b);
    let cfg = jp2k::decode_config(&large, &limits()).unwrap();
    assert_eq!(
        (cfg.width as usize, cfg.height as usize),
        (width(&a), height(&a))
    );
}

#[test]
fn config_padded_stays_small() {
    let stream = read("g64_53.j2k");
    let mut padded = stream.clone();
    padded.extend(vec![0u8; 4 << 20]);
    let data = jp2_file(&with_ihdr(&colr_box(17)), &padded);
    let cfg = jp2k::decode_config(&data, &limits()).unwrap();
    assert_eq!((cfg.width, cfg.height), (64, 64));
}

#[test]
fn config_many_jp2h_boxes_and_large_child() {
    let stream = read("g64_53.j2k");
    let mut data = jp_head();
    for _ in 0..2000 {
        data.extend(jp2_box("jp2h", &with_ihdr(&colr_box(17))));
    }
    data.extend(jp2_box("jp2c", &stream));
    let cfg = jp2k::decode_config(&data, &limits()).unwrap();
    assert_eq!((cfg.width, cfg.height), (64, 64));

    // A large unknown child of jp2h is skipped, not buffered.
    let big = jp2_box("xml ", &vec![0u8; 1 << 20]);
    let data = jp2_file(&with_ihdr(&big), &stream);
    jp2k::decode_config(&data, &limits()).unwrap();
}

/// `decode_config` checks every top-level jp2h box, not just the first, matching `decode`.
#[test]
fn config_validates_every_jp2h() {
    let stream = read("t81_r40_97.j2k");
    // pclr in a second jp2h must be rejected, as decode rejects it.
    let mut data = jp_head();
    data.extend(jp2_box("jp2h", &ihdr_box()));
    data.extend(jp2_box("jp2h", &jp2_box("pclr", &[0u8; 20])));
    data.extend(jp2_box("jp2c", &stream));
    assert!(jp2k::decode_config(&data, &limits()).is_err());

    // ihdr present only in a second jp2h must still be found.
    let mut data = jp_head();
    data.extend(jp2_box("jp2h", &[]));
    data.extend(jp2_box("jp2h", &ihdr_box()));
    data.extend(jp2_box("jp2c", &stream));
    let cfg = jp2k::decode_config(&data, &limits()).unwrap();
    assert!(cfg.width != 0 && cfg.height != 0);
}

#[test]
fn config_large_jp2h_child_accepted() {
    let stream = read("t81_r40_97.j2k");
    let big = vec![0u8; 2 << 20];
    for typ in ["colr", "free"] {
        let mut children = with_ihdr(&colr_box(17));
        children.extend(jp2_box(typ, &big));
        let data = jp2_file(&children, &stream);
        let cfg = jp2k::decode_config(&data, &limits()).unwrap();
        assert!(cfg.width != 0 && cfg.height != 0);
        jp2k::decode(&data, &opts(), &nc()).unwrap();
    }
}

/// The streaming path rejects a cdef too large to validate, instead of accepting what it can't
/// check.
#[test]
fn config_cdef_over_cap_rejected() {
    let stream = read("t81_r40_97.j2k");
    let entries: Vec<[u16; 3]> = (0..700u16).map(|i| [i, 0, i + 1]).collect();
    let mut data = jp_head();
    data.extend(jp2_box("jp2h", &with_ihdr(&cdef_box(&entries))));
    data.extend(jp2_box("jp2c", &stream));
    assert!(jp2k::decode_config(&data, &limits()).is_err());
}

/// A clear error for a missing jp2c or a box truncated before its declared end, not a raw EOF.
#[test]
fn config_truncated_errors() {
    let mut data = jp_head();
    data.extend(jp2_box("jp2h", &ihdr_box()));
    let e = jp2k::decode_config(&data, &limits()).unwrap_err();
    assert!(matches!(e, jp2k::Error::Invalid(_)), "{e}");

    let mut data = jp_head();
    let b = jp2_box("xml ", &[0u8; 1000]);
    data.extend_from_slice(&b[..500]); // truncated mid payload
    let e = jp2k::decode_config(&data, &limits()).unwrap_err();
    assert!(matches!(e, jp2k::Error::Invalid(_)), "{e}");
}

/// A jp2h too short for its next child header is a clean, non-nil error for `decode_config` and
/// `decode`, never a panic.
#[test]
fn config_short_jp2h_child() {
    let stream = read("t81_r40_97.j2k");
    for n in 0..=16usize {
        let mut data = jp_head();
        data.extend(jp2_box("jp2h", &vec![0u8; n]));
        data.extend(jp2_box("jp2h", &ihdr_box()));
        data.extend(jp2_box("jp2c", &stream));
        let cfg = jp2k::decode_config(&data, &limits());
        let dec = jp2k::decode(&data, &opts(), &nc());
        for r in [cfg.as_ref().err(), dec.as_ref().err()]
            .into_iter()
            .flatten()
        {
            assert!(!matches!(r, jp2k::Error::Internal(_)), "payload {n}: {r}");
        }
        if (1..=7).contains(&n) {
            assert!(cfg.is_err(), "payload {n}: decode_config");
            assert!(dec.is_err(), "payload {n}: decode");
        }
    }
}

/// A jp2h child header that declares a length past the jp2h end is a clean error.
#[test]
fn config_jp2h_child_overruns_end() {
    let stream = read("t81_r40_97.j2k");
    let mut free_hdr = vec![0u8; 8];
    free_hdr[0..4].copy_from_slice(&100u32.to_be_bytes()); // LBox past jp2h end
    free_hdr[4..8].copy_from_slice(b"free");
    let mut data = jp_head();
    data.extend(jp2_box("jp2h", &with_ihdr(&free_hdr)));
    data.extend(jp2_box("jp2c", &stream));
    assert!(jp2k::decode_config(&data, &limits()).is_err());
    assert!(jp2k::decode(&data, &opts(), &nc()).is_err());
}

/// `decode_config_reader` reads at most `max_input` bytes, whatever the boxes claim.
#[test]
fn config_reader_stops_at_max_input() {
    use std::io::Read;
    let mut head = vec![0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A];
    // A `free` box whose XLBox is 2^63-1, then zeros without end.
    head.extend_from_slice(&[0, 0, 0, 1, b'f', b'r', b'e', b'e', 0x7F, 0xFF, 0xFF, 0xFF]);
    head.extend_from_slice(&[0xFF; 4]);
    let r = std::io::Cursor::new(head).chain(std::io::repeat(0));
    let l = jp2k::Limits {
        max_input: 1024,
        ..limits()
    };
    let e = jp2k::decode_config_reader(r, &l, &nc()).unwrap_err();
    assert!(matches!(e, jp2k::Error::LimitExceeded(_)), "{e:?}");
}

/// `decode_config` charges its buffers to `max_alloc`.
#[test]
fn config_charges_max_alloc() {
    let data = read("g64_53.j2k");
    let l = jp2k::Limits {
        max_alloc: 0,
        ..limits()
    };
    assert!(matches!(
        jp2k::decode_config(&data, &l),
        Err(jp2k::Error::LimitExceeded(_))
    ));
}
