//! The `Error` variant of one representative input per family of Go errors (the mapping is the
//! table in `docs/parity.md`), and `decode_region` with each rounding against an opened `Image`.
mod common;

use common::*;
use jp2k::{Error, Image, Limits, Options, Rect};
use std::io::{self, Read};

/// Go's `fmt.Errorf` text for malformed input: `Invalid`.
#[test]
fn malformed_input_is_invalid() {
    let e = jp2k::decode(b"not a jpeg 2000 file", &opts(), &nc()).unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "{e:?}");
}

/// Valid input the decoder does not handle (RGN): `Unsupported`.
#[test]
fn rgn_is_unsupported() {
    let e = jp2k::decode(&read("p0_03.j2k"), &opts(), &nc()).unwrap_err();
    assert!(matches!(e, Error::Unsupported(_)), "{e:?}");
}

/// Go's memory-limit errors and the input and output size limits: `LimitExceeded`.
#[test]
fn limits_are_limit_exceeded() {
    let data = read("g64_53.j2k");
    let tiny = |f: fn(&mut Limits)| {
        let mut o = Options::default();
        f(&mut o.limits);
        o
    };
    for o in [
        tiny(|l| l.max_alloc = 1000),
        tiny(|l| l.max_pixels = 10),
        tiny(|l| l.max_output = 10),
    ] {
        let e = jp2k::decode(&data, &o, &nc()).unwrap_err();
        assert!(matches!(e, Error::LimitExceeded(_)), "{e:?}");
    }
    let o = tiny(|l| l.max_input = 100);
    let e = jp2k::decode_reader(&data[..], &o, &nc()).unwrap_err();
    assert!(matches!(e, Error::LimitExceeded(_)), "{e:?}");
}

/// A cancel flag that is set (Go's `ErrStopped` or the context's error): `Cancelled`.
#[test]
fn cancel_is_cancelled() {
    let c = std::sync::atomic::AtomicBool::new(true);
    let e = jp2k::decode(&read("g64_53.j2k"), &opts(), &c).unwrap_err();
    assert_eq!(e, Error::Cancelled);
}

/// A reader that fails (Go returns the reader's error): `Io`, with its kind; an empty reader is
/// Go's `io.ErrUnexpectedEOF`.
#[test]
fn reader_failure_is_io() {
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "boom"))
        }
    }
    let e = jp2k::decode_reader(Broken, &opts(), &nc()).unwrap_err();
    assert!(
        matches!(e, Error::Io(io::ErrorKind::PermissionDenied, _)),
        "{e:?}"
    );
    let e = jp2k::decode_config_reader(io::empty(), &limits(), &nc()).unwrap_err();
    assert!(
        matches!(e, Error::Io(io::ErrorKind::UnexpectedEof, _)),
        "{e:?}"
    );
}

/// `decode_region` with `separate_rounding` or `jj2000` is `Image::open` with the same options
/// and `Image::decode_region`, for every region and rounding.
#[test]
fn decode_region_matches_opened_image_in_every_rounding() {
    let rects = [
        Rect {
            x0: 0,
            y0: 0,
            x1: 1 << 20,
            y1: 1 << 20,
        },
        Rect {
            x0: 5,
            y0: 7,
            x1: 41,
            y1: 33,
        },
        Rect {
            x0: 17,
            y0: 3,
            x1: 18,
            y1: 60,
        },
    ];
    for name in ["region_rgb_97.j2k", "region_gray_97.j2k", "g64_97.j2k"] {
        let data = read(name);
        for (separate_rounding, jj2000) in [(false, false), (true, false), (false, true)] {
            let o = Options {
                separate_rounding,
                jj2000,
                ..Options::default()
            };
            let image = Image::open(&data, &o, &nc()).unwrap();
            for rect in rects {
                for reduce in 0..=2 {
                    let a = jp2k::decode_region(&data, rect, reduce, &o, &nc());
                    let b = image.decode_region(rect, reduce, &nc());
                    assert_eq!(
                        a, b,
                        "{name} {rect:?} reduce {reduce} sep {separate_rounding} jj {jj2000}"
                    );
                }
            }
        }
    }
}
