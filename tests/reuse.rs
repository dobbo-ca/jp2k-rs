//! A reused `Decoder` gives what the free functions give: the same bytes, the same errors and,
//! with one thread, a ledger peak (`jp2k::stats::last`) at least the free function's, call after
//! call, whatever it decoded before. It is higher only when a call finishes with planes past a
//! lane's three spares or with a region's lower wavelet levels, which a `Decoder` keeps charged
//! until the call takes them again or ends, or holds the tier-2 lists of a lane's tile, which a
//! `Decoder` keeps charged for its next tile, up to 256 KiB and within the ledger's peak so far,
//! while a region's skipped tile is charged (docs/limits.md); for the images of
//! `a_reused_decoder_needs_the_same_max_alloc` it is the same. `tests/reuse_alloc.rs` counts its
//! allocations.
mod common;

use common::*;
use jp2k::{Decoder, Error, Limits, Options, Rect};

fn with(max_threads: usize, max_alloc: usize) -> Options {
    Options {
        limits: Limits {
            max_threads,
            max_alloc,
            ..Limits::default()
        },
        ..Options::default()
    }
}

fn fixtures() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(path(""))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| [".j2k", ".j2c", ".jp2"].iter().any(|e| n.ends_with(e)))
        .collect();
    v.sort();
    v
}

/// One `Decoder` over every fixture, whole, as a region, as component 0 and at reduce 1, twice
/// and interleaved (each call follows one of another shape), with one thread and with the
/// default: the result equals the free function's, and with one thread the ledger peak is at
/// least the free function's and under twice it (the calls where it is higher are printed).
#[test]
fn a_reused_decoder_matches_the_free_functions() {
    let nc = nc();
    for t in [1, Limits::default().max_threads] {
        let o = with(t, 4 << 30);
        let mut dec = Decoder::new();
        for pass in 0..2 {
            for name in fixtures() {
                let data = read(&name);
                let rect = jp2k::decode_config(&data, &o.limits).ok().map(|c| Rect {
                    x0: i64::from(c.width) / 5,
                    y0: i64::from(c.height) / 4,
                    x1: i64::from(c.width) * 4 / 5,
                    y1: i64::from(c.height),
                });
                let peak = |what: &str, want: usize| {
                    let got = jp2k::stats::last().peak_alloc;
                    if t == 1 && got != want {
                        eprintln!("{name} {what} pass {pass}: Decoder peak {got}, free {want}");
                    }
                    assert!(
                        t > 1 || (want <= got && got < 2 * want),
                        "{name} {what} pass {pass}: Decoder peak {got}, free function {want}"
                    );
                };

                let want = jp2k::decode(&data, &o, &nc);
                let wp = jp2k::stats::last().peak_alloc;
                let got = dec.decode(&data, &o, &nc).cloned();
                assert_eq!(got, want, "{name} whole, {t} threads, pass {pass}");
                peak("whole", wp);

                let want = jp2k::decode_components(&data, &[0], &o, &nc);
                let wp = jp2k::stats::last().peak_alloc;
                let got = dec
                    .decode_components(&data, &[0], &o, &nc)
                    .map(<[_]>::to_vec);
                assert_eq!(got, want, "{name} c0, {t} threads, pass {pass}");
                peak("c0", wp);

                if let Some(r) = rect {
                    for reduce in [0, 1] {
                        let want = jp2k::decode_region(&data, r, reduce, &o, &nc);
                        let wp = jp2k::stats::last().peak_alloc;
                        let got = dec.decode_region(&data, r, reduce, &o, &nc).cloned();
                        assert_eq!(
                            got, want,
                            "{name} region r{reduce}, {t} threads, pass {pass}"
                        );
                        peak("region", wp);
                    }
                }
            }
        }
    }
}

/// The least `max_alloc` a decode needs on one thread is the same through a warmed-up `Decoder`
/// as through `decode`: at the free function's ledger peak it succeeds with the same bytes, one
/// byte under it or at half it fails with `LimitExceeded`, and the `Decoder` still decodes after
/// the failures.
#[test]
fn a_reused_decoder_needs_the_same_max_alloc() {
    let nc = nc();
    for name in [
        "fl_dense.j2k",
        "g1_colr.j2c",
        "vsi512_dense.j2k",
        "gray16x3_97_nomct.j2k",
        "g64_multitile.j2k",
        "g96_3c16_97.j2k",
        "gray_box_m5.jp2",
    ] {
        let data = read(name);
        let want = jp2k::decode(&data, &with(1, 4 << 30), &nc).unwrap();
        let ledger = jp2k::stats::last().peak_alloc;
        let mut dec = Decoder::new();
        dec.decode(&data, &with(1, 4 << 30), &nc).unwrap();
        assert!(
            dec.decode(&data, &with(1, ledger), &nc).unwrap() == &want,
            "{name}"
        );
        assert_eq!(jp2k::stats::last().peak_alloc, ledger, "{name}");
        for max in [ledger - 1, ledger / 2] {
            let r = dec.decode(&data, &with(1, max), &nc);
            assert!(
                matches!(r, Err(Error::LimitExceeded(_))),
                "{name}: max_alloc {max} (peak {ledger}) gave {:?}",
                r.map(|_| ())
            );
        }
        assert!(
            dec.decode(&data, &with(1, ledger), &nc).unwrap() == &want,
            "{name}"
        );
    }
}
