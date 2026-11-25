//! A deterministic stand-in for Go's `FuzzDecode`, `FuzzDecodeRegion`, `FuzzParseCodestream` and
//! `FuzzTier2` (the fuzz stage runs the real fuzzers): mutated copies of every fixture (truncated,
//! bit-flipped, with length fields and bytes overwritten) must not panic any entry point, which
//! would show as `Error::Internal`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::{Error, Image, Options, Rect};
use std::time::{Duration, Instant};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Every entry point on `data`; none may report an internal error (a caught panic).
fn exercise(name: &str, what: &str, data: &[u8], o: &Options) {
    let cancel = nc();
    let results: Vec<(&str, Option<Error>)> = vec![
        ("decode", jp2k::decode(data, o, &cancel).err()),
        ("decode_config", jp2k::decode_config(data, &o.limits).err()),
        (
            "decode_region",
            jp2k::decode_region(
                data,
                Rect {
                    x0: 3,
                    y0: 2,
                    x1: 40,
                    y1: 33,
                },
                1,
                o,
                &cancel,
            )
            .err(),
        ),
        (
            "decode_components",
            jp2k::decode_components(data, &[0, 1], o, &cancel).err(),
        ),
        ("open", Image::open(data, o, &nc()).map(|_| ()).err()),
    ];
    for (entry, err) in results {
        assert!(
            !matches!(err, Some(Error::Internal(_))),
            "{name} {what}: {entry}: {err:?}"
        );
    }
}

#[test]
fn mutated_fixtures_do_not_panic() {
    let mut o = Options::default();
    o.limits.max_alloc = 256 << 20;
    o.limits.max_pixels = 4 << 20;
    let mut files: Vec<_> = std::fs::read_dir(path(""))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("j2k" | "j2c" | "jp2")
            )
        })
        .collect();
    files.sort();
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let start = Instant::now();
    let mut runs = 0;
    for f in files {
        let data = std::fs::read(&f).unwrap();
        // The slide-sized fixtures are for the benchmarks.
        if data.len() > 200_000 {
            continue;
        }
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        exercise(&name, "original", &data, &o);
        // Truncations.
        for k in 1..=12 {
            let cut = data.len() * k / 13;
            exercise(&name, &format!("truncated at {cut}"), &data[..cut], &o);
        }
        // Bit flips and byte overwrites, mostly in the headers where lengths and counts live.
        for i in 0..40 {
            let mut d = data.clone();
            let span = if i % 2 == 0 {
                d.len().min(300)
            } else {
                d.len()
            };
            let pos = rng.below(span);
            match i % 4 {
                0 | 1 => d[pos] ^= 1 << rng.below(8),
                2 => d[pos] = [0x00, 0xFF, 0x7F, 0x80][rng.below(4)],
                _ => {
                    let end = (pos + 2).min(d.len());
                    for b in &mut d[pos..end] {
                        *b = rng.next_u64() as u8;
                    }
                }
            }
            exercise(&name, &format!("mutation {i} at {pos}"), &d, &o);
            runs += 1;
        }
    }
    assert!(runs > 1000, "{runs} mutations");
    assert!(
        start.elapsed() < Duration::from_secs(600),
        "took {:?}",
        start.elapsed()
    );
}
