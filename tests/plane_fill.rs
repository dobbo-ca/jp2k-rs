//! The decoder zeroes a spare tile-component plane only where its code-blocks do not tile it.
//! Here every plane is first filled with a sentinel (`jp2k::fuzz::plane_fill`), positive and then
//! negative (NaN for 9/7), and each decode must give what it gives with every plane zeroed: a
//! sample no code-block writes would carry the sentinel into the output. Over the fixtures, the
//! bigtiles bench fixture, and truncated and mutated copies (missing packets and code-blocks),
//! through the free functions at one thread and the default, and through one reused `Decoder`.
//! One test: the fill mode is global to the process.
mod common;

use common::*;
use jp2k::{Decoder, Options, Plane, Raster, Result};
use std::hash::{DefaultHasher, Hash, Hasher};

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

/// A digest of a decode's and a component decode's results (the outputs of the big inputs are
/// too big to keep): the error, or the layout and every byte.
type Out = (u64, u64);

fn raster(r: Result<&Raster>) -> u64 {
    let mut h = DefaultHasher::new();
    match r {
        Ok(r) => (
            format!("{:?} {:?} {}", r.color_model, r.bounds, r.stride),
            &r.pix,
        )
            .hash(&mut h),
        Err(e) => e.to_string().hash(&mut h),
    }
    h.finish()
}

fn planes(p: Result<&[Plane]>) -> u64 {
    let mut h = DefaultHasher::new();
    match p {
        Ok(ps) => {
            for p in ps {
                (format!("{:?}", p.bounds), &p.pix).hash(&mut h);
            }
        }
        Err(e) => e.to_string().hash(&mut h),
    }
    h.finish()
}

fn run(data: &[u8], o: &Options) -> Out {
    (
        raster(jp2k::decode(data, o, &nc()).as_ref()),
        planes(jp2k::decode_components(data, &[0], o, &nc()).as_deref()),
    )
}

fn run_reused(d: &mut Decoder, data: &[u8], o: &Options) -> Out {
    let r = raster(d.decode(data, o, &nc()));
    (r, planes(d.decode_components(data, &[0], o, &nc())))
}

#[test]
fn sentinel_planes_decode_as_zeroed_ones() {
    let mut one = Options::default();
    one.limits.max_alloc = 512 << 20;
    one.limits.max_threads = 1;
    let mut all = one.clone();
    all.limits.max_threads = jp2k::Limits::default().max_threads;

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
    files.push(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/parity/bigtiles/bigtiles_rgb97.j2k"),
    );

    // (name, input, whether to also run it at the default threads)
    let mut inputs: Vec<(String, Vec<u8>, bool)> = Vec::new();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for f in &files {
        let data = std::fs::read(f).unwrap();
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let big = data.len() > 200_000;
        // Truncations drop the last packets, and with them whole code-blocks.
        let cuts = if big { 3 } else { 12 };
        for k in 1..=cuts {
            let cut = data.len() * k / (cuts + 1);
            inputs.push((
                format!("{name} truncated at {cut}"),
                data[..cut].to_vec(),
                false,
            ));
        }
        if !big {
            for i in 0..24 {
                let mut d = data.clone();
                let span = if i % 2 == 0 {
                    d.len().min(300)
                } else {
                    d.len()
                };
                let pos = rng.below(span);
                match i % 3 {
                    0 => d[pos] ^= 1 << rng.below(8),
                    1 => d[pos] = [0x00, 0xFF, 0x7F, 0x80][rng.below(4)],
                    _ => {
                        let end = (pos + 2).min(d.len());
                        for b in &mut d[pos..end] {
                            *b = rng.next_u64() as u8;
                        }
                    }
                }
                inputs.push((format!("{name} mutation {i} at {pos}"), d, false));
            }
        }
        inputs.push((name, data, true));
    }

    // The zeroed results, then each sentinel's, compared input by input.
    jp2k::fuzz::plane_fill(1);
    let mut want: Vec<(Out, Option<Out>)> = Vec::new();
    let mut reused = Decoder::new();
    let mut want_reused: Vec<Out> = Vec::new();
    for (_, d, both) in &inputs {
        want.push((run(d, &one), both.then(|| run(d, &all))));
        want_reused.push(run_reused(&mut reused, d, &one));
    }
    let decoded = inputs
        .iter()
        .filter(|(_, d, _)| jp2k::decode_config(d, &one.limits).is_ok())
        .count();
    assert!(
        decoded > 100,
        "only {decoded} inputs have a header that parses"
    );
    for mode in [2, 3] {
        jp2k::fuzz::plane_fill(mode);
        let mut reused = Decoder::new();
        for (k, (name, d, both)) in inputs.iter().enumerate() {
            assert!(run(d, &one) == want[k].0, "mode {mode}, {name}, one thread");
            if *both {
                assert!(
                    Some(run(d, &all)) == want[k].1,
                    "mode {mode}, {name}, default threads"
                );
            }
            assert!(
                run_reused(&mut reused, d, &one) == want_reused[k],
                "mode {mode}, {name}, reused Decoder"
            );
        }
    }
    // The default (old samples where the blocks tile the plane) as well.
    jp2k::fuzz::plane_fill(0);
    let mut reused = Decoder::new();
    for (k, (name, d, _)) in inputs.iter().enumerate() {
        assert!(run(d, &one) == want[k].0, "default, {name}");
        assert!(
            run_reused(&mut reused, d, &one) == want_reused[k],
            "default, {name}, reused Decoder"
        );
    }
}
