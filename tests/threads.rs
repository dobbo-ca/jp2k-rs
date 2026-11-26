//! Threads: the same bytes for every `max_threads`, never more threads than asked (and none for
//! 1), a cancel flag set from another thread stops the decode, concurrent decodes agree. Ports of
//! Go's `TestParallelTilesMatchSerial`, `TestDecodeConcurrent` and `TestCancelWhileDecoding`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::{Error, Image, Limits, Options, Rect, ThreadPool};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

fn with(max_threads: usize) -> Options {
    Options {
        limits: Limits {
            max_threads,
            ..Limits::default()
        },
        ..Options::default()
    }
}

fn default_threads() -> usize {
    Limits::default().max_threads
}

/// The thread counts of the tests: 1, 2 and 3 are fixed, then the default.
fn counts() -> Vec<usize> {
    vec![1, 2, 3, default_threads()]
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

/// Every fixture, decoded whole, by region at two scales and by component, gives the same result
/// (the same bytes, or the same error) with 1, 2, 3 and the default number of threads.
#[test]
fn every_fixture_is_identical_for_every_thread_count() {
    let nc = nc();
    for name in fixtures() {
        let data = read(&name);
        let big = name.starts_with("fl_dense");
        let one = with(1);
        let want = jp2k::decode(&data, &one, &nc);
        let region = jp2k::decode_config(&data, &Limits::default())
            .ok()
            .map(|c| {
                let (w, h) = (i64::from(c.width), i64::from(c.height));
                Rect {
                    x0: w / 5,
                    y0: h / 4,
                    x1: w * 4 / 5,
                    y1: h,
                }
            });
        let want_region = region.map(|r| jp2k::decode_region(&data, r, 1, &one, &nc));
        let want_c0 = jp2k::decode_components(&data, &[0], &one, &nc);
        for t in counts().into_iter().skip(1) {
            let o = with(t);
            assert!(
                jp2k::decode(&data, &o, &nc) == want,
                "{name}: {t} threads differ from 1"
            );
            if big {
                continue;
            }
            if let Some(r) = region {
                assert!(
                    jp2k::decode_region(&data, r, 1, &o, &nc) == want_region.clone().unwrap(),
                    "{name}: region at {t} threads differs from 1"
                );
            }
            assert!(
                jp2k::decode_components(&data, &[0], &o, &nc) == want_c0,
                "{name}: component 0 at {t} threads differs from 1"
            );
        }
    }
}

/// An opened `Image` gives the same regions for every thread count too.
#[test]
fn opened_image_is_identical_for_every_thread_count() {
    let nc = nc();
    for name in ["g64_multitile.j2k", "g1_colr.j2c", "region_rgb_53.j2k"] {
        let data = read(name);
        let one = Image::open(&data, &with(1), &nc).unwrap();
        let rect = Rect {
            x0: 5,
            y0: 7,
            x1: 60,
            y1: 50,
        };
        let want = one.decode_region(rect, 0, &nc);
        for t in [2, 3, default_threads()] {
            let m = Image::open(&data, &with(t), &nc).unwrap();
            assert!(m.decode_region(rect, 0, &nc) == want, "{name} at {t}");
        }
    }
}

/// Peak threads never pass `max_threads`, the calling thread included, and `max_threads` 1 (or
/// 0, which counts as 1) starts no thread at all. The images below use every threaded stage:
/// tier-1 blocks, wavelet rows and columns, the colour transform, output rows and tile lanes.
#[test]
fn peak_threads_stay_within_max_threads() {
    let nc = nc();
    let mut spawned_somewhere = false;
    for name in [
        "fl_dense.j2k",
        "g1_colr.j2c",
        "gray16x3_97_nomct.j2k",
        "vsi512_dense.j2k",
        "g64_multitile.j2k",
        "bench_small_97_mct0.j2k",
    ] {
        let data = read(name);
        for t in [0, 1, 2, 3, default_threads()] {
            jp2k::decode(&data, &with(t), &nc).unwrap();
            let st = jp2k::stats::last();
            let cap = t.max(1);
            assert!(
                st.peak_threads <= cap,
                "{name}: peak {} threads with max_threads {t}",
                st.peak_threads
            );
            if cap == 1 {
                assert_eq!(
                    (st.spawned_threads, st.peak_threads),
                    (0, 1),
                    "{name}: max_threads {t} must not spawn"
                );
            }
            if t == 3 {
                spawned_somewhere |= st.spawned_threads > 0;
            }
        }
    }
    assert!(
        spawned_somewhere,
        "no decode used a thread at max_threads 3"
    );
    // The big image uses all three.
    jp2k::decode(&read("fl_dense.j2k"), &with(3), &nc).unwrap();
    let st = jp2k::stats::last();
    assert!((2..=3).contains(&st.peak_threads), "{st:?}");
    // Two tiles: lanes, then each lane's share.
    jp2k::decode(&read("g1_colr.j2c"), &with(3), &nc).unwrap();
    assert!(jp2k::stats::last().peak_threads <= 3);
}

/// The regions and component decodes obey `max_threads` too.
#[test]
fn region_and_components_stay_within_max_threads() {
    let nc = nc();
    let data = read("fl_dense.j2k");
    let rect = Rect {
        x0: 100,
        y0: 100,
        x1: 1900,
        y1: 1900,
    };
    for t in [1, 2, 3] {
        jp2k::decode_region(&data, rect, 0, &with(t), &nc).unwrap();
        assert!(jp2k::stats::last().peak_threads <= t);
        jp2k::decode_components(&data, &[0, 1], &with(t), &nc).unwrap();
        let st = jp2k::stats::last();
        assert!(st.peak_threads <= t);
        if t == 1 {
            assert_eq!(st.spawned_threads, 0);
        }
    }
}

/// A cancel flag set from another thread while the decode runs stops it with `Cancelled`,
/// quickly, at every thread count; the threads are joined before the call returns.
#[test]
fn cancel_from_another_thread_stops_the_decode() {
    let data = read("fl_dense.j2k");
    for t in [1, 2, 3, default_threads()] {
        let o = with(t);
        let start = Instant::now();
        jp2k::decode(&data, &o, &nc()).unwrap();
        let full = start.elapsed();
        let cancel = AtomicBool::new(false);
        let (r, took) = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(full / 10);
                cancel.store(true, Ordering::Relaxed);
            });
            let start = Instant::now();
            let r = jp2k::decode(&data, &o, &cancel);
            (r, start.elapsed())
        });
        assert_eq!(r.unwrap_err(), Error::Cancelled, "{t} threads");
        assert!(
            took < full,
            "{t} threads: cancelled after {took:?}, full {full:?}"
        );
        let st = jp2k::stats::last();
        assert!(st.peak_threads <= t.max(1));
    }
}

/// A flag already set stops every entry point before any work.
#[test]
fn cancel_set_before_the_call() {
    let data = read("g1_colr.j2c");
    let cancel = AtomicBool::new(true);
    for t in counts() {
        let o = with(t);
        assert_eq!(
            jp2k::decode(&data, &o, &cancel).unwrap_err(),
            Error::Cancelled
        );
        assert_eq!(jp2k::stats::last().spawned_threads, 0);
    }
}

/// Many threads decoding different fixtures at once, each with several threads of its own, get
/// the bytes a lone decode gets (Go: `TestDecodeConcurrent`).
#[test]
fn concurrent_decodes_agree() {
    let names = [
        "bench_small_97_mct0.j2k",
        "g1_colr.j2c",
        "c1_mono.j2c",
        "g96_3c16_97.j2k",
        "g64_multitile.j2k",
        "p1_07.j2k",
        "c_odd_origin_53.j2k",
        "gray_box_m5.jp2",
        "signed32x24.j2k",
    ];
    let data: Vec<Vec<u8>> = names.iter().map(|n| read(n)).collect();
    let want: Vec<_> = data
        .iter()
        .map(|d| jp2k::decode(d, &with(1), &nc()).unwrap())
        .collect();
    std::thread::scope(|s| {
        for g in 0..8 {
            let (data, want) = (&data, &want);
            s.spawn(move || {
                for k in 0..3 {
                    let i = (g + k) % data.len();
                    let got = jp2k::decode(&data[i], &with(2 + g % 3), &nc()).unwrap();
                    assert!(got == want[i], "{}: concurrent decode differs", names[i]);
                }
            });
        }
    });
}

/// `max_threads` 0 counts as 1: no thread, same bytes.
#[test]
fn zero_threads_counts_as_one() {
    let data = read("g1_colr.j2c");
    let a = jp2k::decode(&data, &with(0), &nc()).unwrap();
    assert_eq!(jp2k::stats::last().spawned_threads, 0);
    assert!(a == jp2k::decode(&data, &with(1), &nc()).unwrap());
}

/// How long the big image takes with one thread and with the default, for the log.
#[test]
fn threads_time_the_big_image() {
    let data = read("fl_dense.j2k");
    let time = |t: usize| {
        let o = with(t);
        let start = Instant::now();
        jp2k::decode(&data, &o, &nc()).unwrap();
        start.elapsed()
    };
    let (one, many) = (time(1), time(default_threads()));
    eprintln!(
        "fl_dense: 1 thread {one:?}, {} threads {many:?}",
        default_threads()
    );
}

fn pooled(pool: &ThreadPool, max_threads: usize) -> Options {
    Options {
        pool: Some(pool.clone()),
        ..with(max_threads)
    }
}

/// With a pool: the same bytes as the serial decode for every pool size and `max_threads`, the
/// call's participating threads (counted by the ledger, not OS threads) never pass `max_threads`
/// or the pool's size plus the caller, and `max_threads` 1 (or 0) uses no worker at all.
#[test]
fn pool_decodes_are_identical_and_stay_within_max_threads() {
    let nc = nc();
    let mut used_workers = false;
    for size in [1usize, 2, 3, 8] {
        let pool = ThreadPool::new(size).unwrap();
        assert_eq!(pool.threads(), size);
        for name in [
            "fl_dense.j2k",
            "g1_colr.j2c",
            "gray16x3_97_nomct.j2k",
            "vsi512_dense.j2k",
            "g64_multitile.j2k",
            "bench_small_97_mct0.j2k",
        ] {
            let data = read(name);
            let want = jp2k::decode(&data, &with(1), &nc).unwrap();
            for t in [0, 1, 2, 3, 8, 64] {
                let got = jp2k::decode(&data, &pooled(&pool, t), &nc).unwrap();
                assert!(got == want, "{name}: pool {size}, max_threads {t}");
                let st = jp2k::stats::last();
                let cap = t.max(1).min(size + 1);
                assert!(
                    st.peak_threads <= cap,
                    "{name}: peak {} with pool {size}, max_threads {t}",
                    st.peak_threads
                );
                if t.max(1) == 1 {
                    assert_eq!(
                        (st.spawned_threads, st.peak_threads),
                        (0, 1),
                        "{name}: max_threads {t} must not use the pool"
                    );
                }
                used_workers |= st.spawned_threads > 0;
            }
        }
    }
    assert!(used_workers, "no decode ran on a pool worker");
}

/// Regions and component decodes run on the pool and obey `max_threads` too.
#[test]
fn pool_region_and_components_stay_within_max_threads() {
    let nc = nc();
    let data = read("fl_dense.j2k");
    let rect = Rect {
        x0: 100,
        y0: 100,
        x1: 1900,
        y1: 1900,
    };
    let pool = ThreadPool::new(4).unwrap();
    let want = jp2k::decode_region(&data, rect, 0, &with(1), &nc).unwrap();
    for t in [1, 2, 3] {
        let got = jp2k::decode_region(&data, rect, 0, &pooled(&pool, t), &nc).unwrap();
        assert!(got == want, "region at {t}");
        assert!(jp2k::stats::last().peak_threads <= t);
        jp2k::decode_components(&data, &[0, 1], &pooled(&pool, t), &nc).unwrap();
        let st = jp2k::stats::last();
        assert!(st.peak_threads <= t);
        if t == 1 {
            assert_eq!(st.spawned_threads, 0);
        }
    }
}

/// Many decodes on different threads share one small pool (3 workers, each call up to all of
/// them, in parallel): no deadlock, the same bytes.
#[test]
fn concurrent_decodes_share_one_pool() {
    let names = ["g1_colr.j2c", "g64_multitile.j2k", "vsi512_dense.j2k"];
    let data: Vec<Vec<u8>> = names.iter().map(|n| read(n)).collect();
    let want: Vec<_> = data
        .iter()
        .map(|d| jp2k::decode(d, &with(1), &nc()).unwrap())
        .collect();
    let pool = ThreadPool::new(3).unwrap();
    std::thread::scope(|s| {
        for g in 0..6 {
            let (data, want, pool) = (&data, &want, &pool);
            s.spawn(move || {
                for k in 0..3 {
                    let i = (g + k) % data.len();
                    let got = jp2k::decode(&data[i], &pooled(pool, 2 + g % 3), &nc()).unwrap();
                    assert!(got == want[i], "{}: pooled decode differs", names[i]);
                }
            });
        }
    });
}

/// A cancel flag stops a pooled decode, and the pool is still usable afterwards.
#[test]
fn pool_cancel_stops_the_decode() {
    let data = read("fl_dense.j2k");
    let pool = ThreadPool::new(3).unwrap();
    let cancel = AtomicBool::new(true);
    let o = pooled(&pool, 4);
    assert_eq!(
        jp2k::decode(&data, &o, &cancel).unwrap_err(),
        Error::Cancelled
    );
    assert!(jp2k::decode(&data, &o, &nc()).is_ok());
}

fn per_call(max_threads: usize) -> Options {
    Options {
        per_call_pool: true,
        ..with(max_threads)
    }
}

/// Without a caller pool (the crate's shared pool), with one, and with the per-call workers: the
/// same bytes as one thread for 1, 2, 3 and the default `max_threads` and for whole, region and
/// component decodes, never more threads than `max_threads`, and none for 1.
#[test]
fn every_mode_is_identical_and_stays_within_max_threads() {
    let nc = nc();
    let caller = ThreadPool::new(3).unwrap();
    let rect = Rect {
        x0: 100,
        y0: 100,
        x1: 1900,
        y1: 1900,
    };
    let mut per_call_spawned = false;
    for name in [
        "fl_dense.j2k",
        "g1_colr.j2c",
        "gray16x3_97_nomct.j2k",
        "g64_multitile.j2k",
        "bench_small_97_mct0.j2k",
    ] {
        let data = read(name);
        let one = with(1);
        let want = jp2k::decode(&data, &one, &nc).unwrap();
        let big = name == "fl_dense.j2k";
        let want_region = big.then(|| jp2k::decode_region(&data, rect, 0, &one, &nc));
        let want_c0 = jp2k::decode_components(&data, &[0], &one, &nc);
        for t in counts() {
            let modes = [
                ("shared", with(t)),
                ("caller", pooled(&caller, t)),
                ("per-call", per_call(t)),
            ];
            for (mode, o) in modes {
                let check = |what: &str| {
                    let st = jp2k::stats::last();
                    assert!(
                        st.peak_threads <= t,
                        "{name} {what}, {mode}: peak {} with max_threads {t}",
                        st.peak_threads
                    );
                    if t == 1 {
                        assert_eq!(
                            (st.spawned_threads, st.peak_threads),
                            (0, 1),
                            "{name} {what}, {mode}: max_threads 1 must spawn nothing"
                        );
                    }
                    st
                };
                assert!(
                    jp2k::decode(&data, &o, &nc).unwrap() == want,
                    "{name}: {mode}, {t} threads"
                );
                let st = check("decode");
                per_call_spawned |= mode == "per-call" && st.spawned_threads > 0;
                if let Some(w) = &want_region {
                    assert!(
                        &jp2k::decode_region(&data, rect, 0, &o, &nc) == w,
                        "{name}: region, {mode}, {t} threads"
                    );
                    check("region");
                }
                assert!(
                    jp2k::decode_components(&data, &[0], &o, &nc) == want_c0,
                    "{name}: component 0, {mode}, {t} threads"
                );
                check("components");
            }
        }
    }
    assert!(per_call_spawned, "no per-call decode used a worker");
    assert!(
        jp2k::shared_pool_started(),
        "a decode with max_threads above 1 and no pool must have built the shared pool"
    );
}

/// A pool of 2 workers with `max_threads` 2 gives the call both: a decode big enough to
/// parallelise runs on a worker.
#[test]
fn a_two_worker_pool_runs_a_two_thread_decode_in_parallel() {
    let data = read("fl_dense.j2k");
    let want = jp2k::decode(&data, &with(1), &nc()).unwrap();
    let pool = ThreadPool::new(2).unwrap();
    // A worker may come late to a short pass, but not to every pass of a big decode.
    let mut used = false;
    for _ in 0..5 {
        let got = jp2k::decode(&data, &pooled(&pool, 2), &nc()).unwrap();
        assert!(got == want);
        let st = jp2k::stats::last();
        assert!(st.peak_threads <= 2, "{st:?}");
        used |= st.spawned_threads > 0;
    }
    assert!(
        used,
        "a 2-worker pool with max_threads 2 never used a worker"
    );
}

/// A 1024x1024 grey codestream of 16 tiles of 256x256, no wavelet levels, one empty packet a tile,
/// padded with a comment to 4.4 KiB: at least 2 tile lanes (2 KiB of codestream each) with
/// output rows big enough (65536 samples) to be converted in parallel.
fn sixteen_tile_stream() -> Vec<u8> {
    grid_stream(1024, 4200)
}

/// A `side` x `side` grey codestream of 256x256 tiles (`side` a multiple of 256, at most 4096
/// tiles), as `sixteen_tile_stream`, its comment `pad` bytes long.
fn grid_stream(side: u32, pad: usize) -> Vec<u8> {
    fn seg(b: &mut Vec<u8>, marker: u16, body: &[u8]) {
        b.extend_from_slice(&marker.to_be_bytes());
        b.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
        b.extend_from_slice(body);
    }
    let mut b = vec![0xFF, 0x4F];
    let mut siz = vec![0, 0];
    for v in [side, side, 0, 0, 256, 256, 0, 0] {
        siz.extend_from_slice(&v.to_be_bytes());
    }
    siz.extend_from_slice(&[0, 1, 7, 1, 1]);
    seg(&mut b, 0xFF51, &siz);
    // Scod 0, LRCP, 1 layer, no MCT, 0 levels, 64x64 blocks, no block style, 5/3.
    seg(&mut b, 0xFF52, &[0, 0, 0, 1, 0, 0, 4, 4, 0, 1]);
    seg(&mut b, 0xFF5C, &[0x40, 8 << 3]);
    let mut com = vec![0, 1];
    com.resize(2 + pad, b'x');
    seg(&mut b, 0xFF64, &com);
    for t in 0..(side / 256).pow(2) as u16 {
        b.extend_from_slice(&[0xFF, 0x90, 0, 10]);
        b.extend_from_slice(&t.to_be_bytes());
        b.extend_from_slice(&15u32.to_be_bytes());
        b.extend_from_slice(&[0, 1, 0xFF, 0x93, 0]);
    }
    b.extend_from_slice(&[0xFF, 0xD9]);
    b
}

/// Runs `f` on its own thread and fails if it has not finished in a minute (a deadlock hangs, it
/// does not fail).
fn watchdog(what: &str, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .unwrap_or_else(|e| panic!("{what}: deadlocked or panicked ({e})"));
}

/// `p` threads decode the same multi-tile image at once, `rounds` times over, on one pool, each
/// with `max_threads` threads: every decode ends, with the bytes of the serial one.
fn hammer(
    label: &str,
    data: &Arc<Vec<u8>>,
    want: &Arc<jp2k::Raster>,
    p: usize,
    pool: Option<ThreadPool>,
) {
    for round in 0..4 {
        let (data, want, pool) = (data.clone(), want.clone(), pool.clone());
        watchdog(&format!("{label}, {p} callers, round {round}"), move || {
            std::thread::scope(|s| {
                for _ in 0..p {
                    s.spawn(|| {
                        let o = Options {
                            pool: pool.clone(),
                            ..with(p)
                        };
                        for _ in 0..3 {
                            let got = jp2k::decode(&data, &o, &nc()).unwrap();
                            assert!(got == *want, "{p} callers: bytes differ from serial");
                        }
                    });
                }
            });
        });
    }
}

/// Callers that fill a pool must not wait on each other: with the pool as big as the callers
/// (so every worker is busy with some call's tile lane), no decode may hold a lock while it waits
/// for the pool. 2, 3 and 4 callers on a pool of their size, then the same on the shared pool.
#[test]
fn concurrent_multi_tile_decodes_on_a_full_pool_do_not_deadlock() {
    let data = Arc::new(sixteen_tile_stream());
    assert!((4096..6144).contains(&data.len()), "{}", data.len());
    let want = Arc::new(jp2k::decode(&data, &with(1), &nc()).unwrap());
    for p in [2, 3, 4] {
        hammer(
            "caller pool",
            &data,
            &want,
            p,
            Some(ThreadPool::new(p).unwrap()),
        );
    }
    let cpus = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .clamp(2, 16);
    hammer("shared pool", &data, &want, cpus, None);
}

/// The first two tiles of `tests/parity/bigtiles/bigtiles_rgb97.j2k` (RGB 9/7, 5 wavelet levels, 64x64
/// code-blocks, 1024x1024 tiles) as a 2048x1024 image: 2 tile lanes, so on a pool of 8 (a call
/// gets 8 threads) a lane that was given its share of the threads runs nested passes (code-block,
/// wavelet, colour transform and output rows) with 4 of them, each long enough for them to
/// take part.
fn few_lane_stream() -> Vec<u8> {
    let d = std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/parity/bigtiles/bigtiles_rgb97.j2k"),
    )
    .unwrap();
    let sot = d.windows(4).position(|w| w == [0xFF, 0x90, 0, 10]).unwrap();
    let siz = d.windows(2).position(|w| w == [0xFF, 0x51]).unwrap();
    let mut b = d[..sot].to_vec();
    b[siz + 6..siz + 10].copy_from_slice(&2048u32.to_be_bytes());
    b[siz + 10..siz + 14].copy_from_slice(&1024u32.to_be_bytes());
    let mut at = sot;
    for t in 0..2u16 {
        assert_eq!(&d[at..at + 2], [0xFF, 0x90]);
        assert_eq!(u16::from_be_bytes([d[at + 4], d[at + 5]]), t, "tile order");
        let psot = u32::from_be_bytes(d[at + 6..at + 10].try_into().unwrap()) as usize;
        assert!(psot > 0, "a tile-part that runs to the end");
        b.extend_from_slice(&d[at..at + psot]);
        at += psot;
    }
    b.extend_from_slice(&[0xFF, 0xD9]);
    b
}

/// A lane of one call must not run another call's tile loop: on one shared pool, a cancelled
/// decode A returns long before a big decode B that runs beside it (a lane that waited in the pool
/// could pick up B's lane job and not get back to A until B's lane was done).
///
/// The image has to make a lane wait in the pool, or the test passes on the buggy code too: with
/// 2 lanes and 7 threads, the earlier buggy code gave each lane 3 or 4 threads, and its nested
/// passes (tier 1, the wavelet, the colour transform, the output) on this RGB 9/7 image with
/// code-blocks and 5 levels were real `in_place_scope`s on the pool, whose wait takes any
/// queued job, B's lane included. A grey image with no wavelet levels and empty packets has no
/// nested pass at all and so cannot show it.
#[test]
fn a_cancelled_decode_does_not_wait_for_another_decode_on_the_pool() {
    let data = Arc::new(few_lane_stream());
    let pool = ThreadPool::new(8).unwrap();
    let o = Arc::new(pooled(&pool, 8));
    let t0 = Instant::now();
    jp2k::decode(&data, &o, &nc()).unwrap();
    let full = t0.elapsed();
    for round in 0..3 {
        let (data, o) = (data.clone(), o.clone());
        watchdog(
            &format!("cancel beside a big decode, round {round}"),
            move || {
                let cancel = AtomicBool::new(false);
                std::thread::scope(|s| {
                    let b = s.spawn(|| jp2k::decode(&data, &o, &nc()));
                    std::thread::sleep(full / 20);
                    let a = s.spawn(|| {
                        let start = Instant::now();
                        let r = jp2k::decode(&data, &o, &cancel);
                        (r, start.elapsed())
                    });
                    std::thread::sleep(full / 10);
                    cancel.store(true, Ordering::Relaxed);
                    let (r, took) = a.join().unwrap();
                    assert_eq!(r.unwrap_err(), Error::Cancelled, "round {round}");
                    assert!(
                        took < full,
                        "round {round}: cancelled after {took:?}, a decode takes {full:?}"
                    );
                    assert!(b.join().unwrap().is_ok());
                });
            },
        );
    }
}

/// With few lanes for the threads (2 lanes, 7 threads, or 3) the lanes run on the caller and a
/// `std::thread::scope` thread and each gets its share of the threads for its nested passes: the
/// call uses more threads than lanes (a big-tile image such as `tests/parity/bigtiles` is this case),
/// never more than `max_threads` (the lane threads are counted), and the bytes are the serial ones.
#[test]
fn few_lanes_use_every_thread_and_give_the_serial_bytes() {
    let data = few_lane_stream();
    let want = jp2k::decode(&data, &with(1), &nc()).unwrap();
    let pool = ThreadPool::new(8).unwrap();
    // A pool of 8 gives a call at most 8 threads, the caller included.
    for (max_threads, cap) in [(3usize, 3usize), (8, 8)] {
        // A worker may come late to a short pass, but not to every pass of a big decode.
        let mut peak = 0;
        for _ in 0..5 {
            let got = jp2k::decode(&data, &pooled(&pool, max_threads), &nc()).unwrap();
            assert!(got == want, "bytes differ from serial at {max_threads}");
            let st = jp2k::stats::last();
            assert!(st.peak_threads <= cap, "{max_threads}: {st:?}");
            peak = peak.max(st.peak_threads);
        }
        assert!(
            peak > 2,
            "max_threads {max_threads}: peak {peak} threads, no more than the 2 lanes"
        );
    }
}
