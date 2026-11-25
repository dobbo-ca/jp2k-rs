//! Buffer reuse (docs/performance.md): a `Decoder` that has decoded an image once
//! decodes it again, at `max_threads` 1, without one call to the allocator. This binary counts
//! every allocation (`alloc`, `alloc_zeroed`, `realloc`), so it holds exactly one test: another
//! test running beside it would be counted too. A threaded call on a caller's pool keeps a few
//! allocations that are not the decoder's buffers (one boxed task per extra worker of each
//! parallel pass, which rayon's `spawn` allocates, and the column groups of the threaded
//! wavelet); its count is bounded, not zero.
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

mod common;
use common::*;
use jp2k::{Decoder, Error, Image, Limits, Options, Plane, Raster, Rect, ThreadPool};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

/// Counts the calls that ask the allocator for memory.
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// How many allocator calls of a measured window `LOG` keeps.
const LOG_LEN: usize = 16;
/// `ALLOCS` when the window opened.
static BASE: AtomicUsize = AtomicUsize::new(0);
/// The first `LOG_LEN` calls of the window as `(size << 2) | kind`; atomics, so the allocator
/// itself never allocates to record them.
static LOG: [AtomicUsize; LOG_LEN] = [const { AtomicUsize::new(0) }; LOG_LEN];
const KINDS: [&str; 3] = ["alloc", "alloc_zeroed", "realloc"];

/// Counts one allocator call of `size` bytes (`kind` indexes `KINDS`) and files it in `LOG`.
fn note(size: usize, kind: usize) {
    let k = ALLOCS
        .fetch_add(1, Relaxed)
        .wrapping_sub(BASE.load(Relaxed));
    if k < LOG_LEN {
        LOG[k].store((size << 2) | kind, Relaxed);
    }
}

/// Opens a measured window and returns the count at its start.
fn open_window() -> usize {
    let now = ALLOCS.load(Relaxed);
    BASE.store(now, Relaxed);
    now
}

/// The first allocator calls of the window that just closed with `n` of them, by kind and size.
fn first_allocs(n: usize) -> String {
    (0..n.min(LOG_LEN))
        .map(|k| {
            let v = LOG[k].load(Relaxed);
            format!("{} {}", KINDS[v & 3], v >> 2)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// SAFETY: every method forwards to `System` with the caller's arguments, so System's contract
// carries over; the counter is an atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size(), 0);
        // SAFETY: the caller upholds `alloc`'s contract, which `System.alloc` has too.
        unsafe { System.alloc(l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        note(l.size(), 1);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from System with it.
        unsafe { System.dealloc(p, l) };
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        note(new, 2);
        // SAFETY: `p` came from this allocator with layout `l`, and `new` is valid for it.
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The most allocator calls of a serial second call: none.
const SERIAL_MAX: usize = 0;

/// The most allocator calls of one threaded call on a caller's pool: about 20 for svs256 (a
/// rayon task per extra worker of each parallel pass, and the threaded wavelet's column groups).
const THREADED_MAX: usize = 64;

/// What a case calls.
#[derive(Clone, Copy, Debug)]
enum Call {
    Whole,
    Comps(&'static [usize]),
    Region(Rect, i32),
    /// `Image::open` once, then its region through the `Decoder`.
    Image(Rect, i32),
}

/// A result borrowed from the `Decoder`, and the free function's.
enum Got<'a> {
    R(&'a Raster),
    P(&'a [Plane]),
}

enum Want {
    R(Raster),
    P(Vec<Plane>),
}

fn run<'d>(
    dec: &'d mut Decoder,
    src: &[u8],
    img: Option<&Image<'_>>,
    call: Call,
    o: &Options,
    cancel: &AtomicBool,
) -> Result<Got<'d>, Error> {
    Ok(match (call, img) {
        (Call::Whole, _) => Got::R(dec.decode(src, o, cancel)?),
        (Call::Comps(c), _) => Got::P(dec.decode_components(src, c, o, cancel)?),
        (Call::Region(r, k), _) => Got::R(dec.decode_region(src, r, k, o, cancel)?),
        (Call::Image(r, k), Some(i)) => Got::R(dec.decode_image_region(i, r, k, cancel)?),
        (Call::Image(..), None) => unreachable!(),
    })
}

fn want(
    src: &[u8],
    img: Option<&Image<'_>>,
    call: Call,
    o: &Options,
    cancel: &AtomicBool,
) -> Result<Want, Error> {
    Ok(match (call, img) {
        (Call::Whole, _) => Want::R(jp2k::decode(src, o, cancel)?),
        (Call::Comps(c), _) => Want::P(jp2k::decode_components(src, c, o, cancel)?),
        (Call::Region(r, k), _) => Want::R(jp2k::decode_region(src, r, k, o, cancel)?),
        (Call::Image(r, k), Some(i)) => Want::R(i.decode_region(r, k, cancel)?),
        (Call::Image(..), None) => unreachable!(),
    })
}

/// A new `Decoder` warmed by one call: the allocator calls of its second call (with the first of
/// them named by kind and size), and whether its result is the free function's.
fn second_call(src: &[u8], call: Call, o: &Options) -> Result<(usize, bool, String), Error> {
    let cancel = AtomicBool::new(false);
    let img = match call {
        Call::Image(..) => Some(Image::open(src, o, &cancel)?),
        _ => None,
    };
    let want = want(src, img.as_ref(), call, o, &cancel)?;
    let mut dec = Decoder::new();
    run(&mut dec, src, img.as_ref(), call, o, &cancel)?;
    let before = open_window();
    let r = run(&mut dec, src, img.as_ref(), call, o, &cancel);
    let n = ALLOCS.load(Relaxed) - before;
    let same = match (r?, &want) {
        (Got::R(g), Want::R(w)) => g == w,
        (Got::P(g), Want::P(w)) => g == w.as_slice(),
        _ => false,
    };
    Ok((n, same, first_allocs(n)))
}

fn limits(max_threads: usize) -> Options {
    Options {
        limits: Limits {
            max_threads,
            ..Limits::default()
        },
        ..Options::default()
    }
}

/// The inner half of a `w` x `h` image.
fn inner(src: &[u8]) -> Rect {
    let c = jp2k::decode_config(src, &Limits::default()).unwrap();
    let (w, h) = (i64::from(c.width), i64::from(c.height));
    Rect {
        x0: w / 4,
        y0: h / 4,
        x1: w - w / 4,
        y1: h - h / 4,
    }
}

#[test]
fn second_decode_allocates_nothing() {
    let mut rows: Vec<(String, usize)> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    let mut record = |what: String, r: Result<(usize, bool, String), Error>, max: usize| match r {
        Ok((n, same, first)) => {
            if n > max || !same {
                bad.push(format!(
                    "{what}: {n} allocations (max {max}), same output {same}, first: [{first}]"
                ));
            }
            rows.push((format!("{what}: [{first}]"), n));
        }
        Err(e) => bad.push(format!("{what}: {e:?}")),
    };

    // Serial: no allocation at all, on a fresh Decoder warmed by one call.
    let serial = limits(1);
    let svs = read("svs256.j2k");
    let half = inner(&svs);
    let mut cases: Vec<(&str, Call)> = vec![
        ("svs256.j2k", Call::Whole),
        ("svs256.j2k", Call::Comps(&[0, 1, 2])),
        ("svs256.j2k", Call::Comps(&[1])),
        ("svs256.j2k", Call::Image(half, 0)),
    ];
    for reduce in 0..3 {
        cases.push(("svs256.j2k", Call::Region(half, reduce)));
    }
    for name in [
        "bench_small_97_mct0.j2k",
        "g96_3c16_97.j2k",
        "g64_97.j2k",
        "rgb8_rct_m5.j2k",
        "style_rgb_M63.j2k",
        "g64_53.j2k",
        "c1_mono.j2c",
        "m4_3layer_53.j2k",
        "g64_multitile.j2k",
        "region_rgb_97.j2k",
        "gray16x3_97_nomct.j2k",
        "gray_box_m5.jp2",
    ] {
        cases.push((name, Call::Whole));
    }
    for (name, call) in cases {
        let src = read(name);
        record(
            format!("{name} {call:?}"),
            second_call(&src, call, &serial),
            SERIAL_MAX,
        );
    }

    // Threaded, on a caller's pool: a small bounded number of allocations, not zero.
    let pool = ThreadPool::new(3).unwrap();
    let threaded = Options {
        pool: Some(pool),
        ..limits(4)
    };
    for (name, call) in [
        ("svs256.j2k", Call::Whole),
        ("svs256.j2k", Call::Region(half, 0)),
        ("bench_small_97_mct0.j2k", Call::Whole),
    ] {
        let src = read(name);
        record(
            format!("{name} {call:?}, max_threads 4 on a pool"),
            second_call(&src, call, &threaded),
            THREADED_MAX,
        );
    }

    for (what, n) in &rows {
        println!("{n:5} allocations  {what}");
    }
    assert!(bad.is_empty(), "\n{}\n", bad.join("\n"));
}
