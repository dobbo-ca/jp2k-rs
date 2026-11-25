//! Memory: `max_alloc` is the PEAK working memory of a call, all threads together. A counting
//! global allocator measures the real heap a decode takes on top of what was live when it began,
//! and the tests hold it to the limit; the ledger's own peak (`jp2k::stats::last`) is held to it
//! too. Ports of Go's `TestEstimateCoversPeak`, `TestEstimateCoversPackedHeaders`,
//! `TestHugeCanvasFailsUnderLimit`, `TestBigBoxJP2Decodes`, `TestSizedReaderFailsFast` and
//! `TestDecodeAllocBudget`, for a per-call ledger instead of Go's process-wide budget.
//!
//! This binary has its own allocator, and its tests take a lock so only one runs at a time:
//! the counts are for the whole process.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
mod common;

use common::*;
use jp2k::{Decoder, Error, Limits, Options, Rect, ThreadPool};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(n: usize) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments, so `System`'s contract
// holds; the counters only record sizes.
unsafe impl GlobalAlloc for Counting {
    // SAFETY (for the caller): as `GlobalAlloc::alloc`: `l` has a non-zero size.
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is `System`'s.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    // SAFETY (for the caller): as `GlobalAlloc::alloc_zeroed`: `l` has a non-zero size.
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is `System`'s.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    // SAFETY (for the caller): as `GlobalAlloc::dealloc`: `p` came from this allocator with layout `l`.
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    // SAFETY (for the caller): as `GlobalAlloc::realloc`: `p` came from this allocator with layout `l`, and `new` is non-zero and, rounded up to `l`'s alignment, does not overflow `isize`.
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // SAFETY: the caller's contract is `System`'s.
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if q != p && new > l.size() {
                // A block that moved to grow: old and new were both live while it was copied.
                grew(new);
                LIVE.fetch_sub(l.size(), Relaxed);
            } else if new >= l.size() {
                grew(new - l.size());
            } else {
                LIVE.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// One test at a time.
static ONE: Mutex<()> = Mutex::new(());

/// The slack over `max_alloc` the real heap may take: what the ledger does not count because it
/// is small and not sized by the input (a thread's bookkeeping, error text, a `Vec` of a few
/// slices), plus the test harness's own allocations while a decode runs.
const SLACK: usize = 1 << 20;

/// Runs `f`; returns its result and the most heap it held above what was live when it began
/// (its result included).
fn heap_peak<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = f();
    (r, PEAK.load(Relaxed).saturating_sub(base))
}

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

/// `with`, on the caller's `pool` if there is one, else on per-call workers if `per_call`, else on
/// the crate's shared pool.
fn with_pool(
    pool: &(&str, Option<ThreadPool>, bool),
    max_threads: usize,
    max_alloc: usize,
) -> Options {
    Options {
        pool: pool.1.clone(),
        per_call_pool: pool.2,
        ..with(max_threads, max_alloc)
    }
}

fn default_threads() -> usize {
    Limits::default().max_threads
}

/// The ledger's and the heap's peak for one decode, under no practical limit.
fn measure(data: &[u8], threads: usize) -> (usize, usize) {
    let (r, heap) = heap_peak(|| jp2k::decode(data, &with(threads, 4 << 30), &nc()));
    let ledger = jp2k::stats::last().peak_alloc;
    drop(r);
    (ledger, heap)
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

/// The ledger covers the heap: for every fixture, whole, as a region and as a component, with
/// one thread and with the default, the real peak is within `SLACK` of what the ledger counted
/// (Go: the estimate covers the peak).
#[test]
fn the_ledger_covers_the_heap() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let nc = nc();
    let mut worst = (0usize, String::new());
    for name in fixtures() {
        let data = read(&name);
        for t in [1, default_threads()] {
            let o = with(t, 4 << 30);
            let rect = jp2k::decode_config(&data, &o.limits).ok().map(|c| Rect {
                x0: i64::from(c.width) / 5,
                y0: i64::from(c.height) / 4,
                x1: i64::from(c.width) * 4 / 5,
                y1: i64::from(c.height),
            });
            let runs: Vec<(&str, Box<dyn Fn() -> bool>)> = vec![
                ("whole", Box::new(|| jp2k::decode(&data, &o, &nc).is_ok())),
                (
                    "region",
                    Box::new(|| {
                        rect.is_some_and(|r| jp2k::decode_region(&data, r, 0, &o, &nc).is_ok())
                    }),
                ),
                (
                    "c0",
                    Box::new(|| jp2k::decode_components(&data, &[0], &o, &nc).is_ok()),
                ),
            ];
            for (what, run) in runs {
                let (_, heap) = heap_peak(&run);
                let ledger = jp2k::stats::last().peak_alloc;
                if heap.saturating_sub(ledger) > worst.0 {
                    worst = (heap - ledger, format!("{name} {what} threads {t}"));
                }
                assert!(
                    heap <= ledger + SLACK,
                    "{name} {what} at {t} threads: heap peak {heap} over the ledger's {ledger}"
                );
            }
        }
    }
    eprintln!("most heap over the ledger: {} bytes ({})", worst.0, worst.1);
}

/// `max_alloc` is a cap on the peak: for several fixtures and several limits, a decode either
/// succeeds with a heap peak (and a ledger peak) within the limit, or fails with
/// `LimitExceeded`. With one thread the least limit that works is the ledger's peak, exactly;
/// below it, or at half, the decode fails. From that limit up every thread count, with and
/// on the shared pool, a caller's and per-call workers, succeeds with the serial bytes: more threads never turn a success into
/// `LimitExceeded`.
#[test]
fn max_alloc_caps_the_peak() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
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
        let (_, heap1) = measure(&data, 1);
        eprintln!("{name}: ledger peak {ledger}, heap peak {heap1} (1 thread)");
        let ts = if name == "fl_dense.j2k" {
            vec![1, default_threads()]
        } else {
            vec![1, 2, 3, default_threads()]
        };
        for pool in [
            ("shared", None, false),
            ("caller", Some(ThreadPool::new(4).unwrap()), false),
            ("per-call", None, true),
        ] {
            for &t in &ts {
                let pooled = pool.0;
                let (_, heapn) =
                    heap_peak(|| jp2k::decode(&data, &with_pool(&pool, t, 4 << 30), &nc));
                let ledgern = jp2k::stats::last().peak_alloc;
                eprintln!(
                    "{name}: {t} threads, {pooled} pool: ledger peak {ledgern}, heap peak {heapn}"
                );
                for max in [ledger, 2 * ledger, 4 * ledger] {
                    let (r, heap) =
                        heap_peak(|| jp2k::decode(&data, &with_pool(&pool, t, max), &nc));
                    let st = jp2k::stats::last();
                    assert!(
                        st.peak_alloc <= max,
                        "{name} {t} threads, {pooled} pool, max_alloc {max}: ledger peak {}",
                        st.peak_alloc
                    );
                    assert!(
                        heap <= max + SLACK,
                        "{name} {t} threads, {pooled} pool, max_alloc {max}: heap peak {heap}"
                    );
                    match r {
                        Ok(img) => assert!(
                            img == want,
                            "{name} {t} threads, {pooled} pool, max_alloc {max}"
                        ),
                        Err(e) => {
                            panic!("{name} {t} threads, {pooled} pool, max_alloc {max}: {e}")
                        }
                    }
                }
            }
        }
        for max in [ledger - 1, ledger / 2, 0] {
            let r = jp2k::decode(&data, &with(1, max), &nc);
            assert!(
                matches!(r, Err(Error::LimitExceeded(_))),
                "{name}: max_alloc {max} (peak {ledger}) gave {:?}",
                r.map(|_| ())
            );
        }
    }
}

/// Regions and component decodes obey `max_alloc` the same way.
#[test]
fn max_alloc_caps_regions_and_components() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let nc = nc();
    let data = read("fl_dense.j2k");
    let rect = Rect {
        x0: 300,
        y0: 200,
        x1: 1500,
        y1: 1700,
    };
    jp2k::decode_region(&data, rect, 0, &with(1, 4 << 30), &nc).unwrap();
    let rp1 = jp2k::stats::last().peak_alloc;
    jp2k::decode_components(&data, &[1], &with(1, 4 << 30), &nc).unwrap();
    let cp1 = jp2k::stats::last().peak_alloc;
    let want_r = jp2k::decode_region(&data, rect, 0, &with(1, 4 << 30), &nc).unwrap();
    let want_c = jp2k::decode_components(&data, &[1], &with(1, 4 << 30), &nc).unwrap();
    // From the serial peak up every thread count, on every kind of pool, succeeds with the serial
    // bytes: more threads never turn a success into `LimitExceeded`.
    for pool in [
        ("shared", None, false),
        ("caller", Some(ThreadPool::new(4).unwrap()), false),
        ("per-call", None, true),
    ] {
        for t in [1, 2, 3, default_threads()] {
            for max in [rp1, 2 * rp1] {
                let (r, heap) = heap_peak(|| {
                    jp2k::decode_region(&data, rect, 0, &with_pool(&pool, t, max), &nc)
                });
                assert!(
                    heap <= max + SLACK,
                    "region at {t} threads, {} pool, {max}: {heap}",
                    pool.0
                );
                assert!(
                    r.as_ref().is_ok_and(|r| *r == want_r),
                    "region {t} {} pool {max}: {r:?}",
                    pool.0
                );
            }
            for max in [cp1, 2 * cp1] {
                let (r, heap) = heap_peak(|| {
                    jp2k::decode_components(&data, &[1], &with_pool(&pool, t, max), &nc)
                });
                assert!(
                    heap <= max + SLACK,
                    "component at {t} threads, {} pool, {max}: {heap}",
                    pool.0
                );
                assert!(
                    r.as_ref().is_ok_and(|r| *r == want_c),
                    "component {t} {} pool {max}: {r:?}",
                    pool.0
                );
            }
        }
    }
    for t in [1, 3] {
        let r = jp2k::decode_region(&data, rect, 0, &with(t, rp1 / 2), &nc);
        assert!(matches!(r, Err(Error::LimitExceeded(_))));
        let r = jp2k::decode_components(&data, &[1], &with(t, cp1 / 2), &nc);
        assert!(matches!(r, Err(Error::LimitExceeded(_))));
    }
}

/// Decode's heap is its output image plus one working plane per component in flight and a
/// little scratch per thread (Go: `TestDecodeAllocBudget`): 256x256, 3 components, 16-bit 9/7
/// without MCT is a 512 KiB output and a 256 KiB plane.
#[test]
fn decode_heap_is_the_output_and_a_plane() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let data = read("bench_small_97_mct0.j2k");
    let floor = 256 * 256 * 8 + 256 * 256 * 4;
    for t in [1usize, 4] {
        let (_, heap) = heap_peak(|| jp2k::decode(&data, &with(t, 4 << 30), &nc()).unwrap());
        let budget = floor + (512 << 10) + (t - 1) * (256 << 10);
        eprintln!("bench_small_97_mct0 at {t} threads: heap peak {heap}, budget {budget}");
        assert!(heap <= budget, "{t} threads: {heap} over {budget}");
    }
}

/// 136 bytes that declare a 16384x16384 16-bit RGB image with empty packets: about 6 GB to
/// decode. It fails under the limit at once, having taken next to no memory (Go:
/// `TestHugeCanvasFailsUnderLimit`).
#[test]
fn a_huge_canvas_fails_under_the_limit() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let hex = "ff4fff51002f0000000040000000400000000000000000000000400000004000000000000000000000030f01010f01010f0101\
ff52000c00000001010504040000ff5c0023425000500050005000500050005000500050005000500050005000500050005000\
ff90000a0000000000200001ff93000000000000000000000000000000000000ffd9";
    let data: Vec<u8> = (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    assert_eq!(data.len(), 136);
    for t in [1, default_threads()] {
        let start = std::time::Instant::now();
        let (r, heap) = heap_peak(|| jp2k::decode(&data, &with(t, 256 << 20), &nc()));
        assert!(matches!(r, Err(Error::LimitExceeded(_))), "{r:?}");
        assert!(heap < 1 << 20, "{heap} bytes to fail");
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
    }
}

/// Packed packet headers: 2 tiles of 128 PPT markers of 65532 bytes each. The ledger has to
/// count them (Go: `TestEstimateCoversPackedHeaders`).
#[test]
fn packed_headers_are_charged() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let hex = "ff4f".to_string()
        + "ff5100290000"
        + "00000040000000200000000000000000"
        + "00000020000000200000000000000000"
        + "0001070101"
        + "ff52000c00000001000004040001"
        + "ff5c00044040";
    let mut b: Vec<u8> = (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    const N: usize = 128;
    const SIZE: usize = 65532;
    for tile in 0..2u8 {
        b.extend_from_slice(&[0xff, 0x90, 0, 10, 0, tile]);
        b.extend_from_slice(&((12 + N * (SIZE + 5) + 2 + 1) as u32).to_be_bytes());
        b.extend_from_slice(&[0, 1]);
        for z in 0..N {
            b.extend_from_slice(&[0xff, 0x61]);
            b.extend_from_slice(&((SIZE + 3) as u16).to_be_bytes());
            b.push(z as u8);
            b.resize(b.len() + SIZE, 0);
        }
        b.extend_from_slice(&[0xff, 0x93, 0]);
    }
    b.extend_from_slice(&[0xff, 0xd9]);
    let rect = Rect {
        x0: 0,
        y0: 0,
        x1: 8,
        y1: 8,
    };
    let o = with(default_threads(), 4 << 30);
    for region in [false, true] {
        let (_, heap) = heap_peak(|| {
            if region {
                let _ = jp2k::decode_region(&b, rect, 0, &o, &nc());
            } else {
                let _ = jp2k::decode(&b, &o, &nc());
            }
        });
        let ledger = jp2k::stats::last().peak_alloc;
        eprintln!("packed headers, region {region}: ledger {ledger}, heap {heap}");
        assert!(
            heap <= ledger + SLACK,
            "region {region}: heap peak {heap} over the ledger's {ledger}"
        );
    }
}

/// A JP2 file followed by a 50 MiB `xml ` box decodes within a tight `max_alloc`: from a slice
/// the box costs nothing, and from a reader the whole input is charged, so the limit has to hold
/// the box and the decode (Go: `TestBigBoxJP2Decodes`).
#[test]
fn a_big_box_jp2_decodes_within_a_tight_limit() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let mut data = read("gray_box_m5.jp2");
    let len = 8 + (50usize << 20);
    data.extend_from_slice(&(len as u32).to_be_bytes());
    data.extend_from_slice(b"xml ");
    data.resize(data.len() + len - 8, 0);
    let nc = nc();
    // A slice: the input is the caller's.
    let (r, heap) = heap_peak(|| jp2k::decode(&data, &with(2, 4 << 20), &nc));
    r.unwrap();
    assert!(heap <= (4 << 20) + SLACK, "slice: {heap}");
    // A reader: the input counts.
    // The buffer grows by copying, so the limit holds the old buffer and the new one too.
    for limit in [100usize << 20, 110 << 20] {
        let (r, heap) =
            heap_peak(|| jp2k::decode_reader(std::io::Cursor::new(&data), &with(2, limit), &nc));
        r.unwrap_or_else(|e| panic!("at a {} MiB limit: {e}", limit >> 20));
        assert!(
            heap <= limit + SLACK,
            "reader at {} MiB: heap peak {heap}",
            limit >> 20
        );
        assert!(jp2k::stats::last().peak_alloc <= limit);
    }
    // Under the input size it fails, having read about the limit.
    let r = jp2k::decode_reader(std::io::Cursor::new(&data), &with(2, 50 << 20), &nc);
    assert!(matches!(r, Err(Error::LimitExceeded(_))), "{r:?}");
}

/// A reader that would give more than `max_alloc` fails as soon as the limit is passed, without
/// reading the rest (Go: `TestSizedReaderFailsFast`).
#[test]
fn a_reader_over_max_alloc_fails_fast() {
    use std::io::Read;
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    struct Counted(u64, std::io::Take<std::io::Repeat>);
    impl Read for Counted {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            let n = self.1.read(b)?;
            self.0 += n as u64;
            Ok(n)
        }
    }
    let mut r = Counted(0, std::io::repeat(0).take(100 << 20));
    let (res, heap) = heap_peak(|| jp2k::decode_reader(&mut r, &with(2, 10 << 20), &nc()));
    assert!(matches!(res, Err(Error::LimitExceeded(_))), "{res:?}");
    assert!(r.0 < (11 << 20), "read {} bytes", r.0);
    assert!(heap <= (10 << 20) + SLACK, "{heap}");
}

/// The parse holds heap too, and it counts against what the call already holds: a reader's input
/// of about 0.95 x `max_alloc` that is mostly empty tile-parts would reserve another ~0.4 x
/// `max_alloc` for their records, which is over the limit together. The parse fails first, with
/// the heap still within the limit.
#[test]
fn the_parse_counts_what_the_call_holds() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    const M: usize = 16 << 20;
    // SOC, SIZ (65535 one-pixel tiles), COD, QCD, then k bare 14-byte tile-parts, then one
    // tile-part whose data fills the input to 0.95 x M.
    let k = M / 100;
    let mut data = tile_part_flood(k);
    data.truncate(data.len() - 2); // the EOC
    let fill = M * 95 / 100 - data.len() - 14;
    data.extend_from_slice(&[0xff, 0x90, 0, 10]);
    data.extend_from_slice(&((k % 65535) as u16).to_be_bytes());
    data.extend_from_slice(&((14 + fill) as u32).to_be_bytes());
    data.extend_from_slice(&[(k / 65535) as u8, 0, 0xff, 0x93]);
    data.resize(data.len() + fill, 0);
    data.extend_from_slice(&[0xff, 0xd9]);
    let nc = nc();
    let o = with(2, M);
    let (r, heap) = heap_peak(|| jp2k::decode_reader(std::io::Cursor::new(&data), &o, &nc));
    assert!(matches!(r, Err(Error::LimitExceeded(_))), "{r:?}");
    assert!(heap <= M + SLACK, "reader: heap peak {heap} over {M}");
    let rect = Rect {
        x0: 0,
        y0: 0,
        x1: 65535,
        y1: 1,
    };
    let (r, heap) = heap_peak(|| jp2k::decode_region(&data, rect, 0, &o, &nc));
    assert!(!matches!(r, Err(Error::Internal(_))), "{r:?}");
    assert!(heap <= M + SLACK, "region: heap peak {heap} over {M}");
}

/// 31 POC records that each cover every packet of a tile of 32,772: the sequence used to grow to
/// 31 times its reservation (tens of MiB) before the repeats were dropped. Repeats are now dropped
/// as they are generated, so the heap stays within a 16 MiB limit.
#[test]
fn a_multi_poc_sequence_stays_within_its_charge() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    const M: usize = 16 << 20;
    // 64x64, LRCP, 5 levels of one precinct each (6 precincts), 5462 layers.
    let poc: Vec<u8> = (0..31)
        .flat_map(|_| [0u8, 0, 0x15, 0x56, 6, 1, 0])
        .collect();
    let data = hostile_j2k(
        64,
        0,
        5462,
        5,
        Some(&[0xFF; 6]),
        Some(&poc),
        None,
        None,
        Some(&vec![0u8; 131_072]),
    );
    for t in [1, 3] {
        let (r, heap) = heap_peak(|| jp2k::decode(&data, &with(t, M), &nc()));
        assert!(!matches!(r, Err(Error::Internal(_))), "{r:?}");
        assert!(jp2k::stats::last().peak_alloc <= M);
        assert!(heap <= M + SLACK, "{t} threads: heap peak {heap} over {M}");
    }
}

/// The packet vector is charged before it is allocated: 65535 layers over 6 precincts is 393,210
/// packets, 30 MiB of `PacketRef` that used to be reserved before the charge refused it, with a
/// 16 MiB limit. The heap stays within the limit.
#[test]
fn a_huge_packet_sequence_is_charged_before_it_is_allocated() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    const M: usize = 16 << 20;
    let data = hostile_j2k(
        64,
        0,
        65535,
        5,
        Some(&[0xFF; 6]),
        None,
        None,
        None,
        Some(&vec![0u8; 65_536]),
    );
    for t in [1, 3] {
        let (r, heap) = heap_peak(|| jp2k::decode(&data, &with(t, M), &nc()));
        assert!(!matches!(r, Err(Error::Internal(_))), "{r:?}");
        assert!(jp2k::stats::last().peak_alloc <= M);
        assert!(heap <= M + SLACK, "{t} threads: heap peak {heap} over {M}");
    }
}

/// Packet-header fields `(value, bits)`, most significant bit first, with T.800's bit stuffing (a
/// byte after 0xFF holds 7 bits).
fn header_bits(fields: &[(u32, u32)]) -> Vec<u8> {
    let cap = |out: &Vec<u8>| if out.last() == Some(&0xFF) { 7 } else { 8 };
    let (mut out, mut cur, mut n) = (Vec::new(), 0u32, 0u32);
    for &(v, bits) in fields {
        for i in (0..bits).rev() {
            cur = (cur << 1) | ((v >> i) & 1);
            n += 1;
            if n == cap(&out) {
                out.push(cur as u8);
                (cur, n) = (0, 0);
            }
        }
    }
    if n > 0 {
        out.push((cur << (cap(&out) - n)) as u8);
    }
    if out.last() == Some(&0xFF) {
        out.push(0);
    }
    out
}

/// Bytes each of the two layers gives the one code-block that has data (19 bits: Lblock 3 + 16).
const JOIN_HALF: usize = 1 << 18;

/// An 8-bit gray, 5/3, two-layer stream of two tiles: a 2048x1024 canvas from x 960 in 1024x1024
/// tiles, so tile 0 is 64x1024 and tile 1 1024x1024, with 64x64 precincts of one code-block each.
/// Tile 0's first code-block gets `JOIN_HALF` bytes in each layer, one termination segment, so it
/// is joined (512 KiB); every other packet is empty. Tile 1 then needs a plane of 4 MiB.
fn join_then_bigger_tile() -> Vec<u8> {
    let mut b = vec![0xFF, 0x4F, 0xFF, 0x51, 0, 41, 0, 0];
    for v in [2048u32, 1024, 960, 0, 1024, 1024, 0, 0] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.extend_from_slice(&[0, 1, 7, 1, 1]);
    // COD: user precincts, LRCP, 2 layers, no MCT, 0 levels, 64x64 code-blocks, style 0, 5/3,
    // 64x64 precincts. QCD: no quantization, 2 guard bits, exponent 8.
    b.extend_from_slice(&[0xFF, 0x52, 0, 13, 1, 0, 0, 2, 0, 0, 4, 4, 0, 1, 0x66]);
    b.extend_from_slice(&[0xFF, 0x5C, 0, 4, 0x40, 0x40]);
    let half = JOIN_HALF as u32;
    // Layer 0: present, included, no zero bit-planes, one pass, Lblock + 16, the length.
    let first = header_bits(&[
        (1, 1),
        (1, 1),
        (1, 1),
        (0, 1),
        (0xFFFF, 16),
        (0, 1),
        (half, 19),
    ]);
    // Layer 1: present, included, one pass, Lblock as it is, the length.
    let second = header_bits(&[(1, 1), (1, 1), (0, 1), (0, 1), (half, 19)]);
    let mut t0 = Vec::new();
    for h in [first, second] {
        t0.extend_from_slice(&h);
        t0.resize(t0.len() + JOIN_HALF + 15, 0);
    }
    for (tile, data) in [(0u16, t0), (1, vec![0u8; 512])] {
        b.extend_from_slice(&[0xFF, 0x90, 0, 10]);
        b.extend_from_slice(&tile.to_be_bytes());
        b.extend_from_slice(&(14 + data.len() as u32).to_be_bytes());
        b.extend_from_slice(&[0, 1, 0xFF, 0x93]);
        b.extend_from_slice(&data);
    }
    b.extend_from_slice(&[0xFF, 0xD9]);
    b
}

/// A `Decoder` call holds no more than the free function's: a code-block's 512 KiB join is not
/// kept, uncharged, while the call goes on to a bigger tile. With one thread the heap peak of a
/// fresh `Decoder`'s call is the free function's, but for the join a tier-1 worker may keep (at
/// most 64 KiB) and a little bookkeeping.
#[test]
fn a_decoder_call_holds_no_more_than_the_free_function() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (data, o, nc) = (join_then_bigger_tile(), with(1, 4 << 30), nc());
    let (want, free) = heap_peak(|| jp2k::decode(&data, &o, &nc).unwrap());
    let mut dec = Decoder::new();
    let (ok, kept) = heap_peak(|| dec.decode(&data, &o, &nc).is_ok());
    assert!(ok);
    assert!(dec.decode(&data, &o, &nc).unwrap() == &want);
    eprintln!("join then a bigger tile: heap peak {kept} (Decoder), {free} (decode)");
    assert!(
        kept <= free + (128 << 10),
        "Decoder heap peak {kept}, free function {free}"
    );
}

/// An 8-bit 5/3 codestream with no offsets: a `w` x `h` canvas in `tw` x `th` tiles, a component
/// per `(XRsiz, YRsiz)` of `comps`, `levels` levels, `layers` layers, LRCP, 64x64 code-blocks and
/// one precinct per resolution; `qcd` is the QCD body (Sqcd, then an exponent byte per band),
/// and `tiles` the packets of each tile.
fn stream(
    [w, h, tw, th]: [u32; 4],
    comps: &[(u8, u8)],
    layers: u16,
    levels: u8,
    qcd: &[u8],
    tiles: &[Vec<u8>],
) -> Vec<u8> {
    let mut b = vec![0xFF, 0x4F, 0xFF, 0x51];
    b.extend_from_slice(&(38 + 3 * comps.len() as u16).to_be_bytes());
    b.extend_from_slice(&[0, 0]);
    for v in [w, h, 0, 0, tw, th, 0, 0] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.extend_from_slice(&(comps.len() as u16).to_be_bytes());
    for &(xr, yr) in comps {
        b.extend_from_slice(&[7, xr, yr]);
    }
    b.extend_from_slice(&[0xFF, 0x52, 0, 12, 0, 0]);
    b.extend_from_slice(&layers.to_be_bytes());
    b.extend_from_slice(&[0, levels, 4, 4, 0, 1]);
    b.extend_from_slice(&[0xFF, 0x5C]);
    b.extend_from_slice(&(2 + qcd.len() as u16).to_be_bytes());
    b.extend_from_slice(qcd);
    for (tile, data) in tiles.iter().enumerate() {
        b.extend_from_slice(&[0xFF, 0x90, 0, 10]);
        b.extend_from_slice(&(tile as u16).to_be_bytes());
        b.extend_from_slice(&(14 + data.len() as u32).to_be_bytes());
        b.extend_from_slice(&[0, 1, 0xFF, 0x93]);
        b.extend_from_slice(data);
    }
    b.extend_from_slice(&[0xFF, 0xD9]);
    b
}

/// The packet of a precinct of one code-block with `len` bytes, one coding pass, in each layer
/// of `lens`, followed by `empty` empty packets (one byte each), per layer. The block has no zero
/// bit-planes; each layer's pass continues the segment of the one before, so two layers join.
fn one_block_packets(lens: &[usize], empty: usize) -> Vec<u8> {
    let bits = (usize::BITS - lens.iter().max().unwrap().leading_zeros()).max(3);
    let mut out = Vec::new();
    for (l, &len) in lens.iter().enumerate() {
        let h = if l == 0 {
            // Present, included, no zero bit-planes, one pass, Lblock up to `bits`, the length.
            let up = bits - 3;
            header_bits(&[
                (1, 1),
                (1, 1),
                (1, 1),
                (0, 1),
                ((1 << up) - 1, up),
                (0, 1),
                (len as u32, bits),
            ])
        } else {
            header_bits(&[(1, 1), (1, 1), (0, 1), (0, 1), (len as u32, bits)])
        };
        out.extend_from_slice(&h);
        out.resize(out.len() + len + empty, 0);
    }
    out
}

/// `kept` and `free` heap peaks less the ledger peaks `kl` and `fl`: the Decoder's heap over its
/// ledger is no more than the free function's, but for 128 KiB.
fn covered(what: &str, (kept, kl): (usize, usize), (free, fl): (usize, usize)) {
    eprintln!("{what}: Decoder heap {kept} ledger {kl}; decode heap {free} ledger {fl}");
    assert!(
        kept as i64 - kl as i64 <= free as i64 - fl as i64 + (128 << 10),
        "{what}: Decoder heap {kept} over its ledger {kl} by more than decode's {free} over {fl}"
    );
}

/// The lanes' spare planes are not held, uncharged, while the call upsamples subsampled
/// components: a 1024x1024 YCbCr 4:2:0 stream (no MCT) of one tile, whose 4 MiB plane of
/// component 0 the old code parked before it upsampled components 1 and 2 (4 MiB each).
#[test]
fn a_decoder_call_frees_its_planes_before_upsampling() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let data = stream(
        [1024, 1024, 1024, 1024],
        &[(1, 1), (2, 2), (2, 2)],
        1,
        0,
        &[0x40, 0x40],
        &[vec![0; 3]],
    );
    let (o, nc) = (with(1, 4 << 30), nc());
    let (want, free) = heap_peak(|| jp2k::decode(&data, &o, &nc).unwrap());
    let fl = jp2k::stats::last().peak_alloc;
    let mut dec = Decoder::new();
    let (ok, kept) = heap_peak(|| dec.decode(&data, &o, &nc).is_ok());
    let kl = jp2k::stats::last().peak_alloc;
    assert!(ok);
    assert!(dec.decode(&data, &o, &nc).unwrap() == &want);
    covered("subsampled", (kept, kl), (free, fl));
    assert!(
        kept <= free + (128 << 10),
        "Decoder heap peak {kept}, free function {free}"
    );
}

/// A code-block that fails in tier 1 after its 1 MiB of segments were joined does not leave the
/// join with the `Decoder`: the next call, a 1024x1024 tile whose tier-1 worker the old code
/// took back from the failed call with the join, holds no more than the free function's.
#[test]
fn a_failed_block_leaves_no_join_with_the_decoder() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    // QCD: 7 guard bits, exponent 31: 37 bit-planes, which tier 1 refuses after the join.
    let half = 1 << 19;
    let bad = stream(
        [64, 64, 64, 64],
        &[(1, 1)],
        2,
        0,
        &[0xE0, 0xF8],
        &[one_block_packets(&[half, half], 0)],
    );
    // 5 levels: resolution 0 is one 32x32 code-block, the 5 resolutions above it are empty.
    let good = stream(
        [1024, 1024, 1024, 1024],
        &[(1, 1)],
        1,
        5,
        &[0x40; 17],
        &[one_block_packets(&[4096], 5)],
    );
    let (o, nc) = (with(1, 4 << 30), nc());
    let (want, free) = heap_peak(|| {
        assert!(matches!(
            jp2k::decode(&bad, &o, &nc),
            Err(Error::Invalid(_))
        ));
        jp2k::decode(&good, &o, &nc).unwrap()
    });
    let mut dec = Decoder::new();
    let (ok, kept) = heap_peak(|| {
        assert!(matches!(dec.decode(&bad, &o, &nc), Err(Error::Invalid(_))));
        dec.decode(&good, &o, &nc).is_ok()
    });
    assert!(ok);
    assert!(dec.decode(&good, &o, &nc).unwrap() == &want);
    eprintln!("failed join then a tile: heap peak {kept} (Decoder), {free} (decode)");
    assert!(
        kept <= free + (128 << 10),
        "Decoder heap peak {kept}, free function {free}"
    );
}

/// A region's lower wavelet levels stay charged while the call goes on. Two 1024-high tiles, 1024
/// and 768 wide, 5 levels: the region leaves out column 0, so tile 0 is decoded by levels (its
/// lower ones about 1.3 MiB), then tile 1 whole, with a 2 MiB join in its one code-block that
/// makes tile 1 the peak. The old code released tile 0's lower levels and kept them, uncharged,
/// through that peak; now the `Decoder`'s ledger holds them, so it covers its heap as the free
/// function's does, and its peak is at least the free function's.
#[test]
fn a_region_keeps_its_lower_levels_charged() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let half = 1 << 20;
    let data = stream(
        [1792, 1024, 1024, 1024],
        &[(1, 1)],
        2,
        5,
        &[0x40; 17],
        &[vec![0; 12], one_block_packets(&[half, half], 5)],
    );
    let rect = Rect {
        x0: 1,
        y0: 0,
        x1: 1792,
        y1: 1024,
    };
    let (o, nc) = (with(1, 4 << 30), nc());
    let (want, free) = heap_peak(|| jp2k::decode_region(&data, rect, 0, &o, &nc).unwrap());
    let fl = jp2k::stats::last().peak_alloc;
    let mut dec = Decoder::new();
    let (ok, kept) = heap_peak(|| dec.decode_region(&data, rect, 0, &o, &nc).is_ok());
    let kl = jp2k::stats::last().peak_alloc;
    assert!(ok);
    assert!(dec.decode_region(&data, rect, 0, &o, &nc).unwrap() == &want);
    covered("region levels", (kept, kl), (free, fl));
    assert!(kl >= fl, "Decoder ledger peak {kl} under decode's {fl}");
}

/// An 8-bit gray stream of one component: SIZ `[w, h, x0, y0, tw, th]` (tile grid at 0), the COD
/// body `cod`, QCD with no quantization, 2 guard bits and exponent 8 (one band: no levels), and
/// the packets of each tile.
fn gray_stream([w, h, x0, y0, tw, th]: [u32; 6], cod: &[u8], tiles: &[Vec<u8>]) -> Vec<u8> {
    let mut b = vec![0xFF, 0x4F, 0xFF, 0x51, 0, 41, 0, 0];
    for v in [w, h, x0, y0, tw, th, 0, 0] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.extend_from_slice(&[0, 1, 7, 1, 1, 0xFF, 0x52]);
    b.extend_from_slice(&(2 + cod.len() as u16).to_be_bytes());
    b.extend_from_slice(cod);
    b.extend_from_slice(&[0xFF, 0x5C, 0, 4, 0x40, 0x40]);
    for (tile, data) in tiles.iter().enumerate() {
        b.extend_from_slice(&[0xFF, 0x90, 0, 10]);
        b.extend_from_slice(&(tile as u16).to_be_bytes());
        b.extend_from_slice(&(14 + data.len() as u32).to_be_bytes());
        b.extend_from_slice(&[0, 1, 0xFF, 0x93]);
        b.extend_from_slice(data);
    }
    b.extend_from_slice(&[0xFF, 0xD9]);
    b
}

/// For each leaf of a `w` x `h` tag tree in raster order, the nodes its first decode reads: those
/// on its path to the root that no earlier leaf's path took. With every value 0, each reads one
/// 1 bit.
fn new_tag_nodes(w: usize, h: usize) -> Vec<usize> {
    let mut dims = vec![(w, h)];
    while dims.last().is_some_and(|&(a, b)| a * b > 1) {
        let (a, b) = dims[dims.len() - 1];
        dims.push((a.div_ceil(2), b.div_ceil(2)));
    }
    let mut seen = std::collections::HashSet::new();
    (0..w * h)
        .map(|i| {
            (0..dims.len())
                .filter(|&l| seen.insert((l, (i % w) >> l, (i / w) >> l)))
                .count()
        })
        .collect()
}

/// Tier 2 heavier than the next tile's plane: a 2048x512 canvas from x 992 in 1024x1024 tiles, so
/// tile 0 is 32x512 and tile 1 1024x512, with 4x4 code-blocks, one precinct and 15 layers. Every
/// code-block of tile 0 has one coding pass of no bytes in every layer: 15,360 code-block records
/// of tier 2 (about 3.5 MiB with their segment lists) and 1,024 code-block lists of 15 segments.
/// Every packet of tile 1 is empty; its plane is 2 MiB.
fn heavy_tier_2_then_bigger_tile() -> Vec<u8> {
    const LAYERS: usize = 15;
    let (cw, ch) = (8, 128);
    let new = new_tag_nodes(cw, ch);
    let mut t0 = Vec::new();
    for layer in 0..LAYERS {
        // Present; per block: included (in layer 0 the inclusion and zero bit-plane tree nodes it
        // reads first, all 1), one pass, Lblock as it is, a length of 0 in 3 bits.
        let mut f: Vec<(u32, u32)> = vec![(1, 1)];
        for &n in &new {
            if layer == 0 {
                f.extend(std::iter::repeat_n((1, 1), 2 * n));
            } else {
                f.push((1, 1));
            }
            f.extend([(0, 1), (0, 1), (0, 3)]);
        }
        t0.extend_from_slice(&header_bits(&f));
    }
    // COD: default precincts, LRCP, 15 layers, no MCT, no levels, 4x4 code-blocks, style 0, 5/3.
    let cod = [0, 0, 0, LAYERS as u8, 0, 0, 0, 0, 0, 1];
    gray_stream(
        [2048, 512, 992, 0, 1024, 1024],
        &cod,
        &[t0, vec![0; LAYERS]],
    )
}

/// A `Decoder` call frees the tier-2 lists and code-block lists of a heavy tile before the next,
/// bigger tile: the old code kept both, uncharged, through tile 1's plane, about 4 MiB over the
/// free function's heap peak. With one thread the heap peak of a fresh `Decoder`'s call is the
/// free function's but for 128 KiB, and its heap over its ledger is the free function's but for
/// 128 KiB.
#[test]
fn a_decoder_call_frees_a_heavy_tier_2_before_a_bigger_tile() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (data, o, nc) = (heavy_tier_2_then_bigger_tile(), with(1, 4 << 30), nc());
    let (want, free) = heap_peak(|| jp2k::decode(&data, &o, &nc).unwrap());
    let fl = jp2k::stats::last().peak_alloc;
    let mut dec = Decoder::new();
    let (ok, kept) = heap_peak(|| dec.decode(&data, &o, &nc).is_ok());
    let kl = jp2k::stats::last().peak_alloc;
    assert!(ok);
    assert!(dec.decode(&data, &o, &nc).unwrap() == &want);
    covered("heavy tier 2", (kept, kl), (free, fl));
    assert!(
        kept <= free + (128 << 10),
        "Decoder heap peak {kept}, free function {free}"
    );
}

/// A wide row buffer is not kept, uncharged, into the next tile: a 65536x5 image from y 63 in
/// 65536x64 tiles, 64x64 code-blocks and precincts, 2 layers. Tile 0 is one row (a 256 KiB row
/// buffer), tile 1 four rows whose first code-block joins 1 MiB of segments while tile 1's 1 MiB
/// plane is held: the call's peak, which the old code raised by the row buffer it kept.
#[test]
fn a_decoder_call_frees_a_wide_row_buffer_before_a_bigger_tile() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let half = 1 << 19;
    // COD: user precincts, LRCP, 2 layers, no MCT, no levels, 64x64 code-blocks, style 0, 5/3,
    // 64x64 precincts.
    let cod = [1, 0, 0, 2, 0, 0, 4, 4, 0, 1, 0x66];
    let data = gray_stream(
        [65536, 68, 0, 63, 65536, 64],
        &cod,
        &[vec![0; 2048], one_block_packets(&[half, half], 1023)],
    );
    let (o, nc) = (with(1, 4 << 30), nc());
    let (want, free) = heap_peak(|| jp2k::decode(&data, &o, &nc).unwrap());
    let fl = jp2k::stats::last().peak_alloc;
    let mut dec = Decoder::new();
    let (ok, kept) = heap_peak(|| dec.decode(&data, &o, &nc).is_ok());
    let kl = jp2k::stats::last().peak_alloc;
    assert!(ok);
    assert!(dec.decode(&data, &o, &nc).unwrap() == &want);
    covered("wide row", (kept, kl), (free, fl));
    assert!(
        kept <= free + (128 << 10),
        "Decoder heap peak {kept}, free function {free}"
    );
}

/// A fresh `Decoder`'s call on `data`, with one thread, holds no more than the free function's:
/// its heap peak is the free function's but for 128 KiB, and so is its heap over its ledger.
fn holds_no_more(what: &str, data: &[u8]) {
    let (o, nc) = (with(1, 4 << 30), nc());
    let (want, free) = heap_peak(|| jp2k::decode(data, &o, &nc).unwrap());
    let fl = jp2k::stats::last().peak_alloc;
    let mut dec = Decoder::new();
    let (ok, kept) = heap_peak(|| dec.decode(data, &o, &nc).is_ok());
    let kl = jp2k::stats::last().peak_alloc;
    assert!(ok, "{what}");
    assert!(dec.decode(data, &o, &nc).unwrap() == &want, "{what}");
    covered(what, (kept, kl), (free, fl));
    assert!(
        kept <= free + (128 << 10),
        "{what}: Decoder heap peak {kept}, free function {free}"
    );
}

/// Code-blocks first included a layer later each tile: a 256x2048 canvas in 256x128 tiles (16),
/// one precinct of 4x4 code-blocks (2048 a tile) and 16 layers. In tile `j` every code-block is
/// first included in layer `j`, with one coding pass of no bytes; the tile's other packets are
/// empty. Each tile's tier 2 is the same, but its 2048 code-block records are in packet `j`.
fn later_layer_each_tile() -> Vec<u8> {
    let tiles: Vec<Vec<u8>> = (0..LATER_LAYERS).map(later_layer_packets).collect();
    gray_stream([256, 2048, 0, 0, 256, 128], &LATER_COD, &tiles)
}

const LATER_LAYERS: usize = 16;

/// COD: default precincts, LRCP, 16 layers, no MCT, no levels, 4x4 code-blocks, style 0, 5/3.
const LATER_COD: [u8; 10] = [0, 0, 0, LATER_LAYERS as u8, 0, 0, 0, 0, 0, 1];

/// The packets of `later_layer_each_tile`'s tile `j`.
fn later_layer_packets(j: usize) -> Vec<u8> {
    let new = new_tag_nodes(64, 32);
    let mut t = Vec::new();
    for layer in 0..LATER_LAYERS {
        if layer != j {
            t.push(0);
            continue;
        }
        // Present; the inclusion tree's root reads `j` zeros first (no block is in an earlier
        // layer); per block the inclusion and zero bit-plane tree nodes it reads first, all 1,
        // one pass, Lblock as it is, a length of 0 in 3 bits.
        let mut f: Vec<(u32, u32)> = vec![(1, 1), (0, j as u32)];
        for &n in &new {
            f.extend(std::iter::repeat_n((1, 1), 2 * n));
            f.extend([(0, 1), (0, 1), (0, 3)]);
        }
        t.extend_from_slice(&header_bits(&f));
    }
    t
}

/// A `Decoder` charges the tier-2 lists it keeps for what they hold. Tier 2 reuses a packet's
/// code-block list for the same packet of the next tile, so here the lists a lane kept grew by
/// 2048 records (180 KiB) a tile while each tile's tier 2 charged about 0.7 MB: the old code kept
/// them, charged that much, through every tile, so by the last its heap was about 2.5 MB over
/// its ledger and over the free function's heap.
#[test]
fn a_decoder_charges_tier_2_lists_that_grow_by_position() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    holds_no_more("later layer each tile", &later_layer_each_tile());
}

/// What a `Decoder` keeps between calls does not grow with its calls: one `Decoder`, one
/// thread, decodes `later_layer_each_tile`'s tiles each as a stream of its own, so the 2048
/// code-block records of call `j` are in packet `j`. After each call it holds at most what a
/// fresh `Decoder` holds after that stream plus 256 KiB of tier-2 lists and 64 KiB of
/// code-block lists. The old code kept the lists of a call's last tile on one lane at any size,
/// and tier 2 gives each list to the same packet next call, so the lists of packets 0 to `j`
/// stayed held, 2048 records (180 KiB) more each call: about 2.5 MB over by call 16.
#[test]
fn a_decoder_keeps_bounded_tier_2_lists_between_calls() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (o, nc) = (with(1, 4 << 30), nc());
    let streams: Vec<Vec<u8>> = (0..LATER_LAYERS)
        .map(|j| {
            gray_stream(
                [256, 128, 0, 0, 256, 128],
                &LATER_COD,
                &[later_layer_packets(j)],
            )
        })
        .collect();
    let kept = |dec: &mut Decoder, data: &[u8], base: usize| {
        dec.decode(data, &o, &nc).unwrap();
        LIVE.load(Relaxed).saturating_sub(base)
    };
    let base = LIVE.load(Relaxed);
    let mut dec = Decoder::new();
    for (j, data) in streams.iter().enumerate() {
        let reused = kept(&mut dec, data, base);
        let fresh_base = LIVE.load(Relaxed);
        let mut fresh_dec = Decoder::new();
        let fresh = kept(&mut fresh_dec, data, fresh_base);
        drop(fresh_dec);
        assert!(
            reused <= fresh + (256 << 10) + (64 << 10),
            "call {j}: Decoder keeps {reused}, a fresh Decoder {fresh}"
        );
    }
}

/// A code-block with many segments at another place each tile: a 1024x256 canvas in 64x32 tiles
/// (128), one precinct of 4x4 code-blocks (128 a tile), 60 layers, exponent 20 (21 bit-planes,
/// so tier 2 records 60 passes). Every code-block is included in layer 0, and in tile `j`
/// code-block `j` in every layer, one pass of no bytes each: its code-block list holds 60
/// segments.
fn many_segments_moving() -> Vec<u8> {
    const LAYERS: usize = 60;
    let new = new_tag_nodes(16, 8);
    let tiles: Vec<Vec<u8>> = (0..128)
        .map(|j| {
            // Layer 0 as in `later_layer_each_tile`'s tile 0.
            let mut f: Vec<(u32, u32)> = vec![(1, 1)];
            for &n in &new {
                f.extend(std::iter::repeat_n((1, 1), 2 * n));
                f.extend([(0, 1), (0, 1), (0, 3)]);
            }
            let mut t = header_bits(&f);
            for _ in 1..LAYERS {
                // Present; of the blocks already included only block `j` is again.
                let mut f: Vec<(u32, u32)> = vec![(1, 1)];
                for i in 0..128 {
                    if i == j {
                        f.extend([(1, 1), (0, 1), (0, 1), (0, 3)]);
                    } else {
                        f.push((0, 1));
                    }
                }
                t.extend_from_slice(&header_bits(&f));
            }
            t
        })
        .collect();
    let cod = [0, 0, 0, LAYERS as u8, 0, 0, 0, 0, 0, 1];
    let mut data = gray_stream([1024, 256, 0, 0, 64, 32], &cod, &tiles);
    let q = data
        .windows(6)
        .position(|w| w == [0xFF, 0x5C, 0, 4, 0x40, 0x40])
        .unwrap();
    data[q + 5] = 20 << 3;
    data
}

/// A `Decoder` bounds the code-block lists it keeps between tile components by what they hold.
/// Each component's lists were charged under 64 KiB, so the old code kept them, uncharged; but
/// code-block `i` reuses the segment list of the code-block `i` before it, so the list that grew
/// to 60 segments in tile `j` was kept at position `j`, and by the last tile 128 of them (about
/// 300 KiB) were held over the free function's heap and the ledger.
#[test]
fn a_decoder_charges_code_block_lists_that_grow_by_position() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    holds_no_more("many segments moving", &many_segments_moving());
}

/// Tier 2 heavy in tile 0 and light after it, then a join: a 512x2560 canvas in 512x256 tiles
/// (10), 64x64 code-blocks and precincts (32 a tile) and 72 layers. In tile 0 every code-block
/// has one pass of no bytes in every layer: 2304 code-block records. Tiles 1 to 8 are empty,
/// 2304 packets each. In tile 9 the first code-block has 1 MiB in each of layers 0 and 1, which
/// are joined: the call's peak.
fn heavy_tier_2_then_light_tiles() -> Vec<u8> {
    const LAYERS: usize = 72;
    const PRECINCTS: usize = 32;
    let mut t0 = Vec::new();
    for layer in 0..LAYERS {
        for _ in 0..PRECINCTS {
            // Present, included (in layer 0 the one node of the inclusion and of the zero
            // bit-plane tree, 1), one pass, Lblock as it is, a length of 0 in 3 bits.
            let mut f: Vec<(u32, u32)> = vec![(1, 1), (1, 1)];
            if layer == 0 {
                f.push((1, 1));
            }
            f.extend([(0, 1), (0, 1), (0, 3)]);
            t0.extend_from_slice(&header_bits(&f));
        }
    }
    let mut tiles = vec![t0];
    tiles.resize(9, vec![0; LAYERS * PRECINCTS]);
    let half = 1 << 20;
    let mut last = one_block_packets(&[half, half], PRECINCTS - 1);
    last.resize(last.len() + (LAYERS - 2) * PRECINCTS, 0);
    tiles.push(last);
    // COD: user precincts, LRCP, 72 layers, no MCT, no levels, 64x64 code-blocks, style 0, 5/3,
    // 64x64 precincts.
    let cod = [1, 0, 0, LAYERS as u8, 0, 0, 4, 4, 0, 1, 0x66];
    gray_stream([512, 2560, 0, 0, 512, 256], &cod, &tiles)
}

/// A `Decoder` does not keep a heavy tile's tier-2 lists, uncharged, into lighter tiles. The old
/// code took them into the next tile against a credit of their charge, and a release of tier 2's
/// reserve for the packet sequence (72 KiB here) came off the ledger even when the credit had
/// paid for it, so each light tile gave back that much more of the charge while the lists stayed
/// held: by tile 9 about 1 MB of them was held over the free function's heap and the ledger.
#[test]
fn a_decoder_keeps_no_heavy_tier_2_uncharged_into_light_tiles() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    holds_no_more(
        "heavy tier 2, light tiles",
        &heavy_tier_2_then_light_tiles(),
    );
}
