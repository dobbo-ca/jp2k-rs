#![no_main]
use jp2k::{Image, Limits, Options, Rect, ThreadPool};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

/// `max_alloc` is a quarter of fuzz.yml's -malloc_limit_mb=512 (which bounds one allocation, and
/// -rss_limit_mb=2048 the whole process), so no allocation the ledger allows trips the fuzzer, and
/// a thread count never decides between a result and `LimitExceeded`: the bytes and the error
/// must not depend on threads or pool (README, "Differences from Go", names the one exception: the
/// tile a crafted input names when it exhausts the shared tier-1 work budget).
fn limits(max_threads: usize) -> Limits {
    Limits {
        max_threads,
        max_alloc: 128 << 20,
        max_pixels: 1 << 20,
        max_output: 64 << 20,
        max_input: 1 << 24,
        ..Limits::default()
    }
}

/// One pool for the whole run, as a server would keep: workers are not started per input.
fn pool() -> Option<ThreadPool> {
    static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| ThreadPool::new(3).ok()).clone()
}

// data = [stream][mode: u8]: bit 0 uses the shared 3-thread caller pool, bits 1-2 pick
// `max_threads` (1, 2, 8 or the default), bit 3 `separate_rounding`, bit 4 `jj2000`, bit 5
// per-call workers (without a caller pool; otherwise the crate's shared pool). The region
// and reduce level come from the last 9 bytes of the stream (x0, y0, x1, y1: i16 LE, reduce: u8),
// which stay part of it, so the seeds still decode.
// `decode`, `decode_components`, `decode_region` and `Image::open(..).decode_region` must give
// the same Raster (or the same error) as `max_threads` 1 with no pool.
fuzz_target!(|data: &[u8]| {
    let Some((&mode, data)) = data.split_last() else {
        return;
    };
    let Some(p) = data.last_chunk::<9>() else {
        return;
    };
    let i16_at = |i: usize| i64::from(i16::from_le_bytes([p[i], p[i + 1]]));
    let rect = Rect {
        x0: i16_at(0),
        y0: i16_at(2),
        x1: i16_at(4),
        y1: i16_at(6),
    };
    let reduce = i32::from(p[8] % 40);
    let max_threads = [1, 2, 8, Limits::default().max_threads][usize::from((mode >> 1) & 3)];
    let with = |pool, max_threads| Options {
        separate_rounding: mode & 8 != 0,
        jj2000: mode & 16 != 0,
        limits: limits(max_threads),
        pool,
        per_call_pool: mode & 32 != 0,
    };
    let o = with(if mode & 1 != 0 { pool() } else { None }, max_threads);
    let serial = with(None, 1);
    let cancel = AtomicBool::new(false);
    assert_eq!(
        jp2k::decode(data, &o, &cancel),
        jp2k::decode(data, &serial, &cancel),
        "mode {mode}"
    );
    assert_eq!(
        jp2k::decode_components(data, &[0], &o, &cancel),
        jp2k::decode_components(data, &[0], &serial, &cancel),
        "mode {mode}"
    );
    assert_eq!(
        jp2k::decode_region(data, rect, reduce, &o, &cancel),
        jp2k::decode_region(data, rect, reduce, &serial, &cancel),
        "mode {mode}"
    );
    let opened = |o: &Options| {
        Image::open(data, o, &cancel).and_then(|i| i.decode_region(rect, reduce, &cancel))
    };
    assert_eq!(opened(&o), opened(&serial), "mode {mode}");
});
