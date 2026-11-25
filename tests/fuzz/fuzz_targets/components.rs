#![no_main]
use jp2k::{Error, Limits, Options};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

// data = [stream][4 component indexes][mode: u8]: decode_components with up to 4 of components
// 0 to 4 (repeats, and ones the image lacks, are errors), in each rounding mode.
fuzz_target!(|data: &[u8]| {
    let Some((stream, t)) = data.split_last_chunk::<5>() else {
        return;
    };
    let comps: Vec<usize> = t[..4].iter().map(|&c| usize::from(c % 5)).collect();
    let o = Options {
        separate_rounding: t[4] & 1 != 0,
        jj2000: t[4] & 2 != 0,
        limits: Limits {
            max_alloc: 256 << 20,
            max_pixels: 1 << 22,
            max_output: 64 << 20,
            max_input: 1 << 24,
            ..Limits::default()
        },
        pool: None,
        per_call_pool: false,
    };
    let r = jp2k::decode_components(
        stream,
        &comps[..1 + usize::from(t[4] >> 2) % 4],
        &o,
        &AtomicBool::new(false),
    );
    assert!(!matches!(r, Err(Error::Internal(_))), "{r:?}");
});
