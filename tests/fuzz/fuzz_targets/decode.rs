#![no_main]
use jp2k::{Error, Limits, Options};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

/// Caps that keep an input from costing more than the fuzzer's own limits (-rss_limit_mb,
/// -malloc_limit_mb): an allocation outside them is a crash.
fn limits() -> Limits {
    Limits {
        max_alloc: 256 << 20,
        max_pixels: 1 << 22,
        max_output: 64 << 20,
        max_input: 1 << 24,
        ..Limits::default()
    }
}

// data = [stream][mode: u8]; the mode picks the rounding (a stream, not a prefix, is what the
// seeds are). `decode` and `decode_config` must not report an internal error (a caught panic),
// and agree on the size and format of an image that decodes.
fuzz_target!(|data: &[u8]| {
    let Some((&mode, data)) = data.split_last() else {
        return;
    };
    let o = Options {
        separate_rounding: mode & 1 != 0,
        jj2000: mode & 2 != 0,
        limits: limits(),
        pool: None,
        per_call_pool: false,
    };
    let img = jp2k::decode(data, &o, &AtomicBool::new(false));
    let cfg = jp2k::decode_config(data, &o.limits);
    for e in [img.as_ref().err(), cfg.as_ref().err()]
        .into_iter()
        .flatten()
    {
        assert!(!matches!(e, Error::Internal(_)), "{e}");
    }
    if let Ok(img) = img {
        let cfg = cfg.expect("decode worked, decode_config must");
        assert_eq!(cfg.color_model, img.color_model);
        assert_eq!(i64::from(cfg.width), img.bounds.x1 - img.bounds.x0);
        assert_eq!(i64::from(cfg.height), img.bounds.y1 - img.bounds.y0);
    }
});
