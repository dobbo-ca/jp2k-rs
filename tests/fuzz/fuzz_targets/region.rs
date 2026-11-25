#![no_main]
use jp2k::{Error, Image, Limits, Options, Rect};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

fn no_internal<T: std::fmt::Debug>(r: &jp2k::Result<T>) {
    assert!(!matches!(r, Err(Error::Internal(_))), "{r:?}");
}

/// A limit error depends on how the call charged its memory, which differs between a one-shot
/// decode and a call on an opened Image.
fn limited<T>(r: &jp2k::Result<T>) -> bool {
    matches!(r, Err(Error::LimitExceeded(_) | Error::OutOfMemory))
}

// data = [stream][x0, y0, x1, y1: i16 LE][reduce: u8][mode: u8, bit 0 separate_rounding, bit 1
// jj2000]. The region entry points, and an Image
// opened once and used twice: where Open works, its regions equal the one-shot ones (Go's
// FuzzOpen), also on the second call, which runs on what the first built.
fuzz_target!(|data: &[u8]| {
    let Some((stream, p)) = data.split_last_chunk::<10>() else {
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
    let o = Options {
        limits: Limits {
            max_alloc: 256 << 20,
            max_pixels: 1 << 22,
            max_output: 64 << 20,
            max_input: 1 << 24,
            ..Limits::default()
        },
        separate_rounding: p[9] & 1 != 0,
        jj2000: p[9] & 2 != 0,
        pool: None,
        per_call_pool: false,
    };
    let cancel = AtomicBool::new(false);
    let one = jp2k::decode_region(stream, rect, reduce, &o, &cancel);
    let comps = jp2k::decode_region_components(stream, rect, reduce, &[0], &o, &cancel);
    no_internal(&one);
    no_internal(&comps);
    let Ok(img) = Image::open(stream, &o, &cancel) else {
        return;
    };
    let _ = (img.config(), img.levels());
    let a = img.decode_region(rect, reduce, &cancel);
    let b = img.decode_region(rect, reduce, &cancel);
    let c = img.decode_region_components(rect, reduce, &[0], &cancel);
    no_internal(&a);
    no_internal(&c);
    assert_eq!(a, b);
    // The one-shot call parses strictly when no tile-part of the region survives, and the Image
    // checks the region first, so they can fail with different errors: whether they fail, and
    // what they return, must agree.
    if !limited(&one) && !limited(&a) {
        assert_eq!(one.is_ok(), a.is_ok(), "{one:?} / {a:?}");
        if one.is_ok() {
            assert_eq!(one, a);
        }
    }
    if !limited(&comps) && !limited(&c) {
        assert_eq!(comps.is_ok(), c.is_ok(), "{comps:?} / {c:?}");
        if comps.is_ok() {
            assert_eq!(comps, c);
        }
    }
});
