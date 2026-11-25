#![no_main]
use jp2k::{Error, Image, Limits, Options, Raster, Rect};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

// data = [stream][max_alloc: u32 LE][max_pixels: u16][max_output: u16][max_width: u16]
// [max_height: u16][max_input: u16]: tiny limits drawn from the input. Whatever the limits are,
// a decode, region decode, opened Image or reader decode that works stays within them, and
// nothing fails with an internal error.
fuzz_target!(|data: &[u8]| {
    let Some((stream, t)) = data.split_last_chunk::<14>() else {
        return;
    };
    let u16_at = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]);
    let limits = Limits {
        max_alloc: (u32::from_le_bytes([t[0], t[1], t[2], t[3]]) % (64 << 20)) as usize,
        max_pixels: u64::from(u16_at(4)) * 16,
        max_output: usize::from(u16_at(6)) * 64,
        max_width: u32::from(u16_at(8)),
        max_height: u32::from(u16_at(10)),
        max_input: usize::from(u16_at(12)) * 4,
        ..Limits::default()
    };
    let o = Options {
        limits,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    let whole = Rect {
        x0: 0,
        y0: 0,
        x1: 1 << 20,
        y1: 1 << 20,
    };
    let corner = Rect {
        x0: 0,
        y0: 0,
        x1: 16,
        y1: 16,
    };
    let img = jp2k::decode(stream, &o, &cancel);
    let planes = jp2k::decode_components(stream, &[0], &o, &cancel);
    let cfg = jp2k::decode_config(stream, &limits);
    let cfg_reader = jp2k::decode_config_reader(stream, &limits, &cancel);
    let reader = jp2k::decode_reader(stream, &o, &cancel);
    let region = jp2k::decode_region(stream, whole, 0, &o, &cancel);
    let region_c = jp2k::decode_region_components(stream, corner, 0, &[0], &o, &cancel);
    let open = Image::open(stream, &o, &cancel);
    let (opened, opened_c) = match &open {
        Ok(i) => (
            Some(i.decode_region(corner, 0, &cancel)),
            Some(i.decode_region_components(corner, 0, &[0], &cancel)),
        ),
        Err(_) => (None, None),
    };
    for e in [
        img.as_ref().err(),
        planes.as_ref().err(),
        cfg.as_ref().err(),
        cfg_reader.as_ref().err(),
        reader.as_ref().err(),
        region.as_ref().err(),
        region_c.as_ref().err(),
        open.as_ref().err(),
        opened.as_ref().and_then(|r| r.as_ref().err()),
        opened_c.as_ref().and_then(|r| r.as_ref().err()),
    ]
    .into_iter()
    .flatten()
    {
        assert!(!matches!(e, Error::Internal(_)), "{e}");
    }
    // A reader longer than max_input is refused.
    if stream.len() > limits.max_input {
        assert!(reader.is_err(), "reader over max_input");
    }
    let in_limits = |img: &Raster| {
        let (w, h) = (img.bounds.x1 - img.bounds.x0, img.bounds.y1 - img.bounds.y0);
        assert!(img.pix.len() <= limits.max_output, "output over max_output");
        assert!(img.pix.len() <= limits.max_alloc, "output over max_alloc");
        assert!(w <= i64::from(limits.max_width) && h <= i64::from(limits.max_height));
        assert!((w * h) as u64 <= limits.max_pixels);
    };
    for img in [img, reader, region]
        .iter()
        .chain(opened.iter())
        .filter_map(|r| r.as_ref().ok())
    {
        in_limits(img);
    }
    for planes in [planes, region_c]
        .iter()
        .chain(opened_c.iter())
        .filter_map(|r| r.as_ref().ok())
    {
        let bytes: usize = planes.iter().map(|p| 2 * p.pix.len()).sum();
        assert!(bytes <= limits.max_output, "planes over max_output");
        assert!(bytes <= limits.max_alloc, "planes over max_alloc");
    }
});
