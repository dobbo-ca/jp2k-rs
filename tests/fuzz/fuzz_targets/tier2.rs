#![no_main]
use jp2k::fuzz::{self, Cursor, TileGeom, TileWant};
use jp2k::{Budget, Limits, Rect, Result};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

// Go's FuzzTier2: the packet headers of the first four tiles of a stream that parses. One Cursor
// and one Budget run through the tiles, as in a decode, so state kept between tiles is fuzzed.
// The last five input bytes [mode, a, b, c, d] (still part of the stream, so Go's seeds work)
// also pick what a region decode would read of each tile: everything (mode 0), whole
// resolutions up to a per-component top (mode 1), nothing (mode 2), or a window of each level
// (mode 3). That reaches the pruning of packets and the early stop, which the region target
// only reaches through full decodes.
fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_alloc: 256 << 20,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    let Ok(cs) = fuzz::parse(data, &limits, &cancel) else {
        return;
    };
    let mut sel = [0u8; 5];
    let tail = &data[data.len().saturating_sub(5)..];
    sel[..tail.len()].copy_from_slice(tail);
    let [mode, a, b, c, d] = sel;
    let mut want_of = |tg: &TileGeom| -> Result<Option<TileWant>> {
        if mode % 4 == 0 {
            return Ok(None);
        }
        let mut want = TileWant {
            top: Vec::new(),
            wins: Vec::new(),
        };
        for (i, tc) in tg.comps.iter().enumerate() {
            let top = usize::from(a.wrapping_add(i as u8)) % tc.resolutions.len().max(1);
            want.top.push(top);
            want.wins.push(match mode % 4 {
                1 => None,
                2 => Some(Vec::new()),
                _ => {
                    let (x0, y0) = (i64::from(b % 16), i64::from(c % 16));
                    Some(
                        (0..=top)
                            .map(|r| Rect {
                                x0: x0 >> r,
                                y0: y0 >> r,
                                x1: (x0 + i64::from(d) + 1) >> r,
                                y1: (y0 + i64::from(d >> 4) + 1) >> r,
                            })
                            .collect(),
                    )
                }
            });
        }
        Ok(Some(want))
    };
    let tiles = u64::from(cs.siz.tiles_wide) * u64::from(cs.siz.tiles_high);
    let (mut cur, mut budget) = (Cursor::new(), Budget::new(&limits));
    for tile in 0..tiles.min(4) as u16 {
        let _ = fuzz::decode_tile(
            &cs,
            data,
            tile,
            &mut cur,
            Some(&mut want_of),
            &cancel,
            &mut budget,
        );
        let _ = fuzz::assemble_tile(&cs, data, tile, &mut Budget::new(&limits));
    }
    let _ = fuzz::build_tile_geom(&cs, 0, &mut Budget::new(&limits));
});
