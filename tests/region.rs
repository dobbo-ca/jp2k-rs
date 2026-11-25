//! Ports of `internal/decoder/region_test.go`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::fuzz;
use jp2k::{ColorModel, Image, Raster, Rect};

const REGION_FIXTURES: [&str; 4] = [
    "region_rgb_53",
    "region_rgb_97",
    "region_gray_97",
    "region_sub_53",
];

fn region(data: &[u8], rect: Rect, reduce: i32) -> jp2k::Result<Raster> {
    jp2k::decode_region(data, rect, reduce, &opts(), &nc())
}

fn rect(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
    Rect { x0, y0, x1, y1 }
}

/// The 8-bit sample of output channel `c` at (x, y).
fn channel(img: &Raster, x: i64, y: i64, c: usize) -> i32 {
    let bpp = img.color_model.bytes_per_pixel();
    let off = (y - img.bounds.y0) as usize * img.stride + (x - img.bounds.x0) as usize * bpp;
    match img.color_model {
        ColorModel::Gray => img.pix[off] as i32,
        ColorModel::Rgba => img.pix[off + c] as i32,
        m => panic!("unexpected {m:?}"),
    }
}

/// The whole image at reduce 1 and 2 against opj_decompress -r, component by component. A
/// subsampled component shows sample floor(X/XRsiz) at reduced reference-grid X.
#[test]
fn reduce_matches_openjpeg() {
    for name in REGION_FIXTURES {
        let data = read(&format!("{name}.j2k"));
        let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
        let siz = &cs.siz;
        for r in 1..=2i32 {
            let img = region(&data, rect(0, 0, 1 << 20, 1 << 20), r).unwrap();
            let s = r as u32;
            let (ox, oy) = (
                ceil_div_pow2(siz.xosiz as i64, s),
                ceil_div_pow2(siz.yosiz as i64, s),
            );
            for c in 0..siz.components.len() {
                let pgx = read_pgx(path(&format!("{name}_r{r}_{c}.pgx")));
                let (xr, yr) = (
                    siz.components[c].xrsiz as i64,
                    siz.components[c].yrsiz as i64,
                );
                let cx0 = ceil_div_pow2(ceil_div(siz.xosiz as i64, xr), s);
                let cy0 = ceil_div_pow2(ceil_div(siz.yosiz as i64, yr), s);
                if xr == 1 && yr == 1 {
                    assert_eq!((pgx.w, pgx.h), (width(&img), height(&img)), "{name} r{r}");
                }
                let mut diff = 0;
                for y in img.bounds.y0..img.bounds.y1 {
                    for x in img.bounds.x0..img.bounds.x1 {
                        let (sx, sy) = ((x + ox) / xr - cx0, (y + oy) / yr - cy0);
                        if channel(&img, x, y, c) != pgx.samples[sy as usize * pgx.w + sx as usize]
                        {
                            diff += 1;
                        }
                    }
                }
                assert_eq!(diff, 0, "{name} r{r} component {c}");
            }
        }
    }
}

/// A deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn intn(&mut self, n: i64) -> i64 {
        (self.next_u64() % n as u64) as i64
    }
}

/// DecodeRegion's documented output rect.
fn reduced_bounds(siz: &fuzz::Siz, r: Rect, reduce: u32) -> Rect {
    let (xo, yo) = (siz.xosiz as i64, siz.yosiz as i64);
    let f = |v: i64, o: i64| ceil_div_pow2(v + o, reduce) - ceil_div_pow2(o, reduce);
    Rect {
        x0: f(r.x0, xo),
        y0: f(r.y0, yo),
        x1: f(r.x1, xo),
        y1: f(r.y1, yo),
    }
}

fn crop(img: &Raster, r: Rect) -> Raster {
    let bpp = img.color_model.bytes_per_pixel();
    let w = (r.x1 - r.x0) as usize;
    let mut pix = Vec::new();
    for y in r.y0..r.y1 {
        let off = (y - img.bounds.y0) as usize * img.stride + (r.x0 - img.bounds.x0) as usize * bpp;
        pix.extend_from_slice(&img.pix[off..off + w * bpp]);
    }
    Raster {
        color_model: img.color_model,
        bounds: r,
        stride: w * bpp,
        pix,
    }
}

fn check_regions(data: &[u8], dec: &Raster, rng: &mut Rng, n: usize) {
    let stream = if data.starts_with(&jp2k::fuzz::JP2_SIGNATURE) {
        jp2k::fuzz::jp2_codestream(data, &limits()).unwrap().0
    } else {
        data
    };
    let cs = fuzz::parse(stream, &limits(), &nc()).unwrap();
    let siz = &cs.siz;
    let mut levels = cs.cod.coding_style.num_decomp_levels as i32;
    for c in &cs.coc {
        levels = levels.min(c.coding_style.num_decomp_levels as i32);
    }
    for tp in &cs.tile_parts {
        if let Some(c) = tp.cod() {
            levels = levels.min(c.coding_style.num_decomp_levels as i32);
        }
        for c in tp.coc() {
            levels = levels.min(c.coding_style.num_decomp_levels as i32);
        }
    }
    let (w, h) = (width(dec) as i64, height(dec) as i64);
    let full = rect(0, 0, w, h);
    let got = region(data, full, 0).unwrap();
    assert_eq!(&got, dec, "region(full, 0) differs from decode");
    assert!(
        region(data, full, levels + 1).is_err(),
        "reduce {} past {levels} levels: no error",
        levels + 1
    );

    // Rects that start or end on a tile edge, or one sample either side.
    let mut cuts: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
    for (axis, v) in [
        [siz.xtosiz as i64, siz.xtsiz as i64, siz.xosiz as i64],
        [siz.ytosiz as i64, siz.ytsiz as i64, siz.yosiz as i64],
    ]
    .iter()
    .enumerate()
    {
        let size = [w, h][axis];
        let mut e = v[0] + v[1];
        while e - v[2] < size && cuts[axis].len() < 12 {
            for d in -1..=1 {
                let p = e - v[2] + d;
                if p > 0 && p < size {
                    cuts[axis].push(p);
                }
            }
            e += v[1];
        }
    }
    let pick = |axis: usize, size: i64, rng: &mut Rng| -> i64 {
        if !cuts[axis].is_empty() && rng.intn(2) == 0 {
            let i = rng.intn(cuts[axis].len() as i64) as usize;
            return cuts[axis][i];
        }
        rng.intn(size)
    };
    for r in [0, levels.min(1), levels] {
        let refimg = if r > 0 {
            region(data, full, r).unwrap()
        } else {
            dec.clone()
        };
        let mut rs = vec![
            rect(0, 0, 1, 1),
            rect(w - 1, h - 1, w, h),
            rect(w - 1, 0, w, h),
            rect(1, 1, w - 1, h - 1),
            rect(w / 3, h / 3, w / 3 + 1, h / 3 + 1),
        ];
        for _ in 0..n {
            let (x0, y0) = (pick(0, w, rng), pick(1, h, rng));
            let (mut x1, mut y1) = (x0 + 1 + rng.intn(w - x0), y0 + 1 + rng.intn(h - y0));
            if rng.intn(3) == 0 {
                x1 = pick(0, w + 1, rng);
                y1 = pick(1, h + 1, rng);
                if x1 <= x0 || y1 <= y0 {
                    continue;
                }
            }
            rs.push(rect(x0, y0, x1, y1));
        }
        for rc in rs {
            let want = reduced_bounds(siz, rc, r as u32);
            let got = region(data, rc, r);
            if want.empty() {
                assert!(
                    got.is_err(),
                    "{rc:?} at reduce {r}: empty after reduce but no error"
                );
                continue;
            }
            let got = got.unwrap_or_else(|e| panic!("{rc:?} at reduce {r}: {e}"));
            assert!(
                got.bounds == want && got == crop(&refimg, want),
                "{rc:?} at reduce {r}: bounds {:?}, want {want:?}; or pixels differ from the crop",
                got.bounds
            );
        }
    }
}

/// A region at reduce r is exactly the crop of the whole image at reduce r, and the whole image
/// at reduce 0 is `decode`'s output. Rects are random (seeded) plus the edge cases: 1x1, image
/// corners, odd origins, rects straddling tile edges.
#[test]
fn region_matches_crop() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
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
    let mut n = 0;
    for f in files {
        let data = std::fs::read(f).unwrap();
        let Ok(dec) = jp2k::decode(&data, &opts(), &nc()) else {
            continue;
        };
        // The 2040x2048 slide fixtures are for the benchmarks.
        if width(&dec) * height(&dec) > 1 << 20 {
            continue;
        }
        check_regions(&data, &dec, &mut rng, 12);
        n += 1;
    }
    assert!(n >= 40, "only {n} fixtures checked");
}

/// A region decode runs the windowed inverse DWT in every rounding. For the reversible 5/3
/// fixtures it equals the crop of the whole decode in all of them; the 9/7 JJ2000 region is
/// Go's windowed arithmetic and may differ from that crop by 1 at a pixel (Go's does too).
#[test]
fn region_windowed_all_roundings() {
    let modes = [
        jp2k::Options::default(),
        jp2k::Options {
            separate_rounding: true,
            ..Default::default()
        },
        jp2k::Options {
            jj2000: true,
            ..Default::default()
        },
    ];
    for name in REGION_FIXTURES {
        let data = read(&format!("{name}.j2k"));
        let siz = fuzz::parse(&data, &limits(), &nc()).unwrap().siz;
        for o in &modes {
            let whole = jp2k::decode(&data, o, &nc()).unwrap();
            for rc in [
                rect(0, 0, 1, 1),
                rect(37, 21, 101, 77),
                rect(63, 63, 66, 66),
                rect(1, 1, 149, 109),
            ] {
                let (w, h) = (width(&whole) as i64, height(&whole) as i64);
                let rc = rect(
                    rc.x0.min(w - 1),
                    rc.y0.min(h - 1),
                    rc.x1.min(w),
                    rc.y1.min(h),
                );
                let got = jp2k::decode_region(&data, rc, 0, o, &nc()).unwrap();
                let want = reduced_bounds(&siz, rc, 0);
                assert_eq!(got.bounds, want, "{name} {rc:?}");
                if name.ends_with("_53") {
                    assert_eq!(got, crop(&whole, want), "{name} {rc:?}");
                }
            }
        }
    }
}

#[test]
fn region_errors() {
    let data = read("region_rgb_97.j2k"); // 150x110, 3 decomposition levels
    let cases = [
        (rect(0, 0, 10, 10), -1),
        (rect(0, 0, 10, 10), 4),
        (rect(150, 0, 160, 10), 0),        // outside
        (rect(5, 5, 5, 10), 0),            // empty
        (rect(-10, -10, 0, 0), 0),         // outside
        (rect(9, 9, 10, 10), 3),           // 1x1 at x=9 rounds away at 1/8
        (rect(0, 0, 1 << 30, 1 << 30), 3), // clipped, fine
    ];
    for (rc, reduce) in cases {
        let r = region(&data, rc, reduce);
        assert_eq!(
            r.is_ok(),
            rc.x1 == 1 << 30,
            "region({rc:?}, {reduce}): {r:?}"
        );
    }
}

/// Corrupts every tile-part body except the one tile a region needs: a region decode must not
/// read the rest.
#[test]
fn region_skips_other_tiles() {
    let data = read("region_rgb_97.j2k"); // 3x2 tiles of 64x64
    let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
    let mut bad = data.clone();
    for tp in &cs.tile_parts {
        if tp.sot.isot != 4 {
            for b in &mut bad[tp.data_offset..tp.data_offset + tp.data_length] {
                *b = 0xff;
            }
        }
    }
    assert!(
        jp2k::decode(&bad, &opts(), &nc()).is_err(),
        "corruption not detected by decode; test is not testing anything"
    );
    let rc = rect(70, 70, 120, 100); // inside tile 4
    let got = region(&bad, rc, 0).unwrap();
    let want = jp2k::decode(&data, &opts(), &nc()).unwrap();
    assert_eq!(got, crop(&want, rc), "pixels differ from the crop");
}

/// A tile the region misses gets a tile-part COD with no decomposition levels. OpenJPEG never
/// reads that tile-part, so reduce 1 must work.
#[test]
fn region_reduce_ignores_other_tiles() {
    let data = read("region_rgb_97.j2k"); // 3x2 tiles of 64x64
    let sot = data.windows(2).position(|w| w == [0xFF, 0x90]).unwrap();
    assert_ne!(be16(&data[sot + 4..]), 4, "fixture layout changed");
    let c = data.windows(2).position(|w| w == [0xFF, 0x52]).unwrap();
    let mut cod = data[c..c + 14].to_vec();
    cod[3] = 12; // Lcod
    cod[4] &= !1; // default precincts
    cod[9] = 0; // decomposition levels
    let mut m = data[..sot + 12].to_vec();
    m.extend_from_slice(&cod);
    m.extend_from_slice(&data[sot + 12..]);
    let psot = be32(&data[sot + 6..]) + cod.len() as u32;
    m[sot + 6..sot + 10].copy_from_slice(&psot.to_be_bytes());
    region(&m, rect(70, 70, 120, 100), 1).unwrap(); // inside tile 4
    assert!(
        region(&m, rect(0, 0, 10, 10), 1).is_err(),
        "reduce 1 accepted on the tile with no decomposition levels"
    );
}

/// An opened `Image` gives the same result as the free functions.
#[test]
fn image_open_matches_free_functions() {
    for name in ["region_rgb_53", "region_sub_53", "g64_multitile"] {
        let data = read(&format!("{name}.j2k"));
        let img = Image::open(&data, &opts(), &nc()).unwrap();
        let cfg = img.config();
        let full = rect(0, 0, cfg.width as i64, cfg.height as i64);
        for r in 0..img.levels() as i32 {
            let a = img.decode_region(full, r, &nc()).unwrap();
            let b = region(&data, full, r).unwrap();
            assert_eq!(a, b, "{name} r{r}");
        }
        assert!(img.decode_region(full, img.levels() as i32, &nc()).is_err());
        let p = img.decode_region_components(full, 0, &[0], &nc()).unwrap();
        let q = jp2k::decode_region_components(&data, full, 0, &[0], &opts(), &nc()).unwrap();
        assert_eq!(p, q, "{name}");
    }
}

/// Port of `TestHandleBadTilePart`: a malformed tile-part SOT fails only the regions that read
/// its tile, for `decode_region` as for an opened `Image`.
#[test]
fn bad_tile_part() {
    let data = read("g64_multitile.j2k");
    let sot = data
        .windows(4)
        .position(|w| w == [0xFF, 0x90, 0, 10])
        .unwrap();
    let next = sot + u32::from_be_bytes(data[sot + 6..sot + 10].try_into().unwrap()) as usize;
    for (what, at, v) in [
        ("TPsot out of order", next + 10, 1u8),
        ("Isot out of range", next + 4, 0x40),
    ] {
        let mut d = data.clone();
        d[at] = v;
        let img = Image::open(&d, &opts(), &nc()).unwrap_or_else(|e| panic!("{what}: open: {e}"));
        let mut ok = 0;
        // 4 tiles of 32x32: tile 1 is the bad one at (32..64, 0..32) when it has a real index.
        for (x0, y0) in [(0, 0), (32, 0), (0, 32), (32, 32)] {
            let r = rect(x0, y0, x0 + 32, y0 + 32);
            let a = region(&d, r, 0);
            let b = img.decode_region(r, 0, &nc());
            // The one-shot call falls back to a strict parse of the whole file when no tile-part
            // of the region survives (as Go does), so only whether it fails, and the image, agree.
            assert_eq!(a.is_ok(), b.is_ok(), "{what}: region ({x0},{y0})");
            if let (Ok(a), Ok(b)) = (&a, &b) {
                assert_eq!(a, b, "{what}: region ({x0},{y0})");
            }
            ok += a.is_ok() as usize;
        }
        assert!(ok > 0, "{what}: no region decodes");
        assert_eq!(
            region(&d, rect(0, 0, 32, 32), 0).unwrap().bounds,
            rect(0, 0, 32, 32)
        );
        // The whole image still fails.
        assert!(jp2k::decode(&d, &opts(), &nc()).is_err(), "{what}");
    }
}

/// A malformed marker in a tile-part header (not its SOT) fails `Image::open` in neither Go nor
/// here: only the regions that read that tile fail, as for `decode_region`.
#[test]
fn bad_tile_part_header_marker() {
    let data = read("g64_multitile.j2k");
    let sot = data
        .windows(4)
        .position(|w| w == [0xFF, 0x90, 0, 10])
        .unwrap();
    let next = sot + u32::from_be_bytes(data[sot + 6..sot + 10].try_into().unwrap()) as usize;
    let psot = u32::from_be_bytes(data[next + 6..next + 10].try_into().unwrap()) + 4;
    let mut d = data.clone();
    d[next + 6..next + 10].copy_from_slice(&psot.to_be_bytes());
    // A COM marker whose length is too short, ahead of the tile-part's SOD.
    d.splice(next + 12..next + 12, [0xFF, 0x64, 0, 1]);
    let img = Image::open(&d, &opts(), &nc()).unwrap();
    let r = rect(0, 0, 32, 32);
    let a = region(&d, r, 0).unwrap();
    assert_eq!(a, img.decode_region(r, 0, &nc()).unwrap());
    assert_eq!(a.bounds, r);
    let r = rect(32, 0, 64, 32);
    assert!(region(&d, r, 0).is_err());
    assert!(img.decode_region(r, 0, &nc()).is_err());
}

/// Image::open rejects a header marker id below 0xFF30 as Go's Open does; decode skips it.
#[test]
fn open_rejects_low_marker() {
    let mut d = read("g64_multitile.j2k");
    let com = d.windows(2).position(|w| w == [0xFF, 0x64]).unwrap();
    d[com + 1] = 0x20;
    assert!(Image::open(&d, &opts(), &nc()).is_err());
    assert!(jp2k::decode(&d, &opts(), &nc()).is_ok());
}

/// Go reads a region's tile-parts sparsely, so a flood of tile-part headers in tiles the region
/// misses cannot trip the header flood guard: a reduce the image cannot give is a plain error,
/// not a limit one, and the flood itself still fails a whole decode.
#[test]
fn region_skips_a_header_flood_in_missed_tiles() {
    const SIZE: usize = 4 << 20;
    // One-pixel tiles with no decomposition levels, a QCC in every tile-part header.
    let qcc = [0xff, 0x5d, 0, 4, 0, 0x40];
    let data = header_flood(&[], &qcc, SIZE / 20);
    let mut o = jp2k::Options::default();
    o.limits.max_alloc = 8 * SIZE;
    let flood = jp2k::decode(&data, &o, &nc()).unwrap_err().to_string();
    assert!(
        flood.contains("need more memory than their size allows"),
        "{flood}"
    );
    let e = jp2k::decode_region(&data, rect(0, 0, 1, 1), 1, &o, &nc())
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("reduce 1 exceeds the 0 decomposition levels"),
        "{e}"
    );
}
