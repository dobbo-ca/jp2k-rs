//! Ports of `internal/decoder/components_test.go`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::{ColorModel, Error, Image, Plane, Rect};
use std::collections::HashMap;

/// 3 components, 16 bit, 9/7, no MCT, 2x2 tiles: see its `_ref.sha256` file for how it and its
/// references were made.
const COMPONENT_FIXTURE: &str = "gray16x3_97_nomct.j2k";

const COMPONENT_REQUESTS: [&[usize]; 6] = [&[0], &[1], &[2], &[1, 0], &[2, 0, 1], &[0, 1, 2]];

fn rect(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
    Rect { x0, y0, x1, y1 }
}

fn fixture_sums() -> HashMap<String, String> {
    let text = String::from_utf8(read("gray16x3_97_nomct_ref.sha256")).unwrap();
    let mut sums = HashMap::new();
    for l in text.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() == 2 && !l.starts_with('#') {
            sums.insert(f[1].to_string(), f[0].to_string());
        }
    }
    sums
}

/// Every way of asking for the fixture's planes gives opj_decompress's PGX bytes, whole, at
/// reduce 1, and for a region at reduce 0 and 1.
#[test]
fn components_match_openjpeg() {
    let data = read(COMPONENT_FIXTURE);
    let sums = fixture_sums();
    let img = Image::open(&data, &opts(), &nc()).unwrap();
    let all = rect(0, 0, 1 << 20, 1 << 20);
    let reg = rect(37, 21, 131, 97);
    for (name, rc, reduce) in [
        ("full", all, 0),
        ("r1", all, 1),
        ("d", reg, 0),
        ("dr1", reg, 1),
    ] {
        for comps in COMPONENT_REQUESTS {
            let mut ways: Vec<(&str, Vec<Plane>)> = vec![
                (
                    "region",
                    jp2k::decode_region_components(&data, rc, reduce, comps, &opts(), &nc())
                        .unwrap(),
                ),
                (
                    "handle",
                    img.decode_region_components(rc, reduce, comps, &nc())
                        .unwrap(),
                ),
            ];
            if name == "full" {
                ways.push((
                    "whole",
                    jp2k::decode_components(&data, comps, &opts(), &nc()).unwrap(),
                ));
            }
            for (way, planes) in ways {
                assert_eq!(planes.len(), comps.len(), "{name} {way} {comps:?}");
                for (i, &c) in comps.iter().enumerate() {
                    let got = sha256_hex(&pgx_bytes(&planes[i], 16, false));
                    let file = format!("{name}_{c}.pgx");
                    assert_eq!(
                        got, sums[&file],
                        "{way} {comps:?}: plane {i} differs from {file}"
                    );
                }
            }
        }
    }
}

/// Component `c` of a decode's output at (x, y), for images of `n` components that output does
/// not scale.
fn decoded_sample(img: &jp2k::Raster, x: i64, y: i64, c: usize, n: usize) -> u16 {
    let ch = if n == 2 && c == 1 { 3 } else { c };
    let bpp = img.color_model.bytes_per_pixel();
    let off = (y - img.bounds.y0) as usize * img.stride + (x - img.bounds.x0) as usize * bpp;
    match img.color_model {
        ColorModel::Gray | ColorModel::Rgba | ColorModel::Nrgba => img.pix[off + ch] as u16,
        ColorModel::Gray16 => u16::from_be_bytes([img.pix[off], img.pix[off + 1]]),
        ColorModel::Rgba64 | ColorModel::Nrgba64 => {
            u16::from_be_bytes([img.pix[off + 2 * ch], img.pix[off + 2 * ch + 1]])
        }
    }
}

fn check_plane(what: &str, p: &Plane, img: &jp2k::Raster, c: usize, n: usize) {
    assert_eq!(p.bounds, img.bounds, "{what} component {c}: bounds");
    let w = (p.bounds.x1 - p.bounds.x0) as usize;
    for y in p.bounds.y0..p.bounds.y1 {
        for x in p.bounds.x0..p.bounds.x1 {
            let got = p.pix[(y - p.bounds.y0) as usize * w + (x - p.bounds.x0) as usize];
            assert_eq!(
                got,
                decoded_sample(img, x, y, c, n),
                "{what} component {c} at ({x},{y})"
            );
        }
    }
}

/// Planes equal the channels of `decode` and `decode_region` wherever those hold samples
/// unscaled: 8 or 16 bits, no subsampling, no cdef.
#[test]
fn components_match_decode() {
    let mut checked = 0;
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
    for f in files {
        let data = std::fs::read(&f).unwrap();
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let Ok(img) = jp2k::decode(&data, &opts(), &nc()) else {
            continue;
        };
        if width(&img) * height(&img) > 1 << 20 {
            continue; // the slide fixtures are for the benchmarks
        }
        let (stream, cdef) = if data.starts_with(&jp2k::fuzz::JP2_SIGNATURE) {
            jp2k::fuzz::jp2_codestream(&data, &limits()).unwrap()
        } else {
            (&data[..], None)
        };
        let cs = jp2k::fuzz::parse(stream, &limits(), &nc()).unwrap();
        let siz = &cs.siz;
        let prec = siz.components[0].precision;
        if (prec != 8 && prec != 16) || cdef.is_some() {
            continue;
        }
        let n = siz.components.len();
        if siz.components.iter().any(|c| c.xrsiz != 1 || c.yrsiz != 1) {
            continue;
        }
        let (w, h) = (width(&img) as i64, height(&img) as i64);
        let rc = rect(w / 4, h / 3, w * 3 / 4 + 1, h - 1);
        let mut reqs: Vec<Vec<usize>> = vec![(0..n).map(|c| n - 1 - c).collect()];
        reqs.extend((0..n).map(|c| vec![c]));
        for reduce in [0, 1] {
            let Ok(refimg) = jp2k::decode_region(&data, rc, reduce, &opts(), &nc()) else {
                continue;
            };
            for comps in &reqs {
                let planes =
                    jp2k::decode_region_components(&data, rc, reduce, comps, &opts(), &nc())
                        .unwrap();
                if reduce == 0 {
                    let whole = jp2k::decode_components(&data, comps, &opts(), &nc()).unwrap();
                    for (i, &c) in comps.iter().enumerate() {
                        check_plane(&format!("{name} whole"), &whole[i], &img, c, n);
                    }
                }
                for (i, &c) in comps.iter().enumerate() {
                    check_plane(
                        &format!("{name} {rc:?} reduce {reduce}"),
                        &planes[i],
                        &refimg,
                        c,
                        n,
                    );
                }
            }
        }
        checked += 1;
    }
    assert!(checked >= 20, "checked {checked} files, want at least 20");
}

#[test]
fn components_hostile() {
    let data = read(COMPONENT_FIXTURE);
    let img = Image::open(&data, &opts(), &nc()).unwrap();
    let rc = rect(0, 0, 64, 64);
    let bad: [&[usize]; 7] = [
        &[],
        &[3],
        &[0, 3],
        &[1, 1],
        &[2, 0, 2],
        &[usize::MAX],
        &[1 << 40],
    ];
    for comps in bad {
        let a = jp2k::decode_components(&data, comps, &opts(), &nc());
        let b = jp2k::decode_region_components(&data, rc, 0, comps, &opts(), &nc());
        let c = img.decode_region_components(rc, 0, comps, &nc());
        for (way, r) in [("whole", a), ("region", b), ("handle", c)] {
            assert!(
                matches!(&r, Err(e) if !matches!(e, Error::Internal(_))),
                "{way} {comps:?}: {r:?}"
            );
        }
    }

    // Component 1 at 17 bits: decode refuses the file, and a request for component 1, but
    // component 0 decodes as before.
    let mut hi = data.clone();
    hi[42 + 3] = 16;
    assert!(
        jp2k::decode(&hi, &opts(), &nc()).is_err(),
        "decode of a 17-bit component"
    );
    let e = jp2k::decode_components(&hi, &[0, 1], &opts(), &nc()).unwrap_err();
    assert!(e.to_string().contains("precision 17"), "{e}");
    let want = jp2k::decode_components(&data, &[0], &opts(), &nc()).unwrap();
    let got = jp2k::decode_components(&hi, &[0], &opts(), &nc()).unwrap();
    assert_eq!(got, want, "component 0 beside a 17-bit one");

    let cancelled = std::sync::atomic::AtomicBool::new(true);
    assert_eq!(
        jp2k::decode_components(&data, &[0], &opts(), &cancelled).unwrap_err(),
        Error::Cancelled
    );
}

const BAND_TILES_SIZE: i64 = 128;
const BAND_TILES_TILE: i64 = 64;

fn band_tiles_value(qx: i64, qy: i64) -> u16 {
    (1000 + 20000 * qx + 40000 * qy) as u16
}

/// gray16_bandtiles.j2k, four flat 64x64 tiles: each tile lands at its own origin.
#[test]
fn components_band_non_zero_origin() {
    let data = read("gray16_bandtiles.j2k");
    let planes = jp2k::decode_components(&data, &[0], &opts(), &nc()).unwrap();
    let p = &planes[0];
    assert_eq!(
        (p.bounds.x1 - p.bounds.x0, p.bounds.y1 - p.bounds.y0),
        (BAND_TILES_SIZE, BAND_TILES_SIZE)
    );
    for y in 0..BAND_TILES_SIZE {
        for x in 0..BAND_TILES_SIZE {
            let want = band_tiles_value(x / BAND_TILES_TILE, y / BAND_TILES_TILE);
            assert_eq!(p.pix[(y * BAND_TILES_SIZE + x) as usize], want, "({x},{y})");
        }
    }
}

/// A region of the band tiles: tile 0 whole, the rest cropped.
#[test]
fn region_components_band_then_window() {
    let data = read("gray16_bandtiles.j2k");
    let img = Image::open(&data, &opts(), &nc()).unwrap();
    let planes = img
        .decode_region_components(rect(0, 0, 100, 100), 0, &[0], &nc())
        .unwrap();
    let p = &planes[0];
    for y in 0..100 {
        for x in 0..100 {
            let want = band_tiles_value(x / BAND_TILES_TILE, y / BAND_TILES_TILE);
            assert_eq!(p.pix[(y * 100 + x) as usize], want, "({x},{y})");
        }
    }
}

/// Every component of an MCT image equals its separate decode.
#[test]
fn components_band_mct() {
    let data = read("rgba16_bandtiles.j2k");
    let all = jp2k::decode_components(&data, &[0, 1, 2, 3], &opts(), &nc()).unwrap();
    let rgb = jp2k::decode_components(&data, &[0, 1, 2], &opts(), &nc()).unwrap();
    let a = jp2k::decode_components(&data, &[3], &opts(), &nc()).unwrap();
    let want = [&rgb[0], &rgb[1], &rgb[2], &a[0]];
    for c in 0..4 {
        assert_eq!(
            &all[c], want[c],
            "component {c} differs from its separate decode"
        );
    }
}
