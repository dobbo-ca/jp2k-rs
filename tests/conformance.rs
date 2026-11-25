//! Ports of `internal/decoder/conformance_test.go` and `style_fixtures_test.go`: ISO/IEC 15444-4
//! conformance files (code-block styles, PPM/PPT) and the opj_compress -M style fixtures, with
//! reference output from opj_decompress. Files over 100 KB and the rest of the openjpeg-data
//! corpus are not copied here, so their smoke tests are not ported.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::fuzz;
use jp2k::{ColorModel, Raster};

/// Decodes through the public entry point and compares with an opj_decompress PNM reference.
fn conformance_full(j2k: &str, reference: &str, tol: i32) {
    let img = jp2k::decode(&read(j2k), &opts(), &nc()).unwrap_or_else(|e| panic!("{j2k}: {e}"));
    let want = read_pnm(path(reference));
    let (got, stride) = match img.color_model {
        ColorModel::Gray => (&img.pix, 1),
        ColorModel::Rgba => (&img.pix, 4),
        m => panic!("{j2k}: unexpected {m:?}"),
    };
    assert_eq!((width(&img), height(&img)), (want.w, want.h), "{j2k}");
    let d = max_abs_diff(got, &want, stride);
    assert!(d <= tol, "{j2k}: max abs diff {d}, want <= {tol}");
}

/// Component 0 at its own (possibly subsampled) resolution against an unsigned 8-bit PGX
/// reference: for the subsampled single-component files (p0_02, p1_01).
fn conformance_component(j2k: &str, reference: &str, tol: i32) {
    let planes = jp2k::decode_components(&read(j2k), &[0], &opts(), &nc())
        .unwrap_or_else(|e| panic!("{j2k}: {e}"));
    let p = &planes[0];
    let pgx = read_pgx(path(reference));
    let (w, h) = (
        (p.bounds.x1 - p.bounds.x0) as usize,
        (p.bounds.y1 - p.bounds.y0) as usize,
    );
    assert_eq!((w, h), (pgx.w, pgx.h), "{j2k}: component size");
    let mut max_diff = 0;
    for (i, &v) in p.pix.iter().enumerate() {
        max_diff = max_diff.max((v as i32 - pgx.samples[i]).abs());
    }
    assert!(
        max_diff <= tol,
        "{j2k}: max abs diff {max_diff}, want <= {tol}"
    );
}

#[test]
fn p0_02() {
    // cblksty 0x34: TERMALL, PTERM, SEGSYM. Subsampled (XRsiz=2).
    conformance_component("p0_02.j2k", "conf_p0_02_c0_ref.pgx", 2);
}

#[test]
fn p0_11() {
    // cblksty 0x20: SEGSYM.
    conformance_full("p0_11.j2k", "conf_p0_11_ref.pgm", 0);
}

#[test]
fn p0_12() {
    // cblksty 0x04: TERMALL.
    conformance_full("p0_12.j2k", "conf_p0_12_ref.pgm", 0);
}

#[test]
fn p1_01() {
    // cblksty 0x34: TERMALL, PTERM, SEGSYM. Subsampled (XRsiz=2).
    conformance_component("p1_01.j2k", "conf_p1_01_c0_ref.pgx", 2);
}

#[test]
fn c1_mono() {
    // cblksty 0x01: LAZY (selective bypass).
    conformance_full("c1_mono.j2c", "conf_c1_mono_ref.ppm", 0);
}

#[test]
fn c2_mono() {
    // cblksty 0x2f: LAZY, RESET, TERMALL, VSC, SEGSYM combined.
    conformance_full("c2_mono.j2c", "conf_c2_mono_ref.ppm", 0);
}

#[test]
fn p1_06() {
    // cblksty 0x28: VSC, SEGSYM. 16 tiles x 1 PPT each.
    conformance_full("p1_06.j2k", "conf_p1_06_ref.ppm", 2);
}

#[test]
fn g1_colr() {
    // style 0; 1 PPM marker, 2 Nppm fields, 2 tile-parts.
    conformance_full("g1_colr.j2c", "conf_g1_colr_ref.ppm", 0);
}

#[test]
fn g2_colr() {
    // style 0; 1 PPM marker, 2 Nppm fields, EPH in use (from the PPM stream).
    conformance_full("g2_colr.j2c", "conf_g2_colr_ref.ppm", 0);
}

#[test]
fn g3_colr() {
    // style 0; 214 PPM markers, Zppm descending, one Nppm run spanning ~77 markers: the
    // out-of-Z-order + cross-marker-framing case.
    conformance_full("g3_colr.j2c", "conf_g3_colr_ref.ppm", 0);
}

#[test]
fn g4_colr() {
    // style 0; per-tile-restarting PPT Z values.
    conformance_full("g4_colr.j2c", "conf_g4_colr_ref.ppm", 0);
}

/// A hand-flipped bit in the tail of a SEGSYM (0x20) code-block's data, where the cleanup
/// pass's 4-bit 0b1010 marker lives, is not a decode error: the mismatch is decoded and
/// discarded, matching OpenJPEG's warn-and-continue.
#[test]
fn segsym_corruption_warns_not_errors() {
    let mut data = read("style_M32.j2k");
    let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
    assert_eq!(
        cs.cod.coding_style.code_block_style, 0x20,
        "fixture cblksty"
    );
    let mut b = jp2k::Budget::new(&limits());
    let (pkts, _, _) =
        fuzz::decode_tile(&cs, &data, 0, &mut fuzz::Cursor::new(), None, &nc(), &mut b).unwrap();
    // The last code-block segment's last byte: MQ termination lands the cleanup pass's trailing
    // segmentation-symbol bits right at the tail of the code-block's data.
    let last = pkts
        .iter()
        .flat_map(|p| p.blocks.iter())
        .flat_map(|b| b.segments.iter())
        .map(|s| s.end)
        .max()
        .unwrap();
    let tile_off = cs
        .tile_parts
        .iter()
        .find(|tp| tp.sot.isot == 0)
        .unwrap()
        .data_offset;
    data[tile_off + last - 1] ^= 0xFF;
    jp2k::decode(&data, &opts(), &nc()).expect("warn-and-continue");
}

/// opj_compress -M fixtures, one per code-block style bit (and two combinations), reversible,
/// lossless: -M's value is exactly the SPcod/SPcoc style byte.
#[test]
fn style_fixtures_reversible_exact() {
    for m in [1u8, 2, 4, 8, 16, 32, 13, 63] {
        let name = format!("style_M{m}.j2k");
        let data = read(&name);
        let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
        assert_eq!(cs.cod.coding_style.code_block_style, m, "{name}: cblksty");
        let img: Raster = jp2k::decode(&data, &opts(), &nc()).unwrap();
        assert_eq!(img.color_model, ColorModel::Gray);
        let want = read_pnm(path(&format!("style_M{m}_ref.pgm")));
        assert_eq!((width(&img), height(&img)), (want.w, want.h), "{name}");
        assert_eq!(max_abs_diff(&img.pix, &want, 1), 0, "{name}: lossless");
    }
}

#[test]
fn style_fixture_rgb() {
    let img = jp2k::decode(&read("style_rgb_M63.j2k"), &opts(), &nc()).unwrap();
    assert_eq!(img.color_model, ColorModel::Rgba);
    let want = read_pnm(path("style_rgb_M63_ref.ppm"));
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    assert_eq!(max_abs_diff(&img.pix, &want, 4), 0);
}

/// Style bits on a 16-bit (12-bit precision), 3-component, -mct 0 image: no MCT, so decoded
/// straight through at >8-bit precision alongside the style bits.
#[test]
fn style_fixture_16bit_3comp() {
    let data = read("style_16bit_M63.j2k");
    let cs = fuzz::parse(&data, &limits(), &nc()).unwrap();
    assert_eq!(cs.cod.multiple_component_transform, 0);
    let want = read_pnm(path("style_16bit_M63_ref.ppm"));
    let planes = jp2k::decode_components(&data, &[0, 1, 2], &opts(), &nc()).unwrap();
    for (c, p) in planes.iter().enumerate() {
        let (w, h) = (
            (p.bounds.x1 - p.bounds.x0) as usize,
            (p.bounds.y1 - p.bounds.y0) as usize,
        );
        assert_eq!((w, h), (want.w, want.h), "component {c}");
        for (i, &v) in p.pix.iter().enumerate() {
            assert_eq!(
                v as i32,
                want.samples[i * want.comps + c],
                "component {c} pixel {i}"
            );
        }
    }
}

/// g1_colr.j2c has 1x2 tiles and one main-header PPM marker with two Nppm runs. Swapping both
/// the tile-part order and the Nppm run order is a valid codestream equivalent to the original:
/// T.800 A.7.4 orders Nppm runs by codestream appearance of the tile-parts they serve, not by
/// tile index, so the decode loop must consume the shared PPM cursor in that order too.
fn build_g1_swap() -> Vec<u8> {
    let data = read("g1_colr.j2c");
    const PPM_OFF: usize = 51;
    const TILE0_SOT: usize = 2201;
    const TILE1_SOT: usize = 43324;
    let ppm_l = be16(&data[PPM_OFF + 2..]) as usize;
    let ppm_end = PPM_OFF + 2 + ppm_l;
    let payload = &data[PPM_OFF + 4..ppm_end];
    let z = payload[0];
    let rest = &payload[1..];
    let n1 = be32(rest) as usize;
    let run1 = &rest[4..4 + n1];
    let rest2 = &rest[4 + n1..];
    let n2 = be32(rest2) as usize;
    let run2 = &rest2[4..4 + n2];
    assert_eq!(
        4 + n1 + 4 + n2,
        rest.len(),
        "g1_colr.j2c PPM framing changed shape"
    );
    let mut swapped = vec![z];
    swapped.extend_from_slice(&(n2 as u32).to_be_bytes());
    swapped.extend_from_slice(run2);
    swapped.extend_from_slice(&(n1 as u32).to_be_bytes());
    swapped.extend_from_slice(run1);
    assert_eq!(swapped.len(), payload.len());
    let eoc = data.len() - 2;
    let mut out = data[..PPM_OFF].to_vec();
    out.extend_from_slice(&data[PPM_OFF..PPM_OFF + 4]);
    out.extend_from_slice(&swapped);
    out.extend_from_slice(&data[ppm_end..TILE0_SOT]);
    out.extend_from_slice(&data[TILE1_SOT..eoc]);
    out.extend_from_slice(&data[TILE0_SOT..TILE1_SOT]);
    out.extend_from_slice(&data[eoc..]);
    out
}

#[test]
fn ppm_cursor_follows_tile_part_appearance_order() {
    let want = jp2k::decode(&read("g1_colr.j2c"), &opts(), &nc()).unwrap();
    let got = jp2k::decode(&build_g1_swap(), &opts(), &nc())
        .expect("g1_colr.j2c with tile-parts and Nppm runs both swapped");
    assert_eq!(got, want);
}
