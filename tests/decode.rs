//! Ports of `internal/decoder/decode_test.go`.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::fuzz;
use jp2k::{ColorModel, Error};
use std::io::Read;

fn samples8(p: &Pnm) -> Vec<u8> {
    p.samples.iter().map(|&v| v as u8).collect()
}

#[test]
fn gray8_exact() {
    let img = decode_fixture("g64_53.j2k");
    assert_eq!(img.color_model, ColorModel::Gray);
    let want = read_pnm(path("g64.pgm"));
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    assert_eq!(
        img.pix,
        samples8(&want),
        "pixels not byte-exact vs source PGM"
    );
}

#[test]
fn rgb8_rct_exact() {
    let img = decode_fixture("rgb8_rct_m5.j2k");
    assert_eq!(img.color_model, ColorModel::Rgba);
    let want = read_pnm(path("rgb8_32x24.ppm"));
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    assert_eq!(max_abs_diff(&img.pix, &want, 4), 0, "lossless RCT");
    for i in 0..want.w * want.h {
        assert_eq!(img.pix[4 * i + 3], 0xff, "pixel {i} alpha");
    }
}

#[test]
fn rgb8_ict_tolerance() {
    let img = decode_fixture("rgb8_ict_m5.j2k");
    assert_eq!(img.color_model, ColorModel::Rgba);
    let want = read_pnm(path("rgb8_ict_m5_ref.ppm"));
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    assert!(max_abs_diff(&img.pix, &want, 4) <= 2);
}

#[test]
fn gray16_exact() {
    let img = decode_fixture("g48x40_16_53.j2k");
    assert_eq!(img.color_model, ColorModel::Gray16);
    let want = read_pnm(path("g48x40_16.pgm"));
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    assert_eq!(max_abs_diff16(&img.pix, &want, 2, 1), 0);
}

/// The target profile: 16-bit 3-comp, -I -mct 0, tolerance <= 2.
#[test]
fn target_profile() {
    let img = decode_fixture("g48x40_3c16_ict_mct0_m5.j2k");
    assert_eq!(img.color_model, ColorModel::Rgba64);
    let want = read_pnm(path("g48x40_3c16_ict_mct0_m5_ref.ppm"));
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    assert!(max_abs_diff16(&img.pix, &want, 8, 3) <= 2);
    for i in 0..want.w * want.h {
        let a = ((img.pix[8 * i + 6] as u32) << 8) | img.pix[8 * i + 7] as u32;
        assert_eq!(a, 0xffff, "pixel {i} alpha");
    }
}

#[test]
fn config_fixtures() {
    let cases = [
        ("g64_53.j2k", 64, 64, ColorModel::Gray),
        ("rgb8_rct_m5.j2k", 32, 24, ColorModel::Rgba),
        ("rgb8_ict_m5.j2k", 32, 24, ColorModel::Rgba),
        ("g48x40_16_53.j2k", 48, 40, ColorModel::Gray16),
        ("g48x40_3c16_ict_mct0_m5.j2k", 48, 40, ColorModel::Rgba64),
    ];
    for (name, w, h, cm) in cases {
        let cfg = jp2k::decode_config(&read(name), &limits()).unwrap();
        assert_eq!(
            (cfg.width, cfg.height, cfg.color_model),
            (w, h, cm),
            "{name}"
        );
    }
}

/// A reader that caps every `read` at `n` bytes, modeling a network-style reader.
struct ChunkReader {
    data: Vec<u8>,
    n: usize,
}

impl Read for ChunkReader {
    fn read(&mut self, p: &mut [u8]) -> std::io::Result<usize> {
        let m = self.n.min(p.len()).min(self.data.len());
        p[..m].copy_from_slice(&self.data[..m]);
        self.data.drain(..m);
        Ok(m)
    }
}

#[test]
fn config_short_reads() {
    for name in [
        "g64_53.j2k",
        "rgb8_rct_m5.j2k",
        "g48x40_16_53.j2k",
        "g48x40_3c16_ict_mct0_m5.j2k",
    ] {
        let data = read(name);
        let want = jp2k::decode_config(&data, &limits()).unwrap();
        let got = jp2k::decode_config_reader(
            ChunkReader {
                data: data.clone(),
                n: 7,
            },
            &limits(),
            &nc(),
        )
        .unwrap();
        assert_eq!(got, want, "{name}");
    }
}

struct CountingReader {
    data: Vec<u8>,
    pos: usize,
    n: usize,
}

impl Read for CountingReader {
    fn read(&mut self, p: &mut [u8]) -> std::io::Result<usize> {
        let m = p.len().min(self.data.len() - self.pos);
        p[..m].copy_from_slice(&self.data[self.pos..self.pos + m]);
        self.pos += m;
        self.n += m;
        Ok(m)
    }
}

/// `decode_config` never reads past the largest legal SIZ segment, even when the reader has far
/// more data behind it.
#[test]
fn config_bounded_read() {
    let mut data = read("g64_53.j2k");
    data.extend(vec![0u8; 1 << 20]);
    let mut r = CountingReader { data, pos: 0, n: 0 };
    let cfg = jp2k::decode_config_reader(&mut r, &limits(), &nc()).unwrap();
    assert_eq!((cfg.width, cfg.height), (64, 64));
    assert!(r.n <= 65541, "read {} bytes", r.n);
}

#[test]
fn config_truncated_after_siz() {
    let data = read("g64_53.j2k");
    let (siz, end) = fuzz::parse_siz_segment(&data, 2, &limits()).unwrap();
    let cfg = jp2k::decode_config(&data[..end], &limits()).unwrap();
    assert_eq!(cfg.width, siz.xsiz - siz.xosiz);
    assert_eq!(cfg.height, siz.ysiz - siz.yosiz);
}

#[test]
fn decode_jp2() {
    let data = read("gray_box_m5.jp2");
    let img = jp2k::decode(&data, &opts(), &nc()).unwrap();
    assert_eq!(img.color_model, ColorModel::Gray);
    let cfg = jp2k::decode_config(&data, &limits()).unwrap();
    assert_eq!(
        (cfg.width as usize, cfg.height as usize),
        (width(&img), height(&img))
    );
    assert_eq!(cfg.color_model, ColorModel::Gray);
    let want = read_pnm(path("g64.pgm"));
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    assert_eq!(img.pix, samples8(&want));
}

#[test]
fn config_too_many_components() {
    assert!(jp2k::decode_config(&soc_siz(5, 8, false, 1, 1), &limits()).is_err());
}

/// `decode_config` accepts component subsampling: the reported size is still the reference-grid
/// Xsiz/Ysiz.
#[test]
fn config_subsampling() {
    let cfg = jp2k::decode_config(&soc_siz(3, 8, false, 2, 1), &limits()).unwrap();
    assert_eq!((cfg.width, cfg.height), (64, 64));
}

#[test]
fn config_precision_too_high() {
    assert!(jp2k::decode_config(&soc_siz(1, 17, false, 1, 1), &limits()).is_err());
}

#[test]
fn config_signed_accepted() {
    jp2k::decode_config(&soc_siz(1, 8, true, 1, 1), &limits()).unwrap();
}

/// A SIZ declaring a huge canvas is rejected before any per-component sample buffer is
/// allocated.
#[test]
fn huge_canvas_rejected() {
    // 4 components of 16 bits, 8 x 1409286224, one tile, no tile data.
    let mut p = Vec::new();
    p.extend_from_slice(&0u16.to_be_bytes());
    for v in [8u32, 1409286224, 0, 0, 8, 1409286224, 0, 0] {
        p.extend_from_slice(&v.to_be_bytes());
    }
    p.extend_from_slice(&4u16.to_be_bytes());
    for _ in 0..4 {
        p.extend_from_slice(&[15, 1, 1]);
    }
    let mut b = vec![0xFF, 0x4F, 0xFF, 0x51];
    b.extend_from_slice(&((p.len() + 2) as u16).to_be_bytes());
    b.extend_from_slice(&p);
    // COD: 1 layer, 0 levels, 64x64 blocks (4,4), reversible; QCD: no quantization.
    b.extend_from_slice(&[0xFF, 0x52, 0, 12, 0, 0, 0, 1, 0, 0, 4, 4, 0, 1]);
    b.extend_from_slice(&[0xFF, 0x5C, 0, 4, 0x40, 8 << 3]);
    b.extend_from_slice(&[
        0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 14, 0, 1, 0xFF, 0x93, 0xFF, 0xD9,
    ]);
    let t = std::time::Instant::now();
    let err = jp2k::decode(&b, &opts(), &nc()).unwrap_err();
    assert!(matches!(err, Error::LimitExceeded(_)), "{err}");
    assert!(t.elapsed().as_secs() < 2, "unbounded allocation from SIZ");
}

#[test]
fn multi_tile() {
    let img = decode_fixture("g64_multitile.j2k");
    assert_eq!(img.color_model, ColorModel::Gray);
    assert_eq!(
        (width(&img), height(&img)),
        (64, 64),
        "the full image, not one tile"
    );
    let want = read_pnm(path("g64.pgm"));
    assert_eq!(img.pix, samples8(&want), "lossless 5/3");
    let cfg = jp2k::decode_config(&read("g64_multitile.j2k"), &limits()).unwrap();
    assert_eq!((cfg.width as usize, cfg.height as usize), (64, 64));
}

/// Inserts a COC marker segment for `comp`, with the given code-block style and transformation,
/// immediately after the data's COD segment.
fn inject_coc(data: &[u8], comp: u8, cb_style: u8, transform: u8) -> Vec<u8> {
    let pos = data.windows(2).position(|w| w == [0xff, 0x52]).unwrap();
    let l = be16(&data[pos + 2..]) as usize;
    let seg_end = pos + 2 + l;
    let cod = fuzz::parse_cod(&data[pos + 4..seg_end], pos, &limits()).unwrap();
    let payload = [
        comp,
        0, // Scoc: no custom precincts
        cod.coding_style.num_decomp_levels,
        cod.coding_style.code_block_width_exp - 2,
        cod.coding_style.code_block_height_exp - 2,
        cb_style,
        transform,
    ];
    let mut out = data[..seg_end].to_vec();
    out.extend_from_slice(&[0xff, 0x53, 0, (payload.len() + 2) as u8]);
    out.extend_from_slice(&payload);
    out.extend_from_slice(&data[seg_end..]);
    out
}

/// A code-block style carried in a COC (not just the COD) is enforced.
#[test]
fn coc_code_block_style_rejected() {
    let base = read("rgb8_rct_m5.j2k");
    jp2k::decode(&base, &opts(), &nc()).unwrap();
    let injected = inject_coc(&base, 1, 0x40, 1);
    assert!(jp2k::decode(&injected, &opts(), &nc()).is_err());
}

/// A per-component transform override that disagrees with component 0 under an active MCT is
/// caught rather than silently misdecoded.
#[test]
fn coc_transform_mismatch_rejected() {
    let base = read("rgb8_rct_m5.j2k"); // mct=1, COD transform=1 (5/3)
    let injected = inject_coc(&base, 1, 0, 0); // component 1: transform=0 (9/7)
    assert!(jp2k::decode(&injected, &opts(), &nc()).is_err());
}

/// A 4-component >8-bit image decodes to NRGBA64, not premultiplied.
#[test]
fn four_comp_16bit_not_premultiplied() {
    let img = decode_fixture("a4_16bit_mct.j2k");
    assert_eq!(img.color_model, ColorModel::Nrgba64);
    let be = |b: &[u8]| ((b[0] as u16) << 8) | b[1] as u16;
    let px = &img.pix[0..8];
    let (r, g, b, a) = (be(&px[0..2]), be(&px[2..4]), be(&px[4..6]), be(&px[6..8]));
    assert_eq!((r, g, b, a), (0xffff, 0, 0, 0x8000));
}

/// The public decode biases signed samples by +2^(precision-1) so they land in the unsigned
/// range, matching the PGX reference plus that same bias.
#[test]
fn signed_gray8() {
    let img = decode_fixture("signed32x24.j2k");
    assert_eq!(img.color_model, ColorModel::Gray);
    let want = read_pgx(path("signed32x24_ref.pgx"));
    assert!(want.signed && want.depth == 8);
    assert_eq!((width(&img), height(&img)), (want.w, want.h));
    for (i, &v) in want.samples.iter().enumerate() {
        assert_eq!(img.pix[i] as i32, v + 128, "pixel {i}");
    }
}

/// A region-of-interest marker is a clean error rather than silently decoding with the ROI shift
/// ignored. p0_03.j2k carries one main-header RGN.
#[test]
fn rgn_rejected() {
    let err = jp2k::decode(&read("p0_03.j2k"), &opts(), &nc()).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
}

/// Two components are gray then alpha: R=G=B=gray and A=alpha, as opj_decompress writes them to
/// a gray+alpha PNG.
#[test]
fn gray_alpha() {
    for n in ["ga8_53", "ga16_97"] {
        let img = decode_fixture(&format!("{n}.j2k"));
        let r = read_pam(path(&format!("{n}_ref.pam")));
        let want_model = if r.maxval == 255 {
            ColorModel::Nrgba
        } else {
            ColorModel::Nrgba64
        };
        assert_eq!(img.color_model, want_model, "{n}");
        assert_eq!((width(&img), height(&img)), (r.w, r.h), "{n}");
        assert_eq!(img.pix, r.pix, "{n}: differs from the reference");
    }
}

/// Reads a whole file through `decode_reader`.
#[test]
fn decode_reader_matches_decode() {
    let data = read("g64_53.j2k");
    let a = jp2k::decode(&data, &opts(), &nc()).unwrap();
    let b = jp2k::decode_reader(&data[..], &opts(), &nc()).unwrap();
    assert_eq!(a, b);
    let mut o = opts();
    o.limits.max_input = 100;
    assert!(matches!(
        jp2k::decode_reader(&data[..], &o, &nc()),
        Err(Error::LimitExceeded(_))
    ));
}

#[test]
fn cancel_flag_stops_decode() {
    let data = read("g64_53.j2k");
    let c = std::sync::atomic::AtomicBool::new(true);
    assert_eq!(
        jp2k::decode(&data, &opts(), &c).unwrap_err(),
        Error::Cancelled
    );
}

/// A reader that only ever returns `Interrupted` is stopped by the cancel flag, in `read_full`.
#[test]
fn cancel_stops_interrupted_config_reader() {
    struct Stuck<'a>(&'a std::sync::atomic::AtomicBool);
    impl Read for Stuck<'_> {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            Err(std::io::ErrorKind::Interrupted.into())
        }
    }
    let c = nc();
    assert_eq!(
        jp2k::decode_config_reader(Stuck(&c), &limits(), &c).unwrap_err(),
        Error::Cancelled
    );
}
