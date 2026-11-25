//! What the integration tests share: paths to the committed fixtures, readers for the reference
//! files, and builders of small synthetic codestreams and JP2 files (Go: `internal/fixture`).
#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_memcpy
)]

use jp2k::{Limits, Options, Raster};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

pub fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

pub fn read(name: &str) -> Vec<u8> {
    std::fs::read(path(name)).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

pub fn opts() -> Options {
    Options::default()
}

pub fn nc() -> AtomicBool {
    AtomicBool::new(false)
}

pub fn limits() -> Limits {
    Limits::default()
}

pub fn decode_fixture(name: &str) -> Raster {
    jp2k::decode(&read(name), &opts(), &nc()).unwrap_or_else(|e| panic!("decode({name}): {e}"))
}

/// A PNM (P5/P6) or PGX file: row-major, component-interleaved samples.
pub struct Pnm {
    pub w: usize,
    pub h: usize,
    pub maxval: usize,
    pub comps: usize,
    pub samples: Vec<i32>,
}

pub fn read_pnm(p: PathBuf) -> Pnm {
    let d = std::fs::read(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
    let mut pos = 0;
    let token = |pos: &mut usize| -> String {
        let mut tok = String::new();
        loop {
            let b = d[*pos];
            *pos += 1;
            if b == b'#' {
                while d[*pos] != b'\n' {
                    *pos += 1;
                }
                continue;
            }
            if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
                if !tok.is_empty() {
                    return tok;
                }
                continue;
            }
            tok.push(b as char);
        }
    };
    let comps = match token(&mut pos).as_str() {
        "P5" => 1,
        "P6" => 3,
        m => panic!("{}: unsupported PNM magic {m}", p.display()),
    };
    let w: usize = token(&mut pos).parse().unwrap();
    let h: usize = token(&mut pos).parse().unwrap();
    let maxval: usize = token(&mut pos).parse().unwrap();
    let n = w * h * comps;
    let body = &d[pos..];
    let samples = if maxval < 256 {
        body[..n].iter().map(|&b| b as i32).collect()
    } else {
        (0..n)
            .map(|i| ((body[2 * i] as i32) << 8) | body[2 * i + 1] as i32)
            .collect()
    };
    Pnm {
        w,
        h,
        maxval,
        comps,
        samples,
    }
}

/// An opj_decompress PGX plane: `PG ML [+-] depth w h`, then 1 or 2 bytes per sample.
pub struct Pgx {
    pub w: usize,
    pub h: usize,
    pub depth: usize,
    pub signed: bool,
    pub samples: Vec<i32>,
}

pub fn read_pgx(p: PathBuf) -> Pgx {
    let d = std::fs::read(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
    let nl = d.iter().position(|&b| b == b'\n').unwrap();
    let line = String::from_utf8_lossy(&d[..nl]).to_string();
    let f: Vec<&str> = line.split_whitespace().collect();
    let (signed, depth, w, h) = match f.len() {
        6 => (f[2] == "-", f[3], f[4], f[5]),
        5 => (false, f[2], f[3], f[4]),
        _ => panic!("{}: bad PGX header {line:?}", p.display()),
    };
    let (depth, w, h): (usize, usize, usize) = (
        depth.parse().unwrap(),
        w.parse().unwrap(),
        h.parse().unwrap(),
    );
    let body = &d[nl + 1..];
    let n = w * h;
    let samples = if depth <= 8 {
        body[..n]
            .iter()
            .map(|&b| if signed { b as i8 as i32 } else { b as i32 })
            .collect()
    } else {
        (0..n)
            .map(|i| {
                let v = ((body[2 * i] as u16) << 8) | body[2 * i + 1] as u16;
                if signed { v as i16 as i32 } else { v as i32 }
            })
            .collect()
    };
    Pgx {
        w,
        h,
        depth,
        signed,
        samples,
    }
}

/// A PAM (P7) RGB_ALPHA file: `maxval` 255 gives 4 bytes a pixel, 65535 gives 8 (big-endian).
pub struct Pam {
    pub w: usize,
    pub h: usize,
    pub maxval: usize,
    pub pix: Vec<u8>,
}

pub fn read_pam(p: PathBuf) -> Pam {
    let d = std::fs::read(p).unwrap();
    let end = d.windows(7).position(|w| w == b"ENDHDR\n").unwrap() + 7;
    let hdr = String::from_utf8_lossy(&d[..end]).to_string();
    let get = |k: &str| -> usize {
        hdr.lines()
            .find_map(|l| l.strip_prefix(k).map(|v| v.trim().parse().unwrap()))
            .unwrap()
    };
    Pam {
        w: get("WIDTH "),
        h: get("HEIGHT "),
        maxval: get("MAXVAL "),
        pix: d[end..].to_vec(),
    }
}

/// The largest per-sample absolute difference between an 8-bit-per-channel image's channel
/// bytes (pixel pitch `pix_stride`) and a PNM's samples.
pub fn max_abs_diff(got: &[u8], want: &Pnm, pix_stride: usize) -> i32 {
    let mut max = 0;
    for i in 0..want.w * want.h {
        for c in 0..want.comps {
            let g = got[i * pix_stride + c] as i32;
            let w = want.samples[i * want.comps + c];
            max = max.max((g - w).abs());
        }
    }
    max
}

/// `max_abs_diff` for 16-bit big-endian channel data.
pub fn max_abs_diff16(pix: &[u8], want: &Pnm, pix_stride: usize, comps: usize) -> i32 {
    let mut max = 0;
    for i in 0..want.w * want.h {
        for c in 0..comps {
            let off = i * pix_stride + 2 * c;
            let g = ((pix[off] as i32) << 8) | pix[off + 1] as i32;
            let w = want.samples[i * want.comps + c];
            max = max.max((g - w).abs());
        }
    }
    max
}

pub fn width(r: &Raster) -> usize {
    (r.bounds.x1 - r.bounds.x0) as usize
}

pub fn height(r: &Raster) -> usize {
    (r.bounds.y1 - r.bounds.y0) as usize
}

fn u16be(b: &mut Vec<u8>, v: usize) {
    b.extend_from_slice(&(v as u16).to_be_bytes());
}

fn u32be(b: &mut Vec<u8>, v: usize) {
    b.extend_from_slice(&(v as u32).to_be_bytes());
}

fn seg(b: &mut Vec<u8>, m: usize, body: impl FnOnce(&mut Vec<u8>)) {
    u16be(b, m);
    let at = b.len();
    u16be(b, 0);
    body(b);
    let l = (b.len() - at) as u16;
    b[at..at + 2].copy_from_slice(&l.to_be_bytes());
}

/// Hand-builds a one-tile, one-component 8-bit codestream: the audit's crafted inputs are all
/// headers, so bytes beat checked-in files. `poc`, `extra` (raw main-header markers), `tp_poc`
/// (a tile-part POC body) and `tile_data` may be `None`.
#[allow(clippy::too_many_arguments)]
pub fn hostile_j2k(
    size: usize,
    prog: usize,
    layers: usize,
    nl: usize,
    precincts: Option<&[u8]>,
    poc: Option<&[u8]>,
    extra: Option<&[u8]>,
    tp_poc: Option<&[u8]>,
    tile_data: Option<&[u8]>,
) -> Vec<u8> {
    let mut b = Vec::new();
    u16be(&mut b, 0xFF4F);
    seg(&mut b, 0xFF51, |b| {
        u16be(b, 0);
        u32be(b, size);
        u32be(b, size);
        u32be(b, 0);
        u32be(b, 0);
        u32be(b, size);
        u32be(b, size);
        u32be(b, 0);
        u32be(b, 0);
        u16be(b, 1);
        b.extend_from_slice(&[7, 1, 1]);
    });
    seg(&mut b, 0xFF52, |b| {
        b.push(if precincts.is_some() { 1 } else { 0 });
        b.push(prog as u8);
        u16be(b, layers);
        b.push(0);
        b.extend_from_slice(&[nl as u8, 4, 4, 0, 1]);
        if let Some(p) = precincts {
            b.extend_from_slice(p);
        }
    });
    seg(&mut b, 0xFF5C, |b| {
        b.push(0x40);
        for _ in 0..3 * nl + 1 {
            b.push(8 << 3);
        }
    });
    if let Some(p) = poc {
        seg(&mut b, 0xFF5F, |b| b.extend_from_slice(p));
    }
    if let Some(e) = extra {
        b.extend_from_slice(e);
    }
    u16be(&mut b, 0xFF90);
    u16be(&mut b, 10);
    u16be(&mut b, 0);
    let psot = b.len();
    u32be(&mut b, 0);
    b.extend_from_slice(&[0, 1]);
    if let Some(p) = tp_poc {
        seg(&mut b, 0xFF5F, |b| b.extend_from_slice(p));
    }
    u16be(&mut b, 0xFF93);
    if let Some(t) = tile_data {
        b.extend_from_slice(t);
    }
    let l = (b.len() - psot + 6) as u32;
    b[psot..psot + 4].copy_from_slice(&l.to_be_bytes());
    u16be(&mut b, 0xFFD9);
    b
}

/// `n` bare 14-byte tile-parts spread over a 65535-tile grid of one-pixel tiles.
pub fn tile_part_flood(n: usize) -> Vec<u8> {
    header_flood(&[], &[], n)
}

/// `tile_part_flood` with `main_extra` at the end of the main header and `tp_hdr` in every
/// tile-part header.
pub fn header_flood(main_extra: &[u8], tp_hdr: &[u8], n: usize) -> Vec<u8> {
    let mut b = vec![0xff, 0x4f, 0xff, 0x51, 0, 41, 0, 0];
    for v in [65535usize, 1, 0, 0, 1, 1, 0, 0] {
        u32be(&mut b, v);
    }
    b.extend_from_slice(&[0, 1, 7, 1, 1]);
    b.extend_from_slice(&[0xff, 0x52, 0, 12, 0, 0, 0, 1, 0, 0, 4, 4, 0, 1]);
    b.extend_from_slice(&[0xff, 0x5c, 0, 4, 0x40, 8 << 3]);
    b.extend_from_slice(main_extra);
    for i in 0..n {
        b.extend_from_slice(&[0xff, 0x90, 0, 10]);
        u16be(&mut b, i % 65535);
        u32be(&mut b, 14 + tp_hdr.len());
        b.extend_from_slice(&[(i / 65535) as u8, 0]);
        b.extend_from_slice(tp_hdr);
        b.extend_from_slice(&[0xff, 0x93]);
    }
    b.extend_from_slice(&[0xff, 0xd9]);
    b
}

/// One length-prefixed JP2 box (LBox TBox payload).
pub fn jp2_box(typ: &str, payload: &[u8]) -> Vec<u8> {
    let mut buf = vec![0, 0, 0, 0];
    buf.extend_from_slice(typ.as_bytes());
    buf.extend_from_slice(payload);
    let l = buf.len() as u32;
    buf[0..4].copy_from_slice(&l.to_be_bytes());
    buf
}

/// A minimal ihdr box (HEIGHT WIDTH NC BPC C UnkC IPR payload).
pub fn ihdr_box() -> Vec<u8> {
    let mut p = vec![0u8; 14];
    p[3] = 1; // HEIGHT
    p[7] = 1; // WIDTH
    p[9] = 1; // NC
    p[10] = 7; // BPC (8-bit)
    p[11] = 7; // C
    jp2_box("ihdr", &p)
}

pub fn colr_box(enumcs: u32) -> Vec<u8> {
    let mut p = vec![0u8; 7];
    p[0] = 1; // METH: enumerated
    p[3..7].copy_from_slice(&enumcs.to_be_bytes());
    jp2_box("colr", &p)
}

/// A minimal signature + ftyp + jp2h(+extra) + jp2c file.
pub fn jp2_file(jp2h_children: &[u8], stream: &[u8]) -> Vec<u8> {
    let mut d = vec![0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20];
    d.extend_from_slice(&[0x0d, 0x0a, 0x87, 0x0a]);
    d.extend_from_slice(&jp2_box("ftyp", b"jp2 \x00\x00\x00\x00jp2 "));
    d.extend_from_slice(&jp2_box("jp2h", jp2h_children));
    d.extend_from_slice(&jp2_box("jp2c", stream));
    d
}

/// A cdef box from (Cn, Typ, Asoc) triples.
pub fn cdef_box(entries: &[[u16; 3]]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&(entries.len() as u16).to_be_bytes());
    for e in entries {
        for v in e {
            p.extend_from_slice(&v.to_be_bytes());
        }
    }
    jp2_box("cdef", &p)
}

/// A SOC + SIZ-only stream (no COD/QCD/tile data): enough for `decode_config`.
pub fn soc_siz(csiz: usize, precision: u8, signed: bool, xrsiz: u8, yrsiz: u8) -> Vec<u8> {
    let mut p = Vec::new();
    u16be(&mut p, 0);
    for v in [64usize, 64, 0, 0, 64, 64, 0, 0] {
        u32be(&mut p, v);
    }
    u16be(&mut p, csiz);
    for _ in 0..csiz {
        let mut ssiz = precision - 1;
        if signed {
            ssiz |= 0x80;
        }
        p.extend_from_slice(&[ssiz, xrsiz, yrsiz]);
    }
    let mut b = vec![0xFF, 0x4F, 0xFF, 0x51];
    u16be(&mut b, p.len() + 2);
    b.extend_from_slice(&p);
    b
}

/// SHA-256 (FIPS 180-4), for the reference hashes of the opj_decompress PGX files.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

/// Plane `p` as opj_decompress writes a component of precision `prec` to PGX: signed samples go
/// back to two's complement.
pub fn pgx_bytes(p: &jp2k::Plane, prec: u32, signed: bool) -> Vec<u8> {
    let (w, h) = (p.bounds.x1 - p.bounds.x0, p.bounds.y1 - p.bounds.y0);
    let mut b = format!("PG ML {} {prec} {w} {h}\n", if signed { '-' } else { '+' }).into_bytes();
    for &v in &p.pix {
        let mut v = v as i32;
        if signed {
            v -= 1 << (prec - 1);
        }
        if prec <= 8 {
            b.push(v as u8);
        } else {
            b.extend_from_slice(&(v as u16).to_be_bytes());
        }
    }
    b
}

pub fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

pub fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

pub fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

pub fn ceil_div_pow2(a: i64, b: u32) -> i64 {
    (a + (1i64 << b) - 1) >> b
}
