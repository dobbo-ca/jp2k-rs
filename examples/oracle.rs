//! The Rust side of the parity check: `oracle <inputs> <outputs>` decodes every input like
//! `tests/parity/go/oracle` and writes `<name>.<op>.out` (a description of the result) or
//! `<name>.<op>.err` (the error class). The operations, the description and the class table are
//! in `docs/parity.md`. Go returns nothing with an error, and neither does this crate.

use jp2k::{ColorModel, Config, Error, Image, Limits, Options, Plane, Raster, Rect, ThreadPool};
use std::io::{self, Read};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::{env, fs};

/// Images with more pixels are only configured (hostile-* inputs may be larger), as in the Go
/// oracle.
const MAX_PIXELS: i64 = 1 << 22;
const MAX_HOSTILE_PIXELS: i64 = 1 << 24;

fn fnv(h: u64, b: u8) -> u64 {
    (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
}

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The error class: the message, or `limit` or `internal`, as in the Go oracle.
fn class(e: &Error) -> String {
    match e {
        Error::Internal(_) => return "internal".into(),
        Error::OutOfMemory => return "limit".into(),
        // Go returns the reader's own error, io.ErrUnexpectedEOF, unwrapped.
        Error::Io(_, m) => return m.clone(),
        _ => {}
    }
    let s = e.to_string();
    for k in [
        "working memory",
        "memory limit",
        "input limit",
        "out of memory",
        "more memory than",
    ] {
        if s.contains(k) {
            return "limit".into();
        }
    }
    s.replace('\n', " ")
}

fn model_name(m: ColorModel) -> &'static str {
    match m {
        ColorModel::Gray => "gray",
        ColorModel::Gray16 => "gray16",
        ColorModel::Rgba => "rgba",
        ColorModel::Rgba64 => "rgba64",
        ColorModel::Nrgba => "nrgba",
        ColorModel::Nrgba64 => "nrgba64",
    }
}

fn describe_cfg(c: &Config) -> String {
    format!("{} {} {}\n", model_name(c.color_model), c.width, c.height)
}

fn describe(r: &Raster) -> String {
    let h = r.pix.iter().fold(FNV_BASIS, |h, &b| fnv(h, b));
    format!(
        "{} {} {} {} {} stride={} len={} fnv={:016x}\n",
        model_name(r.color_model),
        r.bounds.x0,
        r.bounds.y0,
        r.bounds.x1,
        r.bounds.y1,
        r.stride,
        r.pix.len(),
        h
    )
}

fn describe_planes(ps: &[Plane]) -> String {
    let mut s = format!("planes {}\n", ps.len());
    for p in ps {
        // Go's Gray16 holds big-endian bytes.
        let h = p
            .pix
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .fold(FNV_BASIS, fnv);
        let b = &p.bounds;
        s += &format!(
            "plane {} {} {} {} len={} fnv={:016x}\n",
            b.x0,
            b.y0,
            b.x1,
            b.y1,
            2 * p.pix.len(),
            h
        );
    }
    s
}

/// xorshift64, the same as the Go oracle, seeded with the FNV-1a 64 of the input name.
struct Rng(u64);

impl Rng {
    fn new(name: &str) -> Rng {
        Rng(name.bytes().fold(FNV_BASIS, fnv) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: i64) -> i64 {
        (self.next_u64() % n as u64) as i64
    }
}

struct Region {
    rect: Rect,
    reduce: i32,
}

/// The whole image at reduce 0 and 1, then four random regions (a random corner, a random extent
/// from it, reduce 0 to 3).
fn regions(name: &str, mut w: i64, mut h: i64) -> Vec<Region> {
    if w < 1 || h < 1 {
        (w, h) = (64, 64);
    }
    let full = Rect {
        x0: 0,
        y0: 0,
        x1: w,
        y1: h,
    };
    let mut rs = vec![
        Region {
            rect: full,
            reduce: 0,
        },
        Region {
            rect: full,
            reduce: 1,
        },
    ];
    let mut r = Rng::new(name);
    for _ in 0..4 {
        let x0 = r.below(w);
        let y0 = r.below(h);
        let x1 = x0 + 1 + r.below(w - x0);
        let y1 = y0 + 1 + r.below(h - y0);
        let reduce = r.below(4) as i32;
        rs.push(Region {
            rect: Rect { x0, y0, x1, y1 },
            reduce,
        });
    }
    rs
}

/// A reader that returns one byte per call, as `iotest.OneByteReader`.
struct OneByte<'a>(&'a [u8]);

impl Read for OneByte<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match (self.0.split_first(), buf.first_mut()) {
            (Some((&b, rest)), Some(o)) => {
                *o = b;
                self.0 = rest;
                Ok(1)
            }
            _ => Ok(0),
        }
    }
}

struct Writer<'a> {
    dir: &'a Path,
    name: &'a str,
}

impl Writer<'_> {
    fn write(&self, op: &str, r: Result<String, Error>) -> io::Result<()> {
        let (ext, body) = match r {
            Ok(s) => ("out", s),
            Err(e) => ("err", class(&e) + "\n"),
        };
        fs::write(self.dir.join(format!("{}.{op}.{ext}", self.name)), body)
    }
}

/// `JP2K_ORACLE_THREADS`: `pool` (default: the default `max_threads` on one shared pool, as a
/// server would keep), `nopool` (the default `max_threads`, no caller pool: the shared pool of
/// `Options::default()`), `percall` (the same with per-call workers) or `1` (`max_threads` 1,
/// the serial path). The output must be the same for all four.
fn threads_mode() -> &'static str {
    static MODE: OnceLock<String> = OnceLock::new();
    MODE.get_or_init(|| env::var("JP2K_ORACLE_THREADS").unwrap_or_else(|_| "pool".into()))
}

fn pool() -> Option<ThreadPool> {
    static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
    if threads_mode() != "pool" {
        return None;
    }
    POOL.get_or_init(|| ThreadPool::new(Limits::default().max_threads).ok())
        .clone()
}

fn opts(limits: Limits, separate_rounding: bool, jj2000: bool) -> Options {
    Options {
        separate_rounding,
        jj2000,
        limits,
        pool: pool(),
        per_call_pool: threads_mode() == "percall",
    }
}

fn run_input(dir: &Path, name: &str, data: &[u8]) -> io::Result<()> {
    let w = Writer { dir, name };
    let cancel = AtomicBool::new(false);
    // The same bound as the Go oracle's SetMemoryLimit.
    let limits = Limits {
        max_alloc: 2 << 30,
        max_threads: if threads_mode() == "1" {
            1
        } else {
            Limits::default().max_threads
        },
        ..Limits::default()
    };
    let plain = opts(limits, false, false);

    let cfg = jp2k::decode_config(data, &limits);
    w.write("cfg", cfg.as_ref().map(describe_cfg).map_err(Error::clone))?;
    let cfg1 = jp2k::decode_config_reader(OneByte(data), &limits, &cancel);
    w.write(
        "cfg1",
        cfg1.as_ref().map(describe_cfg).map_err(Error::clone),
    )?;
    let (mut cw, mut ch) = (64, 64);
    if let Ok(c) = &cfg {
        let max = if name.starts_with("hostile-") {
            MAX_HOSTILE_PIXELS
        } else {
            MAX_PIXELS
        };
        if i64::from(c.width) * i64::from(c.height) > max && !name.starts_with("bigtiles") {
            return w.write("big", Ok(format!("{} {}\n", c.width, c.height)));
        }
        (cw, ch) = (i64::from(c.width), i64::from(c.height));
    }
    let dec = jp2k::decode(data, &plain, &cancel);
    w.write("dec", dec.as_ref().map(describe).map_err(Error::clone))?;
    // bigtiles (33 Mpixel, 1024x1024 tiles, for the tile lanes): the whole image and component 0.
    if name.starts_with("bigtiles") {
        let ps = jp2k::decode_components(data, &[0], &plain, &cancel);
        return w.write(
            "c0",
            ps.as_ref()
                .map(|p| describe_planes(p))
                .map_err(Error::clone),
        );
    }
    let dec1 = jp2k::decode_reader(OneByte(data), &plain, &cancel);
    w.write("dec1", dec1.as_ref().map(describe).map_err(Error::clone))?;

    // The crafted inputs are cheap to fail, not to run a dozen times: they get fewer operations.
    let hostile = name.starts_with("hostile-");
    let rs = regions(name, cw, ch);
    // Region operations run whatever Decode returned, as in the Go oracle.
    let regions_run = if hostile { 2 } else { 6 };
    for (i, r) in rs.iter().enumerate().take(regions_run) {
        let img = jp2k::decode_region(data, r.rect, r.reduce, &plain, &cancel);
        w.write(
            &format!("r{i}"),
            img.as_ref().map(describe).map_err(Error::clone),
        )?;
    }

    let all = [0usize, 1, 2];
    let planes = |op: &str, comps: &[usize], o: &Options| {
        let ps = jp2k::decode_components(data, comps, o, &cancel);
        w.write(
            op,
            ps.as_ref()
                .map(|p| describe_planes(p))
                .map_err(Error::clone),
        )
    };
    planes("c0", &[0], &plain)?;
    planes("cA", &all, &plain)?;
    if !hostile {
        planes("cS", &all, &opts(limits, true, false))?;
        planes("cJ", &all, &opts(limits, false, true))?;
        planes("cS0", &[0], &opts(limits, true, false))?;
        planes("cJ0", &[0], &opts(limits, false, true))?;
    }
    let region_planes = |op: &str, r: &Region, comps: &[usize], o: &Options| {
        let ps = jp2k::decode_region_components(data, r.rect, r.reduce, comps, o, &cancel);
        w.write(
            op,
            ps.as_ref()
                .map(|p| describe_planes(p))
                .map_err(Error::clone),
        )
    };
    if !hostile {
        region_planes("rc0", &rs[2], &[0], &plain)?;
        region_planes("rcA", &rs[3], &all, &opts(limits, true, false))?;
        region_planes("rcJ", &rs[3], &all, &opts(limits, false, true))?;
    }

    // An Image opened once and used for several regions, plain and with each rounding mode.
    open_ops(&w, data, &rs, "open", &plain, &cancel)?;
    if !hostile {
        open_ops(&w, data, &rs, "openS", &opts(limits, true, false), &cancel)?;
        open_ops(&w, data, &rs, "openJ", &opts(limits, false, true), &cancel)?;
    }
    Ok(())
}

/// Opens `data` once and writes one result per call on the Image, as the Go oracle's `openOps`
/// does: `tag` for `Image::open` (config and levels), `tag`0..3 for regions 2, 3, 2 again and 0,
/// `tag`c for the components of region 4. A failed call does not stop the ones after it. For
/// `Image::open` only whether it fails is compared (not the message), besides `limit` and
/// `internal`: Go's `Open` words its header errors differently. The Image's own calls compare the
/// error class, the message, in full.
fn open_ops(
    w: &Writer,
    data: &[u8],
    rs: &[Region],
    tag: &str,
    o: &Options,
    cancel: &AtomicBool,
) -> io::Result<()> {
    let img = match Image::open(data, o, cancel) {
        Ok(img) => img,
        Err(e) => {
            return w.write(tag, Err(open_err(&e)));
        }
    };
    w.write(
        tag,
        Ok(format!(
            "{}levels {}\n",
            describe_cfg(&img.config()),
            img.levels()
        )),
    )?;
    for (k, i) in [2, 3, 2, 0].into_iter().enumerate() {
        let r = img.decode_region(rs[i].rect, rs[i].reduce, cancel);
        w.write(
            &format!("{tag}{k}"),
            r.as_ref().map(describe).map_err(Error::clone),
        )?;
    }
    let ps = img.decode_region_components(rs[4].rect, rs[4].reduce, &[0], cancel);
    w.write(
        &format!("{tag}c"),
        ps.as_ref()
            .map(|p| describe_planes(p))
            .map_err(Error::clone),
    )
}

/// `e` for `Image::open`: limit and internal errors as they are, every other one as "error".
fn open_err(e: &Error) -> Error {
    match class(e).as_str() {
        "limit" | "internal" => e.clone(),
        _ => Error::Io(io::ErrorKind::Other, "error".into()),
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let [_, input, output] = args.as_slice() else {
        eprintln!("usage: oracle <inputs> <outputs>");
        std::process::exit(2);
    };
    fs::create_dir_all(output)?;
    let mut names: Vec<_> = fs::read_dir(input)?
        .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<io::Result<_>>()?;
    names.sort();
    for name in names {
        let path = Path::new(input).join(&name);
        if path.is_dir() {
            continue;
        }
        run_input(Path::new(output), &name, &fs::read(&path)?)?;
    }
    Ok(())
}
