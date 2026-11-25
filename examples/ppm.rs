//! `ppm <tiles> <outputs>`: decodes every `.j2k` in `<tiles>` with the default options and writes
//! `<stem>.pgm` (8-bit gray) or `<stem>.ppm` (8-bit RGB, alpha dropped) to `<outputs>`. The Rust
//! side of a real-file comparison with Go; the Go side is `tests/parity/go/oracle -ppm`. Real-file tiles must decode,
//! so any error, and any other pixel format, stops the run.

use jp2k::{ColorModel, Limits, Options};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let [_, input, output] = args.as_slice() else {
        eprintln!("usage: ppm <tiles> <outputs>");
        std::process::exit(2);
    };
    fs::create_dir_all(output)?;
    let mut names: Vec<PathBuf> = Vec::new();
    for e in fs::read_dir(input)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "j2k") {
            names.push(p);
        }
    }
    names.sort();
    let opts = Options {
        limits: Limits {
            max_pixels: 1 << 26,
            max_alloc: 1 << 30,
            ..Limits::default()
        },
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    for p in &names {
        let fail = |e: &dyn std::fmt::Display| format!("{}: {e}", p.display());
        let src = fs::read(p)?;
        let r = jp2k::decode(&src, &opts, &cancel).map_err(|e| fail(&e))?;
        let w = usize::try_from(r.bounds.x1 - r.bounds.x0)?;
        let h = usize::try_from(r.bounds.y1 - r.bounds.y0)?;
        let (ext, mut buf) = match r.color_model {
            ColorModel::Gray => ("pgm", format!("P5\n{w} {h}\n255\n").into_bytes()),
            ColorModel::Rgba => ("ppm", format!("P6\n{w} {h}\n255\n").into_bytes()),
            m => return Err(fail(&format!("decoded {m:?}, want Gray or Rgba")).into()),
        };
        buf.reserve(3 * w * h);
        for row in r.pix.chunks(r.stride.max(1)).take(h) {
            if r.color_model == ColorModel::Gray {
                buf.extend_from_slice(&row[..w]);
            } else {
                for px in row[..4 * w].as_chunks::<4>().0 {
                    buf.extend_from_slice(&px[..3]);
                }
            }
        }
        // The stem keeps its dots (`CMU-1_svs.ifd0.000123`), so no `with_extension`.
        let stem = p.file_stem().unwrap_or_default().to_string_lossy();
        fs::write(PathBuf::from(output).join(format!("{stem}.{ext}")), buf)?;
    }
    println!("{} tiles decoded", names.len());
    Ok(())
}
