//! `bench <fixtures> <n> [threads]`: prints `<name>:<op>`, the pixels and the ns per decode for each fixture
//! that decodes, for `decode` ("dec") and `decode_components` of component 0 ("c0"), as
//! `oracle -bench` does for Go. Each is run at least n times and for at least 200 ms. See
//! `tests/parity/bench.sh`. `threads` is the `max_threads` of the decodes (0 or absent: the default, run on one
//! `ThreadPool` made up front and reused for every decode, as a server would; `nopool`: the default
//! with `Options::default()`'s no pool, so the crate's shared pool; 1 spawns nothing). The last line,
//! on stderr, is `# files: N`, the fixtures that gave a `dec` line, so a caller can tell a side that
//! dropped fixtures or died.

use jp2k::{Limits, Options, ThreadPool};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use std::{env, fs, io};

/// ns per call of `f`, or None if it fails.
fn time1(n: u32, mut f: impl FnMut() -> bool) -> Option<u128> {
    if !f() {
        return None;
    }
    let t = Instant::now();
    let mut i = 0u32;
    while i < n || t.elapsed() < Duration::from_millis(200) {
        if !std::hint::black_box(f()) {
            return None;
        }
        i += 1;
    }
    Some(t.elapsed().as_nanos() / u128::from(i))
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let (dir, n, threads) = match args.as_slice() {
        [_, dir, n] => (dir, n, "0"),
        [_, dir, n, t] => (dir, n, t.as_str()),
        _ => {
            eprintln!("usage: bench <fixtures> <n> [threads]");
            std::process::exit(2);
        }
    };
    let n: u32 = n.parse().unwrap_or(1).max(1);
    let explicit = threads.parse::<usize>().ok().filter(|&t| t > 0);
    let max_threads = explicit.unwrap_or(Limits::default().max_threads);
    let pool = if explicit.is_none() && threads != "nopool" {
        ThreadPool::new(max_threads).ok()
    } else {
        None
    };
    let opts = Options {
        limits: Limits {
            max_threads,
            ..Limits::default()
        },
        pool,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    // BENCH_OP=dec or c0 times that op only (profile.yml); unset, both.
    let op = env::var("BENCH_OP").unwrap_or_default();
    let mut names: Vec<String> = Vec::new();
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if [".j2k", ".j2c", ".jp2"].iter().any(|e| name.ends_with(e)) {
            names.push(name);
        }
    }
    names.sort();
    let mut files = 0u32;
    for name in names {
        let data = fs::read(format!("{dir}/{name}"))?;
        let Ok(img) = jp2k::decode(&data, &opts, &cancel) else {
            continue;
        };
        let b = img.bounds;
        let pixels = (b.x1 - b.x0) * (b.y1 - b.y0);
        if op == "c0" {
            files += 1;
        } else if let Some(ns) = time1(n, || jp2k::decode(&data, &opts, &cancel).is_ok()) {
            println!("{name}:dec\t{pixels}\t{ns}");
            files += 1;
        }
        if op == "dec" {
            continue;
        }
        if let Some(ns) = time1(n, || {
            jp2k::decode_components(&data, &[0], &opts, &cancel).is_ok()
        }) {
            println!("{name}:c0\t{pixels}\t{ns}");
        }
    }
    eprintln!("# files: {files}");
    Ok(())
}
