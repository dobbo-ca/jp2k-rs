//! `kernels`: times each SIMD kernel at the lengths a decode calls it with, on every tier this CPU
//! has, through the hidden hooks `jp2k::fuzz::simd` (not API). Prints `kernel\tlen\ttier\tns` per
//! call to stdout and a markdown table (ns per call by tier, and AVX-512 over AVX2 where both exist)
//! to stderr. Each case runs for at least 200 ms. Printed by `tests/parity/bench.sh`.
//!
//! Lengths: 16, 32 and 64 (column blocks, code-block rows, small levels), 512 and 1020 (row bands)
//! and 1024 and 2040 (tile rows: the ICT, the RCT and the output packs). `lift` and `lift_to` also
//! run as `cols_97` calls them, over the 64-float rows of a 64x1024 block (256 KB, out of L1), and
//! `lift53_sub` as `cols_53` does: that row is "64x1024", its time per row.

use jp2k::fuzz::{Rounding, simd as k};
use std::hint::black_box;
use std::time::{Duration, Instant};

const LENS: [usize; 7] = [16, 32, 64, 512, 1020, 1024, 2040];
const ROWS: usize = 1024;

/// ns per call of `f`, which makes `per` kernel calls.
fn time(per: usize, mut f: impl FnMut()) -> f64 {
    // Calls in batches of 64, so reading the clock (about 20 ns) is not part of a short kernel's
    // time.
    const BATCH: u64 = 64;
    f();
    let t = Instant::now();
    let mut i = 0u64;
    while t.elapsed() < Duration::from_millis(200) {
        for _ in 0..BATCH {
            f();
        }
        i += BATCH;
    }
    t.elapsed().as_nanos() as f64 / (i * per as u64) as f64
}

/// Values like a decode's: samples of a few hundred, some negative.
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 1024) as f32 - 384.5
        })
        .collect()
}

/// The ns per call of kernel `name` at length `n` on tier `t`.
fn run(name: &str, n: usize, t: usize) -> f64 {
    let (a, b, c) = (data(n, 1), data(n, 2), data(n, 3));
    let r = Rounding::Fused;
    match name {
        "lift" => {
            let mut d = a.clone();
            time(1, || k::lift(t, black_box(&mut d), &b, &c, 0.5, r))
        }
        "lift_to" => {
            let mut o = vec![0f32; n];
            time(1, || k::lift_to(t, black_box(&mut o), &a, &b, &c, -0.25, r))
        }
        "scale" => {
            let mut d = a.clone();
            time(1, || k::scale(t, black_box(&mut d), 1.0))
        }
        "scale_to" => {
            let mut o = vec![0f32; n];
            time(1, || k::scale_to(t, black_box(&mut o), &a, 1.5))
        }
        "ict" => {
            // Repeated in place, the planes grow to infinity and NaN, which cost the same.
            let (mut x, mut y, mut z) = (a, b, c);
            time(1, || k::ict(t, black_box(&mut x), &mut y, &mut z, r))
        }
        "dequant" => {
            let mag: Vec<i32> = a.iter().map(|&v| v.abs() as i32 * 64).collect();
            let flags: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0x2f1b_8d35)).collect();
            let mut o = vec![0f32; n];
            time(1, || {
                k::dequant(t, black_box(&mut o), &mag, &flags, 3, [0.5, 0.25], 0.01)
            })
        }
        "pack_u16" => {
            let mut o = vec![0u16; n];
            let x: Vec<f32> = a.iter().map(|v| v / 2.0).collect();
            time(1, || k::pack_u16(t, black_box(&mut o), &x, 8))
        }
        "pack_u8x4" => {
            let mut o = vec![0u8; 4 * n];
            let x: Vec<f32> = a.iter().map(|v| v / 2.0).collect();
            time(1, || k::pack_u8x4(t, black_box(&mut o), &x, 8, 1))
        }
        "rct" => {
            let (mut x, mut y, mut z) = (ints(&a), ints(&b), ints(&c));
            time(1, || k::rct(t, black_box(&mut x), &mut y, &mut z))
        }
        "lift53_sub" => {
            let (mut d, u, v) = (ints(&a), ints(&b), ints(&c));
            time(1, || k::lift53_sub(t, black_box(&mut d), &u, &v))
        }
        "lift53_add" => {
            let (mut d, u, v) = (ints(&a), ints(&b), ints(&c));
            time(1, || k::lift53_add(t, black_box(&mut d), &u, &v))
        }
        "dequant_rev" => {
            let flags: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0x2f1b_8d35)).collect();
            let mut o = ints(&a);
            time(1, || {
                k::dequant_rev(t, black_box(&mut o), &flags, 3, [16, 8])
            })
        }
        _ => unreachable!(),
    }
}

/// The samples as i32, for the integer kernels.
fn ints(x: &[f32]) -> Vec<i32> {
    x.iter().map(|&v| v as i32).collect()
}

/// `lift53_sub` over the low rows of a 64x`ROWS` block in Mallat order, each with two high rows,
/// as `cols_53` lifts a block of columns: ns per row.
fn run_block53(t: usize) -> f64 {
    const W: usize = 64;
    let mut blk = ints(&data(W * ROWS, 5));
    let sn = ROWS / 2;
    time(sn - 1, || {
        let (lo, hi) = blk.split_at_mut(sn * W);
        for i in 1..sn {
            let d = &mut lo[i * W..(i + 1) * W];
            k::lift53_sub(t, d, &hi[(i - 1) * W..i * W], &hi[i * W..(i + 1) * W]);
        }
        black_box(&mut blk);
    })
}

/// `lift` or `lift_to` (`to`) over the odd rows of a 64x`ROWS` block, each with its two
/// neighbours, as `cols_97` lifts a block of columns: ns per row.
fn run_block(to: bool, t: usize) -> f64 {
    const W: usize = 64;
    let mut blk = data(W * ROWS, 4);
    let mut out = vec![0f32; W * ROWS];
    let rows = (1..ROWS - 1).step_by(2).count();
    time(rows, || {
        for i in (1..ROWS - 1).step_by(2) {
            let (lo, hi) = blk.split_at_mut(i * W);
            let (d, rest) = hi.split_at_mut(W);
            let (u, v) = (&lo[(i - 1) * W..], &rest[..W]);
            if to {
                k::lift_to(
                    t,
                    &mut out[i * W..(i + 1) * W],
                    d,
                    u,
                    v,
                    -0.5,
                    Rounding::Fused,
                );
            } else {
                k::lift(t, d, u, v, 0.5, Rounding::Fused);
            }
        }
        black_box(&mut blk);
    })
}

fn main() {
    let tiers = k::tiers();
    let ix = |name: &str| tiers.iter().position(|&t| t == name);
    let ratio = ix("avx2").zip(ix("avx512"));
    let mut head = format!("| kernel | len | {} |", tiers.join(" | "));
    let mut rule = format!("|---|---|{}", "---|".repeat(tiers.len()));
    if ratio.is_some() {
        head += " avx512/avx2 |";
        rule += "---|";
    }
    eprintln!("{head}\n{rule}");
    let mut cases: Vec<(&str, String)> = Vec::new();
    for name in [
        "lift",
        "lift_to",
        "scale",
        "scale_to",
        "ict",
        "dequant",
        "pack_u16",
        "pack_u8x4",
        "rct",
        "lift53_sub",
        "lift53_add",
        "dequant_rev",
    ] {
        for n in LENS {
            cases.push((name, n.to_string()));
        }
        if ["lift", "lift_to", "lift53_sub"].contains(&name) {
            cases.push((name, format!("64x{ROWS}")));
        }
    }
    for (name, len) in cases {
        let ns: Vec<f64> = (0..tiers.len())
            .map(|t| match len.parse() {
                Ok(n) => run(name, n, t),
                Err(_) if name == "lift53_sub" => run_block53(t),
                Err(_) => run_block(name == "lift_to", t),
            })
            .collect();
        let mut row = format!("| {name} | {len} |");
        for (t, v) in tiers.iter().zip(&ns) {
            println!("{name}\t{len}\t{t}\t{v:.1}");
            row += &format!(" {v:.1} |");
        }
        if let Some((a, b)) = ratio {
            row += &format!(" {:.2} |", ns[b] / ns[a]);
        }
        eprintln!("{row}");
    }
}
