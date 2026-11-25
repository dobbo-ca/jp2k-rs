#![no_main]
use jp2k::fuzz::{Rounding, simd as fuzz};
use libfuzzer_sys::fuzz_target;

/// A runner without AVX2+FMA (x86_64) or NEON (aarch64) would fuzz only the scalar tier against
/// itself: fail instead. Logs the tiers that are fuzzed; with `JP2K_EXPECT_AVX512=1` (fuzz.yml on
/// x86_64) the AVX-512 tier must be one of them.
fn check_tiers() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let names = fuzz::tiers();
        assert!(names.len() > 1, "scalar tier only: nothing to compare");
        eprintln!("kernel tiers fuzzed against scalar: {names:?}");
        if std::env::var("JP2K_EXPECT_AVX512").as_deref() == Ok("1") {
            assert!(
                names.contains(&"avx512"),
                "JP2K_EXPECT_AVX512=1, but this CPU has no AVX-512"
            );
        }
    });
}

/// Finite f32s and u32s drawn from the input, cycling over it (zeros if it is empty).
struct Src<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Src<'_> {
    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        for x in &mut b {
            if !self.data.is_empty() {
                *x = self.data[self.pos % self.data.len()];
                self.pos += 1;
            }
        }
        u32::from_le_bytes(b)
    }

    /// Not NaN: the kernels are bit-exact, but a NaN's payload is the CPU's choice. Infinities
    /// stay: a decode can overflow into them.
    fn f32(&mut self) -> f32 {
        let v = f32::from_bits(self.u32());
        if v.is_nan() { 0.0 } else { v }
    }

    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }

    fn i32s(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| self.u32() as i32).collect()
    }
}

/// Equal bit for bit, except that any NaN equals any NaN (an overflow can make one: inf - inf).
fn same(a: &[f32], b: &[f32]) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

// data = [kernel, length, offset, flags: u8][values]. Every SIMD tier this CPU has, on slices
// that start at any alignment and have any length, equals the scalar tier, bit for bit: the
// lifting step, the band scale, the inverse ICT, the irreversible dequantization, the lifting step and scale with their own destination,
// the 16-bit sample pack, the 8-bit pack into one byte of 4-byte pixels, and the integer kernels of
// the 5/3 path: the inverse RCT, the two 5/3 lifting steps and the reversible dequantization.
fuzz_target!(|data: &[u8]| {
    check_tiers();
    let Some((&[sel, len, off, flags], rest)) = data.split_first_chunk::<4>() else {
        return;
    };
    let (n, off) = (usize::from(len), usize::from(off % 4));
    let r = if flags & 1 == 0 {
        Rounding::Fused
    } else {
        Rounding::Separate
    };
    let mut src = Src { data: rest, pos: 0 };
    let tiers = fuzz::tiers();
    match sel % 12 {
        0 => {
            let (d, u, v, c) = (
                src.f32s(n + off),
                src.f32s(n + off),
                src.f32s(n + off),
                src.f32(),
            );
            let mut want = d.clone();
            fuzz::lift(0, &mut want[off..], &u[off..], &v[off..], c, r);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = d.clone();
                fuzz::lift(t, &mut got[off..], &u[off..], &v[off..], c, r);
                assert!(same(&got, &want), "lift {name} {r:?} n {n} off {off}");
            }
        }
        1 => {
            let (x, k) = (src.f32s(n + off), src.f32());
            let mut want = x.clone();
            fuzz::scale(0, &mut want[off..], k);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = x.clone();
                fuzz::scale(t, &mut got[off..], k);
                assert!(same(&got, &want), "scale {name} n {n} off {off}");
            }
        }
        2 => {
            let p = [src.f32s(n + off), src.f32s(n + off), src.f32s(n + off)];
            let mut want = p.clone();
            let [a, b, c] = &mut want;
            fuzz::ict(0, &mut a[off..], &mut b[off..], &mut c[off..], r);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = p.clone();
                let [a, b, c] = &mut got;
                fuzz::ict(t, &mut a[off..], &mut b[off..], &mut c[off..], r);
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(same(g, w), "ict {name} {r:?} n {n} off {off} plane {k}");
                }
            }
        }
        3 => {
            let mag: Vec<i32> = (0..n + off).map(|_| src.u32() as i32).collect();
            let fl: Vec<u32> = (0..n + off).map(|_| src.u32()).collect();
            let fb = [src.f32().abs(), src.f32().abs()];
            let d = src.f32().abs();
            let sh = u32::from(flags >> 1) % 32;
            let mut want = vec![0f32; n];
            fuzz::dequant(0, &mut want, &mag[off..], &fl[off..], sh, fb, d);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = vec![1f32; n + off];
                fuzz::dequant(t, &mut got[off..], &mag[off..], &fl[off..], sh, fb, d);
                assert!(same(&got[off..], &want), "dequant {name} n {n} off {off}");
            }
        }
        4 => {
            let (d, u, v, c) = (
                src.f32s(n + off),
                src.f32s(n + off),
                src.f32s(n + off),
                src.f32(),
            );
            let mut want = vec![0f32; n];
            fuzz::lift_to(0, &mut want, &d[off..], &u[off..], &v[off..], c, r);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = vec![1f32; n + off];
                fuzz::lift_to(t, &mut got[off..], &d[off..], &u[off..], &v[off..], c, r);
                assert!(
                    same(&got[off..], &want),
                    "lift_to {name} {r:?} n {n} off {off}"
                );
            }
        }
        5 => {
            let (x, k) = (src.f32s(n + off), src.f32());
            let mut want = vec![0f32; n];
            fuzz::scale_to(0, &mut want, &x[off..], k);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = vec![1f32; n + off];
                fuzz::scale_to(t, &mut got[off..], &x[off..], k);
                assert!(same(&got[off..], &want), "scale_to {name} n {n} off {off}");
            }
        }
        6 => {
            let x = src.f32s(n + off);
            let prec = i32::from(flags >> 1) % 30 + 1;
            let mut want = vec![0u16; n];
            fuzz::pack_u16(0, &mut want, &x[off..], prec);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = vec![7u16; n + off];
                fuzz::pack_u16(t, &mut got[off..], &x[off..], prec);
                assert!(
                    got[off..] == want[..],
                    "pack_u16 {name} prec {prec} n {n} off {off}"
                );
            }
        }
        8 => {
            let p = [src.i32s(n + off), src.i32s(n + off), src.i32s(n + off)];
            let mut want = p.clone();
            let [a, b, c] = &mut want;
            fuzz::rct(0, &mut a[off..], &mut b[off..], &mut c[off..]);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = p.clone();
                let [a, b, c] = &mut got;
                fuzz::rct(t, &mut a[off..], &mut b[off..], &mut c[off..]);
                assert!(got == want, "rct {name} n {n} off {off}");
            }
        }
        9 | 10 => {
            let (d, u, v) = (src.i32s(n + off), src.i32s(n + off), src.i32s(n + off));
            let k: fn(usize, &mut [i32], &[i32], &[i32]) = if sel % 12 == 9 {
                fuzz::lift53_sub
            } else {
                fuzz::lift53_add
            };
            let mut want = d.clone();
            k(0, &mut want[off..], &u[off..], &v[off..]);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = d.clone();
                k(t, &mut got[off..], &u[off..], &v[off..]);
                assert!(got == want, "lift53 {name} sel {sel} n {n} off {off}");
            }
        }
        11 => {
            let (o, fl) = (src.i32s(n + off), src.i32s(n + off));
            let fl: Vec<u32> = fl.into_iter().map(|f| f as u32).collect();
            let rb = [src.u32() as i32, src.u32() as i32];
            let sh = u32::from(flags >> 1) % 32;
            let mut want = o.clone();
            fuzz::dequant_rev(0, &mut want[off..], &fl[off..], sh, rb);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = o.clone();
                fuzz::dequant_rev(t, &mut got[off..], &fl[off..], sh, rb);
                assert!(got == want, "dequant_rev {name} n {n} off {off}");
            }
        }
        _ => {
            // The pixels start at any byte offset; their other bytes come from the input too.
            let x = src.f32s(n);
            let px: Vec<u8> = (0..4 * n + off).map(|_| src.u32() as u8).collect();
            let prec = i32::from(flags >> 3) % 30 + 1;
            let ch = usize::from(flags >> 1) % 4;
            let mut want = px.clone();
            fuzz::pack_u8x4(0, &mut want[off..], &x, prec, ch);
            for (t, name) in tiers.iter().enumerate().skip(1) {
                let mut got = px.clone();
                fuzz::pack_u8x4(t, &mut got[off..], &x, prec, ch);
                assert!(
                    got == want,
                    "pack_u8x4 {name} prec {prec} ch {ch} n {n} off {off}"
                );
            }
        }
    }
});
