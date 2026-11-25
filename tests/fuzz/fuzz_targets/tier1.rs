#![no_main]
use jp2k::Limits;
use jp2k::fuzz::{self, Segment, State};
use libfuzzer_sys::fuzz_target;
use std::cell::RefCell;

thread_local! {
    /// One `State` for every input: a decode reuses its worker's, so whatever a block leaves in
    /// it must not change the next block.
    static REUSED: RefCell<State> = RefCell::new(State::default());
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// Go's FuzzTier1, with a fresh State against the reused one. data = [w, h, orientation, zero bit
// planes, Mb, style, passes 1, passes 2, split: u8][segment bytes]: one segment, or two split at
// a byte. The sign-magnitude, the reversible (5/3 write-out, as the decoder uses) and the
// irreversible paths give the same from either State.
fuzz_target!(|data: &[u8]| {
    let Some((&[w, h, o, z, mb, style, p1, p2, split], rest)) = data.split_first_chunk::<9>()
    else {
        return;
    };
    // As Go's FuzzTier1: 1 to 64 on each side, so odd widths, narrow blocks and partial last
    // stripes (heights 1 to 3 past a multiple of 4) are all drawn.
    let (w, h) = (1 + usize::from(w) % 64, 1 + usize::from(h) % 64);
    let (orientation, zbp, mb, style) = (
        usize::from(o & 3),
        i32::from(z % 40),
        i32::from(mb % 32),
        style & 0x3f,
    );
    let (a, b) = rest.split_at(usize::from(split >> 1) % (rest.len() + 1));
    let segs: Vec<Segment> = if split & 1 == 0 {
        vec![Segment {
            data: rest,
            passes: usize::from(p1) % 200,
        }]
    } else {
        vec![
            Segment {
                data: a,
                passes: usize::from(p1) % 200,
            },
            Segment {
                data: b,
                passes: usize::from(p2) % 200,
            },
        ]
    };
    let n = w * h;
    let fresh = fuzz::decode_code_block(
        &segs,
        w as i64,
        h as i64,
        orientation,
        zbp,
        mb,
        style,
        &Limits::default(),
    );
    let reused = REUSED.with(|s| {
        let (mut out, mut qs) = (vec![0i32; n], vec![0i32; n]);
        s.borrow_mut()
            .finish(
                &mut out,
                Some(&mut qs),
                None,
                0.0,
                &segs,
                w,
                h,
                orientation,
                zbp,
                mb,
                style,
            )
            .map(|()| (out, qs))
    });
    assert_eq!(fresh, reused);

    let delta = 1.0 + f64::from(z) / 16.0;
    let irreversible = |s: &mut State| {
        let (mut out, mut f) = (vec![0i32; n], vec![0f32; n]);
        s.finish(
            &mut out,
            None,
            Some(&mut f),
            delta,
            &segs,
            w,
            h,
            orientation,
            zbp,
            mb,
            style,
        )
        .map(|()| bits(&f))
    };
    let fresh = irreversible(&mut State::default());
    let reused = REUSED.with(|s| irreversible(&mut s.borrow_mut()));
    assert_eq!(fresh, reused);

    let reversible = |s: &mut State| {
        let mut out = vec![0i32; n];
        s.finish(
            &mut out,
            None,
            None,
            0.0,
            &segs,
            w,
            h,
            orientation,
            zbp,
            mb,
            style,
        )
        .map(|()| out)
    };
    let fresh = reversible(&mut State::default());
    let reused = REUSED.with(|s| reversible(&mut s.borrow_mut()));
    assert_eq!(fresh, reused);
});
