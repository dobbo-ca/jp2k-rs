//! `max_threads` 1 never touches the crate's shared pool, not even to build it. Alone in its test
//! binary, with every test here on `max_threads` 1, so no other call can have built the pool first.
mod common;

use common::*;
use jp2k::{Image, Limits, Options, Rect};

fn one() -> Options {
    Options {
        limits: Limits {
            max_threads: 1,
            ..Limits::default()
        },
        ..Options::default()
    }
}

#[test]
fn one_thread_spawns_nothing_and_does_not_start_the_shared_pool() {
    let nc = nc();
    let rect = Rect {
        x0: 5,
        y0: 7,
        x1: 60,
        y1: 50,
    };
    for name in ["fl_dense.j2k", "g1_colr.j2c", "g64_multitile.j2k"] {
        let data = read(name);
        let st = |what: &str| {
            let st = jp2k::stats::last();
            assert_eq!(
                (st.spawned_threads, st.peak_threads),
                (0, 1),
                "{name} {what}"
            );
        };
        jp2k::decode(&data, &one(), &nc).unwrap();
        st("decode");
        jp2k::decode_components(&data, &[0], &one(), &nc).unwrap();
        st("components");
        jp2k::decode_region(&data, rect, 0, &one(), &nc).unwrap();
        st("region");
        let img = Image::open(&data, &one(), &nc).unwrap();
        img.decode_region(rect, 0, &nc).unwrap();
        st("image");
    }
    assert!(!jp2k::shared_pool_started());
}
