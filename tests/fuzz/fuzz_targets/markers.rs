#![no_main]
use jp2k::Limits;
use jp2k::fuzz;
use libfuzzer_sys::fuzz_target;

// data = [pos: u8][segment bytes]: the marker-segment parsers on their own, on arbitrary bytes.
fuzz_target!(|data: &[u8]| {
    let Some((&pos, rest)) = data.split_first() else {
        return;
    };
    let limits = Limits::default();
    let pos = usize::from(pos);
    let _ = fuzz::parse_siz_segment(rest, pos, &limits);
    let _ = fuzz::parse_cod(rest, pos, &limits);
    let _ = fuzz::parse_plt(rest, pos, true, &limits);
    let _ = fuzz::parse_plt(rest, pos, false, &limits);
});
