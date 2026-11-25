#![no_main]
use jp2k::Limits;
use jp2k::fuzz;
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

// The codestream parser and the JP2 box walk, on arbitrary bytes: they must return, not panic.
// What the box walk finds is parsed as a codestream too.
fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_alloc: 256 << 20,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    let _ = fuzz::parse(data, &limits, &cancel);
    let _ = fuzz::parse_codestream(data, false, &limits, &cancel);
    if let Ok((stream, _cdef)) = fuzz::jp2_codestream(data, &limits) {
        let _ = fuzz::parse(stream, &limits, &cancel);
    }
});
