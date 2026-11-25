#![no_main]
use jp2k::Limits;
use jp2k::fuzz;
use libfuzzer_sys::fuzz_target;

// Go's FuzzMQ: 10000 MQ decisions on contexts i % 19, and 10000 raw-decoder bits, from arbitrary
// bytes. Neither decoder may panic or loop.
fuzz_target!(|data: &[u8]| {
    let _ = fuzz::mq_decisions(data, &Limits::default());
});
