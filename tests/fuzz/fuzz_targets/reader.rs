#![no_main]
use jp2k::{Error, Limits, Options};
use libfuzzer_sys::fuzz_target;
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, Ordering};

/// A reader that returns 1 to 16 bytes per call, the sizes taken from `sizes` in turn. With
/// `faults`, a size byte with its top two bits 01 returns `Interrupted` instead, and one with its
/// top four bits set a hard error; at most 4 `Interrupted` in a row, so a reader whose every size
/// byte says Interrupted still ends. `cancel_at` raises `cancel` at that call.
struct Chunked<'a> {
    data: &'a [u8],
    sizes: &'a [u8],
    i: usize,
    interrupted: u8,
    faults: bool,
    cancel: Option<(&'a AtomicBool, usize)>,
}

fn faulty<'a>(
    data: &'a [u8],
    sizes: &'a [u8],
    cancel: Option<(&'a AtomicBool, usize)>,
) -> Chunked<'a> {
    Chunked {
        data,
        sizes,
        i: 0,
        interrupted: 0,
        faults: true,
        cancel,
    }
}

impl Read for Chunked<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let b = self.sizes[self.i % self.sizes.len()];
        self.i += 1;
        if let Some((c, at)) = self.cancel
            && self.i == at
        {
            c.store(true, Ordering::Relaxed);
        }
        if self.faults {
            if b & 0xF0 == 0xF0 {
                return Err(io::Error::other("fault"));
            }
            if b & 0xC0 == 0x40 && self.interrupted < 4 {
                self.interrupted += 1;
                return Err(io::ErrorKind::Interrupted.into());
            }
            self.interrupted = 0;
        }
        let want = 1 + usize::from(b & 15);
        let n = want.min(buf.len()).min(self.data.len());
        buf[..n].copy_from_slice(&self.data[..n]);
        self.data = &self.data[n..];
        Ok(n)
    }
}

// data = [stream][8 chunk sizes][cancel_at: u8]. The reader entry points, over a reader that
// returns short reads, give what the slice entry points do (decode_config: see below). Over one
// that also returns Interrupted they give the same; over one that fails they give that or an
// Io error; and when the cancel flag is raised during a read they give that or Cancelled.
fuzz_target!(|data: &[u8]| {
    let Some((rest, &[cancel_at])) = data.split_last_chunk::<1>() else {
        return;
    };
    let Some((stream, sizes)) = rest.split_last_chunk::<8>() else {
        return;
    };
    let limits = Limits {
        max_alloc: 256 << 20,
        max_pixels: 1 << 22,
        max_output: 64 << 20,
        max_input: 1 << 24,
        ..Limits::default()
    };
    let o = Options {
        limits,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    let chunked = || Chunked {
        data: stream,
        sizes,
        i: 0,
        interrupted: 0,
        faults: false,
        cancel: None,
    };
    assert_eq!(
        jp2k::decode_reader(chunked(), &o, &cancel),
        jp2k::decode(stream, &o, &cancel)
    );
    // decode_config on a slice skips a box that runs past the end without error, as Go does on a
    // bytes.Reader (README, Differences from Go), so a truncated JP2 file may fail with another
    // message there; whether it works, and what it returns, must agree.
    match (
        jp2k::decode_config_reader(chunked(), &limits, &cancel),
        jp2k::decode_config(stream, &limits),
    ) {
        (Ok(a), Ok(b)) => assert_eq!(a, b),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("reader {a:?}, slice {b:?}"),
    }
    // Interrupted is retried; a hard error is the reader's; a cancel stops the call.
    let want = jp2k::decode(stream, &o, &cancel);
    let got = jp2k::decode_reader(faulty(stream, sizes, None), &o, &cancel);
    assert!(
        got == want || matches!(got, Err(Error::Io(..))),
        "faulty reader {got:?}, slice {want:?}"
    );
    let cancel2 = AtomicBool::new(false);
    let got = jp2k::decode_reader(
        faulty(stream, sizes, Some((&cancel2, usize::from(cancel_at) + 1))),
        &o,
        &cancel2,
    );
    assert!(
        got == want || matches!(got, Err(Error::Io(..) | Error::Cancelled)),
        "cancelled reader {got:?}, slice {want:?}"
    );
    let cancel3 = AtomicBool::new(false);
    let cfg = jp2k::decode_config_reader(
        faulty(stream, sizes, Some((&cancel3, usize::from(cancel_at) + 1))),
        &limits,
        &cancel3,
    );
    assert!(!matches!(cfg, Err(Error::Internal(_))), "{cfg:?}");
});
