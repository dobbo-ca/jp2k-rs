//! What the last call on this thread charged and started, for the tests. Not part of the API.

use crate::limits::Account;
use std::cell::Cell;
use std::sync::atomic::Ordering;

/// The peaks of one call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The most bytes the call's ledger held at once (what `Limits::max_alloc` caps).
    pub peak_alloc: usize,
    /// The most threads the call had running at once, the calling thread included.
    pub peak_threads: usize,
    /// How many threads the call started in all (0 when `max_threads` is 1).
    pub spawned_threads: usize,
}

thread_local! {
    static LAST: Cell<Stats> = const { Cell::new(Stats { peak_alloc: 0, peak_threads: 0, spawned_threads: 0 }) };
}

pub(crate) fn record(a: &Account) {
    let s = Stats {
        peak_alloc: a.peak(),
        peak_threads: a.peak_threads.load(Ordering::Relaxed),
        spawned_threads: a.spawned.load(Ordering::Relaxed),
    };
    let _ = LAST.try_with(|c| c.set(s));
}

/// The peaks of the last call that ended on the calling thread (any entry point, successful or
/// not). Tests run one call per thread, so concurrent tests do not mix.
pub fn last() -> Stats {
    LAST.with(Cell::get)
}
