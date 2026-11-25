//! Threads for one call, in safe Rust: a rayon scope on a `ThreadPool` (the caller's, the crate's
//! lazy shared one, or one the call builds for itself) over work handed out one item at a time,
//! so every thread owns what it works on. `drive` (on the pool) and `drive_threads` (on std
//! threads, for tile lanes) are the only places work is handed to another thread.
//! After Go's `internal/parallel`: `Workers`, `For`, `RowChunk`.

use crate::Options;
use crate::error::{Error, Result};
use crate::limits::Account;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Worker threads that decodes share. Create one, keep it, and pass it in `Options::pool`: each
/// call then runs its parallel passes on these threads instead of starting new ones, so small
/// images use threads too. A call without one uses the crate's lazy shared pool (see
/// `Options::pool`); a caller's pool always wins. `Limits::max_threads` still caps the threads one
/// call uses, the caller included, and a call on a pool of more than 2 threads uses at most the
/// pool's size minus 1 (a pool of 1 or 2 threads gives them all). Clones share the threads.
#[derive(Clone, Debug)]
pub struct ThreadPool(pub(crate) Arc<rayon::ThreadPool>);

impl ThreadPool {
    /// A pool of `threads` workers (at least 1), started now and stopped when the last clone drops.
    /// It enforces no limit: `threads` is not capped, and the pool takes no `Limits`; each call
    /// on it is bounded by its own `Limits`.
    ///
    /// # Errors
    /// `Error::OutOfMemory` if the operating system will not start the threads.
    pub fn new(threads: usize) -> Result<ThreadPool> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("jp2k-worker-{i}"))
            .build()
            .map(|p| ThreadPool(Arc::new(p)))
            .map_err(|_| Error::OutOfMemory)
    }

    /// How many worker threads the pool has.
    pub fn threads(&self) -> usize {
        self.0.current_num_threads()
    }
}

/// The crate's shared pool, built on first use by a call with `max_threads` above 1 and no caller
/// pool, with `available_parallelism` threads and never more. `None` if the build failed.
static SHARED: OnceLock<Option<ThreadPool>> = OnceLock::new();

fn shared_pool() -> Option<&'static ThreadPool> {
    SHARED
        .get_or_init(|| {
            ThreadPool::new(std::thread::available_parallelism().map_or(1, |n| n.get())).ok()
        })
        .as_ref()
}

/// Whether the shared pool has been built yet (for the tests: `max_threads` 1 never builds it).
#[doc(hidden)]
pub fn shared_pool_started() -> bool {
    SHARED.get().is_some()
}

/// Input bytes that pay for one more thread of a call's own workers: a pool of them costs tens of
/// microseconds per thread to start, so a small input stays on the calling thread.
const PER_CALL_BYTES_PER_THREAD: usize = 8 << 10;

/// The pool one call runs on, and whether the call built it: the caller's, else (with
/// `max_threads` above 1) the shared one, else, when the shared one cannot be built or
/// `Options::per_call_pool` is set, one made now and sized from `input_len`. `None` runs the
/// call on its own thread.
pub(crate) fn pool_for(opts: &Options, input_len: usize) -> (Option<ThreadPool>, bool) {
    if let Some(p) = &opts.pool {
        return (Some(p.clone()), false);
    }
    let max = opts.limits.max_threads;
    if max <= 1 {
        return (None, false);
    }
    if !opts.per_call_pool
        && let Some(p) = shared_pool()
    {
        return (Some(p.clone()), false);
    }
    let n = max.min(1 + input_len / PER_CALL_BYTES_PER_THREAD);
    if n <= 1 {
        return (None, false);
    }
    // The caller is the n-th thread.
    (ThreadPool::new(n - 1).ok(), true)
}

/// How many rows one task of a row pass handles (Go: `dsp.RowChunk`).
pub(crate) const ROW_CHUNK: usize = 16;

/// A mutex's data even if a thread panicked while holding it: the panic is reported as
/// `Error::Internal` and the data is dropped with the call.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The error for a caught panic (Go: `ErrInternal`).
pub(crate) fn panic_error(p: Box<dyn Any + Send>) -> Error {
    let msg = if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic".to_string()
    };
    Error::Internal(msg)
}

struct Ended<'a>(&'a Account);

impl Drop for Ended<'_> {
    fn drop(&mut self) {
        self.0.thread_ended();
    }
}

/// Hands the items of `items` out, in order, to at most `nw` threads (the caller and `nw - 1`
/// more, fewer if `scratch` is shorter; on the pool's workers, and 1 if the call has no pool);
/// `f(scratch, i, item)` is called for item number `i` with the scratch of the thread running it.
/// With `nw` 1 nothing is handed to another thread.
///
/// After the first error no new item starts; items already started finish, so the error returned
/// is that of the lowest-numbered failing item, the one a serial run reports. A set `cancel`
/// fails the next item with `Error::Cancelled`. A panic in `f` on any thread is an
/// `Error::Internal`. Every thread has returned before `drive` does.
pub(crate) fn drive<S, I, F>(
    nw: usize,
    acct: &Account,
    cancel: &AtomicBool,
    scratch: &mut [S],
    items: I,
    f: F,
) -> Result<()>
where
    S: Send,
    I: Iterator + Send,
    F: Fn(&mut S, usize, I::Item) -> Result<()> + Sync,
{
    drive_on(false, nw, acct, cancel, scratch, items, f)
}

/// `drive` on `std::thread::scope` threads started once for the call instead of the pool's
/// workers, for the tile lanes of a call that gives each lane several threads: a lane waits for
/// its nested passes on the pool, and a pool worker that waits there takes other jobs, another
/// call's lane included, while a plain thread only blocks. A thread that cannot be started is
/// skipped (the caller and the threads that did start take its items); a panic joined from a
/// thread is an `Error::Internal`.
pub(crate) fn drive_threads<S, I, F>(
    nw: usize,
    acct: &Account,
    cancel: &AtomicBool,
    scratch: &mut [S],
    items: I,
    f: F,
) -> Result<()>
where
    S: Send,
    I: Iterator + Send,
    F: Fn(&mut S, usize, I::Item) -> Result<()> + Sync,
{
    drive_on(true, nw, acct, cancel, scratch, items, f)
}

fn drive_on<S, I, F>(
    own_threads: bool,
    nw: usize,
    acct: &Account,
    cancel: &AtomicBool,
    scratch: &mut [S],
    items: I,
    f: F,
) -> Result<()>
where
    S: Send,
    I: Iterator + Send,
    F: Fn(&mut S, usize, I::Item) -> Result<()> + Sync,
{
    let mut nw = nw.min(scratch.len()).max(1);
    let Some(pool) = acct.pool() else {
        return drive_serial(scratch, cancel, items, &f);
    };
    // No more threads than items when the count is known: a pool worker that finds nothing to do
    // still has to be woken and waited for.
    if let (lo, Some(hi)) = items.size_hint()
        && lo == hi
    {
        nw = nw.min(hi.max(1));
    }
    if nw == 1 {
        return drive_serial(scratch, cancel, items, &f);
    }
    let Some((head, tail)) = scratch[..nw].split_first_mut() else {
        return Err(Error::Internal("no scratch for a parallel pass".into()));
    };
    let queue = Mutex::new((0usize, items));
    let failed = AtomicBool::new(false);
    let first: Mutex<Option<(usize, Error)>> = Mutex::new(None);
    let work = |s: &mut S| {
        while !failed.load(Ordering::Relaxed) {
            let (i, it) = {
                let mut q = lock(&queue);
                let Some(it) = q.1.next() else { return };
                q.0 += 1;
                (q.0 - 1, it)
            };
            let r = if cancel.load(Ordering::Relaxed) {
                Err(Error::Cancelled)
            } else {
                catch_unwind(AssertUnwindSafe(|| f(s, i, it)))
                    .unwrap_or_else(|p| Err(panic_error(p)))
            };
            if let Err(e) = r {
                let mut fst = lock(&first);
                if fst.as_ref().is_none_or(|(j, _)| i < *j) {
                    *fst = Some((i, e));
                }
                failed.store(true, Ordering::Relaxed);
                return;
            }
        }
    };
    // The caller works too; the pool's workers join as they are free. A worker that comes after
    // the items are gone returns at once.
    if own_threads {
        let r = catch_unwind(AssertUnwindSafe(|| {
            std::thread::scope(|sc| {
                for s in tail {
                    let work = &work;
                    let started = std::thread::Builder::new()
                        .name("jp2k-lane".into())
                        .spawn_scoped(sc, move || {
                            acct.thread_started();
                            let _end = Ended(acct);
                            work(s);
                        });
                    if started.is_err() {
                        break;
                    }
                }
                work(head);
            });
        }));
        if let Err(p) = r {
            lock(&first).get_or_insert((usize::MAX, panic_error(p)));
        }
    } else {
        pool.0.in_place_scope(|sc| {
            for s in tail {
                let work = &work;
                sc.spawn(move |_| {
                    acct.thread_started();
                    let _end = Ended(acct);
                    work(s);
                });
            }
            work(head);
        });
    }
    match first.into_inner().unwrap_or_else(|e| e.into_inner()) {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// The items in order on the calling thread, with the first scratch.
fn drive_serial<S, I, F>(scratch: &mut [S], cancel: &AtomicBool, items: I, f: &F) -> Result<()>
where
    I: Iterator,
    F: Fn(&mut S, usize, I::Item) -> Result<()>,
{
    let Some(head) = scratch.first_mut() else {
        return Err(Error::Internal("no scratch for a parallel pass".into()));
    };
    for (i, it) in items.enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        f(head, i, it)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::{Budget, Limits};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// A ledger whose call runs on a pool of 8 workers (without a pool `drive` is serial).
    fn ledger() -> Budget {
        Budget::with_pool(&Limits::default(), ThreadPool::new(8).ok())
    }

    /// Every item runs once, on at most `nw` threads, and `max_threads` 1 starts none.
    #[test]
    fn items_run_once_on_at_most_nw_threads() {
        for nw in [1usize, 2, 3, 8] {
            let seen = AtomicUsize::new(0);
            let sum = AtomicUsize::new(0);
            {
                let b = ledger();
                let mut scratch = vec![(); nw];
                drive(
                    nw,
                    b.account(),
                    &AtomicBool::new(false),
                    &mut scratch,
                    0..200,
                    |_, i, v| {
                        assert_eq!(i, v);
                        seen.fetch_add(1, Ordering::Relaxed);
                        sum.fetch_add(v, Ordering::Relaxed);
                        std::thread::sleep(Duration::from_micros(50));
                        Ok(())
                    },
                )
                .unwrap();
            }
            assert_eq!(seen.load(Ordering::Relaxed), 200);
            assert_eq!(sum.load(Ordering::Relaxed), 199 * 200 / 2);
            let st = crate::stats::last();
            assert!(st.peak_threads <= nw, "nw {nw}: {st:?}");
            if nw == 1 {
                assert_eq!((st.spawned_threads, st.peak_threads), (0, 1));
            } else {
                assert!(st.spawned_threads < nw);
            }
        }
    }

    /// On a pool: every item runs once, never more than `nw` workers take part, and an error is the
    /// lowest item's. A pool smaller than `nw` still finishes (the caller takes the items).
    #[test]
    fn pool_runs_every_item_within_nw() {
        for (size, nw) in [(1usize, 4usize), (2, 2), (3, 8), (8, 3)] {
            let pool = ThreadPool::new(size).unwrap();
            let b = Budget::with_pool(&Limits::default(), Some(pool));
            let (seen, sum) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let mut scratch = vec![(); nw];
            let nw_eff = b.account().cap_threads(nw);
            drive(
                nw_eff,
                b.account(),
                &AtomicBool::new(false),
                &mut scratch,
                0..200,
                |_, _, v| {
                    seen.fetch_add(1, Ordering::Relaxed);
                    sum.fetch_add(v, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_micros(50));
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(seen.load(Ordering::Relaxed), 200);
            assert_eq!(sum.load(Ordering::Relaxed), 199 * 200 / 2);
            assert!(b.account().peak_threads.load(Ordering::Relaxed) <= nw_eff);
            let r = drive(
                nw_eff,
                b.account(),
                &AtomicBool::new(false),
                &mut scratch,
                0..100,
                |_, _, i| {
                    if i == 3 || i == 5 {
                        return Err(Error::Invalid(format!("boom {i}")));
                    }
                    Ok(())
                },
            );
            assert_eq!(r, Err(Error::Invalid("boom 3".into())));
        }
    }

    /// On std threads: every item runs once on at most `nw` threads, all counted by the ledger,
    /// the error is the lowest item's, and a nested `drive` on the pool inside an item works.
    #[test]
    fn std_threads_run_every_item_within_nw() {
        let b = ledger();
        let sum = AtomicUsize::new(0);
        let mut scratch = vec![(); 3];
        drive_threads(
            3,
            b.account(),
            &AtomicBool::new(false),
            &mut scratch,
            0..60,
            |_, _, v| {
                let mut inner = vec![(); 2];
                drive(
                    2,
                    b.account(),
                    &AtomicBool::new(false),
                    &mut inner,
                    0..4,
                    |_, _, w| {
                        sum.fetch_add(w, Ordering::Relaxed);
                        std::thread::sleep(Duration::from_micros(50));
                        Ok(())
                    },
                )?;
                sum.fetch_add(v * 100, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(sum.load(Ordering::Relaxed), 60 * 6 + 100 * (59 * 60 / 2));
        // 3 lane threads and 1 nested worker each: 3 + 3 at most.
        assert!(b.account().peak_threads.load(Ordering::Relaxed) <= 6);
        let r = drive_threads(
            3,
            b.account(),
            &AtomicBool::new(false),
            &mut scratch,
            0..100,
            |_, _, i| {
                if i == 3 || i == 5 {
                    return Err(Error::Invalid(format!("boom {i}")));
                }
                Ok(())
            },
        );
        assert_eq!(r, Err(Error::Invalid("boom 3".into())));
    }

    /// Each thread gets its own scratch.
    #[test]
    fn scratch_is_per_thread() {
        let b = ledger();
        let mut scratch = vec![0usize; 4];
        drive(
            4,
            b.account(),
            &AtomicBool::new(false),
            &mut scratch,
            0..400,
            |s, _, _| {
                *s += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(scratch.iter().sum::<usize>(), 400);
    }

    /// A failing item stops new items from starting and `drive` does not return until every
    /// thread has (Go: `TestParallelForErrorJoinsWorkers`); the error is the lowest item's.
    #[test]
    fn error_stops_new_items_and_joins_threads() {
        let b = ledger();
        let (running, calls) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let mut scratch = vec![(); 8];
        let r = drive(
            8,
            b.account(),
            &AtomicBool::new(false),
            &mut scratch,
            0..1000,
            |_, _, i| {
                running.fetch_add(1, Ordering::SeqCst);
                calls.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(1));
                running.fetch_sub(1, Ordering::SeqCst);
                if i == 3 || i == 5 {
                    return Err(Error::Invalid(format!("boom {i}")));
                }
                Ok(())
            },
        );
        assert_eq!(r, Err(Error::Invalid("boom 3".into())));
        assert_eq!(running.load(Ordering::SeqCst), 0);
        assert!(calls.load(Ordering::SeqCst) < 1000);
    }

    /// A panic on any thread comes back as `Error::Internal`, with the threads joined.
    #[test]
    fn panic_is_an_internal_error() {
        for nw in [1usize, 3] {
            let b = ledger();
            let mut scratch = vec![(); nw];
            let r = if nw == 1 {
                // On the calling thread a panic propagates to the entry point's guard.
                catch_unwind(AssertUnwindSafe(|| {
                    drive(
                        nw,
                        b.account(),
                        &AtomicBool::new(false),
                        &mut scratch,
                        0..10,
                        |_, _, i| {
                            assert!(i != 4, "injected at {i}");
                            Ok(())
                        },
                    )
                }))
                .unwrap_or_else(|p| Err(panic_error(p)))
            } else {
                drive(
                    nw,
                    b.account(),
                    &AtomicBool::new(false),
                    &mut scratch,
                    0..10,
                    |_, _, i| {
                        assert!(i != 4, "injected at {i}");
                        Ok(())
                    },
                )
            };
            match r {
                Err(Error::Internal(m)) => assert!(m.contains("injected at 4"), "{m}"),
                other => panic!("nw {nw}: {other:?}"),
            }
        }
    }

    /// A set cancel flag stops the next item on every thread.
    #[test]
    fn cancel_stops_every_thread() {
        let b = ledger();
        let cancel = AtomicBool::new(false);
        let calls = AtomicUsize::new(0);
        let mut scratch = vec![(); 4];
        let r = drive(
            4,
            b.account(),
            &cancel,
            &mut scratch,
            0..100_000,
            |_, i, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                if i == 10 {
                    cancel.store(true, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(20));
                Ok(())
            },
        );
        assert_eq!(r, Err(Error::Cancelled));
        assert!(calls.load(Ordering::Relaxed) < 1000);
    }
}
