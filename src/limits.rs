use crate::Options;
use crate::error::{Error, Result};
use crate::parallel::{ThreadPool, pool_for};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Caps on one call. Go had no per-call limits (it had a process-wide memory budget and a
/// fixed 2^28-pixel canvas cap); the defaults are finite and the pixel cap is Go's.
///
/// ```
/// use jp2k::Limits;
/// // A decode on one thread that never holds more than 64 MiB:
/// let l = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
/// assert_eq!(l.max_threads, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most bytes of working memory one call holds at any moment, all threads together: the
    /// parse, tile geometry, every thread's scratch, the sample planes and the output. It is the
    /// PEAK of a ledger that every allocation is charged to before it is made and released from
    /// when it is dropped, not an estimate. The input slice itself is the caller's and is not
    /// charged; a reader entry point charges the bytes it reads. Over it a call fails with
    /// `Error::LimitExceeded`. Default 4 GiB.
    pub max_alloc: usize,
    /// The most OS threads one call uses, the calling thread included. 1 runs everything on the
    /// calling thread and never spawns one. 0 is treated as 1. The threads are split across
    /// tiles, code-blocks, wavelet rows and columns and output rows, never more than this in
    /// total. A call on a pool (its own, the caller's or the crate's shared one) uses at most this
    /// many, and, when the pool has more than 2 threads, at most the pool's size minus 1; a pool
    /// of 1 or 2 threads gives all of them. Output is the same bytes for every value. Default
    /// `std::thread::available_parallelism()`, or 1 if that is unknown.
    pub max_threads: usize,
    /// Most pixels (width times height) of the decoded image or region; over it a call fails
    /// with `Error::LimitExceeded`. Go: 1 << 28.
    pub max_pixels: u64,
    /// Widest decoded image or region, in pixels. `Error::LimitExceeded` over it. Default
    /// `u32::MAX`.
    pub max_width: u32,
    /// Tallest decoded image or region, in pixels. `Error::LimitExceeded` over it. Default
    /// `u32::MAX`.
    pub max_height: u32,
    /// Most bytes a reader entry point reads (`decode_reader`, `decode_config_reader`).
    /// `Error::LimitExceeded` over it. Default 4 GiB.
    pub max_input: usize,
    /// Most bytes of decoded output (pixels or planes); `Error::LimitExceeded` over it. Go:
    /// 8 * (1 << 28).
    pub max_output: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_alloc: 4 << 30,
            max_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_pixels: 1 << 28,
            max_width: u32::MAX,
            max_height: u32::MAX,
            max_input: 1 << 32,
            max_output: 8 << 28,
        }
    }
}

/// The ledger of one call, shared by every thread of it: bytes charged now and at the peak, and
/// the threads the call started.
#[derive(Debug)]
pub struct Account {
    max: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
    /// Set when a charge was refused for want of room.
    denied: AtomicBool,
    /// Live threads of the call, the caller included.
    threads: AtomicUsize,
    pub(crate) peak_threads: AtomicUsize,
    pub(crate) spawned: AtomicUsize,
    pool: Option<ThreadPool>,
    /// The pool was built for this call alone (sized from the input), not a caller's or the shared.
    own_pool: bool,
}

impl Account {
    fn new(max: usize, pool: Option<ThreadPool>, own_pool: bool) -> Account {
        Account {
            max,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            denied: AtomicBool::new(false),
            threads: AtomicUsize::new(1),
            peak_threads: AtomicUsize::new(1),
            spawned: AtomicUsize::new(0),
            pool,
            own_pool,
        }
    }

    /// The pool this call runs its parallel passes on; none means the call runs on its own thread.
    pub(crate) fn pool(&self) -> Option<&ThreadPool> {
        self.pool.as_ref()
    }

    /// `n` threads, or fewer if the pool cannot supply them: 1 with no pool. A caller's or the
    /// shared pool of more than 2 workers gives one fewer than its workers, so a pass never needs
    /// every CPU of a pool sized to them (bench: all of them made default-thread decodes of 4
    /// Mpixel images up to twice as slow on a shared machine); a pool of 1 or 2 workers gives
    /// them all, so a 2-worker pool and a 2-CPU host still decode in parallel. A pool built for
    /// the call gives its workers and the caller.
    pub(crate) fn cap_threads(&self, n: usize) -> usize {
        match &self.pool {
            None => 1,
            Some(p) if self.own_pool => n.min(p.threads() + 1),
            Some(p) if p.threads() <= 2 => n.min(p.threads()),
            Some(p) => n.min(p.threads() - 1),
        }
    }

    /// Takes `n` bytes of the ledger, or fails without taking any.
    pub fn charge(&self, n: usize) -> Result<()> {
        let mut cur = self.used.load(Ordering::Relaxed);
        loop {
            let next = match cur.checked_add(n) {
                Some(u) if u <= self.max => u,
                _ => {
                    self.denied.store(true, Ordering::Relaxed);
                    return Err(limit!(
                        "decode needs more than the {} bytes of working memory the limit allows",
                        self.max
                    ));
                }
            };
            match self
                .used
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::Relaxed);
                    return Ok(());
                }
                Err(c) => cur = c,
            }
        }
    }

    /// Gives `n` bytes back.
    pub fn release(&self, n: usize) {
        let _ = self
            .used
            .try_update(Ordering::AcqRel, Ordering::Relaxed, |u| {
                Some(u.saturating_sub(n))
            });
    }

    /// Bytes still free under the limit.
    pub fn remaining(&self) -> usize {
        self.max.saturating_sub(self.used())
    }

    /// The most bytes charged at once so far.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Whether any charge has been refused for want of room, optional ones included.
    pub(crate) fn denied(&self) -> bool {
        self.denied.load(Ordering::Relaxed)
    }

    /// Bytes charged now.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Counts a thread the call is about to start. The one place threads are counted: see
    /// `parallel::drive`.
    pub(crate) fn thread_started(&self) {
        let t = self.threads.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak_threads.fetch_max(t, Ordering::Relaxed);
        self.spawned.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn thread_ended(&self) {
        self.threads.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One thread's handle on the call's ledger: it remembers what it charged, so it can give that
/// back (`release_to`, or on drop). Threads of one call each hold their own handle
/// (`child`), over the same ledger.
pub struct Budget {
    acct: Arc<Account>,
    held: usize,
    root: bool,
    /// The inverse DWT's scratch, kept between its calls (see `DwtScratch`).
    pub(crate) dwt: crate::dsp::DwtScratch,
}

impl Budget {
    /// A new ledger for one call under `l`.
    pub fn new(l: &Limits) -> Self {
        Self::with_pool(l, None)
    }

    /// A new ledger for one call under `l`, whose parallel passes run on `pool` if there is one
    /// (none: the call stays on its own thread).
    pub fn with_pool(l: &Limits, pool: Option<ThreadPool>) -> Self {
        Budget {
            acct: Arc::new(Account::new(l.max_alloc, pool, false)),
            held: 0,
            root: true,
            dwt: Default::default(),
        }
    }

    /// A new ledger for a call under `opts` on `input_len` bytes of input, with the pool the call
    /// runs on (see `parallel::pool_for`): made once here, so every stage shares it. A `Decoder`
    /// keeps the last call's ledger in `slot`; it is reset in place when nothing else holds it.
    pub(crate) fn for_call_in(
        opts: &Options,
        input_len: usize,
        slot: &mut Option<Arc<Account>>,
    ) -> Self {
        let (pool, own) = pool_for(opts, input_len);
        let fresh = Account::new(opts.limits.max_alloc, pool, own);
        let acct = match slot.take() {
            Some(mut a) => match Arc::get_mut(&mut a) {
                Some(m) => {
                    *m = fresh;
                    a
                }
                None => Arc::new(fresh),
            },
            None => Arc::new(fresh),
        };
        Budget {
            acct,
            held: 0,
            root: true,
            dwt: Default::default(),
        }
    }

    /// Another handle on the same ledger, holding nothing.
    pub fn child(&self) -> Budget {
        Budget {
            acct: Arc::clone(&self.acct),
            held: 0,
            root: false,
            dwt: Default::default(),
        }
    }

    /// The shared ledger.
    pub fn account(&self) -> &Account {
        &self.acct
    }

    /// The ledger, to keep while this handle is borrowed mutably.
    pub fn shared(&self) -> Arc<Account> {
        Arc::clone(&self.acct)
    }

    /// Bytes this handle holds.
    pub fn held(&self) -> usize {
        self.held
    }

    /// The most bytes the whole call has held at once so far.
    pub fn peak(&self) -> usize {
        self.acct.peak()
    }

    pub fn charge(&mut self, n: usize) -> Result<()> {
        self.acct.charge(n)?;
        self.held += n;
        Ok(())
    }

    pub fn release(&mut self, n: usize) {
        let n = n.min(self.held);
        self.held -= n;
        self.acct.release(n);
    }

    /// Releases what this handle charged since it held `mark` bytes.
    pub fn release_to(&mut self, mark: usize) {
        if self.held > mark {
            self.release(self.held - mark);
        }
    }

    /// `n` bytes taken now and given back when the guard drops, for memory a thread holds for a
    /// while (it does not touch this handle's own count, so any thread may call it).
    pub fn guard(&self, n: usize) -> Result<Charge> {
        self.acct.charge(n)?;
        Ok(Charge {
            acct: Arc::clone(&self.acct),
            n,
        })
    }

    /// `vec(n, T::default())`, charged the same, in the buffer `kept` gives after the charge
    /// (a kept one, or an empty one to allocate).
    pub(crate) fn take_vec<T: Clone + Default>(
        &mut self,
        n: usize,
        kept: impl FnOnce() -> Vec<T>,
    ) -> Result<Vec<T>> {
        self.charge(bytes_of::<T>(n)?)?;
        reuse(kept(), n)
    }

    /// `n` zero elements, charged first, in the buffer `kept` gives after the charge when it has
    /// one. Else from a zeroed allocation: the pages are not written here, so a big output costs
    /// its page faults where its rows are written, on every lane, and not all before the first
    /// lane starts. Probed with `try_reserve_exact` first, as `vec` allocates, so a refused
    /// allocation is `Error::OutOfMemory`.
    pub(crate) fn take_zeroed<T: Clone + Default>(
        &mut self,
        n: usize,
        kept: impl FnOnce() -> Vec<T>,
    ) -> Result<Vec<T>> {
        self.charge(bytes_of::<T>(n)?)?;
        let v = kept();
        if v.capacity() == 0 {
            Vec::<T>::new()
                .try_reserve_exact(n)
                .map_err(|_| Error::OutOfMemory)?;
            return Ok(vec![T::default(); n]);
        }
        reuse(v, n)
    }

    /// A vector of `n` copies of `v`, charged first and allocated with `try_reserve_exact`.
    pub fn vec<T: Clone>(&mut self, n: usize, v: T) -> Result<Vec<T>> {
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| limit!("allocation size overflows"))?;
        self.charge(bytes)?;
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        out.resize(n, v);
        Ok(out)
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        self.acct.release(self.held);
        if self.root {
            crate::stats::record(&self.acct);
        }
    }
}

/// Bytes of the ledger held until the guard drops.
pub struct Charge {
    acct: Arc<Account>,
    n: usize,
}

/// The bytes of `n` elements of `T`, or a limit error when that overflows.
fn bytes_of<T>(n: usize) -> Result<usize> {
    n.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| limit!("allocation size overflows"))
}

/// The heap `v` holds: its capacity, not its length.
pub(crate) fn held_by<T>(v: &Vec<T>) -> usize {
    v.capacity() * std::mem::size_of::<T>()
}

/// `v` emptied and refilled with `n` default (zero) elements, as a new vector of them would be:
/// it allocates only when `v` has less room.
pub(crate) fn reuse<T: Clone + Default>(mut v: Vec<T>, n: usize) -> Result<Vec<T>> {
    v.clear();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, T::default());
    Ok(v)
}

/// Buffers a `Decoder` keeps between uses, uncharged: a call charges a buffer as new and then
/// takes one from here (`pick`) instead of allocating.
#[derive(Default)]
pub(crate) struct Pool<T>(pub(crate) Vec<Vec<T>>);

impl<T> Pool<T> {
    /// The kept buffer that fits `n` elements best (the smallest that holds them, else the
    /// largest), or an empty one.
    pub(crate) fn pick(&mut self, n: usize) -> Vec<T> {
        let mut best: Option<usize> = None;
        for (k, v) in self.0.iter().enumerate() {
            let better = match best {
                None => true,
                Some(b) => {
                    let (c, bc) = (v.capacity(), self.0[b].capacity());
                    if bc >= n { c >= n && c < bc } else { c > bc }
                }
            };
            if better {
                best = Some(k);
            }
        }
        best.map_or_else(Vec::new, |k| self.0.swap_remove(k))
    }

    /// Keeps `v` for a later `pick` when `keep` is set, else frees it.
    pub(crate) fn give(&mut self, v: Vec<T>, keep: bool) {
        if keep && v.capacity() > 0 {
            self.0.push(v);
        }
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.acct.release(self.n);
    }
}

pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
