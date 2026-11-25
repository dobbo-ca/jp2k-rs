package decoder

import (
	"container/heap"
	"container/list"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/dobbo-ca/jp2k-go/internal/codestream"
	"github.com/dobbo-ca/jp2k-go/internal/dsp"
	"github.com/dobbo-ca/jp2k-go/internal/tier1"
	"github.com/dobbo-ca/jp2k-go/internal/tier2"
)

// budget admits decodes: at most slotLimit run at once, and together
// they reserve at most the limit in bytes of estimated heap.
//
// The count slots keep heap bounded however many goroutines call Decode:
// the GC lets the heap reach about twice the live set, so unbounded
// callers meant unbounded memory. A fifth of GOMAXPROCS keeps the cores
// busy while tier-2 and other serial parts overlap; a quarter was about
// 5% faster at 14 procs but pushed RSS past 560 MB. 2 at the least for
// the same reason, except at 1 or 2 procs, where a second decode gains
// little and doubles the heap. The limit is read on every acquire so it
// follows GOMAXPROCS changes.
//
// Bytes a decode gives back stay counted, as garbage, until a GC cycle
// that started after has ended: without a runtime memory limit the GC
// lets the heap grow to twice the live data, so the next decode would
// otherwise stack its heap on the last one's.
//
// A decode estimated under smallDecode may take a slot beyond that
// count, up to GOMAXPROCS, as long as no larger decode is waiting for
// one (bigWait): small tiles then keep every core busy without letting
// a burst of them starve a large decode of its slot.
var budget struct {
	mu       sync.Mutex
	explicit bool  // limit was set by SetMemoryLimit
	limit    int64 // SetMemoryLimit's, math.MaxInt64 for none
	used     int64
	parked   int64            // see ticket
	waiting  map[*ticket]bool // parked tickets
	pre      int64            // held or parked by decodes done reading, without a slot
	readers  []*ticket        // decodes reading, oldest first
	queue    ticketQueue      // decodes waiting for first bytes
	slotq    list.List        // waiting only for a slot
	bigWait  int              // big (non-small) tickets in slotq
	tickets  uint64           // counts first reservations, for ticket.seq
	n        int              // slots taken
	freed    chan struct{}    // closed on release, so waiters can also watch a ctx
	garbage  []released
	seq      uint64 // counts releases
	cycles   []metrics.Sample
	kept     int64         // what open Images keep, see keep
	unkept   chan struct{} // closed when unkeep gives some back
}

// smallDecode is the estimate floor below which a decode may take a
// slot beyond slotLimit. It is on the floor, not the estimate, so a
// host process with a big live heap (a reader's own caches) does not
// make every decode look big.
const smallDecode = 24 << 20

// released is n bytes given back when cycle GC cycles had ended.
type released struct {
	cycle, seq uint64
	n          int64
}

func slotLimit() int {
	p := runtime.GOMAXPROCS(0)
	return max(min(2, p-1), (p+4)/5)
}

// SetMemoryLimit implements jp2k.SetMemoryLimit.
func SetMemoryLimit(n int64) int64 {
	if n <= 0 {
		n = math.MaxInt64
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	prev := limitNow()
	budget.explicit, budget.limit = true, n
	wakeWaiters()
	for u := range budget.waiting {
		u.wakeOwn()
	}
	return prev
}

// ErrMemoryLimit is jp2k.ErrMemoryLimit.
var ErrMemoryLimit = errors.New("jp2k: memory limit held by other decodes (transient, retry later)")

// hostHeapError is ErrMemoryLimit without its text.
type hostHeapError struct{}

func (hostHeapError) Error() string {
	return "jp2k: memory held elsewhere in the process fills the memory limit (retry later)"
}
func (hostHeapError) Unwrap() error { return ErrMemoryLimit }

// limitNow is the byte limit; budget.mu held.
func limitNow() int64 {
	if budget.explicit {
		return budget.limit
	}
	return defaultLimit("")
}

// admitLimit is what decodes may reserve: the limit less what open
// Images keep; budget.mu held.
func admitLimit() int64 {
	l := limitNow()
	if l == math.MaxInt64 {
		return l
	}
	return max(0, l-budget.kept)
}

// keep charges n bytes an Image keeps, and reports whether they fit:
// open Images keep at most a quarter of the limit between them.
func keep(n int64) bool {
	budget.mu.Lock()
	defer budget.mu.Unlock()
	if l := limitNow(); l < math.MaxInt64 && n > l/4-budget.kept {
		return false
	}
	budget.kept += n
	return true
}

// keepShare is the most open Images keep between them.
func keepShare() int64 {
	budget.mu.Lock()
	defer budget.mu.Unlock()
	if l := limitNow(); l < math.MaxInt64 {
		return l / 4
	}
	return math.MaxInt64
}

// unkeep gives back n bytes keep charged. A finalizer calls it once
// the GC has freed them: they are not garbage. Evicted ones are.
func unkeep(n int64, evicted bool) {
	budget.mu.Lock()
	budget.kept -= n
	if evicted && n > 0 {
		budget.seq++
		budget.garbage = append(budget.garbage, released{gcCycles(), budget.seq, n})
	}
	if budget.unkept != nil {
		close(budget.unkept)
		budget.unkept = nil
	}
	wakeWaiters()
	budget.mu.Unlock()
}

// reclaimAt is when reclaimKept last ran a GC, in ns.
var reclaimAt atomic.Int64

// keptQuiet is how long reclaimKept waits for another finalizer.
const keptQuiet = 10 * time.Millisecond

// reclaimKept runs a GC, so Images dropped since the last give back
// what they keep, and waits while their finalizers run. It runs one at
// most every ten GC costs.
func reclaimKept() {
	budget.mu.Lock()
	k, seq := budget.kept, budget.seq
	budget.mu.Unlock()
	now := time.Now().UnixNano()
	if k == 0 || now-reclaimAt.Load() < 10*forceGC.cost.Load() {
		return
	}
	reclaimAt.Store(now)
	collectGarbage(seq)
	for {
		budget.mu.Lock()
		if budget.kept == 0 {
			budget.mu.Unlock()
			return
		}
		if budget.unkept == nil {
			budget.unkept = make(chan struct{})
		}
		ch := budget.unkept
		budget.mu.Unlock()
		select {
		case <-ch:
		case <-time.After(keptQuiet):
			return
		}
	}
}

// checkSized fails doomed sized reads early.
// raw, if set, detects bare codestreams.
func checkSized(n int64, raw func() bool) error {
	budget.mu.Lock()
	limit := limitNow()
	budget.mu.Unlock()
	// The estimate scales only codestream bytes.
	if n+1<<20 > limit || raw != nil && float64(n+1<<20)*1.2 > float64(limit) && raw() {
		return errors.New("jp2k: decode needs more than the memory limit")
	}
	return nil
}

// cgroupMax is the cgroup memory limit, read once: a file read per
// decode costs more than a small decode.
var cgroupMax = sync.OnceValue(func() int64 { return cgroupMaxFrom(cgroupRoot(), "/sys/fs/cgroup") })

// cgroupMaxFrom falls back on limitless roots.
func cgroupMaxFrom(root, fallback string) int64 {
	if l := cgroupLimit(root); l > 0 {
		return l
	}
	return cgroupLimit(fallback)
}

// cgroupRoot avoids shared-namespace host limits.
func cgroupRoot() string {
	if b, err := os.ReadFile("/proc/self/cgroup"); err == nil {
		if p, ok := cgroupV2Path(b); ok {
			return filepath.Join("/sys/fs/cgroup", p)
		}
	}
	return "/sys/fs/cgroup"
}

// cgroupV2Path reads the "0::" line's path.
func cgroupV2Path(procSelfCgroup []byte) (string, bool) {
	for _, line := range strings.Split(string(procSelfCgroup), "\n") {
		if p, ok := strings.CutPrefix(line, "0::"); ok {
			return p, true
		}
	}
	return "", false
}

// defaultLimit is the limit SetMemoryLimit's doc gives, with the cgroup
// files under cgroupRoot, or cgroupMax's when it is "".
func defaultLimit(cgroupRoot string) int64 {
	if l := debug.SetMemoryLimit(-1); l < math.MaxInt64 {
		return max(1, l/2)
	}
	if runtime.GOOS == "linux" {
		l := cgroupMax()
		if cgroupRoot != "" {
			l = cgroupLimit(cgroupRoot)
		}
		if l > 0 {
			return max(1, l/2)
		}
	}
	return math.MaxInt64
}

// cgroupLimit reads the cgroup v2 or v1 memory limit under root, or 0.
func cgroupLimit(root string) int64 {
	for _, f := range []string{"memory.max", "memory/memory.limit_in_bytes"} {
		b, err := os.ReadFile(filepath.Join(root, f))
		if err != nil {
			continue
		}
		// v2 says "max" for none, v1 a number near 2^63.
		n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		if err == nil && n > 0 && n < 1<<50 {
			return n
		}
	}
	return 0
}

// wakeWaiters wakes broadcast and parked waiters.
func wakeWaiters() {
	if budget.freed != nil {
		close(budget.freed)
		budget.freed = nil
	}
	wakeParked()
}

// wakeParked shares free bytes out; budget.mu held.
func wakeParked() {
	if len(budget.waiting) == 0 {
		return
	}
	limit := admitLimit()
	free := limit - budget.used - budget.parked
	left, r := free, laterReads()
	rd := r // all readers' bytes
	if len(budget.readers) > 0 {
		rd += budget.readers[0].held + budget.readers[0].parked
	}
	var oldest, newest *ticket
	for u := range budget.waiting {
		if oldest == nil || u.seq < oldest.seq {
			oldest = u
		}
		if newest == nil || u.seq > newest.seq {
			newest = u
		}
	}
	for u := range budget.waiting {
		switch {
		case u.slotEl != nil: // wakeSlot's
		case u.wantN <= left && (u.wantSlot || u == oldest || !u.overCap(u.wantN, limit, r)):
			u.wakeOwn()
			if !u.wantSlot {
				left -= u.wantN
			}
		case !u.inRead && len(budget.readers) > 0 && u.wantN-rd <= free:
			u.wakeOwn() // stalled() may fail it
		}
	}
	if budget.used == 0 {
		newest.wakeOwn() // it may have to give way
	}
}

// gcCycles is how many GC cycles have ended; budget.mu held.
func gcCycles() uint64 {
	if budget.cycles == nil {
		budget.cycles = []metrics.Sample{{Name: "/gc/cycles/total:gc-cycles"}}
	}
	metrics.Read(budget.cycles)
	return budget.cycles[0].Value.Uint64()
}

// garbageNow drops the garbage a GC has collected, that is, released
// before a cycle that has since ended began, or at or before seq, and
// returns the rest; budget.mu held.
func garbageNow(seq uint64) int64 {
	c := gcCycles()
	k, g := 0, int64(0)
	for _, r := range budget.garbage {
		if r.cycle+2 <= c || r.seq <= seq {
			continue
		}
		budget.garbage[k] = r
		k++
		g += r.n
	}
	budget.garbage = budget.garbage[:k]
	return g
}

var forceGC struct {
	mu       sync.Mutex
	done     chan struct{} // non-nil mid-run
	startSeq uint64        // budget.seq when the run began
	cost     atomic.Int64  // last run's duration, ns
}

// collectGarbage shares one forced GC run.
func collectGarbage(since uint64) {
	forceGC.mu.Lock()
	for forceGC.done != nil && forceGC.startSeq < since {
		done := forceGC.done
		forceGC.mu.Unlock()
		<-done
		forceGC.mu.Lock()
	}
	if forceGC.done != nil {
		done := forceGC.done
		forceGC.mu.Unlock()
		<-done
		return
	}
	budget.mu.Lock()
	forceGC.startSeq = budget.seq
	budget.mu.Unlock()
	done := make(chan struct{})
	forceGC.done = done
	forceGC.mu.Unlock()
	start := time.Now()
	runtime.GC()
	forceGC.cost.Store(int64(time.Since(start)))
	forceGC.mu.Lock()
	forceGC.done = nil
	forceGC.mu.Unlock()
	close(done)
}

// A ticket is one decode's hold on the budget.
//
// A decode reads its input and parses it with no global lock held,
// reserving as it goes. Its first reservation may wait, as it holds
// nothing yet. Later ones may not wait holding bytes, as two decodes
// could each wait for the other's: a decode that cannot grow parks
// instead, giving its reservation back (the bytes stay in use, counted
// in budget.parked), and waits to take it back with the rest at once.
// Parked bytes count as taken. When nothing but parked bytes is
// reserved and no parked decode fits, the newest parked decode (by seq)
// fails with ErrMemoryLimit, so the oldest always gets to go on.
// To keep that rare, decodes done reading but not yet running may hold
// no more than a quarter of the limit between them, unless one is alone
// or the oldest parked; that also keeps the unreserved garbage of header
// parses near what one at a time left. Decodes reading, other than from
// memory, share that quarter with them, unless one is the oldest reader
// or the oldest parked.
//
// A read may take any time, and a reader holds what it reserved for it
// until the read returns. So a decode that would go ahead but for the
// bytes of reads that have run for stallTime (and, once one has, of
// readers parked), fails at once with ErrMemoryLimit rather than wait on
// them, unless it is itself reading: that would throw away what it read.
// A decode waiting for its first bytes holds back those that come after
// it, so a big one is not starved by a stream of small ones.
type ticket struct {
	ctx    context.Context
	seq    uint64 // order of first reservation, 0 before
	held   int64  // bytes reserved
	parked int64  // bytes given back while waiting to grow
	pre    int64  // held or parked since reading ended, until running
	parse  bool   // done reading
	slot   bool
	small  bool      // estimate under smallDecode, set once before entering slotq
	inRead bool      // in budget.readers
	since  time.Time // when its latest read began, while reading
	// What a parked t waits for.
	wantN    int64
	wantSlot bool
	// queue position, see ticketQueue.
	inQueue bool
	qidx    int
	own     chan struct{} // wakes this ticket alone
	slotEl  *list.Element // in budget.slotq
}

// stallTime is how long a read runs before decodes waiting for the
// memory it holds give up: long for a local read, short for a request.
const stallTime = 10 * time.Millisecond

// newTicket starts a decode's hold on the budget, unless ctx is done.
func newTicket(ctx context.Context) (*ticket, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("jp2k: waiting to decode: %w", err)
	}
	return &ticket{ctx: ctx}, nil
}

// grow reserves n more bytes.
func (t *ticket) grow(n int64) error {
	return t.take(n, false, n)
}

// growParse reserves n more bytes for a decode done reading, or whose
// reader is in memory, so cannot stall: for its input or header parse.
func (t *ticket) growParse(n int64) error {
	t.parse = true
	return t.take(n, false, n)
}

// run tops the reservation up to need and takes a slot.
// floor excludes otherLive.
func (t *ticket) run(need, floor int64) error {
	if runHook != nil {
		runHook(need)
	}
	return t.take(max(need-t.held, 0), true, max(floor-t.held, 0))
}

// runHook, when set by a test, sees each need run gets.
var runHook func(need int64)

// take reserves n bytes, and a slot too if slot is set, waiting until
// they are free or t.ctx is done.
// floor excludes otherLive: retryable errors.
func (t *ticket) take(n int64, slot bool, floor int64) error {
	// Also stops a decode reading a slow stream.
	if err := t.ctx.Err(); err != nil {
		return fmt.Errorf("jp2k: waiting to decode: %w", err)
	}
	budget.mu.Lock()
	if t.seq == 0 {
		budget.tickets++
		t.seq = budget.tickets
	}
	if t.parse || slot {
		t.leaveReading()
	}
	if slot {
		t.small = t.held+floor <= smallDecode
	}
	entrant := t.held == 0
	var collected uint64 // releases up to this one a runtime.GC collected
	var wait time.Duration
	var gcWait time.Time // when garbage alone first blocked
	reclaimed := false
	for {
		wait = 0
		limit := admitLimit()
		// limit-held, not held+n: that can overflow.
		if n > limit-t.held-t.parked {
			full := limitNow()
			if !reclaimed && budget.kept > 0 && n <= full-t.held-t.parked {
				// Dropped Images or indexes hold it.
				reclaimed = true
				budget.mu.Unlock()
				evictIndexes()
				reclaimKept()
				budget.mu.Lock()
				continue
			}
			t.unqueue()
			t.slotLeave()
			budget.mu.Unlock()
			switch {
			case floor <= limit-t.held-t.parked:
				return hostHeapError{}
			case floor <= full-t.held-t.parked:
				return fmt.Errorf("%w: open Images hold the part of it this decode needs", ErrMemoryLimit)
			}
			return fmt.Errorf("jp2k: decode needs more than the memory limit")
		}
		others := budget.parked - t.parked
		need := t.parked + n
		free := limit - budget.used - others
		capped := t.capped(n, slot, limit, 0)
		haveSlot := !slot || slotFree(t.small)
		queued := entrant && t.queued()
		slotOnly := false
		if need <= free && !capped && !queued {
			g := garbageNow(collected)
			if need <= free-g {
				if haveSlot {
					break
				}
				slotOnly = true
			} else {
				if entrant {
					t.enqueue()
				}
				if gcWait.IsZero() {
					gcWait = time.Now()
				}
				// Batching garbage amortises costly GCs.
				if d := time.Until(gcWait.Add(time.Duration(forceGC.cost.Load()))); budget.n > 0 && d > 0 {
					wait = d
				} else {
					// Force GC; a waiter never allocates.
					seq := budget.seq
					budget.mu.Unlock()
					collectGarbage(seq)
					budget.mu.Lock()
					collected, gcWait = seq, time.Time{}
					continue
				}
			}
		} else if t.held > 0 {
			// Park rather than wait holding bytes.
			budget.used -= t.held
			budget.parked += t.held
			t.parked += t.held
			t.held = 0
			if budget.waiting == nil {
				budget.waiting = map[*ticket]bool{}
			}
			budget.waiting[t] = true
			t.wantN, t.wantSlot = n, slot
			wakeWaiters()
			continue
		} else if budget.used == 0 && t.parked > 0 && t.newest() && !otherFits(t, limit) {
			// All that is reserved waits for more: give way.
			t.giveBack()
			budget.mu.Unlock()
			return fmt.Errorf("%w: others waiting hold the memory limit", ErrMemoryLimit)
		}
		// Capped parses queue; reads use stalled().
		if entrant && (need > free || capped && t.parse || queued) {
			t.enqueue()
		}
		// queued too: held-back entrants fail fast.
		if !t.inRead && haveSlot && (need > free || capped || queued) {
			stalled, inCap, next := t.stalled()
			if stalled > 0 && need <= free+stalled && !t.capped(n, slot, limit, inCap) {
				t.giveBack()
				budget.mu.Unlock()
				return fmt.Errorf("%w: slow reads hold the memory limit", ErrMemoryLimit)
			}
			wait = next
		}
		if budget.freed == nil {
			budget.freed = make(chan struct{})
		}
		freed := budget.freed
		var own chan struct{}
		if slotOnly {
			t.slotWait()
		} else {
			t.slotLeave()
		}
		if queued && t.inQueue || slotOnly || t.parked > 0 {
			// Woken alone, see wakeParked.
			if t.own == nil {
				t.own = make(chan struct{})
			}
			own, freed = t.own, nil
		}
		budget.mu.Unlock()
		var stall <-chan time.Time
		if wait > 0 {
			stall = time.After(wait)
		}
		select {
		case <-freed:
		case <-own:
		case <-stall:
		case <-t.ctx.Done():
			budget.mu.Lock()
			t.unqueue()
			t.slotLeave()
			budget.mu.Unlock()
			return fmt.Errorf("jp2k: waiting to decode: %w", t.ctx.Err())
		}
		budget.mu.Lock()
	}
	wake := budget.waiting[t] // another parked decode may now be the oldest
	delete(budget.waiting, t)
	if !t.parse && !slot {
		if !t.inRead {
			t.inRead = true
			budget.readers = append(budget.readers, t)
		}
		t.since = time.Now()
	}
	if slot && t.pre > 0 {
		budget.pre -= t.pre
		t.pre = 0
		wake = true // a capped parse may now fit
	} else if t.parse && !slot {
		add := n
		if t.pre == 0 {
			add += t.held + t.parked
		}
		budget.pre += add
		t.pre += add
	}
	budget.used += t.parked + n
	budget.parked -= t.parked
	t.held += t.parked + n
	t.parked = 0
	if slot {
		budget.n++
		t.slot = true
	}
	t.slotLeave()
	t.unqueue() // after accounting: wakes see it
	if wake {
		wakeWaiters()
	}
	budget.mu.Unlock()
	return nil
}

// enqueue and unqueue add t to and take it from budget.queue; budget.mu
// held.
func (t *ticket) enqueue() {
	if !t.inQueue {
		t.inQueue = true
		heap.Push(&budget.queue, t)
	}
}

func (t *ticket) unqueue() {
	if t.inQueue {
		head := t.qidx == 0
		t.inQueue = false
		heap.Remove(&budget.queue, t.qidx)
		if head && len(budget.queue) > 0 {
			budget.queue[0].wakeOwn() // only the new head may go
		}
		wakeWaiters()
	}
}

// budget.slotq membership; budget.mu held.
func (t *ticket) slotWait() {
	if t.slotEl == nil {
		t.slotEl = budget.slotq.PushBack(t)
		if !t.small {
			budget.bigWait++
		}
	}
}

func (t *ticket) slotLeave() {
	if t.slotEl != nil {
		budget.slotq.Remove(t.slotEl)
		t.slotEl = nil
		if !t.small {
			budget.bigWait--
		}
		wakeSlot() // pass a free slot on
	}
}

// wakeSlot wakes the oldest; budget.mu held.
func wakeSlot() {
	if e := budget.slotq.Front(); e != nil && slotFree(e.Value.(*ticket).small) {
		e.Value.(*ticket).wakeOwn()
	}
}

// slotFree reports whether a slot is free for a decode of the given
// size: within slotLimit, or small, with no larger decode waiting, and
// within GOMAXPROCS. budget.mu held.
func slotFree(small bool) bool {
	return budget.n < slotLimit() || small && budget.bigWait == 0 && budget.n < runtime.GOMAXPROCS(0)
}

// wakeOwn wakes t alone; budget.mu held.
func (t *ticket) wakeOwn() {
	if t.own != nil {
		close(t.own)
		t.own = nil
	}
}

// queued reports whether an older decode waits for its first bytes;
// budget.mu held.
func (t *ticket) queued() bool {
	return len(budget.queue) > 0 && budget.queue[0].seq < t.seq
}

type ticketQueue []*ticket

func (q ticketQueue) Len() int           { return len(q) }
func (q ticketQueue) Less(i, j int) bool { return q[i].seq < q[j].seq }
func (q ticketQueue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
	q[i].qidx, q[j].qidx = i, j
}

func (q *ticketQueue) Push(x any) {
	t := x.(*ticket)
	t.qidx = len(*q)
	*q = append(*q, t)
}

func (q *ticketQueue) Pop() any {
	old := *q
	n := len(old)
	t := old[n-1]
	old[n-1] = nil
	*q = old[:n-1]
	return t
}

// stalled returns the bytes held by readers other than t whose read has
// run for stallTime, and, if there are any, by readers parked; how many
// of those the read cap counts; and how long until the next read in
// flight stalls, 0 for none. budget.mu held.
func (t *ticket) stalled() (all, inCap int64, next time.Duration) {
	now := time.Now()
	slow := func(u *ticket) bool { return u.held > 0 && now.Sub(u.since) >= stallTime }
	some := false
	for _, u := range budget.readers {
		if u == t {
			continue
		}
		if slow(u) {
			some = true
		} else if u.held > 0 {
			d := u.since.Add(stallTime).Sub(now)
			if next == 0 || d < next {
				next = d
			}
		}
	}
	if !some {
		return 0, 0, next
	}
	for i, u := range budget.readers {
		if u == t || !slow(u) && u.parked == 0 {
			continue
		}
		all += u.held + u.parked
		if i > 0 {
			inCap += u.held + u.parked
		}
	}
	return all, inCap, next
}

// drop gives back n of the bytes t holds, as garbage, which they are.
func (t *ticket) drop(n int64) {
	budget.mu.Lock()
	n = min(n, t.held)
	t.held -= n
	budget.used -= n
	if p := min(n, t.pre); p > 0 {
		t.pre -= p
		budget.pre -= p
	}
	if n > 0 {
		budget.seq++
		budget.garbage = append(budget.garbage, released{gcCycles(), budget.seq, n})
	}
	wakeWaiters()
	budget.mu.Unlock()
}

// done gives back all t holds.
func (t *ticket) done() {
	budget.mu.Lock()
	t.giveBack()
	budget.mu.Unlock()
	*t = ticket{}
}

// giveBack gives back all t holds; budget.mu held.
func (t *ticket) giveBack() {
	budget.used -= t.held
	budget.parked -= t.parked
	delete(budget.waiting, t)
	t.unqueue()
	t.slotLeave()
	t.leaveReading()
	budget.pre -= t.pre
	if t.slot {
		budget.n--
		wakeSlot()
	}
	if n := t.held + t.parked; n > 0 {
		budget.seq++
		budget.garbage = append(budget.garbage, released{gcCycles(), budget.seq, n})
	}
	t.held, t.parked, t.pre, t.slot = 0, 0, 0, false
	wakeWaiters()
}

// oldest and newest report whether t is the parked decode that began
// reserving first or last; budget.mu held.
func (t *ticket) oldest() bool {
	return t.parked > 0 && !t.otherParked(func(u *ticket) bool { return u.seq < t.seq })
}

func (t *ticket) newest() bool {
	return t.parked > 0 && !t.otherParked(func(u *ticket) bool { return u.seq > t.seq })
}

// otherParked reports whether f holds for a parked decode other than t;
// budget.mu held.
func (t *ticket) otherParked(f func(*ticket) bool) bool {
	for u := range budget.waiting {
		if u != t && f(u) {
			return true
		}
	}
	return false
}

// leaveReading takes t out of budget.readers; budget.mu held.
func (t *ticket) leaveReading() {
	if !t.inRead {
		return
	}
	t.inRead = false
	budget.readers = slices.DeleteFunc(budget.readers, func(u *ticket) bool { return u == t })
	wakeWaiters() // a capped reader may now fit
}

// capped reports whether the read or parse cap holds t back from n more
// bytes, not counting skip bytes of readers it counts; budget.mu held.
func (t *ticket) capped(n int64, slot bool, limit, skip int64) bool {
	return !slot && !t.oldest() && t.overCap(n, limit, laterReads()-skip)
}

// laterReads skips the oldest reader; budget.mu held.
func laterReads() (r int64) {
	for _, u := range budget.readers[min(1, len(budget.readers)):] {
		r += u.held + u.parked
	}
	return r
}

// overCap skips capped's slot, oldest checks.
func (t *ticket) overCap(n, limit, r int64) bool {
	if !t.parse {
		return len(budget.readers) > 0 && budget.readers[0] != t && n > limit/4-r-budget.pre
	}
	if t.pre == 0 {
		n += t.held + t.parked // entering: its input counts too
	}
	return budget.pre != t.pre && n > limit/4-budget.pre
}

// otherFits reports whether a parked decode other than t could go
// ahead within the limit; budget.mu held.
func otherFits(t *ticket, limit int64) bool {
	r := laterReads()
	for u := range budget.waiting {
		if u != t && u.parked+u.wantN <= limit-(budget.parked-u.parked) && (u.wantSlot || u.oldest() || !u.overCap(u.wantN, limit, r)) {
			return true
		}
	}
	return false
}

// acquire takes a slot and need bytes, as a running decode holds them.
func acquire(ctx context.Context, need int64) error {
	return (&ticket{ctx: ctx}).take(need, true, need)
}

// release gives back what acquire took.
func release(need int64) {
	(&ticket{held: need, slot: true}).done()
}

// Unit costs, in bytes of heap, of what a tile allocates per precinct,
// code-block and so on, measured on this implementation and rounded up.
const (
	costPrecinct   = int64(unsafe.Sizeof(tier2.Precinct{}))
	costCodeBlock  = int64(unsafe.Sizeof(tier2.CodeblockState{})) + 48 // plus two tag-tree leaves and their parents
	costPacket     = int64(unsafe.Sizeof(tier2.PacketRef{}) + unsafe.Sizeof(tier2.PacketIndex{}))
	costBlockRef   = 2 * int64(unsafe.Sizeof(tier2.CodeBlockRef{})) // append growth
	costSegment    = 2 * int64(unsafe.Sizeof(tier2.CodeBlockSegment{})+40)
	costGroup      = 400 // map entry, cbGroup, order and job
	costTile       = 64 << 10
	costT1Worker   = int64(unsafe.Sizeof(t1Worker{})) + 16<<10 // plus its flags
	costDecodeBase = 1 << 20
	costProc       = 256 << 10 // spans each P caches for the size classes in use
)

// estimateDecode is an upper estimate of the heap a decode of reg holds
// at its peak, from the headers alone: input and its parse, the out
// bytes of output, subsampled canvases unless planar (each component
// then its own plane), and for each tile lane its working planes,
// scratch and one tile's tier-2 state, plus the garbage of the tiles
// already done that the GC may not have collected yet (at most the live
// heap, at the default GOGC, and that is the whole program's, see
// otherLive). tiles are the tiles decoded, lanes how many run at once,
// procs the workers they share.
// plans, when not nil, are an indexed decode's (see Handle), one per
// tile: it holds no input or parse, and reads only what they say.
// floor excludes otherLive.
func estimateDecode(cs *codestream.Codestream, dataLen int, reg region, out int, planar bool, compWin []codestream.SubbandRect, tiles []uint16, lanes, procs int, full bool, plans []*tilePlan) (est, floor int64) {
	siz := cs.SIZ
	w, h := int64(reg.x1-reg.x0), int64(reg.y1-reg.y0)
	live := float64(costDecodeBase + int64(runtime.GOMAXPROCS(0))*costProc + int64(out))
	if plans == nil {
		// Input, its parse and the per-tile index of it.
		live += float64(dataLen) + float64(cs.ParseCost) +
			float64(128*float64(siz.TilesWide)*float64(siz.TilesHigh))
		if len(cs.PPM) > 0 {
			live += float64(dataLen) // the merged PPM stream
		}
		for _, tp := range cs.Parts() {
			for _, e := range tp.PPT {
				live += float64(len(e.Data)) // every tile's merged PPT, region or not
			}
		}
	}
	for c := range siz.Components {
		if !planar && (siz.Components[c].XRsiz != 1 || siz.Components[c].YRsiz != 1) {
			// Its canvas, then an upsampled plane, left as garbage.
			cw := compWin[c]
			live += float64(4*float64(cw.X1-cw.X0)*float64(cw.Y1-cw.Y0)) + float64(4*float64(w*h)) + float64(8*float64(w+h))
		}
	}

	var (
		maxPlane, maxTopW, maxTopH, maxTransient, sumTransient, maxTileBytes, readBytes, ranges float64
		multiPart, multiLayer, haveRev, haveIrr, mct                                            bool
		resBuf                                                                                  []tier2.Resolution
	)
	for i, t := range tiles {
		e := estimateTile(cs, t, reg, compWin, full, &resBuf)
		maxPlane, maxTopW, maxTopH = max(maxPlane, e.plane), max(maxTopW, e.topW), max(maxTopH, e.topH)
		maxTileBytes = max(maxTileBytes, e.tileBytes)
		multiPart, multiLayer = multiPart || e.multiPart, multiLayer || e.multiLayer
		haveRev, haveIrr, mct = haveRev || e.rev, haveIrr || e.irr, mct || e.mct
		transient := e.transient
		if plans != nil {
			transient = plans[i].cost()
			readBytes += float64(plans[i].bytes)
			ranges += float64(plans[i].pieces)
		}
		maxTransient = max(maxTransient, transient)
		sumTransient += transient
	}

	planes := 1.0
	if mct && len(siz.Components) >= 3 {
		planes = 3
	}
	types := 1.0
	if haveRev && haveIrr {
		types = 2
	}
	nw := float64(codestream.CeilDiv(procs, max(1, lanes)))
	cols := float64(dsp.ColBlock53)
	if haveIrr {
		cols = dsp.ColBlock97
	}
	lane := float64(planes*types*(float64(4*maxPlane)+float64(4*nw*max(maxTopW+8, cols*(maxTopH+8))))) + float64(planes*nw*float64(costT1Worker)) + float64(4*nw*maxTopW)
	if multiPart && plans == nil {
		lane += maxTileBytes
	}
	if plans != nil {
		// read before it runs
		live += readBytes + float64(ranges*float64(unsafe.Sizeof(ixRange{})))
	}
	if multiLayer {
		// tier-1 joins a code-block's layers
		if plans == nil {
			live += float64(2 * float64(dataLen))
		} else {
			live += float64(2 * readBytes)
		}
	}
	live += float64(float64(lanes) * (lane + maxTransient))
	if plans != nil {
		// a lane's planes regrown across tiles, as garbage
		sumTransient += float64(float64(lanes) * planes * types * 4 * maxPlane)
	}
	// A fifth for size classes, fragmentation and what the model misses.
	// The GC lets garbage grow to the live heap it last marked, which
	// includes what was allocated while it marked: a quarter more.
	live = float64(live * 1.2)
	other := float64(otherLive())
	grow := 1.25 * gogcFactor()
	e := live + min(sumTransient, float64(grow*live)+other)
	f := live + min(sumTransient, grow*live)
	return int64(min(e, 1<<62)), int64(min(f, 1<<62))
}

// tileCost is estimateTile's result.
type tileCost struct {
	plane, topW, topH float64 // largest working window, and top resolution size
	transient         float64 // the tile's tier-2 state
	tileBytes         float64
	cbs, segs         int64 // code-blocks, and segments tier-2 records, bounded by its bytes
	refs              int64 // code-block inclusions tier-2 records
	precincts         int64
	packets           int64
	multiPart         bool
	multiLayer        bool
	rev, irr, mct     bool
}

// estimateTile is estimateDecode's model of one tile t. resBuf is
// scratch.
func estimateTile(cs *codestream.Codestream, t uint16, reg region, compWin []codestream.SubbandRect, full bool, resBuf *[]tier2.Resolution) tileCost {
	siz := cs.SIZ
	numComps := len(siz.Components)
	ppmBytes := 0
	for _, e := range cs.PPM {
		ppmBytes += len(e.Data)
	}
	var e tileCost
	tileBytes, hdrBytes := 0, 0
	parts := codestream.TilePartIndex(cs, t)
	e.multiPart = len(parts) > 1
	for _, i := range parts {
		tp := &cs.Parts()[i]
		tileBytes += tp.DataLength
		for _, pe := range tp.PPT {
			hdrBytes += len(pe.Data)
		}
	}
	hdrBytes += ppmBytes
	e.tileBytes = float64(tileBytes)
	tileCOD, tileCOC, hasPOC := codestream.TileOverrides(cs, t)
	effCOD := cs.COD
	if tileCOD != nil {
		effCOD = tileCOD
	}
	layers := int64(effCOD.NumberOfLayers)
	e.multiLayer = layers > 1
	e.mct = effCOD.MultipleComponentTransform != 0

	p := int(t) % int(siz.TilesWide)
	q := int(t) / int(siz.TilesWide)
	tx0 := max(int(siz.XTOsiz)+p*int(siz.XTsiz), int(siz.XOsiz))
	tx1 := min(int(siz.XTOsiz)+(p+1)*int(siz.XTsiz), int(siz.Xsiz))
	ty0 := max(int(siz.YTOsiz)+q*int(siz.YTsiz), int(siz.YOsiz))
	ty1 := min(int(siz.YTOsiz)+(q+1)*int(siz.YTsiz), int(siz.Ysiz))
	touched := full || tileTouches(siz, t, compWin, reg.reduce)

	var precincts, precinctsPerLayer, cbsAll, cbsWanted int64
	termAll := false
	for c := 0; c < numComps; c++ {
		style := codestream.EffectiveCodingStyle(cs, tileCOD, tileCOC, c)
		termAll = termAll || style.CodeBlockStyle&(tier1.CbTermAll|tier1.CbBypass) != 0
		rev := style.Transformation == 1
		e.rev, e.irr = e.rev || rev, e.irr || !rev
		xr, yr := int(siz.Components[c].XRsiz), int(siz.Components[c].YRsiz)
		numRes := int(style.NumDecompLevels) + 1
		if numRes > 33 || style.PrecinctSizes != nil && len(style.PrecinctSizes) < numRes {
			continue // BuildTileGeom rejects it
		}
		tc := tier2.TileComponent{Tcx0: codestream.CeilDiv(tx0, xr), Tcx1: codestream.CeilDiv(tx1, xr), Tcy0: codestream.CeilDiv(ty0, yr), Tcy1: codestream.CeilDiv(ty1, yr), NumResolutions: numRes}
		// Reused: a fresh slice per tile is unreserved garbage.
		*resBuf = append((*resBuf)[:0], make([]tier2.Resolution, numRes)...)
		tc.Resolutions = *resBuf
		for r := range tc.Resolutions {
			l := uint(numRes - 1 - r)
			tc.Resolutions[r].Rect = codestream.SubbandRect{X0: codestream.CeilDivPow2(tc.Tcx0, l), Y0: codestream.CeilDivPow2(tc.Tcy0, l), X1: codestream.CeilDivPow2(tc.Tcx1, l), Y1: codestream.CeilDivPow2(tc.Tcy1, l)}
		}
		top := numRes - 1 - reg.reduce
		var wins []codestream.SubbandRect
		win := codestream.SubbandRect{}
		if top >= 0 && touched {
			if full {
				win = tc.Resolutions[top].Rect
			} else {
				win = compWin[c].Intersect(tc.Resolutions[top].Rect)
			}
			if !win.Empty() {
				wins = dwtLevels(&tc, top, win, rev)
				area := float64(win.X1-win.X0) * float64(win.Y1-win.Y0)
				if wins != nil {
					area = 0
					for _, wn := range wins {
						area += float64(float64(wn.X1-wn.X0) * float64(wn.Y1-wn.Y0))
					}
				}
				e.plane = max(e.plane, area)
				tr := tc.Resolutions[top].Rect
				e.topW, e.topH = max(e.topW, float64(tr.X1-tr.X0)), max(e.topH, float64(tr.Y1-tr.Y0))
			}
		}
		for r := 0; r < numRes; r++ {
			rr := tc.Resolutions[r].Rect
			ppx, ppy := uint(15), uint(15)
			if style.PrecinctSizes != nil {
				ppx, ppy = uint(style.PrecinctSizes[r].PPx), uint(style.PrecinctSizes[r].PPy)
			}
			var pw, ph int64
			if rr.X0 != rr.X1 {
				pw = int64(codestream.CeilDivPow2(rr.X1, ppx) - codestream.FloorDivPow2(rr.X0, ppx))
			}
			if rr.Y0 != rr.Y1 {
				ph = int64(codestream.CeilDivPow2(rr.Y1, ppy) - codestream.FloorDivPow2(rr.Y0, ppy))
			}
			nb := int64(1)
			cbw, cbh := min(uint(style.CodeBlockWidthExp), ppx), min(uint(style.CodeBlockHeightExp), ppy)
			if r > 0 {
				nb = 3
				cbw, cbh = min(uint(style.CodeBlockWidthExp), max(ppx, 1)-1), min(uint(style.CodeBlockHeightExp), max(ppy, 1)-1)
			}
			precinctsPerLayer += pw * ph
			precincts += pw * ph * nb
			l := uint(numRes - 1 - r)
			for b := 1; b <= int(nb); b++ {
				sub := b
				br := rr
				if r > 0 {
					x0b, y0b := sub&1, sub>>1
					br = codestream.SubbandRect{
						X0: codestream.CeilDivPow2(tc.Tcx0-x0b<<l, l+1), Y0: codestream.CeilDivPow2(tc.Tcy0-y0b<<l, l+1),
						X1: codestream.CeilDivPow2(tc.Tcx1-x0b<<l, l+1), Y1: codestream.CeilDivPow2(tc.Tcy1-y0b<<l, l+1),
					}
				} else {
					sub = 0
				}
				n := blocksIn(br, cbw, cbh)
				cbsAll += n
				switch {
				case r > top || win.Empty():
				case wins == nil:
					cbsWanted += n
				default:
					bw, _, _ := tier2.BandWindow(wins, r, sub)
					cbsWanted += blocksIn(bw.Intersect(br), cbw, cbh)
				}
			}
		}
	}
	hdr := tileBytes + hdrBytes
	bits := 8 * int64(hdr)
	packets := min(int64(tier2.PacketBudget(tileBytes, hdrBytes)), precinctsPerLayer*layers)
	if hasPOC || len(cs.POC) > 0 {
		packets *= 2 // the de-duplication map
	}
	precincts = min(precincts, 3*int64(tier2.MaxPrecinctsPerTile))
	cbsAll = min(cbsAll, int64(tier2.CodeBlockBudget(hdr)))
	refs := min(cbsWanted*min(layers, 109), bits/6+1)
	segs := refs
	if termAll {
		segs = min(cbsWanted*int64(tier2.PassesRead(cs, t)), bits/3+1)
	}
	groups := min(cbsWanted, refs)
	e.transient = float64(costTile + precincts*costPrecinct + cbsAll*costCodeBlock + packets*costPacket +
		refs*costBlockRef + segs*costSegment + groups*costGroup)
	e.cbs, e.segs, e.refs, e.precincts, e.packets = cbsAll, segs, refs, precincts, packets
	return e
}

// estimateIndex is an upper estimate of the heap building tile t's index
// (see Handle) holds at its peak: its data, tier-2's state, the index and
// its scratch, plus the packets tier-2 hands over and drops, as garbage.
// floor excludes otherLive, see estimateDecode.
func estimateIndex(cs *codestream.Codestream, t uint16, dataBytes int64) (est, floor int64) {
	var resBuf []tier2.Resolution
	e := estimateTile(cs, t, region{}, nil, true, &resBuf)
	live := float64(costDecodeBase + costTile + dataBytes + e.precincts*costPrecinct + e.packets*costPacket +
		e.cbs*(costCodeBlock+cellBytes+4) + e.segs*(int64(unsafe.Sizeof(ixRec{}))+segBytes+2))
	garbage := float64(e.refs*costBlockRef + e.segs*costSegment)
	live = float64(live * 1.2)
	other := float64(otherLive())
	grow := 1.25 * gogcFactor()
	return int64(min(live+min(garbage, float64(grow*live)+other), 1<<62)), int64(min(live+min(garbage, grow*live), 1<<62))
}

// otherLive is the heap live at the last GC that decodes in flight have
// not reserved nor open Images kept: the program's own data, which lets a decode's garbage
// grow past its own live heap before the next GC.
func otherLive() int64 {
	s := []metrics.Sample{{Name: "/gc/heap/live:bytes"}}
	metrics.Read(s)
	budget.mu.Lock()
	u := budget.used + budget.parked + budget.kept
	budget.mu.Unlock()
	return max(0, int64(s[0].Value.Uint64())-u)
}

// gogcFactor is max(1, GOGC/100).
func gogcFactor() float64 {
	s := []metrics.Sample{{Name: "/gc/gogc:percent"}}
	metrics.Read(s)
	if gogc := int64(s[0].Value.Uint64()); gogc > 0 {
		return max(1, float64(gogc)/100)
	}
	return 1
}

// collectFor runs a GC, and reports that it did, when need is over the
// limit but might not be without decodes done since the last GCs ended:
// those marked them live, so otherLive counts them until the next one.
func collectFor(need int64) bool {
	budget.mu.Lock()
	limit, dead := admitLimit(), garbageNow(0)
	seq := budget.seq
	budget.mu.Unlock()
	if need <= limit || need-dead > limit {
		return false
	}
	collectGarbage(seq)
	return true
}

// blocksIn counts the 2^cbw x 2^cbh code-block grid cells r touches.
func blocksIn(r codestream.SubbandRect, cbw, cbh uint) int64 {
	if r.Empty() {
		return 0
	}
	return int64(codestream.CeilDivPow2(r.X1, cbw)-codestream.FloorDivPow2(r.X0, cbw)) * int64(codestream.CeilDivPow2(r.Y1, cbh)-codestream.FloorDivPow2(r.Y0, cbh))
}
