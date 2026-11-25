// Package parallel spreads decode work over goroutines, stops it when a
// decode is cancelled, and hands a worker's panic back to the caller.
package parallel

import (
	"errors"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
)

// ErrStopped is what a decode step returns once plane.done is closed;
// decodeImage reports the ctx error in its place.
var ErrStopped = errors.New("jp2k: decode stopped")

// ErrInternal is jp2k.ErrInternal.
var ErrInternal = errors.New("jp2k: internal error")

// IsDone reports whether done is closed; a nil done never is.
func IsDone(done <-chan struct{}) bool {
	select {
	case <-done:
		return true
	default:
		return false
	}
}

// Workers is how many goroutines to use for n independent tasks, at
// most limit when that is positive. GOMAXPROCS, not NumCPU, so a
// container CPU limit is respected.
func Workers(n, limit int) int {
	w := min(runtime.GOMAXPROCS(0), n)
	if limit > 0 {
		w = min(w, limit)
	}
	return max(1, w)
}

// For calls fn(w, i) for every i in [0,n) from nw goroutines
// (the caller's plus nw-1 more); w in [0,nw) identifies the worker so fn
// can keep per-worker scratch. After the first error no new items start,
// and every worker has returned before For does.
func For(nw, n int, fn func(w, i int) error) error {
	if nw <= 1 {
		for i := 0; i < n; i++ {
			if err := fn(0, i); err != nil {
				return err
			}
		}
		return nil
	}
	var (
		next   atomic.Int64
		failed atomic.Bool
		once   sync.Once
		first  error
		wg     sync.WaitGroup
	)
	fail := func(err error) {
		once.Do(func() { first = err })
		failed.Store(true)
	}
	run := func(w int) {
		// A panic on a worker goroutine would kill the process.
		defer func() {
			if v := recover(); v != nil {
				fail(newPanicError(v))
			}
		}()
		for !failed.Load() {
			i := int(next.Add(1)) - 1
			if i >= n {
				return
			}
			if err := fn(w, i); err != nil {
				fail(err)
				return
			}
		}
	}
	if SpawnHook != nil {
		SpawnHook(nw - 1)
	}
	wg.Add(nw - 1)
	for w := 1; w < nw; w++ {
		go func(w int) {
			defer wg.Done()
			run(w)
		}(w)
	}
	run(0)
	wg.Wait()
	// Re-raised on the caller, for the entry point's recover.
	if pe, ok := first.(*panicError); ok {
		panic(pe)
	}
	return first
}

// panicError is a recovered panic with the stack where it happened.
type panicError struct {
	value any
	stack []byte
}

func newPanicError(v any) *panicError {
	if pe, ok := v.(*panicError); ok {
		return pe
	}
	return &panicError{v, debug.Stack()}
}

func (e *panicError) Error() string {
	return fmt.Sprintf("jp2k: internal error: panic: %v", e.value)
}

// Stack: for logging, not client errors.
func (e *panicError) Stack() []byte {
	return e.stack
}

func (e *panicError) Unwrap() error {
	return ErrInternal
}

// Recover turns a panic in a public entry point into its error
// result; deferred directly so recover sees the panic.
func Recover[T any](v *T, err *error) {
	if p := recover(); p != nil {
		var zero T
		*v, *err = zero, newPanicError(p)
	}
}

// FaultHook, when set by a test, runs at each named site so a test can
// inject a panic there.
var FaultHook func(site string)

// SpawnHook, when set by a test, gets how many goroutines each For
// starts, from any goroutine.
var SpawnHook func(n int)
