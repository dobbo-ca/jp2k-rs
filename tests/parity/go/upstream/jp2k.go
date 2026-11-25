// Package jp2k decodes JPEG 2000 images, raw J2K codestreams and JP2
// files, in pure Go. Importing it registers the "j2k" and "jp2" formats
// with the image package. The code lives in internal/jp2k; this file
// re-exports it.
package jp2k

import (
	"context"
	"image"
	"io"

	ijp2k "github.com/dobbo-ca/jp2k-go/internal/jp2k"
)

var (
	// ErrMemoryLimit is wrapped by the errors of decodes and Opens refused
	// because other decodes, open Images or the process heap held the memory
	// limit (see SetMemoryLimit); retrying may work.
	ErrMemoryLimit = ijp2k.ErrMemoryLimit
	// ErrInternal is wrapped by the errors of decodes that recovered from an
	// internal panic; a client should treat it as a bug report, not retry.
	// errors.As into an interface{ Stack() []byte } gets the panic's stack.
	ErrInternal = ijp2k.ErrInternal
)

type (
	// Image is a JPEG 2000 file opened for serving regions of it, such as
	// the tiles of a tile server: see Open. It holds its parsed headers and
	// the tile indexes it keeps until it is garbage-collected; under a
	// memory limit, that memory counts against the limit (see Open).
	Image = ijp2k.Image

	// Options changes how a decode computes its samples and how many
	// goroutines it uses. The zero value is the default decode. Its
	// SeparateRounding field rounds the
	// multiply and the add of each 9/7 lifting step and of the ICT
	// separately, matching OpenJPEG built for x86-64 without FMA
	// contraction (for example Ubuntu's libopenjp2, which libvips uses);
	// the default rounds each multiply-add once, matching opj_decompress
	// where its compiler contracts them (arm64). Both give the same bytes
	// on every architecture; SeparateRounding runs Go kernels only. Its
	// JJ2000 field reproduces the JJ2000 decode that Bio-Formats uses. Its
	// Workers field caps the goroutines one decode uses; 0 or less
	// means GOMAXPROCS. Output does not depend on it.
	Options = ijp2k.Options
)

// Decode decodes a JPEG 2000 image: either a raw codestream (starting with
// the SOC marker) or a JP2 file, whose jp2c box holds the codestream
// (its ihdr and colr boxes are otherwise unused).
//
// Output type depends on component count and precision:
//
//	1 component,  8-bit    -> *image.Gray
//	1 component,  9-16 bit -> *image.Gray16
//	2 components, 8-bit    -> *image.NRGBA (gray+alpha, R=G=B=Y)
//	2 components, 9-16 bit -> *image.NRGBA64 (gray+alpha)
//	3 components, 8-bit    -> *image.RGBA (opaque)
//	3 components, 9-16 bit -> *image.RGBA64 (opaque)
//	4 components, 8-bit    -> *image.NRGBA
//	4 components, 9-16 bit -> *image.NRGBA64 (with alpha)
//
// Signed components are supported: their true (negative-capable) values are
// biased by +2^(precision-1) into the same output range an unsigned component
// of the same precision would use.
//
// Component subsampling (XRsiz/YRsiz != 1) is supported: a subsampled
// component is upsampled to the reference grid by sample replication.
//
// Any other component count, mismatched per-component precision or
// signedness, precision above 16 bits, or a region-of-interest (RGN) marker
// is an error.
//
// Decode has no ctx, so it waits as long as r.Read does: a server reading
// a client or upstream should call DecodeContext, DecodeRegion or Open
// instead, with a reader whose Read honours cancellation, and should not
// call image.Decode for jp2/j2k (its bufio wrapping also defeats the
// in-memory fast path DecodeContext and DecodeRegion use).
func Decode(r io.Reader) (image.Image, error) { return ijp2k.Decode(r) }

// DecodeContext is Decode, returning early with an error wrapping
// ctx.Err() once ctx is done while waiting for memory or already running.
// It does not return while r.Read is blocked: pass a reader whose Read
// itself honours ctx (e.g. one built on a net.Conn with a deadline), not
// a plain in-memory or file reader. Decodes read r under the memory
// budget (see SetMemoryLimit) but hold no lock while they do, so a slow
// r holds up others only by the memory it has reserved.
func DecodeContext(ctx context.Context, r io.Reader) (image.Image, error) {
	return ijp2k.DecodeContext(ctx, r)
}

// DecodeConfig returns the width, height, and color model of a JPEG 2000
// image without decoding pixel data. For a raw codestream it reads no more
// than the SOC and SIZ marker segments' worth of r (2 + 2 + 2 + 65535, the
// largest a conforming SIZ segment can be), so a truncated-after-SIZ stream
// still succeeds; for a JP2 file it streams boxes unbuffered,
// then reads the same SIZ-only codestream prefix.
// io.Reader permits any Read call to return fewer bytes than the buffer
// (e.g. a network-backed reader), so this loops via io.ReadAll rather than
// trusting a single call to return the whole header.
func DecodeConfig(r io.Reader) (image.Config, error) { return ijp2k.DecodeConfig(r) }

// DecodeConfigContext is DecodeConfig, ctx-checked between reads.
func DecodeConfigContext(ctx context.Context, r io.Reader) (image.Config, error) {
	return ijp2k.DecodeConfigContext(ctx, r)
}

// DecodeRegion decodes only rect of the image at 1/2^reduce scale. Like
// DecodeContext, it returns early with an error wrapping ctx.Err() once
// ctx is done.
//
// rect is in the coordinates of Decode's Bounds() and is clipped to the
// image; an empty intersection is an error. reduce drops that many of the
// highest resolution levels (0 is full size) and may not exceed the
// smallest decomposition level count of any component, as in OpenJPEG.
//
// The result covers exactly rect at the reduced scale, and its Bounds()
// are in reduced coordinates: each edge e becomes ceil(e/2^reduce) (for an
// image with a nonzero XOsiz/YOsiz origin, ceil((e+XOsiz)/2^reduce) -
// ceil(XOsiz/2^reduce)). Adjacent regions therefore tile without gaps or
// overlap, and DecodeRegion(ctx, r, full, 0) equals Decode(r). Pixel values
// equal the same crop of a whole-image decode at the same reduce, as
// opj_decompress -r gives (its -d output strays from that crop on a few
// conformance images). The decode pixel limit applies to the region,
// not the whole image, so images too large for Decode can still
// be read piecewise. When r is also an io.ReaderAt and io.Seeker (an
// *os.File, a bytes.Reader), only the headers and the tile-parts of the
// tiles the region touches are read, unless the codestream uses PPM or
// tile-parts of unstated length; any other r is read to the end. A
// shared *os.File is safe across DecodeRegion calls: it is read through
// ReadAt, so its offset is left alone for the next call.
func DecodeRegion(ctx context.Context, r io.Reader, rect image.Rectangle, reduce int) (image.Image, error) {
	return ijp2k.DecodeRegion(ctx, r, rect, reduce)
}

// DecodeComponents decodes only the listed components of a JPEG 2000
// image, each into its own plane: the result holds one *image.Gray16 per
// entry of comps, in that order. Like DecodeContext, it returns early
// with an error wrapping ctx.Err() once ctx is done.
//
// comps are codestream component indices (a JP2 cdef box is ignored).
// An empty comps, an index out of range, one given twice, or a
// component of more than 16 bits is an error. Component count and mixed
// precision or signedness, which Decode refuses, are fine here.
//
// Plane values are the component's decoded samples, not scaled to 16
// bits: an unsigned component of precision p gives 0 to 2^p-1, and a
// signed one its value plus 2^(p-1), so the same range. That equals
// opj_decompress's PGX output of the component, with a signed PGX
// sample (two's complement) offset by 2^(p-1).
//
// A plane covers the whole component on its own sample grid: for a
// component without subsampling (XRsiz = YRsiz = 1), the bounds of
// Decode's result. A subsampled component's plane is its own size,
// ceil(Xsiz/XRsiz) - ceil(XOsiz/XRsiz) wide and likewise high, with no
// upsampling, and its Bounds() start at (0, 0).
//
// Components not asked for cost no tier-1 decoding, dequantization or
// inverse wavelet, and no plane or its memory reservation. When a tile
// uses the multiple component transform (RCT or ICT, which mix
// components 0-2) and comps holds any of 0-2, all three are decoded and
// transformed, so the saving applies only to components past 2.
func DecodeComponents(ctx context.Context, r io.Reader, comps []int) ([]*image.Gray16, error) {
	return ijp2k.DecodeComponents(ctx, r, comps)
}

// DecodeComponentsOptions is DecodeComponents with o.
func DecodeComponentsOptions(ctx context.Context, r io.Reader, comps []int, o Options) ([]*image.Gray16, error) {
	return ijp2k.DecodeComponentsOptions(ctx, r, comps, o)
}

// DecodeRegionComponents is DecodeComponents for rect at 1/2^reduce
// scale, as DecodeRegion takes them, with the same reading and errors.
// For a component without subsampling, each plane covers exactly what
// DecodeRegion's result does, with the same Bounds(). A subsampled
// component's plane covers the same region on its own sample grid:
// each edge e of DecodeRegion's reduced-coordinate rect (in absolute
// coordinates, XOsiz included) becomes ceil(e/XRsiz) (ceil(e/YRsiz)
// vertically), as T.800 and OpenJPEG map a region to a component, less
// the component's origin at that scale; it may be empty.
func DecodeRegionComponents(ctx context.Context, r io.Reader, rect image.Rectangle, reduce int, comps []int) ([]*image.Gray16, error) {
	return ijp2k.DecodeRegionComponents(ctx, r, rect, reduce, comps)
}

// DecodeRegionComponentsOptions is DecodeRegionComponents with o.
func DecodeRegionComponentsOptions(ctx context.Context, r io.Reader, rect image.Rectangle, reduce int, comps []int, o Options) ([]*image.Gray16, error) {
	return ijp2k.DecodeRegionComponentsOptions(ctx, r, rect, reduce, comps, o)
}

// SetMemoryLimit sets how many bytes of heap all in-flight decodes may
// reserve together, and returns the previous limit. n <= 0 or
// math.MaxInt64 means no byte limit (math.MaxInt64 is also what is
// returned for it). Open Images keep up to a quarter of it, which
// decodes may not reserve (see Open). A decode reserves its input and header parse as it
// reads them, then an estimate of its peak heap, made from the headers
// before it allocates anything else; it waits while others hold the
// budget, and fails at once if it needs more than the whole limit.
// Together they never reserve more than the limit: when decodes that
// have reserved some of it fill it and each waits for more, the one that
// began reserving last fails, until the oldest fits. A decode that
// would otherwise wait on memory held by reads that have run for 10ms
// fails at once, unless it is reading too, as a read may take any time.
// Lowering the limit does not stop running decodes.
//
// A decode refused because other decodes, open Images, or the rest of
// the process's heap, hold the memory it needs, when it would fit
// alone, returns an error wrapping ErrMemoryLimit: that is transient,
// and the decode may succeed if retried. A decode whose own estimate is over the whole
// limit fails with an error that does not wrap it, as it never fits. A decode refused before it has read its
// headers has no estimate yet, so a retry may fail that way.
//
// Until it is called, the limit is half of the Go runtime's memory limit
// (see runtime/debug.SetMemoryLimit and GOMEMLIMIT) when one is set, else,
// on Linux, half of the cgroup memory limit, else none. The default is
// read again for each decode, so it follows changes to the runtime limit.
func SetMemoryLimit(n int64) int64 { return ijp2k.SetMemoryLimit(n) }

// Open reads and checks the headers of the JPEG 2000 file (a JP2 file or
// raw codestream) in r[0:size] and returns an Image to decode regions
// of it. It reads no tile data; reads past size fail. If a tile-part
// header is malformed, only the regions that read it fail, as with the
// package-level DecodeRegion; such an Image decodes each region as that
// does, without an index. A ReadAt that fails, or ends early inside
// size, fails Open. Open reserves memory for the headers as it parses
// them, as a decode does (see SetMemoryLimit): it waits while decodes
// hold it, for as long as that takes (OpenContext bounds the wait), and
// fails with an error wrapping ErrMemoryLimit where a decode would.
//
// The first DecodeRegion that needs a tile parses that tile's packet
// headers, reading the whole tile once, and keeps an index of where each
// code-block's bytes lie. Later DecodeRegion calls on the tile read only
// the code-block bytes they use. An index costs about 8 bytes per
// code-block, 16 per code-block segment in each quality layer, and 400
// per subband of each tile (component, resolution and band).
//
// An Image holds its parsed headers, a few times the size of its
// tile-part headers plus twice any PPM or PPT markers (the first decode
// merges them), and keeps indexes while the two together stay within
// 16 MiB plus the codestream's size, until it is garbage-collected, r
// reaching it or not. Past that, a tile is indexed again by each call
// that needs it, and a tile whose index alone is larger is decoded as
// the package-level DecodeRegion does. Under a memory limit, all open
// Images keep at most a quarter of it between them, and decodes reserve
// only what they leave. Indexes that do not fit are not kept, and an
// Open or a decode that needs the room kept indexes take drops them
// all, from every Image; those a running decode uses stay counted,
// outside the quarter, until it ends. Open fails with an error wrapping
// ErrMemoryLimit when other Images' headers keep the rest of that
// quarter, after a garbage collection finds none dropped; an Image
// whose headers alone are more than the quarter decodes every region as
// the package-level DecodeRegion does. The quarter goes first come,
// first served: one Image near it makes later Opens fail until it is
// dropped, and a file of many small tile-parts keeps about four times
// its size in headers. Callers own
// caching Images (an LRU per file, say) and should drop those they no
// longer use. An Image is safe for concurrent use; closing r is the
// caller's too, once the Image is no longer used.
func Open(r io.ReaderAt, size int64) (*Image, error) { return ijp2k.Open(r, size) }

// OpenOptions is Open with o, which every decode of the Image uses.
func OpenOptions(r io.ReaderAt, size int64, o Options) (*Image, error) {
	return ijp2k.OpenOptions(r, size, o)
}

// OpenContext is Open, returning early with an error wrapping ctx.Err()
// once ctx is done while it waits for memory (see SetMemoryLimit) or
// reads headers. A ReadAt already blocked is waited for: pass an r whose
// ReadAt honours cancellation itself. ctx is not kept: each DecodeRegion
// takes its own.
func OpenContext(ctx context.Context, r io.ReaderAt, size int64) (*Image, error) {
	return ijp2k.OpenContext(ctx, r, size)
}
