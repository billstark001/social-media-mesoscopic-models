//go:build openblas && !accelerate

package numerics

/*
OpenBLAS is intentionally an optional system dependency. Set CGO_CFLAGS and
CGO_LDFLAGS for the local installation when building with -tags openblas.
*/
import "C"

const selectedNetlibName = "openblas"
