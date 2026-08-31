//go:build accelerate && !openblas && darwin

package numerics

/*
#cgo LDFLAGS: -framework Accelerate
*/
import "C"

const selectedNetlibName = "accelerate"
