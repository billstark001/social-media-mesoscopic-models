//go:build accelerate && !openblas && !darwin

package numerics

// Deliberately undefined on non-Darwin targets: the accelerate build tag is
// a request for Apple's Accelerate framework and must fail at compile time.
var _ = accelerateBackendRequiresDarwin
