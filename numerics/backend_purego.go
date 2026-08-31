//go:build !openblas && !accelerate

package numerics

func newDenseBackend() Backend {
	return pureGoBackend{name: "gonum-native-compatible-pure-go"}
}
