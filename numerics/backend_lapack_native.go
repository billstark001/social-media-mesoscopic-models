//go:build openblas || accelerate

package numerics

/*
void dgttrf_(const int *n, double *dl, double *d, double *du, double *du2,
             int *ipiv, int *info);
void dgttrs_(const char *trans, const int *n, const int *nrhs,
             const double *dl, const double *d, const double *du,
             const double *du2, const int *ipiv, double *b,
             const int *ldb, int *info);
*/
import "C"

import "unsafe"

func nonemptyNative(values []float64) []float64 {
	if len(values) == 0 {
		return make([]float64, 1)
	}
	return values
}

func nativeDgttrf(n int, lower, diagonal, upper, upper2 []float64, pivots []int32) bool {
	lower = nonemptyNative(lower)
	upper = nonemptyNative(upper)
	upper2 = nonemptyNative(upper2)
	nativeN := C.int(n)
	info := C.int(0)
	C.dgttrf_(
		&nativeN,
		(*C.double)(unsafe.Pointer(&lower[0])),
		(*C.double)(unsafe.Pointer(&diagonal[0])),
		(*C.double)(unsafe.Pointer(&upper[0])),
		(*C.double)(unsafe.Pointer(&upper2[0])),
		(*C.int)(unsafe.Pointer(&pivots[0])), &info,
	)
	return info == 0
}

func nativeDgttrs(n, rightHandSides int, lower, diagonal, upper, upper2 []float64, pivots []int32, values []float64) bool {
	lower = nonemptyNative(lower)
	upper = nonemptyNative(upper)
	upper2 = nonemptyNative(upper2)
	trans := C.char('N')
	nativeN := C.int(n)
	nativeRHS := C.int(rightHandSides)
	leadingDimension := C.int(n)
	info := C.int(0)
	C.dgttrs_(
		&trans, &nativeN, &nativeRHS,
		(*C.double)(unsafe.Pointer(&lower[0])),
		(*C.double)(unsafe.Pointer(&diagonal[0])),
		(*C.double)(unsafe.Pointer(&upper[0])),
		(*C.double)(unsafe.Pointer(&upper2[0])),
		(*C.int)(unsafe.Pointer(&pivots[0])),
		(*C.double)(unsafe.Pointer(&values[0])), &leadingDimension, &info,
	)
	return info == 0
}
