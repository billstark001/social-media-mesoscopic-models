//go:build openblas || accelerate

package numerics

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

type netlibBackend struct {
	name string
}

func newDenseBackend() Backend {
	blas64.Use(netlib.Implementation{})
	return netlibBackend{name: selectedNetlibName}
}

func (b netlibBackend) Name() string { return b.name }

func general(rows, cols, stride int, data []float64) blas64.General {
	return blas64.General{Rows: rows, Cols: cols, Stride: stride, Data: data}
}

func (b netlibBackend) Sandwich(dst, scratch, matrix, transition []float64, size int) {
	blas64.Gemm(
		blas.Trans, blas.NoTrans, 1,
		general(size, size, size, transition),
		general(size, size, size, matrix),
		0, general(size, size, size, scratch),
	)
	blas64.Gemm(
		blas.NoTrans, blas.NoTrans, 1,
		general(size, size, size, scratch),
		general(size, size, size, transition),
		0, general(size, size, size, dst),
	)
}

func (b netlibBackend) TransportTensor3(dst, scratch1, scratch2, tensor, transition []float64, size int) {
	// First mode: T^T times W flattened as i x (r,j).
	blas64.Gemm(
		blas.Trans, blas.NoTrans, 1,
		general(size, size, size, transition),
		general(size, size*size, size*size, tensor),
		0, general(size, size*size, size*size, scratch1),
	)

	// Pack the centre mode into r x (a,j), multiply, and unpack. Reusing dst
	// as the packed input avoids a fourth B^3 scratch allocation.
	for a := 0; a < size; a++ {
		for r := 0; r < size; r++ {
			for j := 0; j < size; j++ {
				dst[(r*size+a)*size+j] = scratch1[(a*size+r)*size+j]
			}
		}
	}
	blas64.Gemm(
		blas.Trans, blas.NoTrans, 1,
		general(size, size, size, transition),
		general(size, size*size, size*size, dst),
		0, general(size, size*size, size*size, scratch2),
	)
	for bIndex := 0; bIndex < size; bIndex++ {
		for a := 0; a < size; a++ {
			for j := 0; j < size; j++ {
				scratch1[(a*size+bIndex)*size+j] = scratch2[(bIndex*size+a)*size+j]
			}
		}
	}

	// Last mode: flatten (a,b) x j and multiply on the right by T.
	blas64.Gemm(
		blas.NoTrans, blas.NoTrans, 1,
		general(size*size, size, size, scratch1),
		general(size, size, size, transition),
		0, general(size*size, size, size, dst),
	)
}
