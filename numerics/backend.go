package numerics

// Backend owns the model-independent linear algebra kernels. Implementations
// must permit dst to be distinct from every input.
type Backend interface {
	Name() string
	Sandwich(dst, scratch, matrix, transition []float64, size int)
	TransportTensor3(dst, scratch1, scratch2, tensor, transition []float64, size int)
	MultiplyABT(dst, left, right []float64, size int)
	ApplyTransition(dst, source []float64, transition *SparseTransition)
	SandwichTransition(dst, scratch, matrix []float64, transition *SparseTransition)
	TransportTransitionTensor3(dst, scratch, tensor []float64, transition *SparseTransition, channels int)
	ApplyTridiagonal(dst, source []float64, system *TridiagonalSystem)
	TransportTridiagonalMatrix(dst, scratch, source []float64, system *TridiagonalSystem)
	TransportTridiagonalTensor3(dst, scratch, source []float64, system *TridiagonalSystem, channels int)
}

type pureGoBackend struct {
	name string
}

func (b pureGoBackend) Name() string { return b.name }

func (b pureGoBackend) MultiplyABT(dst, left, right []float64, size int) {
	clear(dst)
	for row := 0; row < size; row++ {
		for column := 0; column < size; column++ {
			for inner := 0; inner < size; inner++ {
				dst[row*size+column] += left[row*size+inner] * right[column*size+inner]
			}
		}
	}
}

// Sandwich computes transition^T * matrix * transition for row-major square
// matrices. The loop ordering keeps the inner operations contiguous.
func (b pureGoBackend) Sandwich(dst, scratch, matrix, transition []float64, size int) {
	clear(scratch)
	clear(dst)
	for source := 0; source < size; source++ {
		for left := 0; left < size; left++ {
			weight := transition[source*size+left]
			if weight == 0 {
				continue
			}
			matrixRow := matrix[source*size : (source+1)*size]
			outRow := scratch[left*size : (left+1)*size]
			for target, value := range matrixRow {
				outRow[target] += weight * value
			}
		}
	}
	for left := 0; left < size; left++ {
		outRow := dst[left*size : (left+1)*size]
		for middle := 0; middle < size; middle++ {
			value := scratch[left*size+middle]
			if value == 0 {
				continue
			}
			transitionRow := transition[middle*size : (middle+1)*size]
			for right, weight := range transitionRow {
				outRow[right] += value * weight
			}
		}
	}
}

// TransportTensor3 applies the same row-stochastic transition to all three
// opinion labels of tensor[i,r,j].
func (b pureGoBackend) TransportTensor3(dst, scratch1, scratch2, tensor, transition []float64, size int) {
	clear(scratch1)
	clear(scratch2)
	clear(dst)
	index := func(i, r, j int) int { return (i*size+r)*size + j }
	for i := 0; i < size; i++ {
		for a := 0; a < size; a++ {
			weight := transition[i*size+a]
			if weight == 0 {
				continue
			}
			for r := 0; r < size; r++ {
				input := tensor[index(i, r, 0) : index(i, r, 0)+size]
				output := scratch1[index(a, r, 0) : index(a, r, 0)+size]
				for j, value := range input {
					output[j] += weight * value
				}
			}
		}
	}
	for a := 0; a < size; a++ {
		for r := 0; r < size; r++ {
			for middle := 0; middle < size; middle++ {
				weight := transition[r*size+middle]
				if weight == 0 {
					continue
				}
				input := scratch1[index(a, r, 0) : index(a, r, 0)+size]
				output := scratch2[index(a, middle, 0) : index(a, middle, 0)+size]
				for j, value := range input {
					output[j] += weight * value
				}
			}
		}
	}
	for a := 0; a < size; a++ {
		for middle := 0; middle < size; middle++ {
			output := dst[index(a, middle, 0) : index(a, middle, 0)+size]
			for j := 0; j < size; j++ {
				value := scratch2[index(a, middle, j)]
				if value == 0 {
					continue
				}
				transitionRow := transition[j*size : (j+1)*size]
				for c, weight := range transitionRow {
					output[c] += value * weight
				}
			}
		}
	}
}

var ActiveBackend Backend = newDenseBackend()
