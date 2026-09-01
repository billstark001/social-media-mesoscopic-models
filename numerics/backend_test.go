package numerics

import (
	"math"
	"testing"
)

func TestSandwich(t *testing.T) {
	transition := []float64{0.8, 0.2, 0.1, 0.9}
	matrix := []float64{1, 2, 3, 4}
	dst, scratch := make([]float64, 4), make([]float64, 4)
	ActiveBackend.Sandwich(dst, scratch, matrix, transition, 2)
	expected := []float64{1.08, 2.02, 2.72, 4.18}
	for i := range dst {
		if math.Abs(dst[i]-expected[i]) > 1e-12 {
			t.Fatalf("dst[%d]=%g want %g", i, dst[i], expected[i])
		}
	}
}

func TestTensorIdentity(t *testing.T) {
	const size = 3
	tensor := make([]float64, size*size*size)
	for i := range tensor {
		tensor[i] = float64(i + 1)
	}
	identity := make([]float64, size*size)
	for i := 0; i < size; i++ {
		identity[i*size+i] = 1
	}
	dst := make([]float64, len(tensor))
	ActiveBackend.TransportTensor3(dst, make([]float64, len(tensor)), make([]float64, len(tensor)), tensor, identity, size)
	for i := range dst {
		if dst[i] != tensor[i] {
			t.Fatalf("tensor[%d]=%g want %g", i, dst[i], tensor[i])
		}
	}
}

func TestActiveBackendMatchesPureGoForDenseTransport(t *testing.T) {
	const size = 3
	transition := []float64{
		0.7, 0.2, 0.1,
		0.1, 0.6, 0.3,
		0.25, 0.25, 0.5,
	}
	matrix := make([]float64, size*size)
	tensor := make([]float64, size*size*size)
	for i := range matrix {
		matrix[i] = float64((i*7)%11+1) / 13
	}
	for i := range tensor {
		tensor[i] = float64((i*5)%17+1) / 19
	}
	reference := pureGoBackend{name: "reference"}
	for name, input := range map[string][]float64{"matrix": matrix, "tensor": tensor} {
		expected := make([]float64, len(input))
		observed := make([]float64, len(input))
		if name == "matrix" {
			reference.Sandwich(expected, make([]float64, len(input)), input, transition, size)
			ActiveBackend.Sandwich(observed, make([]float64, len(input)), input, transition, size)
		} else {
			reference.TransportTensor3(expected, make([]float64, len(input)), make([]float64, len(input)), input, transition, size)
			ActiveBackend.TransportTensor3(observed, make([]float64, len(input)), make([]float64, len(input)), input, transition, size)
		}
		for i := range expected {
			if math.Abs(expected[i]-observed[i]) > 1e-12 {
				t.Fatalf("%s[%d]=%g want %g with backend %s", name, i, observed[i], expected[i], ActiveBackend.Name())
			}
		}
	}
}

func TestSparseBackendMatchesDenseBackend(t *testing.T) {
	const size = 3
	dense := []float64{
		0.7, 0.2, 0.1,
		0.1, 0.6, 0.3,
		0.25, 0.25, 0.5,
	}
	transition, err := DenseToSparse(dense, size)
	if err != nil {
		t.Fatal(err)
	}
	vector := []float64{0.2, 0.3, 0.5}
	matrix := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	tensor := make([]float64, 2*size*size*size)
	for index := range tensor {
		tensor[index] = float64(index+1) / 71
	}
	vectorExpected := make([]float64, size)
	pureGoBackend{}.ApplyTransition(vectorExpected, vector, transition)
	vectorObserved := make([]float64, size)
	ActiveBackend.ApplyTransition(vectorObserved, vector, transition)
	matrixExpected := make([]float64, len(matrix))
	pureGoBackend{}.SandwichTransition(matrixExpected, make([]float64, len(matrix)), matrix, transition)
	matrixObserved := make([]float64, len(matrix))
	ActiveBackend.SandwichTransition(matrixObserved, make([]float64, len(matrix)), matrix, transition)
	tensorExpected := make([]float64, len(tensor))
	pureGoBackend{}.TransportTransitionTensor3(tensorExpected, make([]float64, len(tensor)), tensor, transition, 2)
	tensorObserved := make([]float64, len(tensor))
	ActiveBackend.TransportTransitionTensor3(tensorObserved, make([]float64, len(tensor)), tensor, transition, 2)
	for name, pair := range map[string][2][]float64{
		"vector": {vectorObserved, vectorExpected},
		"matrix": {matrixObserved, matrixExpected},
		"tensor": {tensorObserved, tensorExpected},
	} {
		for index := range pair[0] {
			if math.Abs(pair[0][index]-pair[1][index]) > 1e-12 {
				t.Fatalf("%s[%d]=%g want %g", name, index, pair[0][index], pair[1][index])
			}
		}
	}
}

func TestTridiagonalBackendIdentityAndMass(t *testing.T) {
	const size = 3
	identity, err := NewTridiagonalSystem([]float64{0, 0}, []float64{1, 1, 1}, []float64{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	tensor := make([]float64, 2*size*size*size)
	for index := range tensor {
		tensor[index] = float64(index + 1)
	}
	observed := make([]float64, len(tensor))
	if err := ActiveBackend.TransportTridiagonalTensor3(observed, make([]float64, len(tensor)), tensor, identity, 2); err != nil {
		t.Fatal(err)
	}
	for index := range tensor {
		if math.Abs(observed[index]-tensor[index]) > 1e-12 {
			t.Fatalf("identity tensor[%d]=%g want %g", index, observed[index], tensor[index])
		}
	}
	system, err := NewTridiagonalSystem([]float64{-0.2, -0.2}, []float64{1.2, 1.4, 1.2}, []float64{-0.2, -0.2})
	if err != nil {
		t.Fatal(err)
	}
	vector := []float64{0.2, 0.3, 0.5}
	if err := ActiveBackend.ApplyTridiagonal(observed[:size], vector, system); err != nil {
		t.Fatal(err)
	}
	total := observed[0] + observed[1] + observed[2]
	if math.Abs(total-1) > 1e-12 {
		t.Fatalf("tridiagonal solve changed mass: %g", total)
	}
}

func TestMultiplyABT(t *testing.T) {
	left := []float64{1, 2, 3, 4}
	right := []float64{5, 6, 7, 8}
	observed := make([]float64, 4)
	ActiveBackend.MultiplyABT(observed, left, right, 2)
	expected := []float64{17, 23, 39, 53}
	for index := range expected {
		if math.Abs(observed[index]-expected[index]) > 1e-12 {
			t.Fatalf("ABT[%d]=%g want %g", index, observed[index], expected[index])
		}
	}
}

func TestBatchedKernelsMatchScalarReferences(t *testing.T) {
	const size, columns, channels = 3, 4, 3
	transition := []float64{0.7, 0.2, 0.1, 0.1, 0.6, 0.3, 0.25, 0.25, 0.5}
	source := make([]float64, size*columns)
	for index := range source {
		source[index] = float64(index+1) / 17
	}
	observedBatch := make([]float64, len(source))
	ActiveBackend.ApplyDenseTransitionBatch(observedBatch, source, transition, size, columns)
	expectedBatch := make([]float64, len(source))
	pureGoBackend{}.ApplyDenseTransitionBatch(expectedBatch, source, transition, size, columns)

	matrixChannels := make([]float64, size*size*channels)
	for index := range matrixChannels {
		matrixChannels[index] = float64((index*7)%19+1) / 23
	}
	observedChannels := make([]float64, len(matrixChannels))
	ActiveBackend.TransportMatrixChannels(
		observedChannels, make([]float64, len(matrixChannels)), matrixChannels, transition, size, channels,
	)
	expectedChannels := make([]float64, len(matrixChannels))
	matrix := make([]float64, size*size)
	transported := make([]float64, size*size)
	for channel := 0; channel < channels; channel++ {
		for index := range matrix {
			matrix[index] = matrixChannels[index*channels+channel]
		}
		pureGoBackend{}.Sandwich(transported, make([]float64, len(matrix)), matrix, transition, size)
		for index, value := range transported {
			expectedChannels[index*channels+channel] = value
		}
	}
	for name, pair := range map[string][2][]float64{
		"batch": {observedBatch, expectedBatch}, "channels": {observedChannels, expectedChannels},
	} {
		for index := range pair[0] {
			if math.Abs(pair[0][index]-pair[1][index]) > 1e-12 {
				t.Fatalf("%s[%d]=%g want %g", name, index, pair[0][index], pair[1][index])
			}
		}
	}

	weights := []float64{0.2, 0.3, 0.5}
	gram := make([]float64, size*size)
	ActiveBackend.WeightedGram(gram, make([]float64, len(gram)), transition, weights, size)
	for row := 0; row < size; row++ {
		for column := 0; column < size; column++ {
			expected := 0.0
			for inner, weight := range weights {
				expected += transition[row*size+inner] * weight * transition[column*size+inner]
			}
			if math.Abs(gram[row*size+column]-expected) > 1e-12 {
				t.Fatalf("weighted gram (%d,%d)=%g want %g", row, column, gram[row*size+column], expected)
			}
		}
	}
}

func TestSparseTransitionHasCanonicalRepresentations(t *testing.T) {
	transition, err := DenseToSparse([]float64{
		1, 1e-16, 0,
		0.2, 0.3, 0.5,
		0, 0, 7,
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	for source, row := range transition.Rows {
		total := 0.0
		for _, entry := range row {
			total += entry.Weight
			if transition.Dense[source*transition.Size+entry.Index] != entry.Weight {
				t.Fatalf("dense/sparse mismatch in row %d", source)
			}
		}
		if math.Abs(total-1) > 1e-15 {
			t.Fatalf("row %d sum=%g", source, total)
		}
	}
	if transition.Dense[1] != 0 {
		t.Fatalf("pruned value survived in dense representation: %g", transition.Dense[1])
	}
}

func TestSingularTridiagonalReturnsError(t *testing.T) {
	if _, err := NewTridiagonalSystem([]float64{0}, []float64{0, 0}, []float64{0}); err == nil {
		t.Fatal("singular tridiagonal system was accepted")
	}
}

func TestTridiagonalPivoting(t *testing.T) {
	system, err := NewTridiagonalSystem([]float64{1, 1}, []float64{0, 2, 3}, []float64{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	observed := make([]float64, 3)
	if err := ActiveBackend.ApplyTridiagonal(observed, []float64{2, 8, 11}, system); err != nil {
		t.Fatal(err)
	}
	expected := []float64{1, 2, 3}
	for index := range expected {
		if math.Abs(observed[index]-expected[index]) > 1e-12 {
			t.Fatalf("solution[%d]=%g want %g", index, observed[index], expected[index])
		}
	}
}
