package numerics

type TransitionEntry struct {
	Index  int
	Weight float64
}

// SparseTransition is a row-stochastic source-to-destination operator.
type SparseTransition struct {
	Size  int
	Rows  [][]TransitionEntry
	Dense []float64
}

func DenseToSparse(dense []float64, size int) *SparseTransition {
	rows := make([][]TransitionEntry, size)
	for source := 0; source < size; source++ {
		for target, weight := range dense[source*size : (source+1)*size] {
			if weight > 1e-15 {
				rows[source] = append(rows[source], TransitionEntry{Index: target, Weight: weight})
			}
		}
	}
	return &SparseTransition{Size: size, Rows: rows, Dense: append([]float64(nil), dense...)}
}

func NewSparseTransition(size int, rows [][]TransitionEntry) *SparseTransition {
	dense := make([]float64, size*size)
	for source, row := range rows {
		for _, entry := range row {
			dense[source*size+entry.Index] += entry.Weight
		}
	}
	return &SparseTransition{Size: size, Rows: rows, Dense: dense}
}
