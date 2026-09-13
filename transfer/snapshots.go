package transfer

import (
	"smp-meso/lifted"
	"smp-meso/numerics"
	"smp-meso/protocol"
)

type snapshotField struct {
	name           string
	history, final bool
	values         func(*lifted.State) []float64
	matrix         bool
}

type snapshotCollector struct {
	steps  []int
	next   int
	fields []snapshotField
	series map[string][]float64
}

func newSnapshotCollector(options SnapshotsConfig, steps []int) *snapshotCollector {
	return &snapshotCollector{steps: steps, series: map[string][]float64{
		"steps": {}, "score_sum": {}, "change": {},
	}, fields: []snapshotField{
		{"rho", options.Rho, options.FinalRho, func(s *lifted.State) []float64 { return s.Rho }, false},
		{"edge", options.Edge, options.FinalEdge, func(s *lifted.State) []float64 { return s.Edge }, true},
		{"candidate", options.Candidate, options.FinalCandidate, func(s *lifted.State) []float64 { return s.Candidate }, true},
		{"score", options.Score, options.FinalScore, func(s *lifted.State) []float64 { return s.Score }, true},
	}}
}

func (c *snapshotCollector) record(step int, state *lifted.State, change float64) {
	if c.next == len(c.steps) || step != c.steps[c.next] {
		return
	}
	c.series["steps"] = append(c.series["steps"], float64(step))
	c.series["score_sum"] = append(c.series["score_sum"], numerics.Sum(state.Score))
	c.series["change"] = append(c.series["change"], change)
	for _, field := range c.fields {
		if field.history {
			c.series[field.name] = append(c.series[field.name], field.values(state)...)
		}
	}
	c.next++
}

func (c *snapshotCollector) encode(state *lifted.State) (map[string]protocol.EncodedArray, error) {
	result := make(map[string]protocol.EncodedArray)
	for _, name := range []string{"steps", "score_sum", "change"} {
		encoded, err := protocol.EncodeFloat64(c.series[name], c.next)
		if err != nil {
			return nil, err
		}
		result[name] = encoded
	}
	for _, field := range c.fields {
		shape := []int{state.Bins}
		if field.matrix {
			shape = append(shape, state.Bins)
		}
		if field.history {
			encoded, err := protocol.EncodeFloat64(c.series[field.name], append([]int{c.next}, shape...)...)
			if err != nil {
				return nil, err
			}
			result[field.name] = encoded
		}
		if field.final {
			encoded, err := protocol.EncodeFloat64(field.values(state), shape...)
			if err != nil {
				return nil, err
			}
			result["final_"+field.name] = encoded
		}
	}
	return result, nil
}
