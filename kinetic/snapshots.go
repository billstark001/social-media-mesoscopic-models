package kinetic

import (
	"errors"
	"smp-meso/protocol"
)

// SnapshotSeries contains only explicitly requested state fields. Every field
// is binary encoded; snapshots never enter the scalar observable Series.
type SnapshotSeries struct {
	Time *protocol.EncodedArray `json:"time"`
	Rho  *protocol.EncodedArray `json:"rho,omitempty"`
	Edge *protocol.EncodedArray `json:"edge,omitempty"`
	// Velocity is the coefficient-bearing finite-exposure drift. It retains
	// the C=0 no-update event rather than using E[S]/E[C].
	Velocity     *protocol.EncodedArray `json:"velocity,omitempty"`
	RewiringFlux *protocol.EncodedArray `json:"rewiring_flux,omitempty"`
}

type snapshotCollector struct {
	request       RunRequest
	steps         []int
	next          int
	moments       momentBuilder
	time          []float64
	rho           []float64
	edge          []float64
	velocity      []float64
	rewiringFlux  []float64
	requiresField bool
}

func newSnapshotCollector(request RunRequest) (*snapshotCollector, error) {
	steps, err := request.snapshotSteps()
	if err != nil {
		return nil, err
	}
	if len(steps) == 0 {
		return nil, nil
	}
	count := len(steps)
	size := request.OpinionBins
	result := &snapshotCollector{
		request: request, steps: steps, time: make([]float64, 0, count),
		requiresField: request.Snapshots.Velocity || request.Snapshots.RewiringFlux,
	}
	if request.Snapshots.Rho {
		result.rho = make([]float64, 0, count*size)
	}
	if request.Snapshots.Edge {
		result.edge = make([]float64, 0, count*size*size)
	}
	if request.Snapshots.Velocity {
		result.velocity = make([]float64, 0, count*size)
		result.moments = planMomentBuilder(request)
	}
	if request.Snapshots.RewiringFlux {
		result.rewiringFlux = make([]float64, 0, count*size*size)
	}
	return result, nil
}

func (collector *snapshotCollector) wants(step int) bool {
	return collector != nil && collector.next < len(collector.steps) && collector.steps[collector.next] == step
}

func (collector *snapshotCollector) record(step int, current *state, values *fields) error {
	if !collector.wants(step) {
		return nil
	}
	if collector.requiresField && values == nil {
		return errors.New("snapshot fields were not computed")
	}
	collector.time = append(collector.time, float64(step)*collector.request.Dt)
	if collector.request.Snapshots.Rho {
		collector.rho = append(collector.rho, current.Rho...)
	}
	if collector.request.Snapshots.Edge {
		collector.edge = append(collector.edge, current.Edge...)
	}
	if collector.request.Snapshots.Velocity {
		moments := collector.moments(current, *values)
		for _, displacement := range moments.Mean {
			collector.velocity = append(
				collector.velocity,
				collector.request.Dynamics.Influence*displacement,
			)
		}
	}
	if collector.request.Snapshots.RewiringFlux {
		collector.rewiringFlux = append(collector.rewiringFlux, values.RewiringFlux...)
	}
	collector.next++
	return nil
}

func encodeSnapshot(values []float64, shape ...int) (*protocol.EncodedArray, error) {
	if values == nil {
		return nil, nil
	}
	encoded, err := protocol.EncodeFloat64(values, shape...)
	if err != nil {
		return nil, err
	}
	return &encoded, nil
}

func (collector *snapshotCollector) outcome() (*SnapshotSeries, error) {
	if collector == nil {
		return nil, nil
	}
	if collector.next != len(collector.steps) {
		return nil, errors.New("not every requested snapshot step was recorded")
	}
	count := len(collector.steps)
	size := collector.request.OpinionBins
	result := &SnapshotSeries{}
	var err error
	if result.Time, err = encodeSnapshot(collector.time, count); err != nil {
		return nil, err
	}
	if result.Rho, err = encodeSnapshot(collector.rho, count, size); err != nil {
		return nil, err
	}
	if result.Edge, err = encodeSnapshot(collector.edge, count, size, size); err != nil {
		return nil, err
	}
	if result.Velocity, err = encodeSnapshot(collector.velocity, count, size); err != nil {
		return nil, err
	}
	if result.RewiringFlux, err = encodeSnapshot(collector.rewiringFlux, count, size, size); err != nil {
		return nil, err
	}
	return result, nil
}
