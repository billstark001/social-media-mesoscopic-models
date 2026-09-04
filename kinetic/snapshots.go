package kinetic

import (
	"errors"
	"smp-meso/kinetic/statistics"
	"smp-meso/protocol"
)

// SnapshotSeries contains only explicitly requested state fields. Every field
// is binary encoded; snapshots never enter the scalar observable Series.
type SnapshotSeries struct {
	Time               *protocol.EncodedArray `json:"time"`
	Rho                *protocol.EncodedArray `json:"rho,omitempty"`
	Edge               *protocol.EncodedArray `json:"edge,omitempty"`
	FinalRho           *protocol.EncodedArray `json:"final_rho,omitempty"`
	FinalEdge          *protocol.EncodedArray `json:"final_edge,omitempty"`
	FinalNodePotential *protocol.EncodedArray `json:"final_node_potential,omitempty"`
	FinalEdgePotential *protocol.EncodedArray `json:"final_edge_potential,omitempty"`
	// Velocity is the coefficient-bearing finite-exposure drift. It retains
	// the C=0 no-update event rather than using E[S]/E[C].
	Velocity      *protocol.EncodedArray `json:"velocity,omitempty"`
	RewiringFlux  *protocol.EncodedArray `json:"rewiring_flux,omitempty"`
	NodePotential *protocol.EncodedArray `json:"node_potential,omitempty"`
	EdgePotential *protocol.EncodedArray `json:"edge_potential,omitempty"`
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
	nodePotential []float64
	edgePotential []float64
	interaction   *statistics.InteractionPlan
	requiresField bool
}

func newSnapshotCollector(request RunRequest, axis []float64) (*snapshotCollector, error) {
	steps, err := request.snapshotSteps()
	if err != nil {
		return nil, err
	}
	if len(steps) == 0 && !request.Snapshots.FinalRho && !request.Snapshots.FinalEdge &&
		!request.Snapshots.FinalNodePotential && !request.Snapshots.FinalEdgePotential {
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
	if request.Snapshots.NodePotential || request.Snapshots.EdgePotential ||
		request.Snapshots.FinalNodePotential || request.Snapshots.FinalEdgePotential {
		result.interaction = statistics.NewInteractionPlan(
			axis, request.Dynamics.Tolerance, request.OutDegree,
		)
	}
	if request.Snapshots.NodePotential {
		result.nodePotential = make([]float64, 0, count*size)
	}
	if request.Snapshots.EdgePotential {
		result.edgePotential = make([]float64, 0, count*size)
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
	if collector.interaction != nil {
		node := make([]float64, collector.request.OpinionBins)
		edgePotential := make([]float64, collector.request.OpinionBins)
		collector.interaction.Potentials(current.Rho, current.Edge, node, edgePotential)
		if collector.request.Snapshots.NodePotential {
			collector.nodePotential = append(collector.nodePotential, node...)
		}
		if collector.request.Snapshots.EdgePotential {
			collector.edgePotential = append(collector.edgePotential, edgePotential...)
		}
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

func (collector *snapshotCollector) outcome(final *state) (*SnapshotSeries, error) {
	if collector == nil {
		return nil, nil
	}
	count := collector.next
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
	if result.NodePotential, err = encodeSnapshot(collector.nodePotential, count, size); err != nil {
		return nil, err
	}
	if result.EdgePotential, err = encodeSnapshot(collector.edgePotential, count, size); err != nil {
		return nil, err
	}
	if collector.request.Snapshots.FinalRho {
		if result.FinalRho, err = encodeSnapshot(final.Rho, size); err != nil {
			return nil, err
		}
	}
	if collector.request.Snapshots.FinalEdge {
		if result.FinalEdge, err = encodeSnapshot(final.Edge, size, size); err != nil {
			return nil, err
		}
	}
	if collector.request.Snapshots.FinalNodePotential || collector.request.Snapshots.FinalEdgePotential {
		node := make([]float64, size)
		edgePotential := make([]float64, size)
		collector.interaction.Potentials(final.Rho, final.Edge, node, edgePotential)
		if collector.request.Snapshots.FinalNodePotential {
			if result.FinalNodePotential, err = encodeSnapshot(node, size); err != nil {
				return nil, err
			}
		}
		if collector.request.Snapshots.FinalEdgePotential {
			if result.FinalEdgePotential, err = encodeSnapshot(edgePotential, size); err != nil {
				return nil, err
			}
		}
	}
	return result, nil
}
