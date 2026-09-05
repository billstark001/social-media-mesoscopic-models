package statistics

import (
	"math"
	"testing"
)

func TestUniformHomophilyBaselineIsZero(t *testing.T) {
	axis := []float64{-0.75, -0.25, 0.25, 0.75}
	concordance := make([]float64, 16)
	for left := range axis {
		for right := range axis {
			if math.Abs(axis[left]-axis[right]) <= 0.5 {
				concordance[left*4+right] = 1
			}
		}
	}
	plan := NewPlan(axis, concordance, Config{
		Homophily: true, MinimumBandwidth: 0.01, ObjectiveEffectiveSamples: 100,
		DistanceGridSize: 32, Population: 100, OutDegree: 2, Tolerance: 0.5, DomainWidth: 2,
	})
	rho := []float64{0.25, 0.25, 0.25, 0.25}
	edge := make([]float64, 16)
	for index := range edge {
		edge[index] = 2.0 / 16
	}
	plan.Record(0, rho, edge)
	outcome := plan.Outcome()
	if len(outcome.Homophily) != 1 || outcome.Homophily[0] < 0 || outcome.Homophily[0] > 1 {
		t.Fatalf("invalid normalized homophily: %v", outcome.Homophily)
	}
}

func TestFirstPassageRetainsTimeWithoutOutputSeries(t *testing.T) {
	axis := []float64{-0.5, 0.5}
	concordance := []float64{1, 0, 0, 1}
	plan := NewPlan(axis, concordance, Config{
		HomophilyFirstPassage: true, HomophilyThreshold: 0.5,
		MinimumBandwidth: 0.01, ObjectiveEffectiveSamples: 10, DistanceGridSize: 16,
		Population: 10, OutDegree: 1, Tolerance: 0.5, DomainWidth: 2,
	})
	plan.Record(0, []float64{0.5, 0.5}, []float64{0.25, 0.25, 0.25, 0.25})
	plan.Record(2.5, []float64{0.5, 0.5}, []float64{0.5, 0, 0, 0.5})
	outcome := plan.Outcome()
	if outcome.Time != nil {
		t.Fatal("first-passage-only plan exposed a time series")
	}
	if !outcome.HomophilyPass.Reached || outcome.HomophilyPass.Time != 2.5 {
		t.Fatalf("wrong first passage: %+v", outcome.HomophilyPass)
	}
}

func TestPathwayAccumulatesOnlineWithoutRetainingTrajectory(t *testing.T) {
	axis := []float64{-0.5, 0.5}
	concordance := []float64{1, 0, 0, 1}
	plan := NewPlan(axis, concordance, Config{
		Pathway:          true,
		MinimumBandwidth: 0.01, ObjectiveEffectiveSamples: 10, DistanceGridSize: 16,
		Population: 10, OutDegree: 1, Tolerance: 0.5, DomainWidth: 2,
	})
	states := []struct {
		rho  []float64
		edge []float64
	}{
		{[]float64{0.5, 0.5}, []float64{0.25, 0.25, 0.25, 0.25}},
		{[]float64{0.6, 0.4}, []float64{0.45, 0.15, 0.05, 0.35}},
		{[]float64{0.8, 0.2}, []float64{0.75, 0.05, 0.05, 0.15}},
	}
	polarization := make([]float64, len(states))
	homophily := make([]float64, len(states))
	for index, state := range states {
		polarization[index] = plan.calculatePolarization(state.rho)
		homophily[index], _ = plan.calculateHomophily(state.edge)
		plan.ObservePathway(state.rho, state.edge)
	}
	expected := 0.0
	for index := 1; index < len(states); index++ {
		expected += 0.5 * (polarization[index] - polarization[index-1]) *
			(homophily[index] + homophily[index-1])
	}
	last := len(states) - 1
	expected += (1 - polarization[last]) * homophily[last]
	outcome := plan.Outcome()
	if !outcome.HasPathway || math.Abs(outcome.Pathway-expected) > 1e-15 {
		t.Fatalf("wrong online pathway: got %.17g want %.17g", outcome.Pathway, expected)
	}
	if len(plan.polarization) != 0 || len(plan.homophily) != 0 || outcome.Time != nil {
		t.Fatalf("pathway-only plan retained a trajectory: P=%d H=%d time=%v",
			len(plan.polarization), len(plan.homophily), outcome.Time)
	}
}
