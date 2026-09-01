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
