package lifted

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestDeterministicInvariantsAndRepeatability(t *testing.T) {
	r := testRequest()
	for _, layer := range []Layer{LayerNaive, LayerBase} {
		a, err := InitialState(r, layer, ClosureProfile{}, rand.New(rand.NewPCG(501, 502)))
		if err != nil {
			t.Fatal(err)
		}
		b := a.Clone()
		for i := 0; i < 200; i++ {
			if _, err := DeterministicStep(a, r); err != nil {
				t.Fatal(i, err)
			}
			if _, err := DeterministicStep(b, r); err != nil {
				t.Fatal(i, err)
			}
		}
		if !reflect.DeepEqual(a.Rho, b.Rho) || !reflect.DeepEqual(a.Edge, b.Edge) || !reflect.DeepEqual(a.Score, b.Score) {
			t.Fatal("nonrepeatable")
		}
	}
}

func TestDeterministicNodeMeanMatchesSampling(t *testing.T) {
	r := testRequest()
	r.Dynamics.RewiringRate = 0
	a, err := InitialState(r, LayerBase, ClosureProfile{}, rand.New(rand.NewPCG(601, 602)))
	if err != nil {
		t.Fatal(err)
	}
	d := a.Clone()
	if _, err := DeterministicStep(d, r); err != nil {
		t.Fatal(err)
	}
	rec, err := RecommendationKernel(a, r)
	if err != nil {
		t.Fatal(err)
	}
	kernel, err := TransitionKernel(a, r, a.neighborKernel(), rec)
	if err != nil {
		t.Fatal(err)
	}
	samples := 4000
	mean := make([]float64, a.Bins)
	rng := rand.New(rand.NewPCG(603, 604))
	for k := 0; k < samples; k++ {
		rho, _, _ := sampleNodeTransition(a, kernel, rng)
		for i, v := range rho {
			mean[i] += v / float64(samples)
		}
	}
	for j := range mean {
		variance := 0.0
		for i, m := range a.Rho {
			p := kernel[i*a.Bins+j]
			variance += m * p * (1 - p) / float64(a.Population)
		}
		bound := 6*math.Sqrt(variance/float64(samples)) + 1e-12
		if math.Abs(mean[j]-d.Rho[j]) > bound {
			t.Fatalf("bin %d node expectation mismatch", j)
		}
	}
}

func TestDeterministicScoreIrrelevantToRandomAndOpinion(t *testing.T) {
	r := testRequest()
	for _, kind := range []string{"random", "opinion_random"} {
		r.Recommender.Type = kind
		a, err := InitialState(r, LayerNaive, ClosureProfile{}, rand.New(rand.NewPCG(701, 702)))
		if err != nil {
			t.Fatal(err)
		}
		b, err := InitialState(r, LayerBase, ClosureProfile{}, rand.New(rand.NewPCG(701, 702)))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 100; i++ {
			if _, err := DeterministicStep(a, r); err != nil {
				t.Fatal(err)
			}
			if _, err := DeterministicStep(b, r); err != nil {
				t.Fatal(err)
			}
		}
		if !reflect.DeepEqual(a.Rho, b.Rho) || !reflect.DeepEqual(a.Edge, b.Edge) {
			t.Fatal("irrelevant score changed rho/E", kind)
		}
	}
}
