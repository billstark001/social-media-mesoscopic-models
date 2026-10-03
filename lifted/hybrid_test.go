package lifted

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestHybridEndpointsAndInvariants(t *testing.T) {
	for _, recommender := range []string{"random", "opinion_random", "structure_random"} {
		r := testRequest()
		r.Recommender.Type = recommender
		initial, err := InitialState(r, LayerBase, ClosureProfile{}, rand.New(rand.NewPCG(451, 452)))
		if err != nil {
			t.Fatal(err)
		}
		full, reference := initial.Clone(), initial.Clone()
		g1, g2 := rand.New(rand.NewPCG(453, 454)), rand.New(rand.NewPCG(453, 454))
		if _, err := HybridStep(full, r, ClosureProfile{}, g1, true, true); err != nil {
			t.Fatal(err)
		}
		if _, err := Step(reference, r, ClosureProfile{}, g2); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(full.Rho, reference.Rho) || !reflect.DeepEqual(full.Edge, reference.Edge) || !reflect.DeepEqual(full.Score, reference.Score) {
			t.Fatalf("full hybrid differs from S1 for %s", recommender)
		}
		for _, choice := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
			s := initial.Clone()
			g := rand.New(rand.NewPCG(455, 456))
			for step := 0; step < 100; step++ {
				if _, err := HybridStep(s, r, ClosureProfile{}, g, choice[0], choice[1]); err != nil {
					t.Fatalf("%s %v step %d: %v", recommender, choice, step, err)
				}
			}
			if err := s.Validate(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestSplitTransportAndFractionalNoise(t *testing.T) {
	r := testRequest()
	r.Recommender.Type = "structure_random"
	initial, err := InitialState(r, LayerBase, ClosureProfile{}, rand.New(rand.NewPCG(501, 502)))
	if err != nil {
		t.Fatal(err)
	}
	choices := []struct {
		name   string
		rewire bool
		noise  TransportNoise
	}{
		{"node-only", false, TransportNoise{Nodes: 1}},
		{"edge-only", false, TransportNoise{Edges: 1}},
		{"rewire-node", true, TransportNoise{Nodes: 1}},
		{"rewire-edge", true, TransportNoise{Edges: 1}},
		{"fractional", true, TransportNoise{Nodes: 0.35, Edges: 0.25}},
	}
	for _, choice := range choices {
		t.Run(choice.name, func(t *testing.T) {
			s := initial.Clone()
			g := rand.New(rand.NewPCG(503, 504))
			for step := 0; step < 100; step++ {
				if _, err := HybridStepWithNoise(s, r, ClosureProfile{}, g, choice.rewire, choice.noise); err != nil {
					t.Fatalf("step %d: %v", step, err)
				}
			}
			if err := s.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
	mean, edgeOnly := initial.Clone(), initial.Clone()
	if _, err := HybridStepWithNoise(mean, r, ClosureProfile{}, nil, false, TransportNoise{}); err != nil {
		t.Fatal(err)
	}
	if _, err := HybridStepWithNoise(edgeOnly, r, ClosureProfile{}, rand.New(rand.NewPCG(505, 506)), false,
		TransportNoise{Edges: 1}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mean.Rho, edgeOnly.Rho) {
		t.Fatal("edge-only sampling changed the deterministic node mass")
	}
	if reflect.DeepEqual(mean.Edge, edgeOnly.Edge) {
		t.Fatal("edge-only sampling produced no edge variation")
	}
}
