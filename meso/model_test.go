package meso

import (
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/numerics"
	"testing"
)

func testRequest() config.RunRequest {
	return config.RunRequest{
		RequestID: "test", Layer: "base", Population: 60, OpinionBins: 5,
		OutDegree: 4, RecommendationCount: 3, MaxSteps: 10,
		Paths: 2, IntervalPaths: 1, AmbiguitySamples: 1,
		ConfidenceLevel: 0.95, Workers: 1, Seed: 7, MajorClusterMass: 0.02,
		Dynamics: config.DynamicsConfig{Type: "hk", Tolerance: 0.45, Influence: 0.1, RewiringRate: 0.05},
		Recommender: config.RecommenderConfig{
			Type: "structure_random", Steepness: 1, RandomRatio: 0,
			OpinionTolerance: 0.4, NoiseStd: 0, NoiseQuadraturePoints: 3,
		},
		Initial:    config.InitialConfig{Type: "uniform", OpinionMin: -1, OpinionMax: 1, Probabilities: []float64{}},
		Resolution: config.ResolutionConfig{ScoreMax: 8, AvailabilityBins: 5, ComponentSizeBins: 5, OpinionQuadrature: 3},
		Closure:    config.ClosureConfig{MotifRelaxation: 0.2, HistogramRelaxation: 0.2, CandidateRelaxation: 0.2, TopologyRelaxation: 0.2},
		Ambiguity: config.AmbiguityConfig{
			EligibilityCorrelationRadius: 0.5, ScoreAvailabilityRadius: 0.5,
			MotifPersistenceRadius: 0.5, BridgeBiasRadius: 0.5, ComponentMixRadius: 0.5,
		},
	}
}

func TestFiveLayersInitializeAndIncreaseDimension(t *testing.T) {
	request := testRequest()
	previous := 0
	for layer := LayerBase; layer <= LayerTopology; layer++ {
		rng := rand.New(rand.NewPCG(10+uint64(layer), 20+uint64(layer)))
		state, err := InitialState(request, layer, ClosureProfile{}, rng)
		if err != nil {
			t.Fatalf("layer %s: %v", layer.String(), err)
		}
		if state.Dimension() <= previous {
			t.Fatalf("dimension did not increase: %d <= %d", state.Dimension(), previous)
		}
		previous = state.Dimension()
	}
}

func TestCandidateWeightedWedgeProjectsToS1(t *testing.T) {
	request := testRequest()
	request.Recommender.Steepness = 1
	state, err := InitialState(request, LayerWedge, ClosureProfile{}, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < state.Bins; i++ {
		for j := 0; j < state.Bins; j++ {
			projected := 0.0
			for center := 0; center < state.Bins; center++ {
				projected += state.Wedge[state.wedgeIndex(i, center, j)]
			}
			if math.Abs(projected-state.Score[state.matrixIndex(i, j)]) > 1e-8 {
				t.Fatalf("(%d,%d): W=%g S1=%g", i, j, projected, state.Score[state.matrixIndex(i, j)])
			}
		}
	}
}

func TestRecommendationRowsAndSteps(t *testing.T) {
	for _, dynamics := range []string{"hk", "deffuant"} {
		for _, recommender := range []string{"random", "opinion_random", "structure_random"} {
			request := testRequest()
			request.Dynamics.Type = dynamics
			request.Recommender.Type = recommender
			state, err := InitialState(request, LayerTopology, ClosureProfile{}, rand.New(rand.NewPCG(3, 4)))
			if err != nil {
				t.Fatal(err)
			}
			kernel, err := RecommendationKernel(state, request)
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < state.Bins; row++ {
				if math.Abs(numerics.Sum(kernel[row*state.Bins:(row+1)*state.Bins])-1) > 1e-12 {
					t.Fatalf("%s/%s row %d is not normalized", dynamics, recommender, row)
				}
			}
			if _, err := Step(state, request, ClosureProfile{}, rand.New(rand.NewPCG(5, 6))); err != nil {
				t.Fatalf("%s/%s: %v", dynamics, recommender, err)
			}
		}
	}
}

func TestWedgeFirstMomentFeedsZetaOneScore(t *testing.T) {
	request := testRequest()
	request.Recommender.Steepness = 1
	request.Closure.MotifRelaxation = 1
	state, err := InitialState(request, LayerWedge, ClosureProfile{}, rand.New(rand.NewPCG(7, 8)))
	if err != nil {
		t.Fatal(err)
	}
	target := state.Clone()
	for i := 0; i < state.Bins; i++ {
		for j := 0; j < state.Bins; j++ {
			target.Wedge[target.wedgeIndex(i, 0, j)] *= 2
		}
	}
	before := append([]float64(nil), state.Score...)
	centerChange := make([]float64, state.Bins)
	centerChange[0] = 1
	updateRewiredCoordinates(state, target, request, ClosureProfile{}, centerChange, 0.5)
	if err := state.Validate(); err != nil {
		t.Fatal(err)
	}
	changed := false
	for index := range before {
		changed = changed || math.Abs(before[index]-state.Score[index]) > 1e-12
	}
	if !changed {
		t.Fatal("changing retained W did not change S_1")
	}
}

func TestCandidateProjectionReconcilesWedgeAndXi(t *testing.T) {
	request := testRequest()
	state, err := InitialState(request, LayerCandidate, ClosureProfile{}, rand.New(rand.NewPCG(9, 10)))
	if err != nil {
		t.Fatal(err)
	}
	state.Xi[state.xiIndex(0, 1, 1, 1)] += 0.25
	state.projectXi()
	if err := state.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestRetainedStateInvariantsAcrossSteps(t *testing.T) {
	for _, layer := range []Layer{LayerWedge, LayerHistogram, LayerCandidate, LayerTopology} {
		for _, steepness := range []float64{1, 4} {
			request := testRequest()
			request.Recommender.Steepness = steepness
			state, err := InitialState(
				request, layer, ClosureProfile{},
				rand.New(rand.NewPCG(100+uint64(layer), 200+uint64(steepness))),
			)
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewPCG(300+uint64(layer), 400+uint64(steepness)))
			for step := 1; step <= 5; step++ {
				if _, err := Step(state, request, ClosureProfile{}, rng); err != nil {
					t.Fatalf("layer=%s zeta=%g step=%d: %v", layer.String(), steepness, step, err)
				}
			}
		}
	}
}

func TestZeroWeightedLossFallsBackWithoutBreakingEdgeConservation(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	loss := sampleLossWithoutReplacement([]int{2, 3, 0}, 4, []float64{0, 0, 0}, rng)
	if loss[0]+loss[1]+loss[2] != 4 || loss[0] > 2 || loss[1] > 3 {
		t.Fatalf("invalid fallback loss sample %v", loss)
	}
}
