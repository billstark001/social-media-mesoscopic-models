package lifted

import (
	"math"
	"math/rand/v2"
	"reflect"
	"smp-meso/config"
	"smp-meso/numerics"
	"testing"
)

func TestNaiveRetainsOnlyRhoAndEdge(t *testing.T) {
	request := testRequest()
	profile := ClosureProfile{ScoreAvailability: 0.4}
	state, err := InitialState(request, LayerNaive, profile, rand.New(rand.NewPCG(31, 32)))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := state.Dimension(), state.Bins+state.Bins*state.Bins; got != want {
		t.Fatalf("naive dimension=%d want rho+E=%d", got, want)
	}
	if _, err := Step(state, request, profile, rand.New(rand.NewPCG(33, 34))); err != nil {
		t.Fatal(err)
	}
	fresh, err := independentTarget(state, request, profile, state.Edge)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(state.Candidate, fresh.Candidate) || !reflect.DeepEqual(state.Score, fresh.Score) {
		t.Fatal("naive recommendation caches were not reconstructed from current rho/E")
	}
}

func TestFastSlowFallsBackBelowRatioThreshold(t *testing.T) {
	request := testRequest()
	request.FastSlow.Mode = "conditional_absorption"
	request.FastSlow.RatioThreshold = 10
	left, err := InitialState(request, LayerBase, ClosureProfile{}, rand.New(rand.NewPCG(41, 42)))
	if err != nil {
		t.Fatal(err)
	}
	right := left.Clone()
	leftDiagnostics, err := Step(left, request, ClosureProfile{}, rand.New(rand.NewPCG(43, 44)))
	if err != nil {
		t.Fatal(err)
	}
	rightDiagnostics, err := FastSlowStep(right, request, ClosureProfile{}, rand.New(rand.NewPCG(43, 44)))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(left, right) || !reflect.DeepEqual(leftDiagnostics, rightDiagnostics) {
		t.Fatal("below-threshold fast-slow route did not exactly reuse unsplit step")
	}
}

func TestFastSlowFreezesRhoDuringConditionalRewiring(t *testing.T) {
	request := testRequest()
	request.FastSlow.Mode = "conditional_absorption"
	request.FastSlow.RatioThreshold = 1
	request.FastSlow.MaxSubsteps = 40
	request.Dynamics.Influence = 0
	request.Dynamics.RewiringRate = 0.3
	state, err := InitialState(request, LayerNaive, ClosureProfile{}, rand.New(rand.NewPCG(51, 52)))
	if err != nil {
		t.Fatal(err)
	}
	before := append([]float64(nil), state.Rho...)
	diagnostics, err := FastSlowStep(state, request, ClosureProfile{}, rand.New(rand.NewPCG(53, 54)))
	if err != nil {
		t.Fatal(err)
	}
	if !diagnostics.FastSlowApplied || diagnostics.FastSubsteps == 0 {
		t.Fatalf("fast-slow projection was not applied: %+v", diagnostics)
	}
	if !reflect.DeepEqual(before, state.Rho) {
		t.Fatal("rho changed despite zero-influence slow step")
	}
	if err := state.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestFastSlowStepSupportsEveryLayer(t *testing.T) {
	request := testRequest()
	request.FastSlow.Mode = "conditional_absorption"
	request.FastSlow.RatioThreshold = 1
	request.FastSlow.MaxSubsteps = 4
	request.FastSlow.ZeroEventBatches = 2
	request.Dynamics.Influence = 0
	request.Dynamics.RewiringRate = 0.3
	for layer := LayerNaive; layer <= LayerTopology; layer++ {
		state, err := InitialState(
			request,
			layer,
			ClosureProfile{},
			rand.New(rand.NewPCG(61+uint64(layer), 71+uint64(layer))),
		)
		if err != nil {
			t.Fatalf("initialize layer %s: %v", layer, err)
		}
		diagnostics, err := FastSlowStep(
			state,
			request,
			ClosureProfile{},
			rand.New(rand.NewPCG(81+uint64(layer), 91+uint64(layer))),
		)
		if err != nil {
			t.Fatalf("fast-slow layer %s: %v", layer, err)
		}
		if !diagnostics.FastSlowApplied {
			t.Fatalf("fast-slow layer %s was not activated", layer)
		}
		if err := state.Validate(); err != nil {
			t.Fatalf("validate layer %s: %v", layer, err)
		}
	}
}

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
		Initial: config.InitialConfig{Type: "uniform", OpinionMin: -1, OpinionMax: 1, Probabilities: []float64{}},
		Resolution: config.ResolutionConfig{
			ScoreMax: 8, AvailabilityBins: 5, ComponentSizeBins: 5,
			OpinionQuadrature:     3,
			OpinionQuadratureRule: numerics.UnitVarianceQuantileRule,
		},
		Closure: config.ClosureConfig{MotifRelaxation: 0.2, HistogramRelaxation: 0.2, CandidateRelaxation: 0.2, TopologyRelaxation: 0.2},
		FastSlow: config.FastSlowConfig{Mode: "unsplit", RatioThreshold: 10, MaxSubsteps: 50,
			ZeroEventBatches: 3, ResidualTolerance: 1e-12, ZeroEventResidual: 0.25},
		Ambiguity: config.AmbiguityConfig{
			EligibilityCorrelationRadius: 0.5, ScoreAvailabilityRadius: 0.5,
			MotifPersistenceRadius: 0.5, BridgeBiasRadius: 0.5, ComponentMixRadius: 0.5,
		},
	}
}

func TestSixLayersInitializeAndIncreaseDimension(t *testing.T) {
	request := testRequest()
	previous := 0
	expectedBlocks := [...]coordinateBlocks{
		0,
		blockBase,
		blockBase | blockWedge,
		blockBase | blockWedge | blockHistogram,
		blockBase | blockWedge | blockHistogram | blockCandidate,
		blockBase | blockWedge | blockHistogram | blockCandidate | blockTopology,
	}
	for layer := LayerNaive; layer <= LayerTopology; layer++ {
		rng := rand.New(rand.NewPCG(10+uint64(layer), 20+uint64(layer)))
		state, err := InitialState(request, layer, ClosureProfile{}, rng)
		if err != nil {
			t.Fatalf("layer %s: %v", layer.String(), err)
		}
		if state.plan != &layerPlans[layer] || state.plan.layer != layer {
			t.Fatalf("layer %s did not bind its immutable plan", layer.String())
		}
		if state.plan.blocks != expectedBlocks[layer] {
			t.Fatalf("layer %s blocks=%b, want %b", layer.String(), state.plan.blocks, expectedBlocks[layer])
		}
		if state.Clone().plan != state.plan {
			t.Fatalf("layer %s clone changed plan", layer.String())
		}
		if state.Dimension() <= previous {
			t.Fatalf("dimension did not increase: %d <= %d", state.Dimension(), previous)
		}
		previous = state.Dimension()
	}
}

func TestInitialStateRejectsUnknownLayerPlan(t *testing.T) {
	_, err := InitialState(
		testRequest(),
		Layer(99),
		ClosureProfile{},
		rand.New(rand.NewPCG(101, 102)),
	)
	if err == nil {
		t.Fatal("unknown layer plan was accepted")
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

func TestDepositNormalUsesGaussHermiteWeights(t *testing.T) {
	quadrature, err := numerics.NewNormalQuadrature(numerics.GaussHermiteRule, 3)
	if err != nil {
		t.Fatal(err)
	}
	axis := append([]float64(nil), quadrature.Nodes...)
	row := make([]float64, len(axis))
	depositNormal(row, axis, 0, 1, 6, quadrature)
	want := []float64{1, 4, 1}
	for index := range row {
		if math.Abs(row[index]-want[index]) > 1e-14 {
			t.Fatalf("row=%v, want %v", row, want)
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

func TestExtremeClosureProfilesInitializeAllLayers(t *testing.T) {
	profiles := []ClosureProfile{
		{EligibilityCorrelation: 1}, {EligibilityCorrelation: -1},
		{ScoreAvailability: 1}, {ScoreAvailability: -1},
		{MotifPersistence: 1}, {MotifPersistence: -1},
		{BridgeBias: 1}, {BridgeBias: -1},
		{ComponentMix: 1}, {ComponentMix: -1},
	}
	for _, layer := range []Layer{LayerNaive, LayerBase, LayerWedge, LayerHistogram, LayerCandidate, LayerTopology} {
		for profileIndex, profile := range profiles {
			request := testRequest()
			request.Recommender.Steepness = 1
			_, err := InitialState(
				request, layer, profile,
				rand.New(rand.NewPCG(uint64(500+profileIndex), uint64(600+layer))),
			)
			if err != nil {
				t.Fatalf("layer=%s profile=%+v: %v", layer.String(), profile, err)
			}
		}
	}
}
