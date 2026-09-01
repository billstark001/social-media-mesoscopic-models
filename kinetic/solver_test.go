package kinetic

import (
	"encoding/json"
	"math"
	"smp-meso/numerics"
	"smp-meso/protocol"
	"testing"
)

func testRequest(dynamics, method, recommender string) RunRequest {
	empty, _ := protocol.EncodeFloat64([]float64{}, 0)
	return RunRequest{
		RequestID: "test", Population: 500, OpinionBins: 7, OutDegree: 4,
		RecommendationCount: 3, Steps: 4, RecordEvery: 2, Dt: 0.1,
		NoiseDiffusion: 0.001, ConfidenceMode: "cell_average",
		Dynamics:    DynamicsConfig{Type: dynamics, OpinionMethod: method, Tolerance: 0.45, Influence: 0.1, RewiringRate: 0.05},
		Recommender: RecommenderConfig{Type: recommender, Steepness: 2, RandomRatio: 0.1, OpinionTolerance: 0.4},
		Initial:     InitialConfig{Type: "uniform", OpinionMin: -1, OpinionMax: 1, Probabilities: empty},
		Resolution: ResolutionConfig{
			OpinionQuadraturePoints:    5,
			OpinionQuadratureRule:      numerics.UnitVarianceQuantileRule,
			ConfidenceQuadraturePoints: 5, ScoreMax: 20, DistanceGridSize: 64,
		},
		Observables: ObservablesConfig{
			Polarization: true, Subjective: true, Homophily: true, HomophilyRaw: true, Pathway: true,
			PolarizationFirstPassage: true, HomophilyFirstPassage: true,
			PolarizationThreshold: 0.2, HomophilyThreshold: 0.2,
			MinimumBandwidth: 0.01, ObjectiveEffectiveSamples: 10000,
		},
		Snapshots: SnapshotsConfig{RecordSteps: empty},
	}
}

func TestAllDynamicsMethodsAndRecommendersRun(t *testing.T) {
	for _, dynamics := range []string{"hk", "deffuant"} {
		for _, method := range []string{"measure", "fokker_planck"} {
			for _, recommender := range []string{"random", "opinion_random", "structure_random_l0", "structure_random_l1"} {
				request := testRequest(dynamics, method, recommender)
				result, err := Run(request)
				if err != nil {
					t.Fatalf("%s/%s/%s: %v", dynamics, method, recommender, err)
				}
				values, shape, err := result.Series.Time.DecodeFloat64()
				if err != nil || len(shape) != 1 || len(values) != 3 {
					t.Fatalf("invalid encoded time for %s/%s/%s: %v %v", dynamics, method, recommender, shape, err)
				}
				if result.Diagnostics.StateDimension <= request.OpinionBins+request.OpinionBins*request.OpinionBins {
					if recommender == "structure_random_l1" {
						t.Fatal("L1 did not retain directional wedge state")
					}
				} else if recommender != "structure_random_l1" {
					t.Fatal("non-L1 recommender retained wedge state")
				}
			}
		}
	}
}

func TestNoObservableProducesNoSeriesPayload(t *testing.T) {
	request := testRequest("hk", "measure", "random")
	request.Observables = ObservablesConfig{
		PolarizationThreshold: 0.2, HomophilyThreshold: 0.2,
		MinimumBandwidth: 0.01, ObjectiveEffectiveSamples: 100,
	}
	result, err := Run(request)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(result.Series)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != "{}" {
		t.Fatalf("unused observables leaked payload: %s", encoded)
	}
	if result.Snapshots != nil {
		t.Fatal("disabled snapshots allocated an output payload")
	}
}

func TestValidateRejectsOversizedKineticStateBeforeAllocation(t *testing.T) {
	request := testRequest("hk", "measure", "structure_random_l1")
	request.OpinionBins = 1 << 30
	if err := request.Validate(); err == nil {
		t.Fatal("oversized kinetic request was accepted")
	}
}

func TestSelectedSnapshotsAreBinaryAndConservative(t *testing.T) {
	request := testRequest("hk", "measure", "structure_random_l0")
	steps, _ := protocol.EncodeFloat64([]float64{0, 2, 4}, 3)
	request.Snapshots = SnapshotsConfig{
		RecordSteps: steps, Rho: true, Edge: true, Velocity: true,
	}
	result, err := Run(request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Snapshots == nil || result.Snapshots.RewiringFlux != nil {
		t.Fatalf("unexpected snapshot payload: %+v", result.Snapshots)
	}
	times, timeShape, err := result.Snapshots.Time.DecodeFloat64()
	if err != nil || len(timeShape) != 1 || timeShape[0] != 3 {
		t.Fatalf("invalid snapshot time shape %v: %v", timeShape, err)
	}
	if times[0] != 0 || times[1] != 0.2 || times[2] != 0.4 {
		t.Fatalf("unexpected snapshot times %v", times)
	}
	rho, rhoShape, err := result.Snapshots.Rho.DecodeFloat64()
	if err != nil || len(rhoShape) != 2 || rhoShape[0] != 3 || rhoShape[1] != request.OpinionBins {
		t.Fatalf("invalid rho snapshots %v: %v", rhoShape, err)
	}
	edge, edgeShape, err := result.Snapshots.Edge.DecodeFloat64()
	if err != nil || len(edgeShape) != 3 || edgeShape[0] != 3 || edgeShape[1] != request.OpinionBins || edgeShape[2] != request.OpinionBins {
		t.Fatalf("invalid edge snapshots %v: %v", edgeShape, err)
	}
	velocity, velocityShape, err := result.Snapshots.Velocity.DecodeFloat64()
	if err != nil || len(velocityShape) != 2 || len(velocity) != len(rho) {
		t.Fatalf("invalid velocity snapshots %v: %v", velocityShape, err)
	}
	for snapshot := range 3 {
		nodeMass := 0.0
		for bin := range request.OpinionBins {
			nodeMass += rho[snapshot*request.OpinionBins+bin]
		}
		edgeMass := 0.0
		for index := range request.OpinionBins * request.OpinionBins {
			edgeMass += edge[snapshot*request.OpinionBins*request.OpinionBins+index]
		}
		if math.Abs(nodeMass-1) > 1e-12 || math.Abs(edgeMass-float64(request.OutDegree)) > 1e-10 {
			t.Fatalf("snapshot %d mass rho=%g edge=%g", snapshot, nodeMass, edgeMass)
		}
	}
}

func TestFokkerPlanckFiniteVolumeSystemIsPositiveAndConservative(t *testing.T) {
	velocity := []float64{-0.1, -0.05, 0, 0.05, 0.1}
	diffusion := []float64{0.01, 0.02, 0.03, 0.02, 0.01}
	system, err := fokkerPlanckSystem(velocity, diffusion, 0.4, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	source := []float64{0.1, 0.2, 0.4, 0.2, 0.1}
	destination := make([]float64, len(source))
	if err := numerics.ActiveBackend.ApplyTridiagonal(destination, source, system); err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for _, value := range destination {
		if value < -1e-14 {
			t.Fatalf("negative finite-volume mass: %v", destination)
		}
		total += value
	}
	if math.Abs(total-1) > 1e-12 {
		t.Fatalf("finite-volume mass is %g", total)
	}
}

func TestGaussLegendreOrderIsExplicitAndExactForLowPolynomials(t *testing.T) {
	nodes, weights := gaussLegendre(5)
	for power := 0; power <= 9; power++ {
		observed := 0.0
		for index := range nodes {
			observed += weights[index] * math.Pow(nodes[index], float64(power))
		}
		want := 0.0
		if power%2 == 0 {
			want = 2 / float64(power+1)
		}
		if math.Abs(observed-want) > 1e-13 {
			t.Fatalf("power %d integral %g want %g", power, observed, want)
		}
	}
}

func TestL1SteepnessUsesScoreDistributionMoment(t *testing.T) {
	mean := 1.7
	pmf := make([]float64, 101)
	linear := cappedPoissonPower(pmf, mean, 1)
	quadratic := cappedPoissonPower(pmf, mean, 2)
	if math.Abs(linear-mean) > 1e-12 {
		t.Fatalf("linear Poisson moment %g want %g", linear, mean)
	}
	if math.Abs(quadratic-(mean+mean*mean)) > 1e-11 {
		t.Fatalf("quadratic Poisson moment %g", quadratic)
	}
}

func TestL1WedgeScoreHasFinitePopulationCandidateNormalization(t *testing.T) {
	request := testRequest("hk", "measure", "structure_random_l1")
	request.OpinionBins = 5
	request.Population = 500
	request.OutDegree = 15
	grid := &gridGeometry{Axis: []float64{-1, -0.5, 0, 0.5, 1}}
	current, err := newState(request, grid)
	if err != nil {
		t.Fatal(err)
	}
	scoreMass := wedgeScoreMass(current)
	candidateMass := current.Rho[0] * current.Rho[0]
	observed := scoreMass[0] / (float64(request.Population) * candidateMass)
	want := 4 * float64(request.OutDegree*request.OutDegree) / float64(request.Population)
	if math.Abs(observed-want) > 1e-12 {
		t.Fatalf("independent L1 common-neighbor mean=%g want %g", observed, want)
	}
}

func TestHKMeasureRetainsFiniteExposureNoUpdateProbability(t *testing.T) {
	request := testRequest("hk", "measure", "random")
	request.OpinionBins = 5
	request.OutDegree = 1
	request.RecommendationCount = 1
	request.Dt = 0.5
	request.Dynamics.Influence = 0.4
	grid := &gridGeometry{Axis: []float64{-1, -0.5, 0, 0.5, 1}}
	current := &state{request: request, grid: grid}
	values := fields{
		Neighbors: exposureChannel{
			ConcordantMass:     []float64{0, 0, 0.5, 0, 0},
			MeanDisplacement:   []float64{0, 0, 0.4, 0, 0},
			SecondDisplacement: []float64{0, 0, 0.2, 0, 0},
		},
		Recommendations: exposureChannel{
			ConcordantMass:     make([]float64, 5),
			MeanDisplacement:   make([]float64, 5),
			SecondDisplacement: make([]float64, 5),
		},
	}

	moments := hkIncrementMoments(current, values)
	if math.Abs(moments.Mean[2]-0.2) > 1e-14 || math.Abs(moments.Second[2]-0.1) > 1e-14 {
		t.Fatalf("finite-exposure moments are mean=%g second=%g", moments.Mean[2], moments.Second[2])
	}
	quadrature, err := numerics.NewNormalQuadrature(
		request.Resolution.OpinionQuadratureRule,
		request.Resolution.OpinionQuadraturePoints,
	)
	if err != nil {
		t.Fatal(err)
	}
	transition := hkReferenceTransition(current, values, quadrature)
	row := transition[2*request.OpinionBins : 3*request.OpinionBins]
	destinationMean := 0.0
	for index, probability := range row {
		destinationMean += probability * grid.Axis[index]
	}
	want := request.Dt * request.Dynamics.Influence * moments.Mean[2]
	if math.Abs(destinationMean-want) > 1e-14 {
		t.Fatalf("measure destination mean=%g want %g", destinationMean, want)
	}
	if math.Abs(destinationMean-0.08) < 1e-3 {
		t.Fatal("measure collapsed E[1_{C>0} S/C] to E[S]/E[C]")
	}
}

func TestDepositNormalUsesGaussHermiteWeights(t *testing.T) {
	quadrature, err := numerics.NewNormalQuadrature(numerics.GaussHermiteRule, 3)
	if err != nil {
		t.Fatal(err)
	}
	axis := append([]float64(nil), quadrature.Nodes...)
	row := make([]float64, len(axis))
	depositNormal(row, axis, quadrature, 0, 1, 6)
	want := []float64{1, 4, 1}
	for index := range row {
		if math.Abs(row[index]-want[index]) > 1e-14 {
			t.Fatalf("row=%v, want %v", row, want)
		}
	}
}

func TestStrictDecoderRejectsMissingAndUnknownFields(t *testing.T) {
	request := testRequest("hk", "measure", "random")
	data, _ := json.Marshal(request)
	var object map[string]any
	_ = json.Unmarshal(data, &object)
	delete(object, "dt")
	missing, _ := json.Marshal(object)
	if _, err := DecodeRequest(missing); err == nil {
		t.Fatal("missing dt was accepted")
	}
	object["dt"] = 0.1
	object["unexpected"] = 1
	unknown, _ := json.Marshal(object)
	if _, err := DecodeRequest(unknown); err == nil {
		t.Fatal("unknown field was accepted")
	}
}
