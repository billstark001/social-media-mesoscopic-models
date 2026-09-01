package solver

import (
	"math"
	"reflect"
	"smp-meso/config"
	"smp-meso/protocol"
	"testing"
)

func solverRequest() config.RunRequest {
	return config.RunRequest{
		RequestID: "solver", Layer: "base", Population: 40, OpinionBins: 5,
		OutDegree: 3, RecommendationCount: 2, MaxSteps: 4,
		Paths: 4, IntervalPaths: 2, AmbiguitySamples: 3,
		ConfidenceLevel: 0.9, Workers: 2, Seed: 9, MajorClusterMass: 0.02,
		Dynamics:    config.DynamicsConfig{Type: "deffuant", Tolerance: 0.45, Influence: 0.1, RewiringRate: 0.05},
		Recommender: config.RecommenderConfig{Type: "opinion_random", Steepness: 2, RandomRatio: 0.1, OpinionTolerance: 0.4, NoiseStd: 0.1, NoiseQuadraturePoints: 3},
		Initial:     config.InitialConfig{Type: "uniform", OpinionMin: -1, OpinionMax: 1, Probabilities: []float64{}},
		Resolution: config.ResolutionConfig{
			ScoreMax: 6, AvailabilityBins: 4, ComponentSizeBins: 4,
			OpinionQuadrature: 3, OpinionQuadratureRule: "unit_variance_quantile",
		},
		Closure: config.ClosureConfig{MotifRelaxation: 0.2, HistogramRelaxation: 0.2, CandidateRelaxation: 0.2, TopologyRelaxation: 0.2},
		FastSlow: config.FastSlowConfig{Mode: "unsplit", RatioThreshold: 10, MaxSubsteps: 50,
			ZeroEventBatches: 3, ResidualTolerance: 1e-12, ZeroEventResidual: 0.25},
		Ambiguity: config.AmbiguityConfig{EligibilityCorrelationRadius: 0.5, ScoreAvailabilityRadius: 0.5, MotifPersistenceRadius: 0.5, BridgeBiasRadius: 0.5, ComponentMixRadius: 0.5},
	}
}

func TestResultNormalizesAndIntervalContainsPoint(t *testing.T) {
	result, err := Run(solverRequest())
	if err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for category, probability := range result.Point.Probabilities {
		total += probability
		if result.Interval.Lower[category] > probability || result.Interval.Upper[category] < probability {
			t.Fatalf("category %d point %g outside [%g,%g]", category, probability,
				result.Interval.Lower[category], result.Interval.Upper[category])
		}
	}
	if math.Abs(total-1) > 1e-12 {
		t.Fatalf("probability sum=%g", total)
	}
	if result.Interval.ConfidenceScope == "" ||
		len(result.Interval.ScenarioProfiles) != result.Interval.ScenarioCount {
		t.Fatalf("incomplete interval provenance: %+v", result.Interval)
	}
}

func TestRunValidatesProgrammaticRequestsAndNormalizesEnums(t *testing.T) {
	invalid := solverRequest()
	invalid.Workers = 0
	if _, err := Run(invalid); err == nil {
		t.Fatal("programmatic request bypassed validation")
	}
	request := solverRequest()
	request.Dynamics.Type = " HK "
	request.Recommender.Type = " Opinion_Random "
	request.Initial.Type = " Uniform "
	if _, err := Run(request); err != nil {
		t.Fatalf("normalized enum request failed: %v", err)
	}
}

func TestRetainedLayersRemoveAmbiguityCoordinates(t *testing.T) {
	request := solverRequest()
	previous := 6
	for layer := config.LayerBase; layer <= config.LayerTopology; layer++ {
		count := len(activeCoordinates(request, layer))
		if count >= previous {
			t.Fatalf("layer %s has %d active coordinates after %d", layer.String(), count, previous)
		}
		previous = count
	}
}

func TestNaiveAndFastSlowDiagnostics(t *testing.T) {
	request := solverRequest()
	request.Layer = "naive"
	request.FastSlow.Mode = "conditional_absorption"
	request.FastSlow.RatioThreshold = 1
	request.FastSlow.MaxSubsteps = 20
	request.Dynamics.Influence = 0.005
	request.Dynamics.RewiringRate = 0.3
	result, err := Run(request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Diagnostics.Decomposition != "conditional_absorption" {
		t.Fatalf("unexpected decomposition diagnostics: %+v", result.Diagnostics)
	}
	if result.Diagnostics.StateDimension != request.OpinionBins+request.OpinionBins*request.OpinionBins {
		t.Fatalf("naive state dimension=%d", result.Diagnostics.StateDimension)
	}
	if result.Point.FastSlowAppliedPaths != request.Paths || result.Point.MeanFastSubsteps <= 0 {
		t.Fatalf("missing fast-slow path diagnostics: %+v", result.Point)
	}
}

func TestProgressDoesNotChangeNumericalResult(t *testing.T) {
	request := solverRequest()
	quiet, err := Run(request)
	if err != nil {
		t.Fatal(err)
	}
	events := make([]protocol.ProgressEvent, 0)
	observed, err := RunWithProgress(request, 1, func(event protocol.ProgressEvent) {
		events = append(events, event)
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(quiet.Point, observed.Point) ||
		!reflect.DeepEqual(quiet.Interval, observed.Interval) {
		t.Fatal("enabling progress changed the numerical result")
	}
	completedPaths := 0
	requestStarted, requestCompleted := false, false
	for _, event := range events {
		switch event.Event {
		case "request_started":
			requestStarted = true
		case "path_completed":
			completedPaths++
		case "request_completed":
			requestCompleted = true
			if event.CompletedPaths != request.Paths+observed.Interval.ScenarioCount*request.IntervalPaths {
				t.Fatalf("request completion reports %d paths", event.CompletedPaths)
			}
		}
		if event.RequestID != request.RequestID || event.Layer != "base" {
			t.Fatalf("event lacks request context: %+v", event)
		}
	}
	expectedPaths := request.Paths + observed.Interval.ScenarioCount*request.IntervalPaths
	if !requestStarted || !requestCompleted || completedPaths != expectedPaths {
		t.Fatalf("start=%v complete=%v completed paths=%d want %d",
			requestStarted, requestCompleted, completedPaths, expectedPaths)
	}
}
