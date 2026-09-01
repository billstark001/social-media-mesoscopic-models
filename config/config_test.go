package config

import (
	"encoding/json"
	"strings"
	"testing"
)

func validRequest() RunRequest {
	return RunRequest{
		RequestID: "test", Layer: "base", Population: 60, OpinionBins: 5,
		OutDegree: 4, RecommendationCount: 3, MaxSteps: 10,
		Paths: 2, IntervalPaths: 1, AmbiguitySamples: 1,
		ConfidenceLevel: 0.95, Workers: 1, Seed: 7, MajorClusterMass: 0.02,
		Dynamics: DynamicsConfig{Type: "hk", Tolerance: 0.45, Influence: 0.1, RewiringRate: 0.05},
		Recommender: RecommenderConfig{
			Type: "structure_random", Steepness: 1, RandomRatio: 0,
			OpinionTolerance: 0.4, NoiseStd: 0, NoiseQuadraturePoints: 3,
		},
		Initial:    InitialConfig{Type: "uniform", OpinionMin: -1, OpinionMax: 1, Probabilities: []float64{}},
		Resolution: ResolutionConfig{ScoreMax: 8, AvailabilityBins: 5, ComponentSizeBins: 5, OpinionQuadrature: 3},
		Closure:    ClosureConfig{MotifRelaxation: 0.2, HistogramRelaxation: 0.2, CandidateRelaxation: 0.2, TopologyRelaxation: 0.2},
		FastSlow: FastSlowConfig{Mode: "unsplit", RatioThreshold: 10, MaxSubsteps: 50,
			ZeroEventBatches: 3, ResidualTolerance: 1e-12, ZeroEventResidual: 0.25},
		Ambiguity: AmbiguityConfig{
			EligibilityCorrelationRadius: 0.5, ScoreAvailabilityRadius: 0.5,
			MotifPersistenceRadius: 0.5, BridgeBiasRadius: 0.5, ComponentMixRadius: 0.5,
		},
	}
}

func TestDecodeRequestRequiresExplicitZeroFields(t *testing.T) {
	data, err := json.Marshal(validRequest())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeRequest(data); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}
	missing := strings.Replace(string(data), `,"random_ratio":0`, "", 1)
	if _, err := DecodeRequest([]byte(missing)); err == nil || !strings.Contains(err.Error(), "random_ratio") {
		t.Fatalf("missing explicit zero was not reported: %v", err)
	}
}

func TestDecodeRequestRejectsUnknownFields(t *testing.T) {
	data, _ := json.Marshal(validRequest())
	withUnknown := strings.TrimSuffix(string(data), "}") + `,"mystery":1}`
	if _, err := DecodeRequest([]byte(withUnknown)); err == nil {
		t.Fatal("unknown field accepted")
	}
}

func TestFastSlowConfigurationIsExplicitAndValidated(t *testing.T) {
	data, err := json.Marshal(validRequest())
	if err != nil {
		t.Fatal(err)
	}
	missing := strings.Replace(string(data), `,"fast_slow":`, `,"omitted_fast_slow":`, 1)
	if _, err := DecodeRequest([]byte(missing)); err == nil || !strings.Contains(err.Error(), "fast_slow") {
		t.Fatalf("missing fast_slow object was not reported: %v", err)
	}
	request := validRequest()
	request.FastSlow.Mode = "stationary_average"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "fast_slow.mode") {
		t.Fatalf("unsupported mode was not rejected: %v", err)
	}
}

func TestAllLayerNames(t *testing.T) {
	for _, name := range []string{"naive", "base", "wedge", "histogram", "candidate", "topology"} {
		layer, err := ParseLayer(name)
		if err != nil {
			t.Fatal(err)
		}
		if layer.String() != name {
			t.Fatalf("round trip %q -> %q", name, layer.String())
		}
	}
}
