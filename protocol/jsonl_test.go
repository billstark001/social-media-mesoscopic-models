package protocol

import (
	"bytes"
	"encoding/json"
	"smp-meso/config"
	"smp-meso/solver"
	"testing"
)

func protocolRequest(id string) config.RunRequest {
	return config.RunRequest{
		RequestID: id, Layer: "base", Population: 30, OpinionBins: 4,
		OutDegree: 3, RecommendationCount: 2, MaxSteps: 2,
		Paths: 1, IntervalPaths: 1, AmbiguitySamples: 1,
		ConfidenceLevel: 0.95, Workers: 1, Seed: 9, MajorClusterMass: 0.05,
		Dynamics: config.DynamicsConfig{
			Type: "hk", Tolerance: 0.45, Influence: 0.05, RewiringRate: 0.02,
		},
		Recommender: config.RecommenderConfig{
			Type: "random", Steepness: 0, RandomRatio: 0,
			OpinionTolerance: 0.4, NoiseStd: 0, NoiseQuadraturePoints: 1,
		},
		Initial: config.InitialConfig{
			Type: "uniform", OpinionMin: -1, OpinionMax: 1, Probabilities: []float64{},
		},
		Resolution: config.ResolutionConfig{
			ScoreMax: 5, AvailabilityBins: 3, ComponentSizeBins: 3, OpinionQuadrature: 1,
		},
		Closure: config.ClosureConfig{
			MotifRelaxation: 0.2, HistogramRelaxation: 0.2,
			CandidateRelaxation: 0.2, TopologyRelaxation: 0.2,
		},
		FastSlow: config.FastSlowConfig{Mode: "unsplit", RatioThreshold: 10, MaxSubsteps: 50,
			ZeroEventBatches: 3, ResidualTolerance: 1e-12, ZeroEventResidual: 0.25},
		Ambiguity: config.AmbiguityConfig{},
	}
}

func TestJSONLContinuesAfterInvalidRequest(t *testing.T) {
	valid, err := json.Marshal(protocolRequest("good"))
	if err != nil {
		t.Fatal(err)
	}
	input := bytes.NewBuffer(nil)
	input.WriteString(`{"request_id":"bad"}` + "\n")
	input.Write(valid)
	input.WriteByte('\n')
	output := bytes.NewBuffer(nil)
	if err := RunJSONL(input, output); err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(output)
	var first, second Response
	if err := decoder.Decode(&first); err != nil {
		t.Fatal(err)
	}
	if err := decoder.Decode(&second); err != nil {
		t.Fatal(err)
	}
	if first.RequestID != "bad" || first.Error == "" || first.Result != nil {
		t.Fatalf("unexpected invalid response: %+v", first)
	}
	if second.RequestID != "good" || second.Error != "" || second.Result == nil {
		t.Fatalf("unexpected valid response: %+v", second)
	}
}

func TestJSONLProgressCarriesBatchLine(t *testing.T) {
	valid, err := json.Marshal(protocolRequest("good"))
	if err != nil {
		t.Fatal(err)
	}
	input := bytes.NewBuffer(nil)
	input.WriteString(`{"request_id":"bad"}` + "\n")
	input.Write(valid)
	input.WriteByte('\n')
	events := make([]solver.ProgressEvent, 0)
	if err := RunJSONLWithProgress(input, bytes.NewBuffer(nil), 1, func(event solver.ProgressEvent) {
		events = append(events, event)
	}); err != nil {
		t.Fatal(err)
	}
	rejected, completed := false, false
	for _, event := range events {
		if event.Event == "request_rejected" {
			rejected = event.BatchIndex == 1 && event.RequestID == "bad"
		}
		if event.Event == "request_completed" {
			completed = event.BatchIndex == 2 && event.RequestID == "good"
		}
	}
	if !rejected || !completed {
		t.Fatalf("missing line-aware events: %+v", events)
	}
}
