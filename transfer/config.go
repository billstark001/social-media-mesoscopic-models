// Package transfer runs paired deterministic and stochastic terminal experiments
// on the same finite-exposure lifted kernels and explicit initial-state seeds.
package transfer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"smp-meso/config"
	"smp-meso/numerics"
	"smp-meso/protocol"
)

type Evolution string

const (
	Deterministic Evolution = "deterministic"
	Stochastic    Evolution = "stochastic"
)

type Variant struct {
	Layer     string    `json:"layer"`
	Evolution Evolution `json:"evolution"`
}

// Seed separates initialization from continuation. Replicate identifies a
// requested pair, not a source of randomness; repeated InitSeed values are valid.
type Seed struct {
	Replicate   int    `json:"replicate"`
	InitSeed    uint64 `json:"init_seed"`
	ProcessSeed uint64 `json:"process_seed"`
}

type InitialConfig struct {
	Type          string                `json:"type"`
	OpinionMin    float64               `json:"opinion_min"`
	OpinionMax    float64               `json:"opinion_max"`
	Probabilities protocol.EncodedArray `json:"probabilities"`
}

type ResolutionConfig struct {
	ScoreMax              int    `json:"score_max"`
	OpinionQuadrature     int    `json:"opinion_quadrature_points"`
	OpinionQuadratureRule string `json:"opinion_quadrature_rule"`
}

type ClosureConfig struct {
	MotifRelaxation float64 `json:"motif_relaxation"`
}

type SnapshotsConfig struct {
	RecordSteps    protocol.EncodedArray `json:"record_steps"`
	Rho            bool                  `json:"rho"`
	Edge           bool                  `json:"edge"`
	Candidate      bool                  `json:"candidate"`
	Score          bool                  `json:"score"`
	FinalRho       bool                  `json:"final_rho"`
	FinalEdge      bool                  `json:"final_edge"`
	FinalCandidate bool                  `json:"final_candidate"`
	FinalScore     bool                  `json:"final_score"`
}

// RunRequest contains only experiment controls and model parameters used by
// naive/base unsplit evolution. All JSON fields are required, including false
// snapshot switches. Lifted interval and fast-slow controls are not accepted.
type RunRequest struct {
	RequestID                  string                   `json:"request_id"`
	Population                 int                      `json:"population"`
	OpinionBins                int                      `json:"opinion_bins"`
	OutDegree                  int                      `json:"out_degree"`
	RecommendationCount        int                      `json:"recommendation_count"`
	MaxSteps                   int                      `json:"max_steps"`
	Workers                    int                      `json:"workers"`
	MajorClusterMass           float64                  `json:"major_cluster_mass"`
	TerminalPositionResolution float64                  `json:"terminal_position_resolution"`
	TerminalMassResolution     float64                  `json:"terminal_mass_resolution"`
	Dynamics                   config.DynamicsConfig    `json:"dynamics"`
	Recommender                config.RecommenderConfig `json:"recommender"`
	Initial                    InitialConfig            `json:"initial"`
	Resolution                 ResolutionConfig         `json:"resolution"`
	Closure                    ClosureConfig            `json:"closure"`
	Variants                   []Variant                `json:"variants"`
	Seeds                      []Seed                   `json:"seeds"`
	Snapshots                  SnapshotsConfig          `json:"snapshots"`
}

func DecodeRequest(data []byte) (RunRequest, error) {
	if err := requireFields(data, reflect.TypeFor[RunRequest](), "request"); err != nil {
		return RunRequest{}, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var request RunRequest
	if err := decoder.Decode(&request); err != nil {
		return RunRequest{}, err
	}
	if err := request.Validate(); err != nil {
		return RunRequest{}, err
	}
	return request, nil
}

// Walk the typed schema rather than maintaining a second list of required
// fields. In particular, null is not an explicit zero, false, or empty list.
func requireFields(data json.RawMessage, typ reflect.Type, path string) error {
	if len(data) == 0 || bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return fmt.Errorf("missing or null required field %s", path)
	}
	switch typ.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			name := field.Tag.Get("json")
			if err := requireFields(fields[name], field.Type, path+"."+name); err != nil {
				return err
			}
		}
	case reflect.Slice:
		var values []json.RawMessage
		if err := json.Unmarshal(data, &values); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		for i, value := range values {
			if err := requireFields(value, typ.Elem(), fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	}
	return nil
}

// Adapter values for unrelated lifted ensemble/layer controls are inert:
// this runner calls only InitialState, Step and DeterministicStep. Physical
// parameters and their validation remain shared with config.RunRequest.
func (r RunRequest) liftedRequest(layer config.Layer, probabilities []float64) config.RunRequest {
	return config.RunRequest{
		RequestID: r.RequestID, Layer: layer.String(), Population: r.Population,
		OpinionBins: r.OpinionBins, OutDegree: r.OutDegree, RecommendationCount: r.RecommendationCount,
		MaxSteps: r.MaxSteps, Paths: 1, IntervalPaths: 1, AmbiguitySamples: 1,
		ConfidenceLevel: .95, Workers: 1, MajorClusterMass: r.MajorClusterMass,
		TerminalPositionResolution: r.TerminalPositionResolution, TerminalMassResolution: r.TerminalMassResolution,
		Dynamics: r.Dynamics, Recommender: r.Recommender,
		Initial: config.InitialConfig{Type: r.Initial.Type, OpinionMin: r.Initial.OpinionMin,
			OpinionMax: r.Initial.OpinionMax, Probabilities: probabilities},
		Resolution: config.ResolutionConfig{ScoreMax: r.Resolution.ScoreMax, AvailabilityBins: 2,
			ComponentSizeBins: 2, OpinionQuadrature: r.Resolution.OpinionQuadrature,
			OpinionQuadratureRule: r.Resolution.OpinionQuadratureRule},
		Closure:  config.ClosureConfig{MotifRelaxation: r.Closure.MotifRelaxation},
		FastSlow: config.FastSlowConfig{Mode: "unsplit", MaxSubsteps: 1, ZeroEventBatches: 1},
	}
}

func (r RunRequest) snapshotSteps() ([]int, error) {
	values, shape, err := r.Snapshots.RecordSteps.DecodeFloat64()
	if err != nil {
		return nil, fmt.Errorf("snapshots.record_steps: %w", err)
	}
	if len(shape) != 1 {
		return nil, fmt.Errorf("snapshots.record_steps must be one-dimensional")
	}
	steps := make([]int, len(values))
	for i, value := range values {
		// Steps are float64 payloads; require exact integers in both representations.
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > float64(r.MaxSteps) ||
			value >= float64(math.MaxInt) || value > 1<<53 || math.Trunc(value) != value {
			return nil, fmt.Errorf("snapshot step %g is not an integer in [0,max_steps]", value)
		}
		steps[i] = int(value)
		if i > 0 && steps[i] <= steps[i-1] {
			return nil, fmt.Errorf("snapshot steps must be strictly increasing")
		}
	}
	return steps, nil
}

func (r RunRequest) Validate() error {
	probabilities, shape, err := r.Initial.Probabilities.DecodeFloat64()
	if err != nil {
		return fmt.Errorf("initial.probabilities: %w", err)
	}
	if len(shape) != 1 {
		return fmt.Errorf("initial.probabilities must be one-dimensional")
	}
	if err := r.liftedRequest(config.LayerBase, probabilities).Validate(); err != nil {
		return err
	}
	if _, err := numerics.CheckedProduct("population edge count", r.Population, r.OutDegree); err != nil {
		return err
	}
	if r.Workers < 1 || len(r.Seeds) == 0 || len(r.Variants) == 0 {
		return fmt.Errorf("workers, seeds and variants must be nonempty/positive")
	}
	variants := make(map[Variant]bool)
	for _, variant := range r.Variants {
		layer, err := config.ParseLayer(variant.Layer)
		if err != nil {
			return err
		}
		if layer != config.LayerNaive && layer != config.LayerBase {
			return fmt.Errorf("transfer supports naive and base layers only")
		}
		if variant.Evolution != Deterministic && variant.Evolution != Stochastic {
			return fmt.Errorf("unsupported evolution %q", variant.Evolution)
		}
		variant.Layer = layer.String()
		if variants[variant] {
			return fmt.Errorf("duplicate variant %s/%s", variant.Layer, variant.Evolution)
		}
		variants[variant] = true
	}
	ids := make(map[int]bool)
	for _, seed := range r.Seeds {
		if seed.Replicate < 0 || ids[seed.Replicate] {
			return fmt.Errorf("replicate identifiers must be nonnegative and unique")
		}
		ids[seed.Replicate] = true
	}
	steps, err := r.snapshotSteps()
	if err != nil {
		return err
	}
	return r.validateWorkingSet(len(steps))
}

func (r RunRequest) validateWorkingSet(records int) error {
	jobs, err := numerics.CheckedProduct("transfer paths", len(r.Seeds), len(r.Variants))
	if err != nil {
		return err
	}
	square, err := numerics.CheckedProduct("transfer matrix", r.OpinionBins, r.OpinionBins)
	if err != nil {
		return err
	}
	// Include reconstruction targets, hash serialization, sampled kernels and
	// temporary compressed-array buffers; output is retained until batch encoding.
	workspace, err := numerics.CheckedProduct("transfer matrix workspace", 64, square)
	if err != nil {
		return err
	}
	aux, err := numerics.CheckedSum("transfer auxiliary dimensions", r.Resolution.ScoreMax,
		r.Resolution.OpinionQuadrature, r.OutDegree, r.RecommendationCount, 1)
	if err != nil {
		return err
	}
	aux, err = numerics.CheckedProduct("transfer auxiliary workspace", 16, aux)
	if err != nil {
		return err
	}
	workspace, err = numerics.CheckedSum("transfer workspace", workspace, aux)
	if err != nil {
		return err
	}
	workspace, err = numerics.CheckedProduct("transfer workers", min(r.Workers, jobs), workspace)
	if err != nil {
		return err
	}
	historyWidth, finalWidth := 3, 0 // step, score sum, maximum rho change
	for _, field := range []struct {
		history, final bool
		width          int
	}{
		{r.Snapshots.Rho, r.Snapshots.FinalRho, r.OpinionBins},
		{r.Snapshots.Edge, r.Snapshots.FinalEdge, square},
		{r.Snapshots.Candidate, r.Snapshots.FinalCandidate, square},
		{r.Snapshots.Score, r.Snapshots.FinalScore, square},
	} {
		if field.history {
			historyWidth, err = numerics.CheckedSum("snapshot width", historyWidth, field.width)
			if err != nil {
				return err
			}
		}
		if field.final {
			finalWidth, err = numerics.CheckedSum("final width", finalWidth, field.width)
			if err != nil {
				return err
			}
		}
	}
	history, err := numerics.CheckedProduct("snapshot history", records, historyWidth)
	if err != nil {
		return err
	}
	metadata, err := numerics.CheckedProduct("terminal components", 32, r.OpinionBins)
	if err != nil {
		return err
	}
	output, err := numerics.CheckedSum("path output", history, finalWidth, metadata, 256)
	if err != nil {
		return err
	}
	output, err = numerics.CheckedProduct("retained encoded output", jobs, 6, output)
	if err != nil {
		return err
	}
	total, err := numerics.CheckedSum("transfer working set", workspace, output)
	if err != nil {
		return err
	}
	return numerics.CheckFloat64Budget("transfer request", total)
}
