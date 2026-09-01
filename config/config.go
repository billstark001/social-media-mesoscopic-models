package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"smp-meso/numerics"
	"strings"
)

type Layer int

const (
	LayerNaive Layer = iota
	LayerBase
	LayerWedge
	LayerHistogram
	LayerCandidate
	LayerTopology
)

func (l Layer) String() string {
	switch l {
	case LayerNaive:
		return "naive"
	case LayerBase:
		return "base"
	case LayerWedge:
		return "wedge"
	case LayerHistogram:
		return "histogram"
	case LayerCandidate:
		return "candidate"
	case LayerTopology:
		return "topology"
	default:
		return "unknown"
	}
}

func ParseLayer(value string) (Layer, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "naive", "rho_edge", "rho-e", "l-1":
		return LayerNaive, nil
	case "base", "l0":
		return LayerBase, nil
	case "wedge", "l1":
		return LayerWedge, nil
	case "histogram", "h", "l2":
		return LayerHistogram, nil
	case "candidate", "xi", "l3":
		return LayerCandidate, nil
	case "topology", "component", "l4":
		return LayerTopology, nil
	default:
		return 0, fmt.Errorf("unsupported layer %q", value)
	}
}

type DynamicsConfig struct {
	Type         string  `json:"type"`
	Tolerance    float64 `json:"tolerance"`
	Influence    float64 `json:"influence"`
	RewiringRate float64 `json:"rewiring_rate"`
}

type RecommenderConfig struct {
	Type                  string  `json:"type"`
	Steepness             float64 `json:"steepness"`
	RandomRatio           float64 `json:"random_ratio"`
	OpinionTolerance      float64 `json:"opinion_tolerance"`
	NoiseStd              float64 `json:"noise_std"`
	NoiseQuadraturePoints int     `json:"noise_quadrature_points"`
}

type InitialConfig struct {
	Type          string    `json:"type"`
	OpinionMin    float64   `json:"opinion_min"`
	OpinionMax    float64   `json:"opinion_max"`
	Probabilities []float64 `json:"probabilities"`
}

type ResolutionConfig struct {
	ScoreMax          int `json:"score_max"`
	AvailabilityBins  int `json:"availability_bins"`
	ComponentSizeBins int `json:"component_size_bins"`
	OpinionQuadrature int `json:"opinion_quadrature_points"`
}

type ClosureConfig struct {
	MotifRelaxation     float64 `json:"motif_relaxation"`
	HistogramRelaxation float64 `json:"histogram_relaxation"`
	CandidateRelaxation float64 `json:"candidate_relaxation"`
	TopologyRelaxation  float64 `json:"topology_relaxation"`
}

type FastSlowConfig struct {
	Mode              string  `json:"mode"`
	RatioThreshold    float64 `json:"ratio_threshold"`
	MaxSubsteps       int     `json:"max_substeps"`
	ZeroEventBatches  int     `json:"zero_event_batches"`
	ResidualTolerance float64 `json:"residual_tolerance"`
	ZeroEventResidual float64 `json:"zero_event_residual"`
}

type AmbiguityConfig struct {
	EligibilityCorrelationRadius float64 `json:"eligibility_correlation_radius"`
	ScoreAvailabilityRadius      float64 `json:"score_availability_radius"`
	MotifPersistenceRadius       float64 `json:"motif_persistence_radius"`
	BridgeBiasRadius             float64 `json:"bridge_bias_radius"`
	ComponentMixRadius           float64 `json:"component_mix_radius"`
}

// RunRequest is deliberately explicit. The JSON decoder checks that every
// scalar field is present, including fields whose valid value may be zero.
type RunRequest struct {
	RequestID           string            `json:"request_id"`
	Layer               string            `json:"layer"`
	Population          int               `json:"population"`
	OpinionBins         int               `json:"opinion_bins"`
	OutDegree           int               `json:"out_degree"`
	RecommendationCount int               `json:"recommendation_count"`
	MaxSteps            int               `json:"max_steps"`
	Paths               int               `json:"paths"`
	IntervalPaths       int               `json:"interval_paths"`
	AmbiguitySamples    int               `json:"ambiguity_samples"`
	ConfidenceLevel     float64           `json:"confidence_level"`
	Workers             int               `json:"workers"`
	Seed                uint64            `json:"seed"`
	MajorClusterMass    float64           `json:"major_cluster_mass"`
	Dynamics            DynamicsConfig    `json:"dynamics"`
	Recommender         RecommenderConfig `json:"recommender"`
	Initial             InitialConfig     `json:"initial"`
	Resolution          ResolutionConfig  `json:"resolution"`
	Closure             ClosureConfig     `json:"closure"`
	FastSlow            FastSlowConfig    `json:"fast_slow"`
	Ambiguity           AmbiguityConfig   `json:"ambiguity"`
}

var requiredPaths = [][]string{
	{"request_id"}, {"layer"}, {"population"}, {"opinion_bins"},
	{"out_degree"}, {"recommendation_count"}, {"max_steps"}, {"paths"},
	{"interval_paths"}, {"ambiguity_samples"}, {"confidence_level"},
	{"workers"}, {"seed"}, {"major_cluster_mass"},
	{"dynamics", "type"}, {"dynamics", "tolerance"},
	{"dynamics", "influence"}, {"dynamics", "rewiring_rate"},
	{"recommender", "type"}, {"recommender", "steepness"},
	{"recommender", "random_ratio"}, {"recommender", "opinion_tolerance"},
	{"recommender", "noise_std"}, {"recommender", "noise_quadrature_points"},
	{"initial", "type"}, {"initial", "opinion_min"},
	{"initial", "opinion_max"}, {"initial", "probabilities"},
	{"resolution", "score_max"}, {"resolution", "availability_bins"},
	{"resolution", "component_size_bins"},
	{"resolution", "opinion_quadrature_points"},
	{"closure", "motif_relaxation"}, {"closure", "histogram_relaxation"},
	{"closure", "candidate_relaxation"}, {"closure", "topology_relaxation"},
	{"fast_slow", "mode"}, {"fast_slow", "ratio_threshold"},
	{"fast_slow", "max_substeps"}, {"fast_slow", "zero_event_batches"},
	{"fast_slow", "residual_tolerance"}, {"fast_slow", "zero_event_residual"},
	{"ambiguity", "eligibility_correlation_radius"},
	{"ambiguity", "score_availability_radius"},
	{"ambiguity", "motif_persistence_radius"},
	{"ambiguity", "bridge_bias_radius"},
	{"ambiguity", "component_mix_radius"},
}

func DecodeRequest(data []byte) (RunRequest, error) {
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return RunRequest{}, err
	}
	root, ok := raw.(map[string]any)
	if !ok {
		return RunRequest{}, errors.New("request must be a JSON object")
	}
	for _, path := range requiredPaths {
		cursor := any(root)
		for _, key := range path {
			object, ok := cursor.(map[string]any)
			if !ok {
				return RunRequest{}, fmt.Errorf("missing required field %s", strings.Join(path, "."))
			}
			cursor, ok = object[key]
			if !ok {
				return RunRequest{}, fmt.Errorf("missing required field %s", strings.Join(path, "."))
			}
		}
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

func radius(name string, value float64) error {
	if err := numerics.CheckUnit(name, value); err != nil {
		return err
	}
	return nil
}

func (r RunRequest) Validate() error {
	if strings.TrimSpace(r.RequestID) == "" {
		return errors.New("request_id must not be empty")
	}
	layer, err := ParseLayer(r.Layer)
	if err != nil {
		return err
	}
	if r.Population < 2 {
		return fmt.Errorf("population must be >=2, got %d", r.Population)
	}
	if r.OpinionBins < 2 {
		return fmt.Errorf("opinion_bins must be >=2, got %d", r.OpinionBins)
	}
	if r.OutDegree < 1 || r.OutDegree >= r.Population {
		return fmt.Errorf("out_degree must be in [1,population), got %d", r.OutDegree)
	}
	if r.RecommendationCount < 1 || r.RecommendationCount >= r.Population {
		return fmt.Errorf("recommendation_count must be in [1,population), got %d", r.RecommendationCount)
	}
	if r.MaxSteps < 1 || r.Paths < 1 || r.IntervalPaths < 1 || r.AmbiguitySamples < 1 || r.Workers < 1 {
		return errors.New("max_steps, paths, interval_paths, ambiguity_samples, and workers must be positive")
	}
	if r.ConfidenceLevel <= 0 || r.ConfidenceLevel >= 1 || math.IsNaN(r.ConfidenceLevel) {
		return fmt.Errorf("confidence_level must be in (0,1), got %g", r.ConfidenceLevel)
	}
	if err := numerics.CheckUnit("major_cluster_mass", r.MajorClusterMass); err != nil {
		return err
	}
	if r.MajorClusterMass <= 0 {
		return errors.New("major_cluster_mass must be >0")
	}

	dynamics := strings.ToLower(strings.TrimSpace(r.Dynamics.Type))
	if dynamics != "hk" && dynamics != "deffuant" {
		return fmt.Errorf("unsupported dynamics.type %q", r.Dynamics.Type)
	}
	if err := numerics.CheckNonnegative("dynamics.tolerance", r.Dynamics.Tolerance); err != nil {
		return err
	}
	if err := numerics.CheckUnit("dynamics.influence", r.Dynamics.Influence); err != nil {
		return err
	}
	if err := numerics.CheckUnit("dynamics.rewiring_rate", r.Dynamics.RewiringRate); err != nil {
		return err
	}

	recommender := strings.ToLower(strings.TrimSpace(r.Recommender.Type))
	if recommender != "random" && recommender != "opinion_random" && recommender != "structure_random" {
		return fmt.Errorf("unsupported recommender.type %q", r.Recommender.Type)
	}
	if err := numerics.CheckNonnegative("recommender.steepness", r.Recommender.Steepness); err != nil {
		return err
	}
	if recommender != "random" && r.Recommender.Steepness <= 0 {
		return errors.New("recommender.steepness must be >0 for weighted recommenders")
	}
	if err := numerics.CheckUnit("recommender.random_ratio", r.Recommender.RandomRatio); err != nil {
		return err
	}
	if err := numerics.CheckNonnegative("recommender.opinion_tolerance", r.Recommender.OpinionTolerance); err != nil {
		return err
	}
	if err := numerics.CheckNonnegative("recommender.noise_std", r.Recommender.NoiseStd); err != nil {
		return err
	}
	if r.Recommender.NoiseQuadraturePoints < 1 {
		return errors.New("recommender.noise_quadrature_points must be positive")
	}

	initialType := strings.ToLower(strings.TrimSpace(r.Initial.Type))
	if initialType != "uniform" && initialType != "categorical" {
		return fmt.Errorf("unsupported initial.type %q", r.Initial.Type)
	}
	if err := numerics.CheckFinite("initial.opinion_min", r.Initial.OpinionMin); err != nil {
		return err
	}
	if err := numerics.CheckFinite("initial.opinion_max", r.Initial.OpinionMax); err != nil {
		return err
	}
	if r.Initial.OpinionMax <= r.Initial.OpinionMin {
		return errors.New("initial.opinion_max must exceed initial.opinion_min")
	}
	if initialType == "uniform" && len(r.Initial.Probabilities) != 0 {
		return errors.New("initial.probabilities must be [] for uniform initial state")
	}
	if initialType == "categorical" {
		if len(r.Initial.Probabilities) != r.OpinionBins {
			return fmt.Errorf("categorical initial probabilities must have %d entries", r.OpinionBins)
		}
		if _, err := numerics.NormalizedCopy(r.Initial.Probabilities); err != nil {
			return fmt.Errorf("initial.probabilities: %w", err)
		}
	}

	if r.Resolution.ScoreMax < 1 || r.Resolution.AvailabilityBins < 2 ||
		r.Resolution.ComponentSizeBins < 2 || r.Resolution.OpinionQuadrature < 1 {
		return errors.New("all resolution counts must be positive; availability and component bins must be >=2")
	}
	for name, value := range map[string]float64{
		"closure.motif_relaxation":     r.Closure.MotifRelaxation,
		"closure.histogram_relaxation": r.Closure.HistogramRelaxation,
		"closure.candidate_relaxation": r.Closure.CandidateRelaxation,
		"closure.topology_relaxation":  r.Closure.TopologyRelaxation,
	} {
		if err := numerics.CheckUnit(name, value); err != nil {
			return err
		}
	}
	fastSlowMode := strings.ToLower(strings.TrimSpace(r.FastSlow.Mode))
	if fastSlowMode != "unsplit" && fastSlowMode != "conditional_absorption" {
		return fmt.Errorf("unsupported fast_slow.mode %q", r.FastSlow.Mode)
	}
	if err := numerics.CheckNonnegative("fast_slow.ratio_threshold", r.FastSlow.RatioThreshold); err != nil {
		return err
	}
	if r.FastSlow.MaxSubsteps < 1 {
		return errors.New("fast_slow.max_substeps must be positive")
	}
	if r.FastSlow.ZeroEventBatches < 1 {
		return errors.New("fast_slow.zero_event_batches must be positive")
	}
	if err := numerics.CheckNonnegative("fast_slow.residual_tolerance", r.FastSlow.ResidualTolerance); err != nil {
		return err
	}
	if err := numerics.CheckNonnegative("fast_slow.zero_event_residual", r.FastSlow.ZeroEventResidual); err != nil {
		return err
	}
	for name, value := range map[string]float64{
		"ambiguity.eligibility_correlation_radius": r.Ambiguity.EligibilityCorrelationRadius,
		"ambiguity.score_availability_radius":      r.Ambiguity.ScoreAvailabilityRadius,
		"ambiguity.motif_persistence_radius":       r.Ambiguity.MotifPersistenceRadius,
		"ambiguity.bridge_bias_radius":             r.Ambiguity.BridgeBiasRadius,
		"ambiguity.component_mix_radius":           r.Ambiguity.ComponentMixRadius,
	} {
		if err := radius(name, value); err != nil {
			return err
		}
	}
	return r.validateWorkingSet(layer)
}

func (r RunRequest) validateWorkingSet(layer Layer) error {
	size := r.OpinionBins
	square, err := numerics.CheckedProduct("lifted opinion matrix", size, size)
	if err != nil {
		return err
	}
	linear, err := numerics.CheckedProduct("lifted vector state", 2, size)
	if err != nil {
		return err
	}
	matrices, err := numerics.CheckedProduct("lifted matrix state", 3, square)
	if err != nil {
		return err
	}
	coordinates, err := numerics.CheckedSum("lifted state", linear, matrices)
	if err != nil {
		return err
	}
	if layer >= LayerWedge {
		cube, err := numerics.CheckedProduct("lifted wedge tensor", size, size, size)
		if err != nil {
			return err
		}
		coordinates, err = numerics.CheckedSum("lifted state", coordinates, cube)
		if err != nil {
			return err
		}
	}
	if layer >= LayerHistogram {
		degreeBins, err := numerics.CheckedSum("lifted degree bins", r.OutDegree, 1)
		if err != nil {
			return err
		}
		histogram, err := numerics.CheckedProduct("lifted histogram", size,
			degreeBins, degreeBins, r.Resolution.AvailabilityBins)
		if err != nil {
			return err
		}
		coordinates, err = numerics.CheckedSum("lifted state", coordinates, histogram)
		if err != nil {
			return err
		}
	}
	if layer >= LayerCandidate {
		scoreBins, err := numerics.CheckedSum("lifted score bins", r.Resolution.ScoreMax, 1)
		if err != nil {
			return err
		}
		xi, err := numerics.CheckedProduct("lifted candidate tensor", size, size, 2, scoreBins)
		if err != nil {
			return err
		}
		coordinates, err = numerics.CheckedSum("lifted state", coordinates, xi)
		if err != nil {
			return err
		}
	}
	if layer >= LayerTopology {
		topology, err := numerics.CheckedProduct("lifted component state", size, r.Resolution.ComponentSizeBins)
		if err != nil {
			return err
		}
		coordinates, err = numerics.CheckedSum("lifted state", coordinates, topology, square)
		if err != nil {
			return err
		}
	}
	// A retained path temporarily holds reconstructed targets and transported
	// coordinate buffers. Six copies is a conservative peak estimate.
	perPath, err := numerics.CheckedProduct("lifted per-path workspace", 6, coordinates)
	if err != nil {
		return err
	}
	workers := min(r.Workers, max(r.Paths, r.IntervalPaths))
	total, err := numerics.CheckedProduct("lifted concurrent working set", workers, perPath)
	if err != nil {
		return err
	}
	return numerics.CheckFloat64Budget("lifted request", total)
}
