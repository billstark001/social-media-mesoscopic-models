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
	LayerBase Layer = iota
	LayerWedge
	LayerHistogram
	LayerCandidate
	LayerTopology
)

func (l Layer) String() string {
	switch l {
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

func finite(name string, value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return fmt.Errorf("%s must be finite", name)
	}
	return nil
}

func unit(name string, value float64) error {
	if err := finite(name, value); err != nil {
		return err
	}
	if value < 0 || value > 1 {
		return fmt.Errorf("%s must be in [0,1], got %g", name, value)
	}
	return nil
}

func nonnegative(name string, value float64) error {
	if err := finite(name, value); err != nil {
		return err
	}
	if value < 0 {
		return fmt.Errorf("%s must be nonnegative, got %g", name, value)
	}
	return nil
}

func radius(name string, value float64) error {
	if err := unit(name, value); err != nil {
		return err
	}
	return nil
}

func (r RunRequest) Validate() error {
	if strings.TrimSpace(r.RequestID) == "" {
		return errors.New("request_id must not be empty")
	}
	if _, err := ParseLayer(r.Layer); err != nil {
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
	if err := unit("major_cluster_mass", r.MajorClusterMass); err != nil {
		return err
	}
	if r.MajorClusterMass <= 0 {
		return errors.New("major_cluster_mass must be >0")
	}

	dynamics := strings.ToLower(strings.TrimSpace(r.Dynamics.Type))
	if dynamics != "hk" && dynamics != "deffuant" {
		return fmt.Errorf("unsupported dynamics.type %q", r.Dynamics.Type)
	}
	if err := nonnegative("dynamics.tolerance", r.Dynamics.Tolerance); err != nil {
		return err
	}
	if err := unit("dynamics.influence", r.Dynamics.Influence); err != nil {
		return err
	}
	if err := unit("dynamics.rewiring_rate", r.Dynamics.RewiringRate); err != nil {
		return err
	}

	recommender := strings.ToLower(strings.TrimSpace(r.Recommender.Type))
	if recommender != "random" && recommender != "opinion_random" && recommender != "structure_random" {
		return fmt.Errorf("unsupported recommender.type %q", r.Recommender.Type)
	}
	if err := nonnegative("recommender.steepness", r.Recommender.Steepness); err != nil {
		return err
	}
	if recommender != "random" && r.Recommender.Steepness <= 0 {
		return errors.New("recommender.steepness must be >0 for weighted recommenders")
	}
	if err := unit("recommender.random_ratio", r.Recommender.RandomRatio); err != nil {
		return err
	}
	if err := nonnegative("recommender.opinion_tolerance", r.Recommender.OpinionTolerance); err != nil {
		return err
	}
	if err := nonnegative("recommender.noise_std", r.Recommender.NoiseStd); err != nil {
		return err
	}
	if r.Recommender.NoiseQuadraturePoints < 1 {
		return errors.New("recommender.noise_quadrature_points must be positive")
	}

	initialType := strings.ToLower(strings.TrimSpace(r.Initial.Type))
	if initialType != "uniform" && initialType != "categorical" {
		return fmt.Errorf("unsupported initial.type %q", r.Initial.Type)
	}
	if err := finite("initial.opinion_min", r.Initial.OpinionMin); err != nil {
		return err
	}
	if err := finite("initial.opinion_max", r.Initial.OpinionMax); err != nil {
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
		if err := unit(name, value); err != nil {
			return err
		}
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
	return nil
}
