package kinetic

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"smp-meso/numerics"
	"smp-meso/protocol"
	"strings"
)

type DynamicsConfig struct {
	Type          string  `json:"type"`
	OpinionMethod string  `json:"opinion_method"`
	Tolerance     float64 `json:"tolerance"`
	Influence     float64 `json:"influence"`
	RewiringRate  float64 `json:"rewiring_rate"`
}

type RecommenderConfig struct {
	Type             string  `json:"type"`
	Steepness        float64 `json:"steepness"`
	RandomRatio      float64 `json:"random_ratio"`
	OpinionTolerance float64 `json:"opinion_tolerance"`
}

type InitialConfig struct {
	Type          string                `json:"type"`
	OpinionMin    float64               `json:"opinion_min"`
	OpinionMax    float64               `json:"opinion_max"`
	Probabilities protocol.EncodedArray `json:"probabilities"`
}

type ResolutionConfig struct {
	OpinionQuadraturePoints    int    `json:"opinion_quadrature_points"`
	OpinionQuadratureRule      string `json:"opinion_quadrature_rule"`
	ConfidenceQuadraturePoints int    `json:"confidence_quadrature_points"`
	ScoreMax                   int    `json:"score_max"`
	DistanceGridSize           int    `json:"distance_grid_size"`
}

type ObservablesConfig struct {
	Polarization              bool    `json:"polarization"`
	Subjective                bool    `json:"subjective"`
	Homophily                 bool    `json:"homophily"`
	HomophilyRaw              bool    `json:"homophily_raw"`
	Pathway                   bool    `json:"pathway"`
	PolarizationFirstPassage  bool    `json:"polarization_first_passage"`
	HomophilyFirstPassage     bool    `json:"homophily_first_passage"`
	PolarizationThreshold     float64 `json:"polarization_threshold"`
	HomophilyThreshold        float64 `json:"homophily_threshold"`
	MinimumBandwidth          float64 `json:"minimum_bandwidth"`
	ObjectiveEffectiveSamples int     `json:"objective_effective_samples"`
}

// SnapshotsConfig requests selected state fields at explicitly encoded
// numerical steps. With every field false, record_steps must encode shape 0
// and the solver does not allocate a snapshot collector.
type SnapshotsConfig struct {
	RecordSteps  protocol.EncodedArray `json:"record_steps"`
	Rho          bool                  `json:"rho"`
	Edge         bool                  `json:"edge"`
	Velocity     bool                  `json:"velocity"`
	RewiringFlux bool                  `json:"rewiring_flux"`
	FinalRho     bool                  `json:"final_rho"`
	FinalEdge    bool                  `json:"final_edge"`
}

// RunRequest is fully explicit: the strict decoder requires every field even
// when an observable is disabled or the uniform initial law ignores its
// encoded probability payload.
type RunRequest struct {
	RequestID           string            `json:"request_id"`
	Population          int               `json:"population"`
	OpinionBins         int               `json:"opinion_bins"`
	OutDegree           int               `json:"out_degree"`
	RecommendationCount int               `json:"recommendation_count"`
	Steps               int               `json:"steps"`
	RecordEvery         int               `json:"record_every"`
	Dt                  float64           `json:"dt"`
	NoiseDiffusion      float64           `json:"noise_diffusion"`
	ConfidenceMode      string            `json:"confidence_mode"`
	Dynamics            DynamicsConfig    `json:"dynamics"`
	Recommender         RecommenderConfig `json:"recommender"`
	Initial             InitialConfig     `json:"initial"`
	Resolution          ResolutionConfig  `json:"resolution"`
	Observables         ObservablesConfig `json:"observables"`
	Snapshots           SnapshotsConfig   `json:"snapshots"`
}

var requiredPaths = [][]string{
	{"request_id"}, {"population"}, {"opinion_bins"}, {"out_degree"},
	{"recommendation_count"}, {"steps"}, {"record_every"}, {"dt"},
	{"noise_diffusion"}, {"confidence_mode"},
	{"dynamics", "type"}, {"dynamics", "opinion_method"},
	{"dynamics", "tolerance"}, {"dynamics", "influence"},
	{"dynamics", "rewiring_rate"},
	{"recommender", "type"}, {"recommender", "steepness"},
	{"recommender", "random_ratio"}, {"recommender", "opinion_tolerance"},
	{"initial", "type"}, {"initial", "opinion_min"},
	{"initial", "opinion_max"}, {"initial", "probabilities"},
	{"initial", "probabilities", "encoding"},
	{"initial", "probabilities", "shape"}, {"initial", "probabilities", "data"},
	{"resolution", "opinion_quadrature_points"},
	{"resolution", "opinion_quadrature_rule"},
	{"resolution", "confidence_quadrature_points"},
	{"resolution", "score_max"}, {"resolution", "distance_grid_size"},
	{"observables", "polarization"}, {"observables", "subjective"},
	{"observables", "homophily"}, {"observables", "homophily_raw"},
	{"observables", "pathway"},
	{"observables", "polarization_first_passage"},
	{"observables", "homophily_first_passage"},
	{"observables", "polarization_threshold"},
	{"observables", "homophily_threshold"},
	{"observables", "minimum_bandwidth"},
	{"observables", "objective_effective_samples"},
	{"snapshots", "record_steps"},
	{"snapshots", "record_steps", "encoding"},
	{"snapshots", "record_steps", "shape"},
	{"snapshots", "record_steps", "data"},
	{"snapshots", "rho"}, {"snapshots", "edge"},
	{"snapshots", "velocity"}, {"snapshots", "rewiring_flux"},
	{"snapshots", "final_rho"}, {"snapshots", "final_edge"},
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

func oneOf(value string, choices ...string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}

func (request RunRequest) Validate() error {
	if strings.TrimSpace(request.RequestID) == "" {
		return errors.New("request_id must not be empty")
	}
	if request.Population < 2 || request.OpinionBins < 2 {
		return errors.New("population and opinion_bins must be at least 2")
	}
	if request.OutDegree < 1 || request.OutDegree >= request.Population {
		return errors.New("out_degree must be in [1,population)")
	}
	if request.RecommendationCount < 1 || request.RecommendationCount >= request.Population {
		return errors.New("recommendation_count must be in [1,population)")
	}
	if request.Steps < 1 || request.RecordEvery < 1 {
		return errors.New("steps and record_every must be positive")
	}
	for name, value := range map[string]float64{
		"dt": request.Dt, "noise_diffusion": request.NoiseDiffusion,
		"opinion_min":       request.Initial.OpinionMin,
		"opinion_max":       request.Initial.OpinionMax,
		"tolerance":         request.Dynamics.Tolerance,
		"steepness":         request.Recommender.Steepness,
		"opinion_tolerance": request.Recommender.OpinionTolerance,
		"minimum_bandwidth": request.Observables.MinimumBandwidth,
	} {
		if err := numerics.CheckFinite(name, value); err != nil {
			return err
		}
	}
	if request.Dt <= 0 || request.NoiseDiffusion < 0 {
		return errors.New("dt must be positive and noise_diffusion nonnegative")
	}
	if request.Initial.OpinionMax <= request.Initial.OpinionMin {
		return errors.New("opinion_max must exceed opinion_min")
	}
	if request.Dynamics.Tolerance <= 0 || request.Dynamics.Tolerance > request.Initial.OpinionMax-request.Initial.OpinionMin {
		return errors.New("tolerance must lie in the opinion-domain width")
	}
	if err := numerics.CheckUnit("influence", request.Dynamics.Influence); err != nil {
		return err
	}
	if err := numerics.CheckUnit("rewiring_rate", request.Dynamics.RewiringRate); err != nil {
		return err
	}
	if request.Dt*request.Dynamics.Influence > 1+1e-12 || request.Dt*request.Dynamics.RewiringRate > 1+1e-12 {
		return errors.New("dt*influence and dt*rewiring_rate must not exceed 1")
	}
	if !oneOf(request.Dynamics.Type, "hk", "deffuant") {
		return fmt.Errorf("unsupported dynamics %q", request.Dynamics.Type)
	}
	if !oneOf(request.Dynamics.OpinionMethod, "measure", "fokker_planck") {
		return fmt.Errorf("unsupported opinion_method %q", request.Dynamics.OpinionMethod)
	}
	if !oneOf(request.ConfidenceMode, "center", "cell_average") {
		return fmt.Errorf("unsupported confidence_mode %q", request.ConfidenceMode)
	}
	if !oneOf(request.Recommender.Type, "random", "opinion_random", "structure_random_l0", "structure_random_l1") {
		return fmt.Errorf("unsupported recommender %q", request.Recommender.Type)
	}
	if request.Recommender.Steepness <= 0 || request.Recommender.OpinionTolerance <= 0 {
		return errors.New("steepness and opinion_tolerance must be positive")
	}
	if err := numerics.CheckUnit("random_ratio", request.Recommender.RandomRatio); err != nil {
		return err
	}
	if request.Resolution.OpinionQuadraturePoints < 1 || request.Resolution.ConfidenceQuadraturePoints < 1 || request.Resolution.ScoreMax < 1 || request.Resolution.DistanceGridSize < 3 {
		return errors.New("quadrature points and score_max must be positive; distance_grid_size must be >=3")
	}
	if err := numerics.CheckNormalQuadratureRule(request.Resolution.OpinionQuadratureRule); err != nil {
		return err
	}
	if request.Observables.MinimumBandwidth <= 0 || request.Observables.ObjectiveEffectiveSamples < 1 {
		return errors.New("minimum_bandwidth and objective_effective_samples must be positive")
	}
	if err := numerics.CheckUnit("polarization_threshold", request.Observables.PolarizationThreshold); err != nil {
		return err
	}
	if err := numerics.CheckUnit("homophily_threshold", request.Observables.HomophilyThreshold); err != nil {
		return err
	}
	snapshotSteps, err := request.snapshotSteps()
	if err != nil {
		return err
	}
	if err := request.validateWorkingSet(len(snapshotSteps)); err != nil {
		return err
	}
	values, shape, err := request.Initial.Probabilities.DecodeFloat64()
	if err != nil {
		return fmt.Errorf("initial.probabilities: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(request.Initial.Type)) {
	case "uniform":
		if len(shape) != 1 || shape[0] != 0 || len(values) != 0 {
			return errors.New("uniform initial probabilities must have shape 0")
		}
	case "categorical":
		if len(shape) != 1 || shape[0] != request.OpinionBins {
			return errors.New("categorical initial probabilities must match opinion_bins")
		}
		total := 0.0
		for _, value := range values {
			if err := numerics.CheckFinite("initial probability", value); err != nil || value < 0 {
				return errors.New("initial probabilities must be finite and nonnegative")
			}
			total += value
		}
		if total <= 0 {
			return errors.New("categorical initial probabilities must have positive mass")
		}
	default:
		return fmt.Errorf("unsupported initial type %q", request.Initial.Type)
	}
	return nil
}

func (request RunRequest) validateWorkingSet(snapshotCount int) error {
	size := request.OpinionBins
	square, err := numerics.CheckedProduct("kinetic opinion matrix", size, size)
	if err != nil {
		return err
	}
	base, err := numerics.CheckedProduct("kinetic base workspace", 20, square)
	if err != nil {
		return err
	}
	linear, err := numerics.CheckedProduct("kinetic vector workspace", 10, size)
	if err != nil {
		return err
	}
	total, err := numerics.CheckedSum("kinetic working set", base, linear)
	if err != nil {
		return err
	}
	if normalize(request.Recommender.Type) == "structure_random_l1" {
		cube, err := numerics.CheckedProduct("kinetic wedge tensor", size, size, size)
		if err != nil {
			return err
		}
		wedge, err := numerics.CheckedProduct("kinetic wedge workspace", 12, cube)
		if err != nil {
			return err
		}
		total, err = numerics.CheckedSum("kinetic working set", total, wedge)
		if err != nil {
			return err
		}
	}
	perSnapshot := 0
	if request.Snapshots.Rho {
		perSnapshot, err = numerics.CheckedSum("kinetic snapshot", perSnapshot, size)
	}
	if err == nil && request.Snapshots.Edge {
		perSnapshot, err = numerics.CheckedSum("kinetic snapshot", perSnapshot, square)
	}
	if err == nil && request.Snapshots.Velocity {
		perSnapshot, err = numerics.CheckedSum("kinetic snapshot", perSnapshot, size)
	}
	if err == nil && request.Snapshots.RewiringFlux {
		perSnapshot, err = numerics.CheckedSum("kinetic snapshot", perSnapshot, square)
	}
	if err != nil {
		return err
	}
	snapshots, err := numerics.CheckedProduct("kinetic snapshot output", snapshotCount, perSnapshot)
	if err != nil {
		return err
	}
	total, err = numerics.CheckedSum("kinetic working set", total, snapshots)
	if err != nil {
		return err
	}
	finalOutput := 0
	if request.Snapshots.FinalRho {
		finalOutput, err = numerics.CheckedSum("kinetic final snapshot", finalOutput, size)
	}
	if err == nil && request.Snapshots.FinalEdge {
		finalOutput, err = numerics.CheckedSum("kinetic final snapshot", finalOutput, square)
	}
	if err != nil {
		return err
	}
	total, err = numerics.CheckedSum("kinetic working set", total, finalOutput)
	if err != nil {
		return err
	}
	return numerics.CheckFloat64Budget("kinetic request", total)
}

func (request RunRequest) snapshotSteps() ([]int, error) {
	values, shape, err := request.Snapshots.RecordSteps.DecodeFloat64()
	if err != nil {
		return nil, fmt.Errorf("snapshots.record_steps: %w", err)
	}
	if len(shape) != 1 {
		return nil, errors.New("snapshots.record_steps must be one-dimensional")
	}
	requested := request.Snapshots.Rho || request.Snapshots.Edge ||
		request.Snapshots.Velocity || request.Snapshots.RewiringFlux
	if !requested {
		if shape[0] != 0 {
			return nil, errors.New("snapshots.record_steps must have shape 0 when no snapshot field is requested")
		}
		return nil, nil
	}
	if shape[0] == 0 {
		return nil, errors.New("snapshots.record_steps must not be empty when snapshot fields are requested")
	}
	steps := make([]int, len(values))
	previous := -1
	for index, value := range values {
		if err := numerics.CheckFinite("snapshot step", value); err != nil || math.Trunc(value) != value {
			return nil, errors.New("snapshot steps must be finite integers")
		}
		step := int(value)
		if step < 0 || step > request.Steps {
			return nil, fmt.Errorf("snapshot step %d must be in [0,steps]", step)
		}
		if step <= previous {
			return nil, errors.New("snapshot steps must be strictly increasing")
		}
		steps[index] = step
		previous = step
	}
	return steps, nil
}

func (request RunRequest) initialProbabilities() ([]float64, error) {
	if strings.EqualFold(strings.TrimSpace(request.Initial.Type), "uniform") {
		values := make([]float64, request.OpinionBins)
		for index := range values {
			values[index] = 1 / float64(request.OpinionBins)
		}
		return values, nil
	}
	values, _, err := request.Initial.Probabilities.DecodeFloat64()
	if err != nil {
		return nil, err
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	for index := range values {
		values[index] /= total
	}
	return values, nil
}
