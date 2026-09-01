package solver

import (
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/lifted"
	"smp-meso/numerics"
	"smp-meso/protocol"
)

type intervalCoordinate struct {
	radius float64
	set    func(*lifted.ClosureProfile, float64)
}

type intervalCoordinateSpec struct {
	resolvedAt config.Layer
	radius     func(config.AmbiguityConfig) float64
	set        func(*lifted.ClosureProfile, float64)
}

var intervalCoordinateSpecs = [...]intervalCoordinateSpec{
	{
		resolvedAt: config.LayerWedge,
		radius:     func(value config.AmbiguityConfig) float64 { return value.MotifPersistenceRadius },
		set:        func(profile *lifted.ClosureProfile, value float64) { profile.MotifPersistence = value },
	},
	{
		resolvedAt: config.LayerHistogram,
		radius:     func(value config.AmbiguityConfig) float64 { return value.EligibilityCorrelationRadius },
		set:        func(profile *lifted.ClosureProfile, value float64) { profile.EligibilityCorrelation = value },
	},
	{
		resolvedAt: config.LayerCandidate,
		radius:     func(value config.AmbiguityConfig) float64 { return value.ScoreAvailabilityRadius },
		set:        func(profile *lifted.ClosureProfile, value float64) { profile.ScoreAvailability = value },
	},
	{
		resolvedAt: config.LayerTopology,
		radius:     func(value config.AmbiguityConfig) float64 { return value.BridgeBiasRadius },
		set:        func(profile *lifted.ClosureProfile, value float64) { profile.BridgeBias = value },
	},
	{
		resolvedAt: config.LayerTopology,
		radius:     func(value config.AmbiguityConfig) float64 { return value.ComponentMixRadius },
		set:        func(profile *lifted.ClosureProfile, value float64) { profile.ComponentMix = value },
	},
}

func activeCoordinates(request config.RunRequest, layer config.Layer) []intervalCoordinate {
	coordinates := make([]intervalCoordinate, 0, 5)
	for _, spec := range intervalCoordinateSpecs {
		if layer < spec.resolvedAt {
			coordinate := intervalCoordinate{radius: spec.radius(request.Ambiguity), set: spec.set}
			if coordinate.radius > 0 {
				coordinates = append(coordinates, coordinate)
			}
		}
	}
	return coordinates
}

func ambiguityProfiles(request config.RunRequest, layer config.Layer) []lifted.ClosureProfile {
	coordinates := activeCoordinates(request, layer)
	if len(coordinates) == 0 {
		return []lifted.ClosureProfile{{}}
	}
	profiles := make([]lifted.ClosureProfile, 0, request.AmbiguitySamples)
	profiles = append(profiles, lifted.ClosureProfile{})
	for index := 0; len(profiles) < request.AmbiguitySamples && index < 2*len(coordinates); index++ {
		coordinate := coordinates[index/2]
		value := coordinate.radius
		if index%2 == 1 {
			value = -value
		}
		profile := lifted.ClosureProfile{}
		coordinate.set(&profile, value)
		profiles = append(profiles, profile)
	}
	rng := rand.New(rand.NewPCG(splitMix64(request.Seed^0xa0761d6478bd642f), splitMix64(request.Seed^0xe7037ed1a0b428db)))
	for len(profiles) < request.AmbiguitySamples {
		profile := lifted.ClosureProfile{}
		for _, coordinate := range coordinates {
			coordinate.set(&profile, (2*rng.Float64()-1)*coordinate.radius)
		}
		profiles = append(profiles, profile)
	}
	return profiles
}

func wilson(successes, trials int, confidence float64) (float64, float64) {
	if trials <= 0 {
		return 0, 1
	}
	z := math.Sqrt2 * math.Erfinv(confidence)
	n := float64(trials)
	p := float64(successes) / n
	z2 := z * z
	denominator := 1 + z2/n
	center := (p + z2/(2*n)) / denominator
	half := z / denominator * math.Sqrt(p*(1-p)/n+z2/(4*n*n))
	return numerics.Clamp(center-half, 0, 1), numerics.Clamp(center+half, 0, 1)
}

type IntervalEstimate struct {
	Method              string                  `json:"method"`
	ConfidenceLevel     float64                 `json:"confidence_level"`
	ConfidenceScope     string                  `json:"confidence_scope"`
	ScenarioCount       int                     `json:"scenario_count"`
	PathsPerScenario    int                     `json:"paths_per_scenario"`
	ClosureLower        []float64               `json:"closure_lower"`
	ClosureUpper        []float64               `json:"closure_upper"`
	ClosureWidth        []float64               `json:"closure_width"`
	Lower               []float64               `json:"lower"`
	Upper               []float64               `json:"upper"`
	Width               []float64               `json:"width"`
	ScenarioProfiles    []lifted.ClosureProfile `json:"scenario_profiles"`
	ScenarioProbability [][]float64             `json:"scenario_probabilities"`
}

func runInterval(
	request config.RunRequest,
	layer config.Layer,
	point Ensemble,
	progressStepInterval int,
	progress protocol.ProgressFunc,
) (IntervalEstimate, error) {
	profiles := ambiguityProfiles(request, layer)
	estimate := IntervalEstimate{
		Method:          "sampled_closure_envelope_with_marginal_wilson_bounds",
		ConfidenceLevel: request.ConfidenceLevel,
		ConfidenceScope: "per-scenario per-category marginal; no simultaneous correction",
		ScenarioCount:   len(profiles), PathsPerScenario: request.IntervalPaths,
		ScenarioProfiles: append([]lifted.ClosureProfile(nil), profiles...),
		ClosureLower:     make([]float64, len(Categories)),
		ClosureUpper:     make([]float64, len(Categories)),
		Lower:            append([]float64(nil), point.Probabilities...),
		Upper:            append([]float64(nil), point.Probabilities...),
	}
	for category := range Categories {
		estimate.ClosureLower[category] = math.Inf(1)
		estimate.ClosureUpper[category] = math.Inf(-1)
	}
	for scenario, profile := range profiles {
		if progress != nil {
			progress(protocol.ProgressEvent{
				Event: "scenario_started", Stage: "interval",
				ScenarioIndex: scenario + 1, ScenarioCount: len(profiles),
				TotalPaths: request.IntervalPaths,
			})
		}
		scenarioProgress := func(event protocol.ProgressEvent) {
			event.Stage = "interval"
			event.ScenarioIndex = scenario + 1
			event.ScenarioCount = len(profiles)
			progress(event)
		}
		if progress == nil {
			scenarioProgress = nil
		}
		ensemble, err := runEnsembleWithProgress(
			request, layer, profile, request.IntervalPaths,
			0x6a09e667f3bcc909^uint64(scenario+1)*0x9e3779b97f4a7c15,
			progressStepInterval, scenarioProgress,
		)
		if err != nil {
			return IntervalEstimate{}, err
		}
		estimate.ScenarioProbability = append(estimate.ScenarioProbability,
			append([]float64(nil), ensemble.Probabilities...))
		for category, probability := range ensemble.Probabilities {
			estimate.ClosureLower[category] = math.Min(estimate.ClosureLower[category], probability)
			estimate.ClosureUpper[category] = math.Max(estimate.ClosureUpper[category], probability)
			lower, upper := wilson(ensemble.Counts[category], ensemble.Paths, request.ConfidenceLevel)
			estimate.Lower[category] = math.Min(estimate.Lower[category], lower)
			estimate.Upper[category] = math.Max(estimate.Upper[category], upper)
		}
		if progress != nil {
			progress(protocol.ProgressEvent{
				Event: "scenario_completed", Stage: "interval",
				ScenarioIndex: scenario + 1, ScenarioCount: len(profiles),
				CompletedPaths: ensemble.Paths, TotalPaths: ensemble.Paths,
			})
		}
	}
	estimate.ClosureWidth = make([]float64, len(Categories))
	estimate.Width = make([]float64, len(Categories))
	for category := range Categories {
		estimate.ClosureWidth[category] = estimate.ClosureUpper[category] - estimate.ClosureLower[category]
		estimate.Width[category] = estimate.Upper[category] - estimate.Lower[category]
	}
	return estimate, nil
}
