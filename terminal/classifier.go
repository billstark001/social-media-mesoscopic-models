// Package terminal defines the common measure-level terminal classifier used
// by microscopic and lifted simulations.
package terminal

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

const (
	StatusAbsorbed      = "absorbed"
	StatusNonterminal   = "nonterminal"
	StatusGridAmbiguous = "grid_ambiguous"
)

type Options struct {
	Epsilon            float64 `json:"epsilon"`
	OccupiedMass       float64 `json:"occupied_mass"`
	MajorMass          float64 `json:"major_mass"`
	PositionResolution float64 `json:"position_resolution"`
	MassResolution     float64 `json:"mass_resolution"`
}

type Component struct {
	Minimum float64 `json:"minimum"`
	Maximum float64 `json:"maximum"`
	Mass    float64 `json:"mass"`
	Major   bool    `json:"major"`
}

type Margins struct {
	GapToEpsilon      float64 `json:"gap_to_epsilon"`
	DiameterToEpsilon float64 `json:"diameter_to_epsilon"`
	MassToMajorCutoff float64 `json:"mass_to_major_cutoff"`
}

type Result struct {
	Status     string      `json:"status"`
	Category   string      `json:"category"`
	KAll       int         `json:"k_all"`
	KMajor     int         `json:"k_major"`
	Components []Component `json:"components"`
	Margins    Margins     `json:"margins"`
}

type atom struct{ position, mass float64 }

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validate(positions, masses []float64, options Options) error {
	if len(positions) != len(masses) {
		return errors.New("positions and masses must have equal length")
	}
	if !finite(options.Epsilon) || options.Epsilon < 0 ||
		!finite(options.OccupiedMass) || options.OccupiedMass < 0 ||
		!finite(options.PositionResolution) || options.PositionResolution < 0 ||
		!finite(options.MassResolution) || options.MassResolution < 0 {
		return errors.New("epsilon, occupied mass, and resolutions must be finite and non-negative")
	}
	if !finite(options.MajorMass) || options.MajorMass <= 0 {
		return errors.New("major_mass must be finite and positive")
	}
	for index := range positions {
		if !finite(positions[index]) {
			return fmt.Errorf("position %d is not finite", index)
		}
		if !finite(masses[index]) || masses[index] < 0 {
			return fmt.Errorf("mass %d must be finite and non-negative", index)
		}
	}
	return nil
}

func minimum(current, candidate float64) float64 {
	if current < 0 || candidate < current {
		return candidate
	}
	return current
}

func category(count int) string {
	switch count {
	case 1:
		return "k1"
	case 2:
		return "k2"
	case 3:
		return "k3"
	default:
		return "k4plus"
	}
}

// Classify applies the same support-component functional to any atomic
// probability measure. Positions may be unsorted and duplicated.
func Classify(positions, masses []float64, options Options) (Result, error) {
	if err := validate(positions, masses, options); err != nil {
		return Result{}, err
	}
	atoms := make([]atom, 0, len(positions))
	for index := range positions {
		if masses[index] > 0 {
			atoms = append(atoms, atom{positions[index], masses[index]})
		}
	}
	sort.Slice(atoms, func(i, j int) bool { return atoms[i].position < atoms[j].position })
	aggregated := make([]atom, 0, len(atoms))
	for _, item := range atoms {
		last := len(aggregated) - 1
		if last >= 0 && aggregated[last].position == item.position {
			aggregated[last].mass += item.mass
		} else {
			aggregated = append(aggregated, item)
		}
	}
	occupied := make([]atom, 0, len(aggregated))
	for _, item := range aggregated {
		if item.mass >= options.OccupiedMass {
			occupied = append(occupied, item)
		}
	}
	result := Result{
		Status:   StatusNonterminal,
		Category: "censored",
		Margins: Margins{
			GapToEpsilon: -1, DiameterToEpsilon: -1, MassToMajorCutoff: -1,
		},
	}
	if len(occupied) == 0 {
		return result, nil
	}

	type span struct{ first, last int }
	spans := []span{{0, 0}}
	ambiguous := false
	for index := 1; index < len(occupied); index++ {
		gap := occupied[index].position - occupied[index-1].position
		margin := math.Abs(gap - options.Epsilon)
		result.Margins.GapToEpsilon = minimum(result.Margins.GapToEpsilon, margin)
		ambiguous = ambiguous || options.PositionResolution > 0 && margin <= options.PositionResolution
		if gap > options.Epsilon {
			spans = append(spans, span{index, index})
		} else {
			spans[len(spans)-1].last = index
		}
	}

	absorbed := true
	largestMass := 0.0
	for _, current := range spans {
		component := Component{
			Minimum: occupied[current.first].position,
			Maximum: occupied[current.last].position,
		}
		for index := current.first; index <= current.last; index++ {
			component.Mass += occupied[index].mass
		}
		diameter := component.Maximum - component.Minimum
		margin := math.Abs(options.Epsilon - diameter)
		result.Margins.DiameterToEpsilon = minimum(result.Margins.DiameterToEpsilon, margin)
		absorbed = absorbed && diameter <= options.Epsilon
		ambiguous = ambiguous || options.PositionResolution > 0 && margin <= options.PositionResolution
		massMargin := math.Abs(component.Mass - options.MajorMass)
		result.Margins.MassToMajorCutoff = minimum(result.Margins.MassToMajorCutoff, massMargin)
		ambiguous = ambiguous || options.MassResolution > 0 && massMargin <= options.MassResolution
		component.Major = component.Mass >= options.MajorMass
		if component.Major {
			result.KMajor++
		}
		largestMass = math.Max(largestMass, component.Mass)
		result.Components = append(result.Components, component)
	}
	result.KAll = len(result.Components)
	if result.KMajor == 0 && largestMass > 0 {
		result.KMajor = 1
	}
	if ambiguous {
		result.Status = StatusGridAmbiguous
		return result, nil
	}
	if !absorbed {
		return result, nil
	}
	result.Status = StatusAbsorbed
	result.Category = category(result.KMajor)
	return result, nil
}
