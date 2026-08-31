package solver

import (
	"math"
	"smp-meso/config"
	"smp-meso/meso"
)

var Categories = []string{"k1", "k2", "k3", "k4plus", "censored"}

func categoryIndex(clusterCount int) int {
	switch clusterCount {
	case 1:
		return 0
	case 2:
		return 1
	case 3:
		return 2
	default:
		return 3
	}
}

// terminalCategory returns an absorbing major-cluster category. Every occupied
// confidence component must have diameter no larger than tolerance; minor
// components remain part of the invariance check but are not counted as major.
func terminalCategory(state *meso.State, request config.RunRequest) (int, bool) {
	occupiedThreshold := 0.5 / float64(state.Population)
	occupied := make([]int, 0, state.Bins)
	for index, mass := range state.Rho {
		if mass >= occupiedThreshold {
			occupied = append(occupied, index)
		}
	}
	if len(occupied) == 0 {
		return 0, false
	}
	components := [][]int{{occupied[0]}}
	for _, index := range occupied[1:] {
		last := components[len(components)-1]
		if state.Axis[index]-state.Axis[last[len(last)-1]] > request.Dynamics.Tolerance {
			components = append(components, []int{index})
		} else {
			components[len(components)-1] = append(last, index)
		}
	}
	majorCount := 0
	largestMass := 0.0
	for _, component := range components {
		if state.Axis[component[len(component)-1]]-state.Axis[component[0]] > request.Dynamics.Tolerance {
			return 0, false
		}
		mass := 0.0
		for _, index := range component {
			mass += state.Rho[index]
		}
		largestMass = math.Max(largestMass, mass)
		if mass >= request.MajorClusterMass {
			majorCount++
		}
	}
	if majorCount == 0 && largestMass > 0 {
		majorCount = 1
	}
	return categoryIndex(majorCount), true
}
