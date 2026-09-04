package solver

import (
	"smp-meso/config"
	"smp-meso/lifted"
	"smp-meso/terminal"
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

// terminalCategory applies the common measure-level classifier. Ambiguous or
// nonterminal states continue until a later step or are censored at the horizon.
func terminalCategory(state *lifted.State, request config.RunRequest) (int, bool, error) {
	result, err := terminal.Classify(state.Axis, state.Rho, terminal.Options{
		Epsilon:            request.Dynamics.Tolerance,
		OccupiedMass:       0.5 / float64(state.Population),
		MajorMass:          request.MajorClusterMass,
		PositionResolution: request.TerminalPositionResolution,
		MassResolution:     request.TerminalMassResolution,
	})
	if err != nil {
		return 0, false, err
	}
	if result.Status != terminal.StatusAbsorbed {
		return 0, false, nil
	}
	return categoryIndex(result.KMajor), true, nil
}
