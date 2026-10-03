package trajectory

import (
	"smp-meso/config"
	"smp-meso/terminal"
)

// TerminalOptions applies the shared finite-population threshold conventions
// to a lifted request. Experiments may adjust the resolutions for a separate
// point-hit diagnostic without changing the primary stopping rule.
func TerminalOptions(request config.RunRequest) terminal.Options {
	return terminal.Options{
		Epsilon:            request.Dynamics.Tolerance,
		OccupiedMass:       0.5 / float64(request.Population),
		MajorMass:          request.MajorClusterMass,
		PositionResolution: request.TerminalPositionResolution,
		MassResolution:     request.TerminalMassResolution,
	}
}
