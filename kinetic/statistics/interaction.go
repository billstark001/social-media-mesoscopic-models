package statistics

import "math"

// InteractionPlan evaluates the bounded-confidence interaction kernel on the
// finite-volume cell centres used by the kinetic solver. Density arrays are
// cell masses, so the discrete sums do not carry grid-spacing factors.
type InteractionPlan struct {
	size        int
	outDegree   float64
	interaction []float64
}

// NewInteractionPlan constructs W_ij = 0.5*min((x_i-x_j)^2, epsilon^2).
func NewInteractionPlan(axis []float64, epsilon float64, outDegree int) *InteractionPlan {
	size := len(axis)
	interaction := make([]float64, size*size)
	cutoff := epsilon * epsilon
	for source, x := range axis {
		for target, y := range axis {
			difference := x - y
			interaction[source*size+target] = 0.5 * math.Min(difference*difference, cutoff)
		}
	}
	return &InteractionPlan{size: size, outDegree: float64(outDegree), interaction: interaction}
}

// Energies returns U_rho and U_E for one kinetic state.
func (plan *InteractionPlan) Energies(rho, edge []float64) (float64, float64) {
	node, edgeEnergy := 0.0, 0.0
	for source := 0; source < plan.size; source++ {
		for target := 0; target < plan.size; target++ {
			index := source*plan.size + target
			weight := plan.interaction[index]
			node += weight * rho[source] * rho[target]
			edgeEnergy += weight * edge[index]
		}
	}
	return 0.5 * node, edgeEnergy / plan.outDegree
}

// Potentials returns Phi_rho on every cell and conditional Phi_E on every
// occupied source cell. Phi_E is NaN where the source row has zero mass.
func (plan *InteractionPlan) Potentials(rho, edge, node, edgePotential []float64) {
	for source := 0; source < plan.size; source++ {
		nodeValue, edgeValue, rowMass := 0.0, 0.0, 0.0
		for target := 0; target < plan.size; target++ {
			index := source*plan.size + target
			weight := plan.interaction[index]
			nodeValue += weight * rho[target]
			edgeValue += weight * edge[index]
			rowMass += edge[index]
		}
		node[source] = nodeValue
		if rowMass > 1e-15 {
			edgePotential[source] = edgeValue / rowMass
		} else {
			edgePotential[source] = math.NaN()
		}
	}
}
