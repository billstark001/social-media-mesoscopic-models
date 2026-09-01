package kinetic

import (
	"fmt"
	"math"
	"smp-meso/numerics"
)

const wedgeChannels = 4

type statePlan struct {
	initialize       func(*state)
	structuralScore  func(*state) []float64
	rewire           func(*state, []float64, []float64)
	transportMeasure func(*state, *numerics.SparseTransition)
	transportPDE     func(*state, *numerics.TridiagonalSystem)
	validate         func(*state) error
}

type state struct {
	request RunRequest
	grid    *gridGeometry
	plan    statePlan
	Rho     []float64
	Edge    []float64
	Wedge   []float64
	work1   []float64
	work2   []float64
}

func inertInitialize(*state)                            {}
func inertScore(*state) []float64                       { return nil }
func inertRewire(*state, []float64, []float64)          {}
func inertTransport(*state, *numerics.SparseTransition) {}
func inertDiffuse(*state, *numerics.TridiagonalSystem)  {}
func inertValidate(*state) error                        { return nil }

func planForState(request RunRequest) statePlan {
	if normalize(request.Recommender.Type) != "structure_random_l1" {
		return statePlan{
			initialize: inertInitialize, structuralScore: inertScore,
			rewire: inertRewire, transportMeasure: inertTransport,
			transportPDE: inertDiffuse, validate: inertValidate,
		}
	}
	return statePlan{
		initialize: initializeWedge, structuralScore: wedgeScoreMass,
		rewire: rewireWedge, transportMeasure: transportWedge,
		transportPDE: diffuseWedge, validate: validateWedge,
	}
}

func newState(request RunRequest, grid *gridGeometry) (*state, error) {
	rho, err := request.initialProbabilities()
	if err != nil {
		return nil, err
	}
	size := request.OpinionBins
	edge := make([]float64, size*size)
	for source := 0; source < size; source++ {
		for target := 0; target < size; target++ {
			edge[source*size+target] = float64(request.OutDegree) * rho[source] * rho[target]
		}
	}
	result := &state{request: request, grid: grid, Rho: rho, Edge: edge}
	result.plan = planForState(request)
	result.plan.initialize(result)
	if err := result.validate(); err != nil {
		return nil, err
	}
	return result, nil
}

func initializeWedge(current *state) {
	size := current.request.OpinionBins
	current.Wedge = make([]float64, wedgeChannels*size*size*size)
	current.work1 = make([]float64, len(current.Wedge))
	current.work2 = make([]float64, len(current.Wedge))
	fillIndependentWedge(current.Wedge, current.Rho, current.Edge, size)
}

func wedgeIndex(channel, left, center, right, size int) int {
	return ((channel*size+left)*size+center)*size + right
}

func fillIndependentWedge(output, rho, edge []float64, size int) {
	clear(output)
	perCenter := make([]float64, 2*size*size)
	for endpoint := 0; endpoint < size; endpoint++ {
		for center := 0; center < size; center++ {
			if rho[center] <= 1e-15 {
				continue
			}
			perCenter[endpoint*size+center] = edge[endpoint*size+center] / rho[center]
			perCenter[size*size+endpoint*size+center] = edge[center*size+endpoint] / rho[center]
		}
	}
	directions := [wedgeChannels][2]int{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for channel, direction := range directions {
		leftBlock := direction[0] * size * size
		rightBlock := direction[1] * size * size
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				leftValue := perCenter[leftBlock+left*size+center]
				for right := 0; right < size; right++ {
					output[wedgeIndex(channel, left, center, right, size)] =
						leftValue * perCenter[rightBlock+right*size+center] * rho[center]
				}
			}
		}
	}
}

func wedgeScoreMass(current *state) []float64 {
	size := current.request.OpinionBins
	result := make([]float64, size*size)
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			total := 0.0
			for channel := 0; channel < wedgeChannels; channel++ {
				for center := 0; center < size; center++ {
					total += current.Wedge[wedgeIndex(channel, left, center, right, size)]
				}
			}
			result[left*size+right] = total
		}
	}
	return result
}

func rewireWedge(current *state, edgeBefore, edgeAfter []float64) {
	size := current.request.OpinionBins
	target := current.work1
	fillIndependentWedge(target, current.Rho, edgeAfter, size)
	incomingDenominator := make([]float64, size)
	outgoingDenominator := make([]float64, size)
	incomingChange := make([]float64, size)
	outgoingChange := make([]float64, size)
	for source := 0; source < size; source++ {
		for targetIndex := 0; targetIndex < size; targetIndex++ {
			index := source*size + targetIndex
			value := math.Abs(edgeAfter[index] - edgeBefore[index])
			outgoingDenominator[source] += edgeBefore[index]
			incomingDenominator[targetIndex] += edgeBefore[index]
			outgoingChange[source] += value
			incomingChange[targetIndex] += value
		}
	}
	for index := 0; index < size; index++ {
		if incomingDenominator[index] > 1e-15 {
			incomingChange[index] = clamp(incomingChange[index]/incomingDenominator[index], 0, 1)
		}
		if outgoingDenominator[index] > 1e-15 {
			outgoingChange[index] = clamp(outgoingChange[index]/outgoingDenominator[index], 0, 1)
		}
	}
	directionChange := [2][]float64{incomingChange, outgoingChange}
	directions := [wedgeChannels][2]int{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for channel, direction := range directions {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				persistence := (1 - directionChange[direction[0]][center]) *
					(1 - directionChange[direction[1]][center])
				for right := 0; right < size; right++ {
					index := wedgeIndex(channel, left, center, right, size)
					current.Wedge[index] = persistence*current.Wedge[index] +
						(1-persistence)*target[index]
				}
			}
		}
	}
}

func transportWedge(current *state, transition *numerics.SparseTransition) {
	numerics.ActiveBackend.TransportTransitionTensor3(current.work1, current.work2, current.Wedge, transition, wedgeChannels)
	current.Wedge, current.work1 = current.work1, current.Wedge
}

func diffuseWedge(current *state, system *numerics.TridiagonalSystem) {
	numerics.ActiveBackend.TransportTridiagonalTensor3(current.work1, current.work2, current.Wedge, system, wedgeChannels)
	current.Wedge, current.work1 = current.work1, current.Wedge
}

func validateWedge(current *state) error {
	expected := wedgeChannels * current.request.OpinionBins * current.request.OpinionBins * current.request.OpinionBins
	if len(current.Wedge) != expected {
		return fmt.Errorf("wedge dimension %d, expected %d", len(current.Wedge), expected)
	}
	for index, value := range current.Wedge {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < -1e-10 {
			return fmt.Errorf("invalid wedge mass at %d: %g", index, value)
		}
		if value < 0 {
			current.Wedge[index] = 0
		}
	}
	return nil
}

func (current *state) validate() error {
	for index, value := range current.Rho {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < -1e-10 {
			return fmt.Errorf("invalid rho at %d: %g", index, value)
		}
		if value < 0 {
			current.Rho[index] = 0
		}
	}
	for index, value := range current.Edge {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < -1e-10 {
			return fmt.Errorf("invalid edge at %d: %g", index, value)
		}
		if value < 0 {
			current.Edge[index] = 0
		}
	}
	total := 0.0
	for _, value := range current.Rho {
		total += value
	}
	if math.Abs(total-1) > 1e-9 {
		return fmt.Errorf("node mass error %.3e", math.Abs(total-1))
	}
	size := current.request.OpinionBins
	degree := float64(current.request.OutDegree)
	for source := 0; source < size; source++ {
		row := 0.0
		for target := 0; target < size; target++ {
			row += current.Edge[source*size+target]
		}
		if math.Abs(row-degree*current.Rho[source]) > 1e-8 {
			return fmt.Errorf("edge row %d violates fixed out-degree", source)
		}
	}
	return current.plan.validate(current)
}
