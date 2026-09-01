package lifted

import (
	"fmt"
	"smp-meso/config"
)

type coordinateBlocks uint8

const (
	blockBase coordinateBlocks = 1 << iota
	blockWedge
	blockHistogram
	blockCandidate
	blockTopology
)

type rewireWedgeKernel func(
	*State, *State, config.ClosureConfig, ClosureProfile, []float64, float64,
) []float64
type rewireCoordinateKernel func(
	*State, *State, config.ClosureConfig, ClosureProfile, []float64, float64,
)
type transportBaseKernel func(*State, []float64, []float64) ([]float64, []float64)
type transportSliceKernel func(*State, []float64) []float64
type transportTopologyKernel func(*State, []float64, []float64) ([]float64, []float64)
type installOpinionKernel func(
	*State, ClosureProfile,
	[]float64, []float64, []float64, []float64, []float64, []float64, []float64,
)

// layerPlan resolves layer-dependent behavior once when a path is created.
// Its function fields are invoked at stage boundaries, keeping dispatch out of
// the numerical inner loops.
type layerPlan struct {
	layer  Layer
	blocks coordinateBlocks

	scoreCore           func(*State, []float64, float64)
	rebuildWedge        func(*State, []float64, float64)
	initializeHistogram func(*State, config.RunRequest, ClosureProfile) error
	initializeTopology  func(*State, config.RunRequest)

	rewiringEligibility func(*State, config.RunRequest, ClosureProfile, []float64, []float64) []float64
	bridgeFractions     func(*State) []float64
	rewireWedge         rewireWedgeKernel
	rewireScore         rewireCoordinateKernel
	rewireHistogram     rewireCoordinateKernel
	rewireTopology      rewireCoordinateKernel
	afterFastRewire     func(*State, *State)

	transportBase      transportBaseKernel
	transportWedge     transportSliceKernel
	transportHistogram transportSliceKernel
	transportCandidate transportSliceKernel
	transportTopology  transportTopologyKernel
	installOpinion     installOpinionKernel
	rescaleCandidate   func(*State, []float64)
	relaxOpinion       func(*State, config.RunRequest, ClosureProfile) error
	relaxHistogram     func(*State, *State, config.RunRequest)
	relaxCandidate     func(*State, *State, config.RunRequest)
	relaxTopology      func(*State, *State, config.RunRequest)

	validateHistogram func(*State) error
	validateTopology  func(*State) error
	validateWedge     func(*State) error
}

const layerCount = int(LayerTopology) + 1

var layerPlans [layerCount]layerPlan

func init() {
	layerPlans = buildLayerPlans()
}

func noInitializeHistogram(*State, config.RunRequest, ClosureProfile) error { return nil }
func noInitializeTopology(*State, config.RunRequest)                        {}
func noRewireWedge(
	*State, *State, config.ClosureConfig, ClosureProfile, []float64, float64,
) []float64 {
	return nil
}
func noRewire(
	*State, *State, config.ClosureConfig, ClosureProfile, []float64, float64,
) {
}
func noAfterFastRewire(*State, *State) {}
func noTransportBase(*State, []float64, []float64) ([]float64, []float64) {
	return nil, nil
}
func noTransportSlice(*State, []float64) []float64 { return nil }
func noTransportTopology(*State, []float64, []float64) ([]float64, []float64) {
	return nil, nil
}
func noRescaleCandidate(*State, []float64)                           {}
func noRelaxOpinion(*State, config.RunRequest, ClosureProfile) error { return nil }
func noRelax(*State, *State, config.RunRequest)                      {}
func noValidate(*State) error                                        { return nil }

func buildLayerPlans() [layerCount]layerPlan {
	// Plans are accumulated in layer order. Each layer inherits every earlier
	// decision and overrides only the stages introduced or replaced there.
	plan := layerPlan{
		scoreCore:           rebuildScoreWithoutXi,
		rebuildWedge:        noRebuildWedge,
		initializeHistogram: noInitializeHistogram,
		initializeTopology:  noInitializeTopology,
		rewiringEligibility: inferredRewiringEligibility,
		bridgeFractions:     inferredBridgeFractions,
		rewireWedge:         noRewireWedge,
		rewireScore:         rewireNaiveScore,
		rewireHistogram:     noRewire,
		rewireTopology:      noRewire,
		afterFastRewire:     noAfterFastRewire,
		transportBase:       noTransportBase,
		transportWedge:      noTransportSlice,
		transportHistogram:  noTransportSlice,
		transportCandidate:  noTransportSlice,
		transportTopology:   noTransportTopology,
		installOpinion:      installNaiveOpinion,
		rescaleCandidate:    noRescaleCandidate,
		relaxOpinion:        noRelaxOpinion,
		relaxHistogram:      noRelax,
		relaxCandidate:      noRelax,
		relaxTopology:       noRelax,
		validateHistogram:   noValidate,
		validateTopology:    noValidate,
		validateWedge:       noValidate,
	}

	var result [layerCount]layerPlan
	plan.layer = LayerNaive
	result[LayerNaive] = plan

	plan.layer = LayerBase
	plan.blocks |= blockBase
	plan.rewireScore = rewireBaseScore
	plan.afterFastRewire = copyFastCandidate
	plan.transportBase = transportBaseOpinion
	plan.installOpinion = installBaseOpinion
	plan.rescaleCandidate = rescaleBaseCandidate
	result[LayerBase] = plan

	plan.layer = LayerWedge
	plan.blocks |= blockWedge
	plan.rebuildWedge = rebuildWedgeFromPoisson
	plan.rewireWedge = rewireWedgeCoordinate
	plan.rewireScore = rewireWedgeScore
	plan.transportWedge = transportWedgeOpinion
	plan.rescaleCandidate = rescaleWedgeCandidate
	plan.validateWedge = validateWedgeAgainstScore
	result[LayerWedge] = plan

	plan.layer = LayerHistogram
	plan.blocks |= blockHistogram
	plan.initializeHistogram = initializeHistogram
	plan.rewiringEligibility = histogramRewiringEligibility
	plan.rewireHistogram = rewireHistogramCoordinate
	plan.transportHistogram = transportHistogramOpinion
	plan.relaxOpinion = relaxRetainedOpinion
	plan.relaxHistogram = relaxHistogramOpinion
	plan.validateHistogram = validateHistogramMass
	result[LayerHistogram] = plan

	plan.layer = LayerCandidate
	plan.blocks |= blockCandidate
	plan.scoreCore = rebuildScoreWithXi
	plan.rebuildWedge = rebuildWedgeFromXi
	plan.rewireScore = rewireCandidateScore
	plan.afterFastRewire = noAfterFastRewire
	plan.transportBase = noTransportBase
	plan.transportCandidate = transportCandidateOpinion
	plan.installOpinion = installCandidateOpinion
	plan.rescaleCandidate = noRescaleCandidate
	plan.relaxCandidate = relaxCandidateOpinion
	plan.validateWedge = validateWedgeAgainstXi
	result[LayerCandidate] = plan

	plan.layer = LayerTopology
	plan.blocks |= blockTopology
	plan.initializeTopology = initializeTopology
	plan.rewiringEligibility = topologyRewiringEligibility
	plan.bridgeFractions = topologyBridgeFractions
	plan.rewireTopology = rewireTopologyCoordinate
	plan.transportTopology = transportTopologyOpinion
	plan.relaxTopology = relaxTopologyOpinion
	plan.validateTopology = validateTopologyMass
	result[LayerTopology] = plan

	return result
}

func planFor(layer Layer) (*layerPlan, error) {
	index := int(layer)
	if index < 0 || index >= len(layerPlans) {
		return nil, fmt.Errorf("unsupported layer index %d", index)
	}
	return &layerPlans[index], nil
}

func mustPlanFor(layer Layer) *layerPlan {
	plan, err := planFor(layer)
	if err != nil {
		panic(err)
	}
	return plan
}

func (p *layerPlan) allocate(state *State, request config.RunRequest) {
	if p.blocks&blockWedge != 0 {
		state.Wedge = make([]float64, state.Bins*state.Bins*state.Bins)
	}
	if p.blocks&blockHistogram != 0 {
		state.Histogram = make([]float64,
			state.Bins*(request.OutDegree+1)*(request.OutDegree+1)*request.Resolution.AvailabilityBins)
	}
	if p.blocks&blockCandidate != 0 {
		state.Xi = make([]float64, state.Bins*state.Bins*2*(request.Resolution.ScoreMax+1))
	}
	if p.blocks&blockTopology != 0 {
		state.Components = make([]float64, state.Bins*request.Resolution.ComponentSizeBins)
		state.Bridges = make([]float64, state.Bins*state.Bins)
	}
}

func (p *layerPlan) dimension(state *State) int {
	dimension := len(state.Rho) + len(state.Edge)
	if p.blocks&blockBase != 0 {
		dimension += len(state.Candidate) + len(state.Score)
	}
	if p.blocks&blockWedge != 0 {
		dimension += len(state.Wedge)
	}
	if p.blocks&blockHistogram != 0 {
		dimension += len(state.Histogram)
	}
	if p.blocks&blockCandidate != 0 {
		dimension += len(state.Xi)
	}
	if p.blocks&blockTopology != 0 {
		dimension += len(state.Components) + len(state.Bridges)
	}
	return dimension
}

func (p *layerPlan) initializeAuxiliary(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
) error {
	if err := p.initializeHistogram(state, request, profile); err != nil {
		return err
	}
	p.initializeTopology(state, request)
	return nil
}
