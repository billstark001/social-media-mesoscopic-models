package meso

import (
	"fmt"
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/numerics"
	"strings"
)

type Layer = config.Layer

const (
	LayerNaive     = config.LayerNaive
	LayerBase      = config.LayerBase
	LayerWedge     = config.LayerWedge
	LayerHistogram = config.LayerHistogram
	LayerCandidate = config.LayerCandidate
	LayerTopology  = config.LayerTopology
)

type ClosureProfile struct {
	EligibilityCorrelation float64 `json:"eligibility_correlation"`
	ScoreAvailability      float64 `json:"score_availability"`
	MotifPersistence       float64 `json:"motif_persistence"`
	BridgeBias             float64 `json:"bridge_bias"`
	ComponentMix           float64 `json:"component_mix"`
}

type State struct {
	Layer             Layer
	Population        int
	Bins              int
	Degree            int
	Recommendations   int
	ScoreMax          int
	AvailabilityBins  int
	ComponentSizeBins int
	Steepness         float64

	Axis       []float64
	Rho        []float64
	Edge       []float64
	Candidate  []float64
	Score      []float64
	Wedge      []float64
	Histogram  []float64
	Xi         []float64
	Components []float64
	Bridges    []float64
}

func newEmptyState(request config.RunRequest, layer Layer) *State {
	bins := request.OpinionBins
	state := &State{
		Layer: layer, Population: request.Population, Bins: bins,
		Degree: request.OutDegree, Recommendations: request.RecommendationCount,
		ScoreMax:          request.Resolution.ScoreMax,
		AvailabilityBins:  request.Resolution.AvailabilityBins,
		ComponentSizeBins: request.Resolution.ComponentSizeBins,
		Steepness:         request.Recommender.Steepness,
		Axis:              make([]float64, bins), Rho: make([]float64, bins),
		Edge: make([]float64, bins*bins),
	}
	// Naive keeps only rho/E dynamically. Candidate and Score are workspace
	// caches reconstructed from rho/E before they are used, so they are not
	// counted as retained coordinates.
	state.Candidate = make([]float64, bins*bins)
	state.Score = make([]float64, bins*bins)
	dx := (request.Initial.OpinionMax - request.Initial.OpinionMin) / float64(bins)
	for i := range state.Axis {
		state.Axis[i] = request.Initial.OpinionMin + (float64(i)+0.5)*dx
	}
	if layer >= LayerWedge {
		state.Wedge = make([]float64, bins*bins*bins)
	}
	if layer >= LayerHistogram {
		state.Histogram = make([]float64,
			bins*(request.OutDegree+1)*(request.OutDegree+1)*request.Resolution.AvailabilityBins)
	}
	if layer >= LayerCandidate {
		state.Xi = make([]float64, bins*bins*2*(request.Resolution.ScoreMax+1))
	}
	if layer >= LayerTopology {
		state.Components = make([]float64, bins*request.Resolution.ComponentSizeBins)
		state.Bridges = make([]float64, bins*bins)
	}
	return state
}

func (s *State) Clone() *State {
	result := *s
	result.Axis = append([]float64(nil), s.Axis...)
	result.Rho = append([]float64(nil), s.Rho...)
	result.Edge = append([]float64(nil), s.Edge...)
	result.Candidate = append([]float64(nil), s.Candidate...)
	result.Score = append([]float64(nil), s.Score...)
	result.Wedge = append([]float64(nil), s.Wedge...)
	result.Histogram = append([]float64(nil), s.Histogram...)
	result.Xi = append([]float64(nil), s.Xi...)
	result.Components = append([]float64(nil), s.Components...)
	result.Bridges = append([]float64(nil), s.Bridges...)
	return &result
}

func (s *State) matrixIndex(i, j int) int { return i*s.Bins + j }
func (s *State) wedgeIndex(i, r, j int) int {
	return (i*s.Bins+r)*s.Bins + j
}
func (s *State) histogramIndex(i, k, d, c int) int {
	dCount := s.Degree + 1
	return (((i*dCount+k)*dCount+d)*s.AvailabilityBins + c)
}
func (s *State) xiIndex(i, j, available, score int) int {
	return (((i*s.Bins+j)*2+available)*(s.ScoreMax+1) + score)
}
func (s *State) componentIndex(i, sizeBin int) int {
	return i*s.ComponentSizeBins + sizeBin
}

func initialProbabilities(request config.RunRequest) []float64 {
	if strings.ToLower(strings.TrimSpace(request.Initial.Type)) == "categorical" {
		result, _ := numerics.NormalizedCopy(request.Initial.Probabilities)
		return result
	}
	result := make([]float64, request.OpinionBins)
	for i := range result {
		result[i] = 1 / float64(len(result))
	}
	return result
}

func InitialState(request config.RunRequest, layer Layer, profile ClosureProfile, rng *rand.Rand) (*State, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	state := newEmptyState(request, layer)
	probabilities := initialProbabilities(request)
	nodeCounts := make([]int, state.Bins)
	numerics.SampleMultinomial(state.Population, probabilities, rng, nodeCounts)
	for i, count := range nodeCounts {
		state.Rho[i] = float64(count) / float64(state.Population)
	}

	edgeCounts := make([]int, state.Bins)
	for source, count := range nodeCounts {
		numerics.SampleMultinomial(state.Degree*count, state.Rho, rng, edgeCounts)
		for target, value := range edgeCounts {
			state.Edge[state.matrixIndex(source, target)] = float64(value) / float64(state.Population)
		}
	}
	state.rebuildCandidate()
	state.rebuildScoreState(profile.ScoreAvailability)
	if layer >= LayerHistogram {
		if err := state.rebuildHistogram(request, profile.EligibilityCorrelation); err != nil {
			return nil, err
		}
	}
	if layer >= LayerTopology {
		state.rebuildTopology(request)
	}
	if err := state.Validate(); err != nil {
		return nil, err
	}
	return state, nil
}

func (s *State) neighborKernel() []float64 {
	result := make([]float64, len(s.Edge))
	for i := 0; i < s.Bins; i++ {
		fallback := s.Rho
		row := result[i*s.Bins : (i+1)*s.Bins]
		if s.Rho[i] > numerics.ProbabilityEpsilon {
			denominator := float64(s.Degree) * s.Rho[i]
			for j := range row {
				row[j] = s.Edge[s.matrixIndex(i, j)] / denominator
			}
		}
		numerics.NormalizeInPlace(row, fallback)
	}
	return result
}

func (s *State) rebuildCandidate() {
	neighbors := s.neighborKernel()
	for i := 0; i < s.Bins; i++ {
		for j := 0; j < s.Bins; j++ {
			perSource := float64(s.Population)*s.Rho[j] - float64(s.Degree)*neighbors[s.matrixIndex(i, j)]
			if i == j {
				perSource--
			}
			s.Candidate[s.matrixIndex(i, j)] = s.Rho[i] * math.Max(perSource, 0)
		}
	}
}

func (s *State) undirectedAdjacencyProbabilities() []float64 {
	result := make([]float64, s.Bins*s.Bins)
	for i := 0; i < s.Bins; i++ {
		for j := 0; j < s.Bins; j++ {
			denominator := float64(s.Population) * s.Rho[i] * s.Rho[j]
			if denominator > numerics.ProbabilityEpsilon {
				mass := s.Edge[s.matrixIndex(i, j)] + s.Edge[s.matrixIndex(j, i)]
				result[s.matrixIndex(i, j)] = numerics.Clamp(mass/denominator, 0, 1)
			}
		}
	}
	return result
}

func (s *State) scoreMean(i, j int, adjacency []float64) float64 {
	mean := 0.0
	for center := 0; center < s.Bins; center++ {
		mean += float64(s.Population) * s.Rho[center] *
			adjacency[s.matrixIndex(i, center)] * adjacency[s.matrixIndex(j, center)]
	}
	return math.Max(mean, 0)
}

func availabilityByScore(pmf []float64, availableFraction, correlation float64) []float64 {
	availableFraction = numerics.Clamp(availableFraction, 0, 1)
	correlation = numerics.Clamp(correlation, -1, 1)
	independent := make([]float64, len(pmf))
	for i, probability := range pmf {
		independent[i] = availableFraction * probability
	}
	if math.Abs(correlation) <= numerics.ProbabilityEpsilon || availableFraction == 0 || availableFraction == 1 {
		return independent
	}
	extreme := make([]float64, len(pmf))
	remaining := availableFraction
	if correlation > 0 {
		for score := len(pmf) - 1; score >= 0 && remaining > 0; score-- {
			take := math.Min(pmf[score], remaining)
			extreme[score] = take
			remaining -= take
		}
	} else {
		for score := 0; score < len(pmf) && remaining > 0; score++ {
			take := math.Min(pmf[score], remaining)
			extreme[score] = take
			remaining -= take
		}
	}
	weight := math.Abs(correlation)
	for score := range independent {
		independent[score] = (1-weight)*independent[score] + weight*extreme[score]
	}
	return independent
}

func (s *State) rebuildScoreState(availabilityCorrelation float64) {
	adjacency := s.undirectedAdjacencyProbabilities()
	if s.Layer >= LayerCandidate {
		clear(s.Xi)
	}
	for i := 0; i < s.Bins; i++ {
		for j := 0; j < s.Bins; j++ {
			index := s.matrixIndex(i, j)
			pairMass := float64(s.Population) * s.Rho[i] * s.Rho[j]
			if i == j {
				pairMass = math.Max(pairMass-s.Rho[i], 0)
			}
			availableFraction := 0.0
			if pairMass > numerics.ProbabilityEpsilon {
				availableFraction = numerics.Clamp(s.Candidate[index]/pairMass, 0, 1)
			}
			pmf := numerics.PoissonPMF(s.scoreMean(i, j, adjacency), s.ScoreMax)
			available := availabilityByScore(pmf, availableFraction, availabilityCorrelation)
			scoreMoment := 0.0
			for score, mass := range available {
				if score > 0 {
					scoreMoment += mass * math.Pow(float64(score), s.Steepness)
				}
				if s.Layer >= LayerCandidate {
					s.Xi[s.xiIndex(i, j, 1, score)] = pairMass * mass
					s.Xi[s.xiIndex(i, j, 0, score)] = pairMass * math.Max(pmf[score]-mass, 0)
				}
			}
			s.Score[index] = pairMass * scoreMoment
		}
	}
	if s.Layer >= LayerWedge {
		s.rebuildWedge(adjacency, availabilityCorrelation)
	}
}

func (s *State) rebuildWedge(adjacency []float64, availabilityCorrelation float64) {
	clear(s.Wedge)
	for center := 0; center < s.Bins; center++ {
		if s.Rho[center] <= numerics.ProbabilityEpsilon {
			continue
		}
		degrees := make([]float64, s.Bins)
		for endpoint := 0; endpoint < s.Bins; endpoint++ {
			undirectedMass := s.Edge[s.matrixIndex(endpoint, center)] + s.Edge[s.matrixIndex(center, endpoint)]
			degrees[endpoint] = undirectedMass / s.Rho[center]
		}
		for i := 0; i < s.Bins; i++ {
			for j := 0; j < s.Bins; j++ {
				value := s.Rho[center] * degrees[i] * degrees[j]
				if i == j {
					value = s.Rho[center] * math.Max(degrees[i]*(degrees[i]-1), 0)
				}
				pairMass := float64(s.Population) * s.Rho[i] * s.Rho[j]
				if i == j {
					pairMass = math.Max(pairMass-s.Rho[i], 0)
				}
				availableFraction := 0.0
				if pairMass > numerics.ProbabilityEpsilon {
					availableFraction = numerics.Clamp(s.Candidate[s.matrixIndex(i, j)]/pairMass, 0, 1)
				}
				s.Wedge[s.wedgeIndex(i, center, j)] = value * availableFraction
			}
		}
	}

	// Candidate-weighted W must project to the first score moment. Match the
	// candidate histogram when it is present, otherwise match the Poisson target.
	for i := 0; i < s.Bins; i++ {
		for j := 0; j < s.Bins; j++ {
			firstMoment := 0.0
			if s.Layer >= LayerCandidate {
				for score := 1; score <= s.ScoreMax; score++ {
					firstMoment += float64(score) * s.Xi[s.xiIndex(i, j, 1, score)]
				}
			} else {
				pairMass := float64(s.Population) * s.Rho[i] * s.Rho[j]
				if i == j {
					pairMass = math.Max(pairMass-s.Rho[i], 0)
				}
				availableFraction := 0.0
				if pairMass > numerics.ProbabilityEpsilon {
					availableFraction = s.Candidate[s.matrixIndex(i, j)] / pairMass
				}
				// Use the same right-censored Poisson law as S_zeta.  Using the
				// uncensored analytic mean here would make W and S_1 disagree by
				// exactly the mass accumulated in the last score bin.
				pmf := numerics.PoissonPMF(s.scoreMean(i, j, adjacency), s.ScoreMax)
				available := availabilityByScore(pmf, availableFraction, availabilityCorrelation)
				for score, mass := range available {
					firstMoment += pairMass * mass * float64(score)
				}
			}
			current := 0.0
			for center := 0; center < s.Bins; center++ {
				current += s.Wedge[s.wedgeIndex(i, center, j)]
			}
			if current > numerics.ProbabilityEpsilon {
				scale := firstMoment / current
				for center := 0; center < s.Bins; center++ {
					s.Wedge[s.wedgeIndex(i, center, j)] *= scale
				}
			} else if firstMoment > numerics.ProbabilityEpsilon {
				for center, mass := range s.Rho {
					s.Wedge[s.wedgeIndex(i, center, j)] = firstMoment * mass
				}
			}
		}
	}
}

func (s *State) projectXi() {
	if s.Layer < LayerCandidate {
		return
	}
	clear(s.Candidate)
	clear(s.Score)
	for i := 0; i < s.Bins; i++ {
		for j := 0; j < s.Bins; j++ {
			index := s.matrixIndex(i, j)
			for score := 0; score <= s.ScoreMax; score++ {
				mass := s.Xi[s.xiIndex(i, j, 1, score)]
				s.Candidate[index] += mass
				if score > 0 {
					s.Score[index] += mass * math.Pow(float64(score), s.Steepness)
				}
			}
		}
	}
	s.reconcileWedgeToXi()
}

func (s *State) wedgeFirstMoment(i, j int) float64 {
	total := 0.0
	for center := 0; center < s.Bins; center++ {
		total += s.Wedge[s.wedgeIndex(i, center, j)]
	}
	return total
}

func (s *State) xiFirstMoment(i, j int) float64 {
	total := 0.0
	for score := 1; score <= s.ScoreMax; score++ {
		total += float64(score) * s.Xi[s.xiIndex(i, j, 1, score)]
	}
	return total
}

// reconcileWedgeToXi keeps the nested candidate layer projectively
// consistent. Xi is the richer coordinate, so its first score moment is
// authoritative; W retains its centre allocation whenever it has mass.
func (s *State) reconcileWedgeToXi() {
	if s.Layer < LayerCandidate {
		return
	}
	for i := 0; i < s.Bins; i++ {
		for j := 0; j < s.Bins; j++ {
			desired := s.xiFirstMoment(i, j)
			current := s.wedgeFirstMoment(i, j)
			if current > numerics.ProbabilityEpsilon {
				scale := desired / current
				for center := 0; center < s.Bins; center++ {
					s.Wedge[s.wedgeIndex(i, center, j)] *= scale
				}
				continue
			}
			if desired <= numerics.ProbabilityEpsilon {
				continue
			}
			for center, mass := range s.Rho {
				s.Wedge[s.wedgeIndex(i, center, j)] = desired * mass
			}
		}
	}
}

func frechetJoint(left, right []float64, correlation float64) []float64 {
	rows, cols := len(left), len(right)
	independent := make([]float64, rows*cols)
	for i, p := range left {
		for j, q := range right {
			independent[i*cols+j] = p * q
		}
	}
	if math.Abs(correlation) <= numerics.ProbabilityEpsilon {
		return independent
	}
	extreme := make([]float64, rows*cols)
	leftRemaining := append([]float64(nil), left...)
	rightRemaining := append([]float64(nil), right...)
	i, j := 0, 0
	if correlation < 0 {
		j = cols - 1
	}
	for i < rows && j >= 0 && j < cols {
		mass := math.Min(leftRemaining[i], rightRemaining[j])
		extreme[i*cols+j] += mass
		leftRemaining[i] -= mass
		rightRemaining[j] -= mass
		if leftRemaining[i] <= numerics.ProbabilityEpsilon {
			i++
		}
		if rightRemaining[j] <= numerics.ProbabilityEpsilon {
			if correlation > 0 {
				j++
			} else {
				j--
			}
		}
	}
	weight := math.Abs(correlation)
	for index := range independent {
		independent[index] = (1-weight)*independent[index] + weight*extreme[index]
	}
	return independent
}

func (s *State) rebuildHistogram(request config.RunRequest, correlation float64) error {
	if s.Layer < LayerHistogram {
		return nil
	}
	clear(s.Histogram)
	neighbors := s.neighborKernel()
	recommendations, err := RecommendationKernel(s, request)
	if err != nil {
		return err
	}
	for i := 0; i < s.Bins; i++ {
		pDiscordant := 0.0
		pConcordantRecommendation := 0.0
		for j := 0; j < s.Bins; j++ {
			concordant := math.Abs(s.Axis[i]-s.Axis[j]) <= request.Dynamics.Tolerance
			if concordant {
				pConcordantRecommendation += recommendations[s.matrixIndex(i, j)]
			} else {
				pDiscordant += neighbors[s.matrixIndex(i, j)]
			}
		}
		dMarginal := numerics.BinomialPMF(s.Degree, numerics.Clamp(pDiscordant, 0, 1))
		cMarginal := numerics.BinomialPMF(s.AvailabilityBins-1, numerics.Clamp(pConcordantRecommendation, 0, 1))
		joint := frechetJoint(dMarginal, cMarginal, correlation)
		for d := 0; d <= s.Degree; d++ {
			for c := 0; c < s.AvailabilityBins; c++ {
				s.Histogram[s.histogramIndex(i, s.Degree, d, c)] = s.Rho[i] * joint[d*s.AvailabilityBins+c]
			}
		}
	}
	return nil
}

func (s *State) histogramEligibility(source int) float64 {
	if s.Layer < LayerHistogram || s.Rho[source] <= numerics.ProbabilityEpsilon {
		return 0
	}
	eligible := 0.0
	for d := 1; d <= s.Degree; d++ {
		for c := 1; c < s.AvailabilityBins; c++ {
			concordantFraction := float64(c) / float64(s.AvailabilityBins-1)
			feedAvailable := 1 - math.Pow(1-concordantFraction, float64(s.Recommendations))
			eligible += s.Histogram[s.histogramIndex(source, s.Degree, d, c)] * feedAvailable
		}
	}
	return numerics.Clamp(eligible/s.Rho[source], 0, 1)
}

func (s *State) rebuildTopology(request config.RunRequest) {
	if s.Layer < LayerTopology {
		return
	}
	clear(s.Components)
	neighbors := s.neighborKernel()
	adjacency := s.undirectedAdjacencyProbabilities()
	for i := 0; i < s.Bins; i++ {
		concordantDegree := 0.0
		for j := 0; j < s.Bins; j++ {
			if math.Abs(s.Axis[i]-s.Axis[j]) <= request.Dynamics.Tolerance {
				concordantDegree += float64(s.Degree) * neighbors[s.matrixIndex(i, j)]
			}
		}
		giantFraction := numerics.Clamp((concordantDegree-1)/math.Max(concordantDegree, 1), 0, 1)
		s.Components[s.componentIndex(i, s.ComponentSizeBins-1)] = s.Rho[i] * giantFraction
		smallBin := int(math.Min(float64(s.ComponentSizeBins-2), math.Max(concordantDegree, 0)))
		s.Components[s.componentIndex(i, smallBin)] += s.Rho[i] * (1 - giantFraction)
	}
	for i := 0; i < s.Bins; i++ {
		for j := 0; j < s.Bins; j++ {
			meanScore := s.scoreMean(i, j, adjacency)
			s.Bridges[s.matrixIndex(i, j)] = s.Edge[s.matrixIndex(i, j)] / (1 + meanScore)
		}
	}
}

func (s *State) Validate() error {
	for name, values := range map[string][]float64{
		"rho": s.Rho, "edge": s.Edge, "candidate": s.Candidate,
		"score": s.Score, "wedge": s.Wedge, "histogram": s.Histogram,
		"xi": s.Xi, "components": s.Components, "bridges": s.Bridges,
	} {
		if !numerics.FiniteSlice(values) {
			return fmt.Errorf("%s contains non-finite values", name)
		}
		for _, value := range values {
			if value < -1e-9 {
				return fmt.Errorf("%s contains negative value %g", name, value)
			}
		}
	}
	if math.Abs(numerics.Sum(s.Rho)-1) > 1e-9 {
		return fmt.Errorf("rho is not normalized: %.16g", numerics.Sum(s.Rho))
	}
	for i := 0; i < s.Bins; i++ {
		row := numerics.Sum(s.Edge[i*s.Bins : (i+1)*s.Bins])
		expected := float64(s.Degree) * s.Rho[i]
		if math.Abs(row-expected) > 1e-8 {
			return fmt.Errorf("edge row %d has mass %.16g, expected %.16g", i, row, expected)
		}
		if s.Layer >= LayerHistogram {
			histogramMass := 0.0
			for k := 0; k <= s.Degree; k++ {
				for d := 0; d <= s.Degree; d++ {
					for c := 0; c < s.AvailabilityBins; c++ {
						histogramMass += s.Histogram[s.histogramIndex(i, k, d, c)]
					}
				}
			}
			if math.Abs(histogramMass-s.Rho[i]) > 1e-8 {
				return fmt.Errorf("histogram bin %d has mass %.16g, expected %.16g", i, histogramMass, s.Rho[i])
			}
		}
		if s.Layer >= LayerTopology {
			componentMass := numerics.Sum(s.Components[i*s.ComponentSizeBins : (i+1)*s.ComponentSizeBins])
			if math.Abs(componentMass-s.Rho[i]) > 1e-8 {
				return fmt.Errorf("component bin %d has mass %.16g, expected %.16g", i, componentMass, s.Rho[i])
			}
		}
	}
	if err := s.validateWedgeProjection(); err != nil {
		return err
	}
	return nil
}

func (s *State) validateWedgeProjection() error {
	if s.Layer >= LayerWedge {
		for i := 0; i < s.Bins; i++ {
			for j := 0; j < s.Bins; j++ {
				projected := s.wedgeFirstMoment(i, j)
				desired := projected
				switch {
				case s.Layer >= LayerCandidate:
					desired = s.xiFirstMoment(i, j)
				case math.Abs(s.Steepness-1) <= 1e-12:
					desired = s.Score[s.matrixIndex(i, j)]
				default:
					continue
				}
				if math.Abs(projected-desired) > 1e-7*math.Max(1, desired) {
					return fmt.Errorf("wedge (%d,%d) projects to %.16g, expected %.16g", i, j, projected, desired)
				}
			}
		}
	}
	return nil
}

func (s *State) Dimension() int {
	dimension := len(s.Rho) + len(s.Edge)
	if s.Layer >= LayerBase {
		dimension += len(s.Candidate) + len(s.Score)
	}
	return dimension +
		len(s.Wedge) + len(s.Histogram) + len(s.Xi) + len(s.Components) + len(s.Bridges)
}
