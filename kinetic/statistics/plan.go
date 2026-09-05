// Package statistics computes density observables without retaining kinetic
// state trajectories. It depends only on density arrays and grid geometry.
package statistics

import "math"

type Config struct {
	Polarization              bool
	Subjective                bool
	Homophily                 bool
	HomophilyRaw              bool
	NodeEnergy                bool
	EdgeEnergy                bool
	Pathway                   bool
	PolarizationFirstPassage  bool
	HomophilyFirstPassage     bool
	PolarizationThreshold     float64
	HomophilyThreshold        float64
	MinimumBandwidth          float64
	ObjectiveEffectiveSamples int
	DistanceGridSize          int
	Population                int
	OutDegree                 int
	Tolerance                 float64
	DomainWidth               float64
}

type FirstPassage struct {
	Reached bool
	Time    float64
}

type Outcome struct {
	Time             []float64
	Polarization     []float64
	Subjective       []float64
	Homophily        []float64
	HomophilyRaw     []float64
	NodeEnergy       []float64
	EdgeEnergy       []float64
	Pathway          float64
	HasPathway       bool
	PolarizationPass FirstPassage
	HomophilyPass    FirstPassage
}

type Plan struct {
	config           Config
	axis             []float64
	concordance      []float64
	distanceBins     []int
	distances        []float64
	distanceAxis     []float64
	randomPDF        []float64
	subjectiveWorst  []float64
	subjectiveScale  float64
	needPolarization bool
	needSubjective   bool
	needHomophily    bool
	storeTime        bool
	retainTime       bool
	time             []float64
	polarization     []float64
	subjective       []float64
	homophily        []float64
	homophilyRaw     []float64
	nodeEnergy       []float64
	edgeEnergy       []float64
	interaction      *InteractionPlan
	pathwayValue     float64
	pathwayPreviousP float64
	pathwayPreviousH float64
	pathwayStarted   bool
}

func NewPlan(axis, concordance []float64, config Config) *Plan {
	result := &Plan{
		config: config, axis: append([]float64(nil), axis...),
		concordance:      append([]float64(nil), concordance...),
		needPolarization: config.Polarization || config.PolarizationFirstPassage,
		needSubjective:   config.Subjective,
		needHomophily:    config.Homophily || config.HomophilyRaw || config.HomophilyFirstPassage,
		storeTime:        config.Polarization || config.Subjective || config.Homophily || config.HomophilyRaw || config.NodeEnergy || config.EdgeEnergy,
		retainTime:       config.Polarization || config.Subjective || config.Homophily || config.HomophilyRaw || config.NodeEnergy || config.EdgeEnergy || config.PolarizationFirstPassage || config.HomophilyFirstPassage,
	}
	if result.needPolarization || result.needSubjective || config.Pathway {
		result.prepareDistances()
	}
	if config.NodeEnergy || config.EdgeEnergy {
		result.interaction = NewInteractionPlan(axis, config.Tolerance, config.OutDegree)
	}
	return result
}

func (plan *Plan) prepareDistances() {
	size := len(plan.axis)
	dx := plan.axis[1] - plan.axis[0]
	plan.distances = make([]float64, size)
	for index := range plan.distances {
		plan.distances[index] = float64(index) * dx
	}
	plan.distanceBins = make([]int, size*size)
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			plan.distanceBins[left*size+right] = absInt(left - right)
		}
	}
	errorRange := 4 * plan.config.MinimumBandwidth
	plan.distanceAxis = linspace(-errorRange, plan.config.DomainWidth+errorRange, plan.config.DistanceGridSize)
	uniformPair := make([]float64, size*size)
	for index := range uniformPair {
		uniformPair[index] = 1 / float64(size*size)
	}
	randomMass := plan.distanceMass(uniformPair)
	randomBandwidth := bandwidth(plan.distances, randomMass, plan.config.ObjectiveEffectiveSamples, plan.config.MinimumBandwidth)
	plan.randomPDF = mixturePDF(plan.distanceAxis, plan.distances, randomMass, randomBandwidth)
	if plan.needSubjective {
		plan.subjectiveWorst = normalPDF(plan.distanceAxis, 0, plan.config.MinimumBandwidth)
		plan.subjectiveScale = jsDistance(plan.subjectiveWorst, plan.randomPDF, plan.distanceAxis)
	}
}

// ObservePathway accumulates the trapezoidal line integral H dP at every
// solver step without retaining the polarization or homophily trajectory.
// The completion term is added only when Outcome is requested.
func (plan *Plan) ObservePathway(rho, edge []float64) {
	if !plan.config.Pathway {
		return
	}
	polarization := plan.calculatePolarization(rho)
	homophily, _ := plan.calculateHomophily(edge)
	if plan.pathwayStarted {
		plan.pathwayValue += 0.5 * (polarization - plan.pathwayPreviousP) *
			(homophily + plan.pathwayPreviousH)
	}
	plan.pathwayPreviousP = polarization
	plan.pathwayPreviousH = homophily
	plan.pathwayStarted = true
}

func (plan *Plan) Record(time float64, rho, edge []float64) {
	if plan.retainTime {
		plan.time = append(plan.time, time)
	}
	if plan.needPolarization {
		plan.polarization = append(plan.polarization, plan.calculatePolarization(rho))
	}
	if plan.needSubjective {
		plan.subjective = append(plan.subjective, plan.calculateSubjective(edge))
	}
	if plan.needHomophily {
		homophily, raw := plan.calculateHomophily(edge)
		plan.homophily = append(plan.homophily, homophily)
		plan.homophilyRaw = append(plan.homophilyRaw, raw)
	}
	if plan.interaction != nil {
		node, edgeValue := plan.interaction.Energies(rho, edge)
		if plan.config.NodeEnergy {
			plan.nodeEnergy = append(plan.nodeEnergy, node)
		}
		if plan.config.EdgeEnergy {
			plan.edgeEnergy = append(plan.edgeEnergy, edgeValue)
		}
	}
}

func firstPassage(values, times []float64, threshold float64) FirstPassage {
	for index, value := range values {
		if value >= threshold {
			return FirstPassage{Reached: true, Time: times[index]}
		}
	}
	return FirstPassage{}
}

func (plan *Plan) Outcome() Outcome {
	result := Outcome{}
	if plan.storeTime {
		result.Time = append([]float64(nil), plan.time...)
	}
	if plan.config.Polarization {
		result.Polarization = append([]float64(nil), plan.polarization...)
	}
	if plan.config.Subjective {
		result.Subjective = append([]float64(nil), plan.subjective...)
	}
	if plan.config.Homophily {
		result.Homophily = append([]float64(nil), plan.homophily...)
	}
	if plan.config.HomophilyRaw {
		result.HomophilyRaw = append([]float64(nil), plan.homophilyRaw...)
	}
	if plan.config.NodeEnergy {
		result.NodeEnergy = append([]float64(nil), plan.nodeEnergy...)
	}
	if plan.config.EdgeEnergy {
		result.EdgeEnergy = append([]float64(nil), plan.edgeEnergy...)
	}
	if plan.config.Pathway {
		result.HasPathway = true
		if plan.pathwayStarted {
			result.Pathway = plan.pathwayValue +
				(1-plan.pathwayPreviousP)*plan.pathwayPreviousH
		}
	}
	if plan.config.PolarizationFirstPassage {
		result.PolarizationPass = firstPassage(plan.polarization, plan.time, plan.config.PolarizationThreshold)
	}
	if plan.config.HomophilyFirstPassage {
		result.HomophilyPass = firstPassage(plan.homophily, plan.time, plan.config.HomophilyThreshold)
	}
	return result
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

func linspace(lower, upper float64, count int) []float64 {
	result := make([]float64, count)
	step := (upper - lower) / float64(count-1)
	for index := range result {
		result[index] = lower + float64(index)*step
	}
	return result
}

func trapezoid(values, axis []float64) float64 {
	total := 0.0
	for index := 1; index < len(values); index++ {
		total += 0.5 * (values[index] + values[index-1]) * (axis[index] - axis[index-1])
	}
	return total
}

func jsDistance(left, right, axis []float64) float64 {
	integrand := make([]float64, len(axis))
	for index := range integrand {
		p := math.Max(left[index], 1e-13)
		q := math.Max(right[index], 1e-13)
		middle := 0.5 * (p + q)
		integrand[index] = 0.5 * (p*(math.Log(p)-math.Log(middle)) + q*(math.Log(q)-math.Log(middle)))
	}
	return math.Sqrt(math.Max(trapezoid(integrand, axis), 0))
}

func bandwidth(distances, weights []float64, effectiveSamples int, minimum float64) float64 {
	mean := 0.0
	for index := range distances {
		mean += distances[index] * weights[index]
	}
	variance := 0.0
	for index := range distances {
		variance += math.Pow(distances[index]-mean, 2) * weights[index]
	}
	return math.Max(minimum, math.Sqrt(math.Max(variance, 0))*math.Pow(float64(effectiveSamples), -0.2))
}

func normalPDF(axis []float64, location, standardDeviation float64) []float64 {
	result := make([]float64, len(axis))
	coefficient := 1 / (standardDeviation * math.Sqrt(2*math.Pi))
	for index, value := range axis {
		z := (value - location) / standardDeviation
		result[index] = coefficient * math.Exp(-0.5*z*z)
	}
	return result
}

func mixturePDF(axis, locations, weights []float64, bandwidth float64) []float64 {
	result := make([]float64, len(axis))
	coefficient := 1 / (bandwidth * math.Sqrt(2*math.Pi))
	for index, value := range axis {
		for component, location := range locations {
			z := (value - location) / bandwidth
			result[index] += weights[component] * coefficient * math.Exp(-0.5*z*z)
		}
	}
	return result
}

func (plan *Plan) distanceMass(pair []float64) []float64 {
	result := make([]float64, len(plan.axis))
	total := 0.0
	for index, value := range pair {
		result[plan.distanceBins[index]] += value
		total += value
	}
	if total > 0 {
		for index := range result {
			result[index] /= total
		}
	}
	return result
}

func (plan *Plan) calculatePolarization(rho []float64) float64 {
	size := len(rho)
	mass := make([]float64, len(plan.axis))
	total := 0.0
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			value := rho[left] * rho[right]
			mass[absInt(left-right)] += value
			total += value
		}
	}
	if total > 0 {
		for index := range mass {
			mass[index] /= total
		}
	}
	density := mixturePDF(plan.distanceAxis, plan.distances, mass,
		bandwidth(plan.distances, mass, plan.config.ObjectiveEffectiveSamples, plan.config.MinimumBandwidth))
	polarizedMass, weightedDistance := 0.0, 0.0
	for index, distance := range plan.distances {
		if distance >= plan.config.Tolerance {
			polarizedMass += mass[index]
			weightedDistance += distance * mass[index]
		}
	}
	clusterDistance := plan.config.Tolerance
	if polarizedMass > 1e-15 {
		clusterDistance = weightedDistance / polarizedMass
	}
	worstBandwidth := math.Max(plan.config.MinimumBandwidth,
		0.5*clusterDistance*math.Pow(float64(plan.config.ObjectiveEffectiveSamples), -0.2))
	zero := normalPDF(plan.distanceAxis, 0, worstBandwidth)
	far := normalPDF(plan.distanceAxis, clusterDistance, worstBandwidth)
	worst := make([]float64, len(zero))
	for index := range worst {
		worst[index] = 0.5 * (zero[index] + far[index])
	}
	scale := jsDistance(worst, plan.randomPDF, plan.distanceAxis)
	return math.Min(math.Max(1-jsDistance(density, worst, plan.distanceAxis)/math.Max(scale, 1e-15), 0), 1)
}

func (plan *Plan) calculateSubjective(edge []float64) float64 {
	// distanceMass normalizes its input, so dividing every edge coordinate by
	// the common degree first only allocated and copied an unnecessary B^2 slice.
	mass := plan.distanceMass(edge)
	effectiveSamples := max(plan.config.Population*plan.config.OutDegree, 1)
	density := mixturePDF(plan.distanceAxis, plan.distances, mass,
		bandwidth(plan.distances, mass, effectiveSamples, plan.config.MinimumBandwidth))
	return math.Min(math.Max(1-jsDistance(density, plan.subjectiveWorst, plan.distanceAxis)/math.Max(plan.subjectiveScale, 1e-15), 0), 1)
}

func (plan *Plan) calculateHomophily(edge []float64) (float64, float64) {
	raw := 0.0
	for index, value := range edge {
		raw += value * plan.concordance[index]
	}
	raw = math.Min(math.Max(raw/float64(plan.config.OutDegree), 0), 1)
	ratio := plan.config.Tolerance / plan.config.DomainWidth
	baseline := 2*ratio - ratio*ratio
	normalized := 0.0
	if baseline >= 1 {
		if raw >= 1 {
			normalized = 1
		}
	} else {
		normalized = math.Max(0, (raw-baseline)/(1-baseline))
	}
	return math.Min(normalized, 1), raw
}
