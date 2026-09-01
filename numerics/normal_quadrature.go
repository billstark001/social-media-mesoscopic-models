package numerics

import (
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/integrate/quad"
)

const (
	UnitVarianceQuantileRule = "unit_variance_quantile"
	GaussHermiteRule         = "gauss_hermite"
)

// NormalQuadrature is a finite probability rule with centered nodes and unit
// weighted variance whenever it contains more than one point. A one-point
// rule is the deterministic mass at zero.
type NormalQuadrature struct {
	Nodes   []float64
	Weights []float64
}

func CheckNormalQuadratureRule(rule string) error {
	switch strings.ToLower(strings.TrimSpace(rule)) {
	case UnitVarianceQuantileRule, GaussHermiteRule:
		return nil
	default:
		return fmt.Errorf("unsupported normal quadrature rule %q", rule)
	}
}

// NewNormalQuadrature constructs a standard-normal quadrature rule. Both
// supported rules are normalized after construction so their finite nodes,
// rather than only their continuum target, have mean zero and variance one.
func NewNormalQuadrature(rule string, count int) (NormalQuadrature, error) {
	if count < 1 {
		return NormalQuadrature{}, fmt.Errorf("normal quadrature count must be positive")
	}
	if err := CheckNormalQuadratureRule(rule); err != nil {
		return NormalQuadrature{}, err
	}
	result := NormalQuadrature{
		Nodes:   make([]float64, count),
		Weights: make([]float64, count),
	}
	if count == 1 {
		result.Weights[0] = 1
		return result, nil
	}
	switch strings.ToLower(strings.TrimSpace(rule)) {
	case UnitVarianceQuantileRule:
		weight := 1 / float64(count)
		for index := range result.Nodes {
			result.Nodes[index] = QuantileNormal(index, count)
			result.Weights[index] = weight
		}
	case GaussHermiteRule:
		quad.Hermite{}.FixedLocations(
			result.Nodes, result.Weights, math.Inf(-1), math.Inf(1),
		)
		for index := range result.Nodes {
			// Gonum integrates exp(-x^2). Transform to the standard-normal
			// density exp(-z^2/2)/sqrt(2*pi).
			result.Nodes[index] *= math.Sqrt2
			result.Weights[index] /= math.SqrtPi
		}
	}
	if err := normalizeNormalQuadrature(&result); err != nil {
		return NormalQuadrature{}, err
	}
	return result, nil
}

func normalizeNormalQuadrature(rule *NormalQuadrature) error {
	total := 0.0
	for index := range rule.Nodes {
		node, weight := rule.Nodes[index], rule.Weights[index]
		if math.IsNaN(node) || math.IsInf(node, 0) || math.IsNaN(weight) ||
			math.IsInf(weight, 0) || weight <= 0 {
			return fmt.Errorf("normal quadrature contains an invalid node or weight")
		}
		total += weight
	}
	if total <= 0 || math.IsInf(total, 0) {
		return fmt.Errorf("normal quadrature has invalid total weight")
	}
	mean := 0.0
	for index := range rule.Nodes {
		rule.Weights[index] /= total
		mean += rule.Weights[index] * rule.Nodes[index]
	}
	variance := 0.0
	for index := range rule.Nodes {
		rule.Nodes[index] -= mean
		variance += rule.Weights[index] * rule.Nodes[index] * rule.Nodes[index]
	}
	if variance <= 0 || math.IsNaN(variance) || math.IsInf(variance, 0) {
		return fmt.Errorf("normal quadrature has invalid variance")
	}
	scale := 1 / math.Sqrt(variance)
	for index := range rule.Nodes {
		rule.Nodes[index] *= scale
	}
	return nil
}
