package numerics

import (
	"math"
	"testing"
)

func TestNormalQuadratureMoments(t *testing.T) {
	for _, rule := range []string{UnitVarianceQuantileRule, GaussHermiteRule} {
		for _, count := range []int{2, 3, 5, 7, 9, 15} {
			quadrature, err := NewNormalQuadrature(rule, count)
			if err != nil {
				t.Fatalf("%s/%d: %v", rule, count, err)
			}
			total, mean, second := 0.0, 0.0, 0.0
			for index, node := range quadrature.Nodes {
				weight := quadrature.Weights[index]
				total += weight
				mean += weight * node
				second += weight * node * node
			}
			if math.Abs(total-1) > 2e-15 || math.Abs(mean) > 2e-15 ||
				math.Abs(second-1) > 2e-15 {
				t.Fatalf("%s/%d: mass=%g mean=%g variance=%g", rule, count, total, mean, second)
			}
		}
	}
}

func TestGaussHermiteFourthMoment(t *testing.T) {
	quadrature, err := NewNormalQuadrature(GaussHermiteRule, 5)
	if err != nil {
		t.Fatal(err)
	}
	fourth := 0.0
	for index, node := range quadrature.Nodes {
		fourth += quadrature.Weights[index] * math.Pow(node, 4)
	}
	if math.Abs(fourth-3) > 2e-14 {
		t.Fatalf("fourth moment=%g", fourth)
	}
}

func TestOnePointNormalQuadratureIsDeterministic(t *testing.T) {
	for _, rule := range []string{UnitVarianceQuantileRule, GaussHermiteRule} {
		quadrature, err := NewNormalQuadrature(rule, 1)
		if err != nil {
			t.Fatal(err)
		}
		if len(quadrature.Nodes) != 1 || quadrature.Nodes[0] != 0 ||
			quadrature.Weights[0] != 1 {
			t.Fatalf("%s: %+v", rule, quadrature)
		}
	}
}

func TestNormalQuadratureRejectsInvalidConfiguration(t *testing.T) {
	if _, err := NewNormalQuadrature("legacy", 7); err == nil {
		t.Fatal("unsupported rule was accepted")
	}
	if _, err := NewNormalQuadrature(UnitVarianceQuantileRule, 0); err == nil {
		t.Fatal("zero points were accepted")
	}
}
