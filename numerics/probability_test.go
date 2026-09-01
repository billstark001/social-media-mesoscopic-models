package numerics

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestBinomialPMF(t *testing.T) {
	pmf := BinomialPMF(15, 0.37)
	if math.Abs(Sum(pmf)-1) > 1e-14 {
		t.Fatalf("pmf sum=%g", Sum(pmf))
	}
	mean := 0.0
	for k, probability := range pmf {
		mean += float64(k) * probability
	}
	if math.Abs(mean-15*0.37) > 1e-12 {
		t.Fatalf("mean=%g", mean)
	}
}

func TestSampleBinomialMean(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	total := 0
	const samples = 20000
	for range samples {
		total += SampleBinomial(100, 0.31, rng)
	}
	mean := float64(total) / samples
	if math.Abs(mean-31) > 0.12 {
		t.Fatalf("sample mean=%g", mean)
	}
}

func TestLargeBinomialAvoidsTailUnderflow(t *testing.T) {
	const trials = 2000
	pmf := BinomialPMF(trials, 0.5)
	mean, variance := 0.0, 0.0
	for k, probability := range pmf {
		mean += float64(k) * probability
	}
	for k, probability := range pmf {
		difference := float64(k) - mean
		variance += difference * difference * probability
	}
	if math.Abs(Sum(pmf)-1) > 1e-13 || math.Abs(mean-1000) > 1e-10 || math.Abs(variance-500) > 1e-8 {
		t.Fatalf("sum=%g mean=%g variance=%g", Sum(pmf), mean, variance)
	}
	rng := rand.New(rand.NewPCG(3, 4))
	for range 20 {
		sample := SampleBinomial(trials, 0.5, rng)
		if sample < 800 || sample > 1200 {
			t.Fatalf("implausible large-binomial sample %d", sample)
		}
	}
}

func TestLargePoissonAvoidsZeroAnchorUnderflow(t *testing.T) {
	pmf := PoissonPMF(800, 1000)
	mean := 0.0
	for score, probability := range pmf {
		mean += float64(score) * probability
	}
	if math.Abs(Sum(pmf)-1) > 1e-13 {
		t.Fatalf("pmf sum=%g", Sum(pmf))
	}
	if math.Abs(mean-800) > 1e-9 {
		t.Fatalf("large Poisson mean=%g", mean)
	}
	if pmf[800] <= 0.01 || pmf[len(pmf)-1] >= 1e-8 {
		t.Fatalf("large Poisson collapsed: mode=%g capped-tail=%g", pmf[800], pmf[len(pmf)-1])
	}
}

func TestNormalizeInPlaceRejectsNonFiniteMass(t *testing.T) {
	values := []float64{1, math.Inf(1), math.NaN()}
	NormalizeInPlace(values, []float64{0, 1, 0})
	if !FiniteSlice(values) || math.Abs(Sum(values)-1) > 1e-14 || values[0] != 1 {
		t.Fatalf("unexpected normalized values %v", values)
	}
}
