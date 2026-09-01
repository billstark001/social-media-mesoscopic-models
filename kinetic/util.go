package kinetic

import (
	"math"
	"strings"
)

func normalize(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func clamp(value, lower, upper float64) float64 {
	return math.Min(math.Max(value, lower), upper)
}

func normalizeMass(values []float64) {
	total := 0.0
	for index, value := range values {
		if value < 0 && value > -1e-12 {
			values[index] = 0
			value = 0
		}
		total += value
	}
	if total <= 0 {
		return
	}
	for index := range values {
		values[index] /= total
	}
}

func normalizeRow(values []float64, fallback []float64) {
	total := 0.0
	for _, value := range values {
		total += value
	}
	if total <= 1e-15 {
		copy(values, fallback)
		total = 0
		for _, value := range values {
			total += value
		}
	}
	if total <= 0 {
		return
	}
	for index := range values {
		values[index] /= total
	}
}
