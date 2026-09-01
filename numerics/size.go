package numerics

import (
	"fmt"
	"math"
)

// MaxRequestWorkingSetBytes is the largest estimated working set accepted for
// one request. The solvers use this explicit guard instead of letting an
// overflowing dimension or an accidental multi-gigabyte request reach make.
const MaxRequestWorkingSetBytes int64 = 512 << 20

func CheckedProduct(name string, factors ...int) (int, error) {
	result := 1
	for _, factor := range factors {
		if factor < 0 {
			return 0, fmt.Errorf("%s contains a negative dimension", name)
		}
		if factor != 0 && result > math.MaxInt/factor {
			return 0, fmt.Errorf("%s overflows int", name)
		}
		result *= factor
	}
	return result, nil
}

func CheckedSum(name string, values ...int) (int, error) {
	result := 0
	for _, value := range values {
		if value < 0 || result > math.MaxInt-value {
			return 0, fmt.Errorf("%s overflows int", name)
		}
		result += value
	}
	return result, nil
}

func CheckFloat64Budget(name string, elements int) error {
	if elements < 0 || int64(elements) > MaxRequestWorkingSetBytes/8 {
		return fmt.Errorf("%s requires an estimated %d float64 values; limit is %d bytes",
			name, elements, MaxRequestWorkingSetBytes)
	}
	return nil
}

func CheckFinite(name string, value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return fmt.Errorf("%s must be finite", name)
	}
	return nil
}

func CheckUnit(name string, value float64) error {
	if err := CheckFinite(name, value); err != nil {
		return err
	}
	if value < 0 || value > 1 {
		return fmt.Errorf("%s must be in [0,1], got %g", name, value)
	}
	return nil
}

func CheckNonnegative(name string, value float64) error {
	if err := CheckFinite(name, value); err != nil {
		return err
	}
	if value < 0 {
		return fmt.Errorf("%s must be nonnegative, got %g", name, value)
	}
	return nil
}
