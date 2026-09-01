package numerics

import (
	"math"
	"testing"
)

func TestCheckedDimensionsRejectOverflowAndBudget(t *testing.T) {
	if _, err := CheckedProduct("test", math.MaxInt, 2); err == nil {
		t.Fatal("overflowing product was accepted")
	}
	if _, err := CheckedSum("test", math.MaxInt, 1); err == nil {
		t.Fatal("overflowing sum was accepted")
	}
	if err := CheckFloat64Budget("test", int(MaxRequestWorkingSetBytes/8)+1); err == nil {
		t.Fatal("oversized working set was accepted")
	}
}
