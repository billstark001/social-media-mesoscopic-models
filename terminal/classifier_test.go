package terminal

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestClassifierFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/classifier_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name       string      `json:"name"`
		Positions  []float64   `json:"positions"`
		Masses     []float64   `json:"masses"`
		Options    Options     `json:"options"`
		Status     string      `json:"status"`
		Category   string      `json:"category"`
		KAll       int         `json:"k_all"`
		KMajor     int         `json:"k_major"`
		Components []Component `json:"components"`
		Margins    Margins     `json:"margins"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, item := range cases {
		t.Run(item.Name, func(t *testing.T) {
			result, err := Classify(item.Positions, item.Masses, item.Options)
			if err != nil {
				t.Fatal(err)
			}
			if result.Status != item.Status || result.Category != item.Category ||
				result.KAll != item.KAll || result.KMajor != item.KMajor {
				t.Fatalf("got %+v, expected %+v", result, item)
			}
			if len(result.Components) != len(item.Components) {
				t.Fatalf("component count got %+v, expected %+v", result.Components, item.Components)
			}
			for index := range result.Components {
				got, want := result.Components[index], item.Components[index]
				if math.Abs(got.Minimum-want.Minimum) > 1e-12 ||
					math.Abs(got.Maximum-want.Maximum) > 1e-12 ||
					math.Abs(got.Mass-want.Mass) > 1e-12 || got.Major != want.Major {
					t.Fatalf("component %d got %+v, expected %+v", index, got, want)
				}
			}
			if math.Abs(result.Margins.GapToEpsilon-item.Margins.GapToEpsilon) > 1e-12 ||
				math.Abs(result.Margins.DiameterToEpsilon-item.Margins.DiameterToEpsilon) > 1e-12 ||
				math.Abs(result.Margins.MassToMajorCutoff-item.Margins.MassToMajorCutoff) > 1e-12 {
				t.Fatalf("margins got %+v, expected %+v", result.Margins, item.Margins)
			}
		})
	}
}
