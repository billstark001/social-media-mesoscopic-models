package lifted

import (
	"math/rand/v2"
	"reflect"
	"smp-meso/protocol"
	"testing"
)

func TestFromProjectedUsesSharedNodeEdgeSchema(t *testing.T) {
	r := testRequest()
	for _, layer := range []Layer{LayerNaive, LayerBase} {
		initial, err := InitialState(r, layer, ClosureProfile{}, rand.New(rand.NewPCG(17, 23)))
		if err != nil {
			t.Fatal(err)
		}
		projection, err := protocol.EncodeNodeEdgeProjection(initial.Rho, initial.Edge, r.OpinionBins)
		if err != nil {
			t.Fatal(err)
		}
		projected, err := FromProjected(r, layer, projection, initial.Score, ClosureProfile{})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(projected.Rho, initial.Rho) || !reflect.DeepEqual(projected.Edge, initial.Edge) {
			t.Fatal("projection changed node or edge coordinates")
		}
		if layer == LayerBase && !reflect.DeepEqual(projected.Score, initial.Score) {
			t.Fatal("base projection changed its supplied score")
		}
		bad := projection
		bad.Edge, _ = protocol.EncodeFloat64([]float64{1}, 1, 1)
		if _, err := FromProjected(r, layer, bad, initial.Score, ClosureProfile{}); err == nil {
			t.Fatal("wrong edge shape accepted")
		}
	}
}
