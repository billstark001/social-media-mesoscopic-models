package protocol

import "fmt"

// NodeEdgeProjection is the common, compressed node/edge part of a coarse state.
// It is not a full stochastic checkpoint: motifs, model parameters and RNG
// position must be supplied separately for exact lifted continuation.
type NodeEdgeProjection struct {
	Rho  EncodedArray `json:"rho"`
	Edge EncodedArray `json:"edge"`
}

func EncodeNodeEdgeProjection(rho, edge []float64, bins int) (NodeEdgeProjection, error) {
	if bins < 1 || len(rho) != bins || len(edge)/bins != bins || len(edge)%bins != 0 {
		return NodeEdgeProjection{}, fmt.Errorf("node/edge projection dimensions")
	}
	p, err := EncodeFloat64(rho, bins)
	if err != nil {
		return NodeEdgeProjection{}, err
	}
	e, err := EncodeFloat64(edge, bins, bins)
	if err != nil {
		return NodeEdgeProjection{}, err
	}
	return NodeEdgeProjection{p, e}, nil
}
func (p NodeEdgeProjection) Decode(bins int) ([]float64, []float64, error) {
	rho, shape, err := p.Rho.DecodeFloat64()
	if err != nil {
		return nil, nil, err
	}
	if len(shape) != 1 || shape[0] != bins {
		return nil, nil, fmt.Errorf("rho projection shape")
	}
	edge, shape, err := p.Edge.DecodeFloat64()
	if err != nil {
		return nil, nil, err
	}
	if len(shape) != 2 || shape[0] != bins || shape[1] != bins {
		return nil, nil, fmt.Errorf("edge projection shape")
	}
	return rho, edge, nil
}
