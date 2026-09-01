package protocol

import (
	"bytes"
	"compress/zlib"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

const Float64Encoding = "base64+zlib+f64le"

// EncodedArray keeps numerical payloads out of JSON arrays. Shape uses an
// "x"-separated scalar string so array metadata has one representation too.
type EncodedArray struct {
	Encoding string `json:"encoding"`
	Shape    string `json:"shape"`
	Data     string `json:"data"`
}

func shapeString(shape []int) (string, int, error) {
	if len(shape) == 0 {
		return "", 0, errors.New("array shape must have at least one dimension")
	}
	parts := make([]string, len(shape))
	size := 1
	for index, value := range shape {
		if value < 0 {
			return "", 0, errors.New("array dimensions must be nonnegative")
		}
		if value != 0 && size > int(^uint(0)>>1)/value {
			return "", 0, errors.New("array shape overflows int")
		}
		size *= value
		parts[index] = strconv.Itoa(value)
	}
	return strings.Join(parts, "x"), size, nil
}

func parseShape(value string) ([]int, int, error) {
	if value == "" {
		return nil, 0, errors.New("array shape must not be empty")
	}
	parts := strings.Split(value, "x")
	shape := make([]int, len(parts))
	size := 1
	for index, part := range parts {
		dimension, err := strconv.Atoi(part)
		if err != nil || dimension < 0 {
			return nil, 0, fmt.Errorf("invalid array shape %q", value)
		}
		if dimension != 0 && size > int(^uint(0)>>1)/dimension {
			return nil, 0, errors.New("array shape overflows int")
		}
		shape[index] = dimension
		size *= dimension
	}
	return shape, size, nil
}

func EncodeFloat64(values []float64, shape ...int) (EncodedArray, error) {
	shapeValue, size, err := shapeString(shape)
	if err != nil {
		return EncodedArray{}, err
	}
	if size != len(values) {
		return EncodedArray{}, fmt.Errorf("shape contains %d values, received %d", size, len(values))
	}
	var compressed bytes.Buffer
	writer := zlib.NewWriter(&compressed)
	var bits [8]byte
	for _, value := range values {
		binary.LittleEndian.PutUint64(bits[:], math.Float64bits(value))
		if _, err := writer.Write(bits[:]); err != nil {
			return EncodedArray{}, err
		}
	}
	if err := writer.Close(); err != nil {
		return EncodedArray{}, err
	}
	return EncodedArray{
		Encoding: Float64Encoding,
		Shape:    shapeValue,
		Data:     base64.StdEncoding.EncodeToString(compressed.Bytes()),
	}, nil
}

func (array EncodedArray) DecodeFloat64() ([]float64, []int, error) {
	if array.Encoding != Float64Encoding {
		return nil, nil, fmt.Errorf("unsupported array encoding %q", array.Encoding)
	}
	shape, size, err := parseShape(array.Shape)
	if err != nil {
		return nil, nil, err
	}
	compressed, err := base64.StdEncoding.DecodeString(array.Data)
	if err != nil {
		return nil, nil, fmt.Errorf("decode array base64: %w", err)
	}
	reader, err := zlib.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return nil, nil, fmt.Errorf("open array zlib stream: %w", err)
	}
	raw, readErr := io.ReadAll(io.LimitReader(reader, int64(size)*8+1))
	closeErr := reader.Close()
	if readErr != nil {
		return nil, nil, fmt.Errorf("read array zlib stream: %w", readErr)
	}
	if closeErr != nil {
		return nil, nil, fmt.Errorf("close array zlib stream: %w", closeErr)
	}
	if len(raw) != size*8 {
		return nil, nil, fmt.Errorf("decoded array has %d bytes, expected %d", len(raw), size*8)
	}
	values := make([]float64, size)
	for index := range values {
		values[index] = math.Float64frombits(binary.LittleEndian.Uint64(raw[index*8 : index*8+8]))
	}
	return values, shape, nil
}
