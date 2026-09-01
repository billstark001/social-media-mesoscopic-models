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
	"sync"
)

const Float64Encoding = "base64+zlib+f64le"
const maxDecodedArrayBytes = 512 << 20

var compressedBufferPool = sync.Pool{New: func() any { return new(bytes.Buffer) }}
var zlibWriterPool = sync.Pool{New: func() any { return zlib.NewWriter(io.Discard) }}
var zlibReaderPool sync.Pool

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
	if size > maxDecodedArrayBytes/8 {
		return EncodedArray{}, fmt.Errorf("encoded array exceeds %d-byte limit", maxDecodedArrayBytes)
	}
	raw := make([]byte, len(values)*8)
	for index, value := range values {
		binary.LittleEndian.PutUint64(raw[index*8:index*8+8], math.Float64bits(value))
	}
	compressed := compressedBufferPool.Get().(*bytes.Buffer)
	compressed.Reset()
	writer := zlibWriterPool.Get().(*zlib.Writer)
	writer.Reset(compressed)
	_, writeErr := writer.Write(raw)
	closeErr := writer.Close()
	if writeErr != nil {
		writer.Reset(io.Discard)
		zlibWriterPool.Put(writer)
		compressedBufferPool.Put(compressed)
		return EncodedArray{}, writeErr
	}
	if closeErr != nil {
		writer.Reset(io.Discard)
		zlibWriterPool.Put(writer)
		compressedBufferPool.Put(compressed)
		return EncodedArray{}, closeErr
	}
	result := EncodedArray{
		Encoding: Float64Encoding,
		Shape:    shapeValue,
		Data:     base64.StdEncoding.EncodeToString(compressed.Bytes()),
	}
	writer.Reset(io.Discard)
	zlibWriterPool.Put(writer)
	compressedBufferPool.Put(compressed)
	return result, nil
}

func (array EncodedArray) DecodeFloat64() ([]float64, []int, error) {
	if array.Encoding != Float64Encoding {
		return nil, nil, fmt.Errorf("unsupported array encoding %q", array.Encoding)
	}
	shape, size, err := parseShape(array.Shape)
	if err != nil {
		return nil, nil, err
	}
	if size > maxDecodedArrayBytes/8 {
		return nil, nil, fmt.Errorf("decoded array exceeds %d-byte limit", maxDecodedArrayBytes)
	}
	compressed, err := base64.StdEncoding.DecodeString(array.Data)
	if err != nil {
		return nil, nil, fmt.Errorf("decode array base64: %w", err)
	}
	var reader io.ReadCloser
	pooled := zlibReaderPool.Get()
	if pooled == nil {
		reader, err = zlib.NewReader(bytes.NewReader(compressed))
	} else {
		reader = pooled.(io.ReadCloser)
		err = pooled.(zlib.Resetter).Reset(bytes.NewReader(compressed), nil)
	}
	if err != nil {
		return nil, nil, fmt.Errorf("open array zlib stream: %w", err)
	}
	raw := make([]byte, size*8)
	_, readErr := io.ReadFull(reader, raw)
	var extra [1]byte
	if readErr == nil {
		_, readErr = reader.Read(extra[:])
		if errors.Is(readErr, io.EOF) {
			readErr = nil
		} else if readErr == nil {
			readErr = errors.New("decoded array contains trailing bytes")
		}
	}
	closeErr := reader.Close()
	zlibReaderPool.Put(reader)
	if readErr != nil {
		return nil, nil, fmt.Errorf("read array zlib stream: %w", readErr)
	}
	if closeErr != nil {
		return nil, nil, fmt.Errorf("close array zlib stream: %w", closeErr)
	}
	values := make([]float64, size)
	for index := range values {
		values[index] = math.Float64frombits(binary.LittleEndian.Uint64(raw[index*8 : index*8+8]))
	}
	return values, shape, nil
}
