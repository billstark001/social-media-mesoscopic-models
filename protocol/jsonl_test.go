package protocol

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

func stubExecute(data []byte, _ int, progress ProgressFunc) Response {
	var request struct {
		RequestID string `json:"request_id"`
		Fail      bool   `json:"fail"`
	}
	if err := json.Unmarshal(data, &request); err != nil {
		return Response{Error: err.Error()}
	}
	if request.Fail {
		if progress != nil {
			progress(ProgressEvent{Event: "request_rejected", RequestID: request.RequestID})
		}
		return Response{RequestID: request.RequestID, Error: "rejected"}
	}
	if progress != nil {
		progress(ProgressEvent{Event: "request_completed", RequestID: request.RequestID})
	}
	return Response{RequestID: request.RequestID, Result: map[string]any{"ok": true}}
}

func TestJSONLContinuesAfterInvalidRequest(t *testing.T) {
	input := bytes.NewBufferString("{\"request_id\":\"bad\",\"fail\":true}\n{\"request_id\":\"good\"}\n")
	output := bytes.NewBuffer(nil)
	if err := RunJSONL(input, output, stubExecute); err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(output)
	var first, second Response
	if err := decoder.Decode(&first); err != nil {
		t.Fatal(err)
	}
	if err := decoder.Decode(&second); err != nil {
		t.Fatal(err)
	}
	if first.RequestID != "bad" || first.Error == "" || first.Result != nil {
		t.Fatalf("unexpected invalid response: %+v", first)
	}
	if second.RequestID != "good" || second.Error != "" || second.Result == nil {
		t.Fatalf("unexpected valid response: %+v", second)
	}
}

func TestJSONLProgressCarriesBatchLine(t *testing.T) {
	input := bytes.NewBufferString("{\"request_id\":\"bad\",\"fail\":true}\n{\"request_id\":\"good\"}\n")
	events := make([]ProgressEvent, 0)
	if err := RunJSONLWithProgress(input, bytes.NewBuffer(nil), stubExecute, 1, func(event ProgressEvent) {
		events = append(events, event)
	}); err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 || events[0].BatchIndex != 1 || events[1].BatchIndex != 2 {
		t.Fatalf("missing line-aware events: %+v", events)
	}
}

func TestFloat64ArrayRoundTrip(t *testing.T) {
	values := []float64{0, -1.25, 3.5, 1e100}
	encoded, err := EncodeFloat64(values, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	decoded, shape, err := encoded.DecodeFloat64()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, values) || !reflect.DeepEqual(shape, []int{2, 2}) {
		t.Fatalf("decoded=%v shape=%v", decoded, shape)
	}
	encoded.Data += "not-base64"
	if _, _, err := encoded.DecodeFloat64(); err == nil {
		t.Fatal("corrupt array was accepted")
	}
}

func TestFloat64ArrayCodecIsConcurrentAndBounded(t *testing.T) {
	const workers = 16
	var wait sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wait.Add(1)
		go func(seed int) {
			defer wait.Done()
			values := []float64{float64(seed), -0.5, 1e20}
			encoded, err := EncodeFloat64(values, len(values))
			if err != nil {
				t.Errorf("encode: %v", err)
				return
			}
			decoded, _, err := encoded.DecodeFloat64()
			if err != nil || !reflect.DeepEqual(decoded, values) {
				t.Errorf("round trip: %v %v", decoded, err)
			}
		}(worker)
	}
	wait.Wait()
	oversized := EncodedArray{
		Encoding: Float64Encoding,
		Shape:    fmt.Sprint(maxDecodedArrayBytes/8 + 1),
	}
	if _, _, err := oversized.DecodeFloat64(); err == nil {
		t.Fatal("oversized encoded array was accepted")
	}
}
