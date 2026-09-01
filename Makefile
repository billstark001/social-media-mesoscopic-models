.PHONY: build build-lifted build-kinetic build-all install-python test test-python benchmark fmt clean

GO ?= go
PYTHON ?= python
BIN_DIR ?= bin
GO_TAGS ?=

GO_BUILD_FLAGS = $(if $(strip $(GO_TAGS)),-tags $(GO_TAGS),)

build: build-all

build-lifted:
	mkdir -p $(BIN_DIR)
	$(GO) build $(GO_BUILD_FLAGS) -o $(BIN_DIR)/smp-lifted ./cmd/smp-lifted

build-kinetic:
	mkdir -p $(BIN_DIR)
	$(GO) build $(GO_BUILD_FLAGS) -o $(BIN_DIR)/smp-kinetic ./cmd/smp-kinetic

build-all: build-lifted build-kinetic

install-python:
	$(PYTHON) -m pip install -e .

test:
	$(GO) test $(GO_BUILD_FLAGS) ./...

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

benchmark:
	$(GO) test $(GO_BUILD_FLAGS) ./... -run '^$$' -bench . -benchmem

fmt:
	$(GO) fmt ./...

clean:
	$(RM) $(BIN_DIR)/smp-lifted $(BIN_DIR)/smp-lifted.exe
	$(RM) $(BIN_DIR)/smp-kinetic $(BIN_DIR)/smp-kinetic.exe
