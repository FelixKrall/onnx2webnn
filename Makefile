.PHONY: build test fmt check clean help

build:
	cargo build

test:
	cargo test

fmt:
	cargo fmt

check:
	cargo check

clean:
	cargo clean

help:
	@echo "onnx2webnn - ONNX to WebNN converter"
	@echo "  make build  - build binary"
	@echo "  make test   - run tests"
	@echo "  make fmt    - format Rust code"
	@echo "  make check  - check compilation"
	@echo "  make clean  - remove build artifacts"
	@echo ""
	@echo "Compilation features (cargo build --features NAME):"
	@echo "  default         - enables env_logging"
	@echo "  env_logging     - logging via RUST_LOG (enabled by default)"
	@echo "  dynamic-inputs  - build graphs with dynamic dimensions"
	@echo "  coreml          - Apple CoreML backend (macOS/iOS)"
	@echo "  trtx            - TensorRT-RTX backend (compatible NVIDIA GPU required)"
	@echo "  ONNX backend support is always compiled in; execution requires ONNX Runtime."
	@echo ""
	@echo "Build examples:"
	@echo "  cargo build --release --features dynamic-inputs"
	@echo "  cargo build --release --features dynamic-inputs,trtx"
