> **Last edited:** `2026-11-04T18:17:07Z`<br>
> **Checkout:** `feat/consolidate-cli` at `098e8aa`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

# onnx2webnn

onnx2webnn lowers ONNX opsets **1–26** through [RustNN](https://github.com/rustnn/rustnn)'s WebNN builder.
`convert` exports reloadable `.webnn`/Safetensors artifacts. Add `--backend onnx` to build with
CPU ONNX Runtime, or select another available RustNN backend. Add `--validate` to reload the
artifacts and numerically compare execution against native CPU ORT.
`validate-cached` validates existing graphs without conversion; `validate-manifest` converts
and validates models selected from a manifest. See [operator coverage](docs/operator-conversion-status.md)
and [model results](docs/model-validation-status-triage.md).

## Build

### Prerequisites

- A stable Rust toolchain.
- `protoc` (Debian/Ubuntu: `protobuf-compiler`; macOS: Homebrew `protobuf`).
- Cargo downloads the tested RustNN revision recorded in `Cargo.lock`; `Cargo.toml` tracks `main`.
- ONNX Runtime 1.29+ for ORT graph builds and validation. Export-only conversion does not load it.
  An optional sibling RustNN checkout can download the runtime:

```powershell
make -C ../rustnn onnxruntime-download
```

Set `ORT_DYLIB_PATH` to the downloaded shared library before ORT use; see
[CI setup](.github/workflows/ci.yml) for platform paths.

```powershell
cargo build --locked
# or
cargo build --release --locked
```

The `Makefile` also provides `build`, `test`, `fmt`, and `check` targets.

Run `cargo update -p rustnn` to advance RustNN, test compatibility, and commit the updated
`Cargo.lock`. The weekly
[compatibility workflow](.github/workflows/rustnn-compatibility.yml) also checks compilation
against RustNN `main` on Linux and macOS; it is diagnostic and can be run manually.
For local RustNN development, override the Git dependency with
`cargo --config 'patch."https://github.com/rustnn/rustnn".rustnn.path="../rustnn"' check`.

## Convert

```powershell
cargo run -- convert --model model.onnx --optimize --override-dim batch_size=1
```

Dynamic ONNX inputs (unresolved symbolic dims kept as WebNN dynamic metadata):

```powershell
cargo run -- convert --model model.onnx `
  --experimental-dynamic-inputs `
  --override-dim batch_size=1 `
  --override-dim sequence_length=1
```

Without CLI overrides, `model.dims.json` can supply dimension bindings. WebNN exports default
to the OS cache's `onnx2webnn/webnn` directory; source ONNX copies go to `onnx2webnn/onnx`.
Set `O2W_CACHE_DIR`, `O2W_ONNX_CACHE`, or `O2W_WEBNN_CACHE` to override locations.

WebNN lacks fused `MatMulNBits`; conversion dequantizes packed q4/q8 weights before floating-point
matmul. ORT `accuracy_level=4` can instead use Int8 activations, so numerical parity is not assured.

Merged decoders (optimum's `decoder_model_merged*.onnx`) branch at runtime on `use_cache_branch`.
WebNN has no runtime `If`, so pin the input and convert each branch separately:

```powershell
cargo run -- convert --model decoder_model_merged.onnx --optimize `
  --pin-input use_cache_branch=false `
  --override-dim batch_size=1 --override-dim decoder_sequence_length=4 `
  --override-dim past_decoder_sequence_length=0 ...
```

Use distinct `--output` locations for pinned branches or other specializations of the same
source model: the default WebNN cache is keyed by source path and overwritten on conversion.
Each exported branch has its own `.webnn`/Safetensors pair. Shared weights are not deduplicated;
loading both graphs may duplicate RAM/VRAM use. Proper growing-KV-cache execution still needs
dynamic shapes or further specialization.

| Flag | Purpose |
|------|---------|
| `--model PATH` | ONNX model path (required, repeatable) |
| `-i`, `--input PATH` | Optional complete SafeTensors execution inputs, paired with models in argument order; requires `--validate` on `convert` |
| `-o`, `--output PATH` | Write to a `.webnn` file for one model, or an extensionless directory containing `<model-stem>.webnn`. Multiple models require a directory; colliding outputs are rejected before writes. Omit to use the cache. |
| `--optimize` | Enable constant folding during conversion (off by default) |
| `--override-dim NAME=VALUE` | Single-model only. Bind a symbolic dim (repeatable); unnamed zero dims are addressed as `<input>_dim<axis>` |
| `--pin-input NAME=VALUE` | Single-model only. Freeze a graph input to `true`/`false`/an integer (repeatable) |
| `--override-dims-pininput-file PATH` | JSON array matching each `--model` to its own dimension overrides and input pins; available on `convert` and `validate-cached` |
| `--allow-missing-external-data` | Zero-fill absent external tensors for structural conversion |
| `--experimental-dynamic-inputs` | Preserve unresolved symbolic dims as dynamic metadata during conversion |
| `--debug` | Verbose converter diagnostics (shared across modes); `RUST_LOG` controls environment logging |
| `--backend NAME` | Optional graph-build backend; required for every numerical validation. Names: `onnx`, `coreml`, `trtx`, `litert`, `cann`, `burn-cpu`; the runtime must be compiled in and available |
| `--validate` | On `convert`, export, build, reload, and numerically compare against native CPU ORT |

Shared `--backend` and `--debug` options must follow the mode name, for example
`onnx2webnn convert --backend onnx --model model.onnx`. Options before the mode are rejected.

`convert` accepts multiple `--model` paths. Backend and optimization apply to every model.
With multiple models, `--override-dim` and `--pin-input` are rejected; use
`--override-dims-pininput-file` for model-specific settings. Batch commands continue after
individual model failures, print a summary, and exit
nonzero if any model fails. Conversion always exports; `--backend` additionally builds the graph.

Pair every model with a complete `--input` SafeTensors archive, or omit all input files for
deterministic validation feeds. Input names, types, and shapes must match the feedable ONNX
inputs. Supplied tensor shapes can resolve symbolic dimensions. External ONNX data stays beside
the source model. Validation requires `--backend` before any model processing or output writes.

```bash
cargo run -- convert --model first.onnx --input first.safetensors \
  --model second.onnx --input second.safetensors \
  --output exported --backend onnx --validate
```

### Per-model dimensions and input pins

Pass `--override-dims-pininput-file settings.json` to configure models independently. The file
contains an array with exactly one entry per selected model; entries are matched by model path,
not array order. Relative `model` paths resolve from the JSON file's directory; absolute paths
are also accepted. Duplicate, missing, and unmatched entries are rejected before processing.
The file cannot be combined with `--override-dim` or `--pin-input`, even
for a single model. Omitted `override_dims` and `pin_inputs` fields default to empty maps.
Dimensions must be `u32` integers; pins accept booleans or signed `i64` integers.

```json
[
  {
    "model": "encoder.onnx",
    "override_dims": {"batch_size": 1, "sequence_length": 128}
  },
  {
    "model": "decoder.onnx",
    "override_dims": {"batch_size": 1, "sequence_length": 1},
    "pin_inputs": {"use_cache_branch": false}
  }
]
```

With `settings.json` and both models in the current directory:

```bash
cargo run -- convert --model encoder.onnx --model decoder.onnx \
  --override-dims-pininput-file settings.json --output exported
```

### Validate existing artifacts

`validate-cached` never converts and errors if a graph is missing. Pass one `--graph` per
`--model` to select existing WebNN files; omit all `--graph` options to use the same WebNN cache
lookup as `convert`. The reference ONNX file is read directly from `--model`, so an explicit
model/graph pair needs no ONNX cache copy. Relative and absolute paths are accepted. Reuse the
conversion's dimension overrides and pins. Like `convert --validate`, this command accepts
optional `--input` tensor files, otherwise it uses deterministic feeds.

```bash
cargo run -- validate-cached --model first.onnx --graph exported/first.webnn \
  --input first.safetensors --backend onnx
```

The previous `validate-files` mode is now `convert --validate`; `validate-models` is renamed
`validate-manifest`. Model paths use `--model`; tensor files use singular `--input` instead of
`--inputs`. The `convert --validate-cached` and `--build-and-export` flags are removed; use the
`validate-cached` subcommand and `convert --backend NAME`, respectively.

Operator statistics are code-backed; “fully supported” is a lowering classification, not measured
hardware parity. Performance-limited paths remain functional. Cargo features make CoreML/TensorRT
available, while `--backend` selects one; native comparison remains CPU ORT.
Deterministic fallback feeds are reproducible. Scale-aware quantization feeds use fixed model
scale/zero-point values; ambiguous quantization patterns warn and use the generic deterministic
feed, while malformed quantization parameters still fail validation.

## Model sweep

[`manifest.json`](tests/models/manifest.json) is a generated Transformers.js selection, not a
pinned CI gate. The skeleton sweep fetches graph structure and small constants with HTTP ranges,
zero-fills missing large weights, and builds each graph in ORT or CI-selected CoreML. It checks
construction, not numerical correctness.

### Numerical-validation CI

PR CI numerically validates the curated
[`ci-validation.json`](tests/models/ci-validation.json): Linux/ORT blocks; macOS/CoreML is
experimental; Windows does not run this check. Each entry pins a Hugging Face commit and ONNX
SHA-256, verified on download and reuse. To add one, test both backends and record its file,
revision, digest, dimensions, and pinned inputs in the curated file—not the generated manifest.
Downloads use the standard Hugging Face cache unless `O2W_ONNX_CACHE` or `O2W_CACHE_DIR` overrides
it.

```bash
target/release/onnx2webnn validate-manifest --backend onnx \
  --manifest tests/models/ci-validation.json \
  --selection all --weights real --jobs 1
```

The manual full-manifest real-weight workflow needs a dedicated Linux runner (≥100 GiB disk,
~32 GiB RAM). Model failures are diagnostic; setup failures are fatal. Logs persist for 14 days.

For the skeleton test, `O2W_MODELS=hub|dir=<path>|strip=<path>` selects source models; unset
outside CI skips it. `O2W_MODEL_FETCH_JOBS`, `O2W_MODEL_TEST_JOBS`, and
`O2W_MODEL_TEST_SKIP_HEAVY` control concurrency and large cases.

```powershell
$env:O2W_MODELS = "dir=..\transformers_js_experiments\models"; cargo test --release --test model_skeletons
```

Library API:

```rust
use onnx2webnn::{convert_onnx, ConvertOptions};

let graph = convert_onnx("model.onnx", ConvertOptions::default())?;
```

Operator handlers live in `src/onnx/ops/`; graph conversion is in `src/onnx/convert.rs`.
See [webnn-graph](../webnn-graph) for the `.webnn` format.
