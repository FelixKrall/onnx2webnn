# Real-weight validation failures

This document is the current triage record for failures from full-model validation with publisher
weights. Generated-weight failures are intentionally out of scope. The summary and complete case
ledger remain in [Full-model numerical validation status](model-validation-status.md).

## Recorded run

- Date: 2026-09-15
- Tested executable: onnx2webnn `ec5ba275` plus current uncommitted validator fixes
- RustNN: `7f07a5e1`
- Manifest: `tests/models/manifest.json` (52 cases)
- Runtime: CPU ONNX Runtime 1.29.0
- Cache state: complete; no model downloads or skips
- Result: 42 passed and 10 failed
- Warm wall time: 5m 9.9s, one validation worker

```bash
ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

The recorded result for each case is its first blocker. A case that fails before comparison may
contain further converter, serialization, reload, execution, or numerical issues that are not yet
observable.

## Failure groups

| Code | Cases | Furthest stage | Classification | Current ownership |
|------|------:|----------------|----------------|-------------------|
| N1 | 4 | Output comparison | Confirmed numerical disagreement | Localize between onnx2webnn lowering, RustNN serialization/reload, and the RustNN ORT backend. |
| E1 | 4 | WebNN export | Uint4 constants cannot be represented by the current Safetensors export | RustNN graph serialization/external-weight format. |
| O1 | 2 | Native ORT model load | The source ONNX is rejected before conversion can be compared | Source model/export or ORT compatibility. |

The real sweep has no deterministic-input or cached-interface failures. N1 proves that both
execution paths completed and returned different values. E1 is a cache export limitation, while O1
fails in the reference model itself.

## N1: numerical disagreement

| # | Case | First mismatch |
|--:|------|----------------|
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | `logits[0]`: ORT `0.6691238284111023`, WebNN `1.2111917734146118`, tolerance `0.00007691238284111024` |
| 33 | `Xenova--detr-resnet-50 :: model_quantized.onnx` | `logits[0]`: ORT `-15.032588958740234`, WebNN `-14.491597175598145`, tolerance `0.0015132588958740236` |
| 34 | `Xenova--donut-base-finetuned-docvqa :: encoder_model_quantized.onnx` | `last_hidden_state[0]`: ORT `-0.06940168142318726`, WebNN `-0.6064815521240234`, tolerance `0.000016940168142318727` |
| 48 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: decoder_model_merged_quantized.onnx` (`past=0`) | `logits[19286]`: ORT `1.0281682014465332`, WebNN `1.0283994674682617`, tolerance `0.00011281682014465332` |

Compare intermediate tensors to locate the first divergent operation. Donut remains the strongest
first target because its mismatch also occurs with generated weights. FastVLM and Qwen became
observable only after zero-element input generation was corrected.

## E1: Uint4 external-weight serialization

RustNN reports at export that a Uint4 constant cannot be stored in Safetensors.

| # | Case |
|--:|------|
| 4 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=64`, `past=0`) |
| 5 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=1`, `past=64`) |
| 18 | `onnx-community--Janus-Pro-1B-ONNX :: language_model_q4.onnx` |
| 28 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: decoder_model_merged_q4.onnx` |

This is four cases but three ONNX files. The fix needs a lossless packed Int4/Uint4 representation
and executable reload; widening values without preserving the descriptor dtype is not a valid
round trip.

## O1: source ONNX rejected by native ORT

Both Chronos files fail during native ORT model loading. A `Gather` indices input is the float
output of `ConstantOfShape`; Gather indices must be integer.

| # | Case |
|--:|------|
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` |

Verify the publisher artifacts and intended ORT/opset version. Until the source model runs in the
reference runtime, these cases cannot provide a numerical oracle and are not evidence for or
against onnx2webnn correctness.

## Repair order

1. Localize N1, beginning with Donut encoder.
2. Add lossless Uint4 cache serialization for E1.
3. Resolve or replace the invalid Chronos reference exports for O1.

## Maintenance rule

This file describes only the latest complete real-weight run. Replace counts, cases, diagnoses,
and exact errors when a new run changes first blockers; do not append obsolete full failure tables.
Record revision-attributed coverage changes in the main status file's history section.
