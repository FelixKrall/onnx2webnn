# Real-weight validation failures

This document is the current triage record for failures from full-model validation with publisher
weights. Generated-weight failures are intentionally out of scope. The summary and complete case
ledger remain in [Full-model numerical validation status](model-validation-status.md).

## Recorded run

- Date: 2026-09-14
- Tested executable: onnx2webnn `aaa33e2` plus current uncommitted validator fixes
- RustNN: `7f07a5e1`
- Manifest: `tests/models/manifest.json` (52 cases)
- Runtime: CPU ONNX Runtime 1.29.0
- Cache state: complete; no model downloads or skips
- Result: 32 passed and 20 failed
- Warm wall time: 5m 8.8s, one validation worker

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
| V1 | 10 | Cached-interface validation | Cached graph interface differs from the source ONNX interface | Determine whether each descriptor is intentionally pruned or lost during the cache round trip. |
| E1 | 4 | WebNN export | Uint4 constants cannot be represented by the current Safetensors export | RustNN graph serialization/external-weight format. |
| O1 | 2 | Native ORT model load | The source ONNX is rejected before conversion can be compared | Source model/export or ORT compatibility. |

The real sweep has no remaining deterministic-input failures. Only N1 proves that both execution
paths completed and returned different values. V1 and E1 are conversion/cache-path limitations or
bugs. O1 fails in the reference model itself.

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

## V1: cached graph interface mismatch

The five prefill cases now pass valid empty cache tensors to native ORT, then discover a non-empty
source input absent from the cached graph. The five cache-enabled cases retain the previously known
interface differences.

| # | Case | Missing descriptor |
|--:|------|--------------------|
| 7 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=0`) | input `past_key_values_0_decoder_key` |
| 8 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=1`) | input `encoder_hidden_states` |
| 24 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=0`) | input `past_key_values_0_encoder_key` |
| 25 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=1`) | input `encoder_hidden_states` |
| 31 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=0`) | input `past_key_values_0_encoder_key` |
| 32 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=1`) | output `present_0_encoder_key` |
| 35 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=0`) | input `past_key_values_0_encoder_key` |
| 36 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=1`) | input `encoder_hidden_states` |
| 39 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=0`) | input `past_key_values_0_encoder_key` |
| 40 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=1`) | input `encoder_hidden_states` |

Compare the converted `GraphInfo` interface before export with the reloaded interface:

- A descriptor that disappears only after reload identifies a lossy RustNN cache round trip.
- A descriptor already absent before export identifies conversion pruning; validation must then
  distinguish the source ONNX feed interface from the converted branch's dispatch interface.
- For the missing output, determine whether the branch aliases or intentionally omits the
  encoder-cache output.

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
2. Determine whether each V1 descriptor is intentionally pruned or lost during serialization.
3. Add lossless Uint4 cache serialization for E1.
4. Resolve or replace the invalid Chronos reference exports for O1.

## Maintenance rule

This file describes only the latest complete real-weight run. Replace counts, cases, diagnoses,
and exact errors when a new run changes first blockers; do not append obsolete full failure tables.
Record revision-attributed coverage changes in the main status file's history section.
