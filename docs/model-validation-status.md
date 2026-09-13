# Full-model numerical validation status

This document tracks the **latest recorded** full-model numerical-validation sweep and a concise
history of changes that materially altered coverage. Only the latest sweep keeps a complete
per-case ledger; older ledgers and duplicated historical failure tables are intentionally omitted.

The skeleton sweep establishes broad graph-construction coverage. Numerical validation additionally
exports the converted graph, reloads its `.webnn` and Safetensors artifacts, executes both the
original ONNX model and the reloaded graph on CPU ONNX Runtime, and compares matching outputs.

## Latest recorded sweep

- Date: 2026-09-12
- onnx2webnn: `4926c3e` on `fkrall/cache-backed-validation`
- RustNN: `7f07a5e1` on `fkrall/executable-webnn-reload`
- ORT: repository-local Linux x64 1.29.0 build
- Manifest: `tests/models/manifest.json` (52 cases, 45 unique ONNX files)
- Execution: one validation worker; ORT may use multiple CPU threads inside a case

These revisions identify the latest **tested** state. A newer checkout is not considered the current
validation baseline until both sweeps have been rerun and this section is replaced.

```bash
ORT_DYLIB_PATH=../tools/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights generated --jobs 1

ORT_DYLIB_PATH=../tools/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

### Summary

| Weight mode | Pass | Fail | Download skipped | Result |
|-------------|-----:|-----:|-----------------:|--------|
| Generated (`g4`) | 21 | 31 | 0 | Complete (52/52) |
| Real | 28 | 24 | 0 | Complete (52/52) |

Twenty cases pass in both modes. Eight generated cases and two real cases reach comparison but
differ beyond tolerance. Generated and real weights can expose different first blockers; a failure
before comparison is not evidence of a numerical mismatch.

### Current failure families

| Code | Generated | Real | Stage | Current cause / next action |
|------|----------:|-----:|-------|-----------------------------|
| N1 | 8 | 2 | Comparison | Native ORT and reloaded WebNN differ. Localize the first divergent operation, starting with Donut encoder because both modes reproduce it. |
| V1 | 0 | 5 | Cached graph validation | Four cache-branch decoders lose `encoder_hidden_states`; one loses `present_0_encoder_key`. Reconcile optimized ONNX and serialized graph interfaces. |
| E1 | 4 | 4 | Export | Uint4 constants have no current Safetensors representation. |
| G1 | 10 | 0 | Generated preparation | Initializer has no recorded consumers; extend consumer analysis while keeping generation fail-closed. |
| G2 | 1 | 0 | Generated preparation | Large tensor-valued Constant has an ambiguous role. |
| I1 | 3 | 3 | Native ORT input | Deterministic token-type ID 2 exceeds a two-row embedding. |
| I2 | 2 | 7 | Native ORT input | A zero-sized past-key tensor receives one generated element. |
| I3 | 1 | 1 | Native ORT input | Generated `seqlens_k` is invalid for GroupQueryAttention. |
| O1 | 2 | 2 | Native ORT load | Chronos feeds a float ConstantOfShape result to Gather indices. |

Generated totals: 11 generated-model preparation, 4 export, 6 native-ORT input, 2 native-ORT
model-load, 8 comparison failures, and 21 passes. Real totals: 4 export, 5 cached-interface,
11 native-ORT input, 2 native-ORT model-load, 2 comparison failures, and 28 passes.

### Current case ledger

`PASS` means export, reload, both executions, and numerical comparison completed. Any other value
is the current first-blocker code from the table above.

| # | Manifest case | Generated | Real |
|---:|---------------|-----------|------|
| 0 | `briaai--RMBG-1.4 :: model_quantized.onnx` | PASS | PASS |
| 1 | `openai--privacy-filter :: model_quantized.onnx` | N1 | PASS |
| 2 | `nomic-ai--nomic-embed-text-v1.5 :: model_quantized.onnx` | I1 | I1 |
| 3 | `mixedbread-ai--mxbai-embed-large-v1 :: model_quantized.onnx` | I1 | I1 |
| 4 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=64`, `past=0`) | E1 | E1 |
| 5 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=1`, `past=64`) | E1 | E1 |
| 6 | `distil-whisper--distil-large-v2 :: encoder_model_quantized.onnx` | N1 | PASS |
| 7 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 8 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | V1 |
| 9 | `jinaai--jina-reranker-v2-base-multilingual :: model_quantized.onnx` | PASS | PASS |
| 10 | `onnx-community--FastVLM-0.5B-ONNX :: embed_tokens_quantized.onnx` | PASS | PASS |
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | I2 | I2 |
| 12 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=1`, `past=64`) | I3 | I3 |
| 13 | `onnx-community--FastVLM-0.5B-ONNX :: vision_encoder_quantized.onnx` | PASS | PASS |
| 14 | `Marqo--marqo-fashionSigLIP :: text_model_quantized.onnx` | PASS | PASS |
| 15 | `Marqo--marqo-fashionSigLIP :: vision_model_quantized.onnx` | N1 | PASS |
| 16 | `AdamCodd--vit-base-nsfw-detector :: model_quantized.onnx` | PASS | PASS |
| 17 | `Xenova--nllb-200-distilled-600M :: encoder_model_quantized.onnx` | G2 | PASS |
| 18 | `onnx-community--Janus-Pro-1B-ONNX :: language_model_q4.onnx` | E1 | E1 |
| 19 | `onnx-community--Janus-Pro-1B-ONNX :: lm_head.onnx` | PASS | PASS |
| 20 | `onnx-community--Janus-Pro-1B-ONNX :: gen_head.onnx` | PASS | PASS |
| 21 | `onnx-community--Janus-Pro-1B-ONNX :: gen_img_embeds.onnx` | PASS | PASS |
| 22 | `onnx-community--Janus-Pro-1B-ONNX :: image_decode.onnx` | PASS | PASS |
| 23 | `Xenova--musicgen-small :: text_encoder_quantized.onnx` | PASS | PASS |
| 24 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 25 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | V1 |
| 26 | `Mozilla--distilvit :: encoder_model_quantized.onnx` | N1 | PASS |
| 27 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: embed_tokens_fp16.onnx` | PASS | PASS |
| 28 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: decoder_model_merged_q4.onnx` | E1 | E1 |
| 29 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: audio_encoder_quantized.onnx` | N1 | PASS |
| 30 | `Xenova--LaMini-Flan-T5-783M :: encoder_model_quantized.onnx` | PASS | PASS |
| 31 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 32 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | V1 |
| 33 | `Xenova--detr-resnet-50 :: model_quantized.onnx` | PASS | N1 |
| 34 | `Xenova--donut-base-finetuned-docvqa :: encoder_model_quantized.onnx` | N1 | N1 |
| 35 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 36 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | V1 |
| 37 | `onnx-community--dinov3-vits16-pretrain-lvd1689m-ONNX :: model.onnx` | PASS | PASS |
| 38 | `Xenova--distilbart-cnn-6-6 :: encoder_model_quantized.onnx` | PASS | PASS |
| 39 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 40 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | V1 |
| 41 | `prithivMLmods--Common-Voice-Gender-Detection-ONNX :: model_quantized.onnx` | N1 | PASS |
| 42 | `Xenova--bert-base-multilingual-cased :: model_quantized.onnx` | I1 | I1 |
| 43 | `Xenova--distilbert-base-cased-distilled-squad :: model_quantized.onnx` | PASS | PASS |
| 44 | `onnx-community--vitpose-base-simple :: model_quantized.onnx` | PASS | PASS |
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` | O1 | O1 |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` | O1 | O1 |
| 47 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: embed_tokens_quantized.onnx` | PASS | PASS |
| 48 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: decoder_model_merged_quantized.onnx` | I2 | I2 |
| 49 | `onnx-community--timesformer-base-finetuned-k400 :: model_quantized.onnx` | N1 | PASS |
| 50 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model_quantized.onnx` | PASS | PASS |
| 51 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model.onnx` | PASS | PASS |

### Current timing and storage

| Run | Cases | Warm wall time |
|-----|------:|---------------:|
| Generated | 52 | 7m 47.5s |
| Real, cache-complete | 52 | 6m 47.2s |

At the end of the sweep, `.onnx-cache` occupied 76 GB and `.webnn-cache` 75 GB. Neither
completed run needed or skipped a download.

### Current repair order

1. Localize N1, beginning with Donut encoder, then DETR and generated-only mismatches.
2. Reconcile the five V1 cached graph interfaces.
3. Preserve zero-element buffers for I2 and provide semantic values for I1/I3.
4. Define lossless Int4/Uint4 external-weight serialization for E1.
5. Extend generated-weight role/consumer analysis for G1/G2 without weakening fail-closed behavior.
6. Verify or regenerate the two Chronos exports rejected by native ORT.

## Coverage change history

This history records only changes that explain coverage movement or establish that an apparent
movement was not a functional improvement. Detailed obsolete ledgers are available through Git.

| Date / tested revisions | Change | Comparable coverage effect |
|-------------------------|--------|----------------------------|
| 2026-09-12 — onnx2webnn `4926c3e`, RustNN `7f07a5e1` | RustNN `2e22b3db` unified MLGraphBuilder recording and GraphJSON loader inference, replacing the loader-only string-based inference loop. Typed inference added the previously missing/rejected Resample2d, RoundEven/Clamp, Conv2d, LogicalNot, and LogicalAnd reload paths. | On the same 52-case manifest, generated passes rose **7 → 21** and real passes **6 → 28**. Former reload families R1–R5 were eliminated; newly reachable cases exposed N1 and V1 instead. |
| 2026-09-12 — RustNN `7f07a5e1` | Made `[]` unambiguously scalar and required completed GraphInfo descriptors to have known shapes. | Hardened the shared inference path but did not itself add the dispatch that cleared R1–R5; no separate coverage gain is attributed without a bisect. |
| 2026-09-09 — onnx2webnn `e5a5f88c`, RustNN `2783a191` | Rebased onto the newer upstream model manifest and converter work. The manifest changed from 63 to 52 cases: 21 retained, 31 added, 42 removed. | Retained cases preserved their outcomes. The aggregate changed to 7 generated and 6 real passes because the population changed, not because comparable coverage improved. One real Voxtral download was skipped. |
| 2026-09-08/09 — onnx2webnn `f1ca548`, RustNN `b64cf495` | First complete generated/real full-model baseline with cache-backed export, reload, deterministic execution, and comparison. | 6/63 generated and 7/63 real cases passed. Every case that reached comparison matched; failures established the original reload, serialization, generator, and input blocker families. |

### RustNN inference-refactor attribution

RustNN `2e22b3db` removed the separate ten-pass GraphJSON shape-inference implementation. Both
normal graph construction and GraphJSON loading now pass typed operations through
`GraphRecorder::record_operation` and `infer_operation_descriptors`. This directly removed:

- R1: missing Resample2d output-shape inference.
- R2: inconsistent RoundEven spelling and missing Clamp inference.
- R3: missing Conv2d output-shape inference.
- R4: inconsistent LogicalNot spelling.
- R5: inconsistent LogicalAnd spelling.

The full manifest was tested at `7f07a5e1`, not bisected at `2e22b3db`; attribution is based on
the code changes between the recorded revisions. A focused reload benchmark found the changed
graph reconstruction/inference stage about 6% faster on seven cached graphs, while end-to-end
weight-heavy loading was effectively unchanged.

## Maintenance rule

When a new complete sweep is recorded:

1. Replace the latest metadata, summary, failure counts, timing, and the **single** case ledger.
2. Add one concise history row only when a code/revision change explains coverage movement.
3. Do not append another historical case ledger or duplicate obsolete failure-family tables.
4. Distinguish manifest/population changes from improvements on comparable cases.
