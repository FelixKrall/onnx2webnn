# Full-model numerical validation status

This document tracks full-model numerical validation independently from the skeleton sweep. The skeleton manifest contains 63 cases that construct an ORT-backed graph successfully; this validation additionally exports the converted graph, reloads its `.webnn` and Safetensors artifacts, executes both the original ONNX model and the reloaded graph on CPU ORT, and compares their outputs.

## Baseline

- Generated sweep: 2026-09-08
- Real sweep: 2026-09-09
- onnx2webnn: `f1ca548` on `fkrall/cache-backed-validation`
- rustnn: `b64cf495` on `fkrall/executable-webnn-reload`
- ORT: repository-local Linux x64 1.29.0 build
- Rust/Cargo: 1.97.0
- Manifest: `tests/models/manifest.json` (63 cases, including repeated files with distinct dimensions or pinned inputs)
- Execution: one worker; heavy cases remained sequential

Both weight modes were tested with the same command, changing only `--weights`:

```sh
ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights generated --jobs 1
```

Repeat the command with `--weights real` for the real-weight sweep.

The real sweep started with 250.9 GB available and completed with 196.2 GB available; the safety stop was not needed. Generated ONNX artifacts use 29.9 GB and generated WebNN artifacts use 21.6 GB. Real ONNX artifacts use 31.6 GB and real WebNN artifacts use 23.1 GB. The real run therefore added 54.7 GB, about 3.2 GB more than the generated run because it materialized models and exports that generated-fixture checks had rejected.

## Summary

| Weight mode | Pass | Fail | Not run | Result |
| --- | ---: | ---: | ---: | --- |
| Generated (`g4`) | 6 | 57 | 0 | Complete |
| Real | 7 | 56 | 0 | Complete |

No case reached output comparison and then mismatched in either mode. The six generated cases that reached comparison passed, as did all seven real cases. Six cases passed both modes. DistilVIT decoder (#32) additionally passed with real weights after its generated fixture was rejected before conversion.

The passes in both modes are:

- #12 Distil-Whisper `decoder_with_past_model.onnx`
- #21 FashionSigLIP `text_model.onnx`
- #36 FastVLM `embed_tokens_quantized.onnx`
- #44 Janus `embed_tokens_q4.onnx`
- #60 Qwen2.5-VL `embed_tokens_q4.onnx`
- #62 Tiny RoFormer `model.onnx`

The additional real-only pass is:

- #32 DistilVIT `decoder_model.onnx`

These are marked explicitly in the case ledger below. Tiny RoFormer remains the existing `smoke` manifest case; this report does not change manifest tiers.

## Differences between generated and real weights

Fifty-five cases retained the same pass or first-blocker family. Eight changed outcome or stage:

| # | Generated | Real | Analysis |
| ---: | --- | --- | --- |
| 13 | G1 | I2 | Real weights bypassed ambiguous-initializer generation, then exposed the validator's incorrect one-element allocation for a zero-element past-key tensor. |
| 14 | G1 | R2 | Real weights bypassed generation and reached the known `roundEven`/`clamp` reload gap. |
| 29 | G2 | R4 | Real weights bypassed a large tensor-valued Constant restriction and reached the `logicalNot` reload-name mismatch. |
| 30 | G2 | R2 | Real weights bypassed a large tensor-valued Constant restriction and reached the known quantization reload gap. |
| 32 | G2 | PASS | The real model completed export, reload, both executions, and comparison. This is a generated-fixture limitation, not a model/converter failure. |
| 40 | G1 | R3 | Real weights bypassed ambiguous-initializer generation and reached the `conv2d` reload gap. |
| 51 | G1 | E1 | Real weights bypassed ambiguous-initializer generation and reached Uint4 Safetensors export rejection. |
| 52 | G1 | E1 | Same as #51 for the alternate pinned cache branch. |

There was no evidence that real tensor values changed numerical behavior: every case that completed comparison passed. The observed differences are reachability differences—real weights bypass generator safeguards and reveal later blockers.

Operational note: FastVLM vision encoder exhausted five non-resuming HTTP attempts during the main sweep. An immediate single-case retry downloaded successfully and reproduced R3. This did not change the functional result, but resumable downloads would avoid repeating large transfers after transient timeouts.

## Failure families

Each code identifies the first blocker observed. Large graphs often contain later operations that could reveal additional issues after the first blocker is fixed.

| Code | Generated | Real | Stage | Likely owner | Cause and next action |
| --- | ---: | ---: | --- | --- | --- |
| R1 | 1 | 1 | Reload | RustNN | `resample2d` is absent from `webnn_json::infer_output_shapes`, although RustNN already has `infer_resample2d_shape`. Add the reload inference case and test its serialized options. |
| R2 | 10 | 12 | Reload | RustNN | The first unresolved value is a `roundEven` output in generated quantization code. `op_type()` is lowercased to `roundeven`, but the inference match uses `roundEven`; `clamp`, immediately downstream, is also missing from the shape-preserving unary set. Fix both before reassessing the remaining chain. |
| R3 | 9 | 10 | Reload | RustNN | The first unresolved value is a `conv2d` output. The reload pass has pool inference but no `conv2d` branch, despite the normal builder already using `infer_conv2d_shape`. Add equivalent inference using input/filter shapes and serialized options. Real weights additionally move #40 past its generated-fixture blocker into R3. |
| R4 | 1 | 2 | Reload | RustNN | `logicalNot` becomes `logicalnot` after lowercasing, while the reload matcher accepts only `logical_not`. Normalize operation names consistently. |
| R5 | 3 | 3 | Reload | RustNN | `logicalAnd` becomes `logicaland`, while the binary matcher accepts only `logical_and`. Normalize logical operation names consistently. |
| E1 | 16 | 18 | Conversion/export | RustNN serialization | `webnn_save` explicitly maps Int4 and Uint4 to no Safetensors dtype and rejects the constant. Define a lossless packed representation plus metadata, or use a sidecar format that represents WebNN 4-bit operands directly. |
| G1 | 5 | 0 | Model preparation | Generated fixture | The fail-closed generator finds an initializer with no recorded consumers. Determine whether it belongs to a pinned/unused branch and can be pruned, or extend consumer analysis across the relevant graph/subgraph boundary. Do not randomize it blindly. |
| G2 | 3 | 0 | Model preparation | Generated fixture | A large tensor-valued `Constant` attribute is deliberately not generated because its role is ambiguous. Add role-aware preservation/generation for tensor attributes or retain its exact bytes. |
| I1 | 6 | 6 | Native ORT input | Validator/manifest | Deterministic integer input generation emits `2` for an input used by a token-type embedding with size 2, so native ORT rejects the Gather index before conversion can be judged. Pin the semantic input (normally token type IDs) to a valid value or derive bounds from its Gather table. |
| I2 | 0 | 1 | Native ORT input | Validator | `deterministic_inputs` forces the element count to at least one with `.max(1)`, even when a fixed dimension is zero. Preserve a zero-element buffer for zero-sized tensors. |
| M1 | 3 | 3 | Input materialization | Manifest | The three RMBG cases still have an unresolved `batch_size` input dimension. Add the fixed dimension override required by numerical validation. |

Generated totals: 24 reload, 16 export, 8 generated-model preparation, 6 native-ORT input, and 3 missing-dimension failures. Real totals: 28 reload, 18 export, 7 native-ORT input, and 3 missing-dimension failures.

## Case ledger

`PASS` means the complete numerical round trip passed in that mode. `FAIL` is followed by the first-blocker code above.

| # | Manifest case | Generated | Real |
| ---: | --- | --- | --- |
| 0 | `nomic-ai--nomic-embed-text-v1.5 :: model.onnx` | FAIL (I1) | FAIL (I1) |
| 1 | `nomic-ai--nomic-embed-text-v1.5 :: model_fp16.onnx` | FAIL (I1) | FAIL (I1) |
| 2 | `nomic-ai--nomic-embed-text-v1.5 :: model_quantized.onnx` | FAIL (I1) | FAIL (I1) |
| 3 | `nomic-ai--nomic-embed-text-v1.5 :: model_bnb4.onnx` | FAIL (I1) | FAIL (I1) |
| 4 | `briaai--RMBG-1.4 :: model.onnx` | FAIL (M1) | FAIL (M1) |
| 5 | `briaai--RMBG-1.4 :: model_fp16.onnx` | FAIL (M1) | FAIL (M1) |
| 6 | `briaai--RMBG-1.4 :: model_quantized.onnx` | FAIL (M1) | FAIL (M1) |
| 7 | `mixedbread-ai--mxbai-embed-large-v1 :: model_quantized.onnx` | FAIL (I1) | FAIL (I1) |
| 8 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=64`, `past=0`) | FAIL (E1) | FAIL (E1) |
| 9 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=1`, `past=64`) | FAIL (E1) | FAIL (E1) |
| 10 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4f16.onnx` | FAIL (E1) | FAIL (E1) |
| 11 | `distil-whisper--distil-large-v2 :: encoder_model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 12 | `distil-whisper--distil-large-v2 :: decoder_with_past_model.onnx` | PASS | PASS |
| 13 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`use_cache_branch=0`) | FAIL (G1) | FAIL (I2) |
| 14 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`use_cache_branch=1`) | FAIL (G1) | FAIL (R2) |
| 15 | `openai--privacy-filter :: model.onnx` | FAIL (R5) | FAIL (R5) |
| 16 | `openai--privacy-filter :: model_fp16.onnx` | FAIL (R5) | FAIL (R5) |
| 17 | `openai--privacy-filter :: model_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 18 | `openai--privacy-filter :: model_q4f16.onnx` | FAIL (E1) | FAIL (E1) |
| 19 | `openai--privacy-filter :: model_quantized.onnx` | FAIL (R5) | FAIL (R5) |
| 20 | `jinaai--jina-reranker-v2-base-multilingual :: model.onnx` | FAIL (R4) | FAIL (R4) |
| 21 | `Marqo--marqo-fashionSigLIP :: text_model.onnx` | PASS | PASS |
| 22 | `Marqo--marqo-fashionSigLIP :: vision_model.onnx` | FAIL (R3) | FAIL (R3) |
| 23 | `AdamCodd--vit-base-nsfw-detector :: model.onnx` | FAIL (R3) | FAIL (R3) |
| 24 | `Xenova--detr-resnet-50 :: model.onnx` | FAIL (R3) | FAIL (R3) |
| 25 | `Xenova--detr-resnet-50 :: model_fp16.onnx` | FAIL (R3) | FAIL (R3) |
| 26 | `Xenova--LaMini-Flan-T5-783M :: encoder_model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 27 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 28 | `Xenova--distilbart-cnn-6-6 :: encoder_model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 29 | `Xenova--nllb-200-distilled-600M :: encoder_model_quantized.onnx` | FAIL (G2) | FAIL (R4) |
| 30 | `Xenova--nllb-200-distilled-600M :: decoder_model_quantized.onnx` | FAIL (G2) | FAIL (R2) |
| 31 | `Mozilla--distilvit :: encoder_model.onnx` | FAIL (R3) | FAIL (R3) |
| 32 | `Mozilla--distilvit :: decoder_model.onnx` | FAIL (G2) | PASS |
| 33 | `prithivMLmods--Common-Voice-Gender-Detection-ONNX :: model.onnx` | FAIL (R3) | FAIL (R3) |
| 34 | `Xenova--donut-base-finetuned-docvqa :: encoder_model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 35 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 36 | `onnx-community--FastVLM-0.5B-ONNX :: embed_tokens_quantized.onnx` | PASS | PASS |
| 37 | `onnx-community--FastVLM-0.5B-ONNX :: vision_encoder_q4f16.onnx` | FAIL (R3) | FAIL (R3) |
| 38 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_q4f16.onnx` (`sequence=64`, `past=0`) | FAIL (E1) | FAIL (E1) |
| 39 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_q4f16.onnx` (`sequence=1`, `past=64`) | FAIL (E1) | FAIL (E1) |
| 40 | `onnx-community--pyannote-segmentation-3.0 :: model.onnx` | FAIL (G1) | FAIL (R3) |
| 41 | `onnx-community--dinov3-vits16-pretrain-lvd1689m-ONNX :: model.onnx` | FAIL (R3) | FAIL (R3) |
| 42 | `onnx-community--dinov3-vits16-pretrain-lvd1689m-ONNX :: model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 43 | `onnx-community--dinov3-vits16-pretrain-lvd1689m-ONNX :: model_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 44 | `onnx-community--Janus-Pro-1B-ONNX :: embed_tokens_q4.onnx` | PASS | PASS |
| 45 | `onnx-community--Janus-Pro-1B-ONNX :: gen_head_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 46 | `onnx-community--Janus-Pro-1B-ONNX :: gen_img_embeds_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 47 | `onnx-community--Janus-Pro-1B-ONNX :: image_decode_q4.onnx` | FAIL (R3) | FAIL (R3) |
| 48 | `onnx-community--Janus-Pro-1B-ONNX :: lm_head_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 49 | `onnx-community--Janus-Pro-1B-ONNX :: language_model_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 50 | `Xenova--musicgen-small :: text_encoder_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 51 | `Xenova--musicgen-small :: decoder_model_merged_q4.onnx` (`use_cache_branch=0`) | FAIL (G1) | FAIL (E1) |
| 52 | `Xenova--musicgen-small :: decoder_model_merged_q4.onnx` (`use_cache_branch=1`) | FAIL (G1) | FAIL (E1) |
| 53 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: audio_encoder_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 54 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: embed_tokens_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 55 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: decoder_model_merged_q4.onnx` | FAIL (E1) | FAIL (E1) |
| 56 | `Xenova--bert-base-multilingual-cased :: model_quantized.onnx` | FAIL (I1) | FAIL (I1) |
| 57 | `Xenova--distilbert-base-cased-distilled-squad :: model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 58 | `onnx-community--vitpose-base-simple :: model_quantized.onnx` | FAIL (R2) | FAIL (R2) |
| 59 | `onnx-community--timesformer-base-finetuned-k400 :: model_quantized.onnx` | FAIL (R1) | FAIL (R1) |
| 60 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: embed_tokens_q4.onnx` | PASS | PASS |
| 61 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: decoder_model_merged_q4f16.onnx` | FAIL (E1) | FAIL (E1) |
| 62 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model.onnx` | PASS | PASS |

## Suggested repair order

1. Fix M1 and I1 with manifest/input semantics; these are small validator setup changes and may unlock nine cases without converter work.
2. Fix RustNN reload inference in focused tests: operation-name normalization and `clamp`, then `conv2d`, then `resample2d`. Rerun the affected subsets after each fix because the reported operand is only the first blocker.
3. Design an explicit Int4/Uint4 external-weight encoding for `.webnn` caches before changing E1. Treating packed 4-bit data as an ordinary 8-bit tensor without metadata would be lossy or ambiguous.
4. Extend generated-weight graph analysis for G1/G2 while preserving its fail-closed rule; #32 is the proof that G2 can be a fixture-only false negative.
5. Add resumable downloads to make large real-weight retries cheaper. Rerun affected subsets in both modes after each functional fix and update this ledger.
