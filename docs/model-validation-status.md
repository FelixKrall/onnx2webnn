# Full-model numerical validation status

This document tracks full-model numerical validation independently from the skeleton sweep. The skeleton manifest contains cases that construct an ORT-backed graph successfully; this validation additionally exports the converted graph, reloads its `.webnn` and Safetensors artifacts, executes both the original ONNX model and the reloaded graph on CPU ORT, and compares their outputs.

## Current post-refactor baseline

- Sweep date: 2026-09-12
- onnx2webnn: `4926c3e` on `fkrall/cache-backed-validation`
- rustnn: `7f07a5e1` on `fkrall/executable-webnn-reload`
- ORT: repository-local Linux x64 1.29.0 build
- Manifest: `tests/models/manifest.json` (52 cases, 45 unique ONNX files)
- Execution: one validation worker; ORT may use multiple CPU threads inside a case

Both modes were run from `onnx2webnn/` with the current release binary:

```sh
ORT_DYLIB_PATH=../tools/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights generated --jobs 1

ORT_DYLIB_PATH=../tools/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

### Current summary

| Weight mode | Pass | Functional fail | Download skipped | Result |
| --- | ---: | ---: | ---: | --- |
| Generated (`g4`) | 21 | 31 | 0 | Complete (52/52) |
| Real | 28 | 24 | 0 | Complete (52/52) |

Twenty cases pass in both modes. All five former RustNN reload families (R1-R5) are cleared: the affected graphs now reload and proceed to execution or comparison. Relative to the 2026-09-09 baseline, generated passes increased from 7 to 21 and real passes from 6 to 28; the previously skipped real Voxtral embedding now passes from the completed cache.

### Resolved RustNN reload families

All five families were directly resolved by RustNN commit `2e22b3db` (`Unify graph recording and WebNN loader inference`). Before that commit, GraphJSON reload used a separate string-based, ten-pass `infer_output_shapes` implementation. It lowercased `Operation::op_type()` and then compared it with inconsistent spellings, and it lacked shape branches for some operations. The commit removed that loader-only inference loop. Both `MLGraphBuilder` and GraphJSON loading now use `GraphRecorder::record_operation`, which calls the typed `infer_operation_descriptors` dispatcher before inserting an operation or its outputs.

| Family | Previous generated / real | Previously affected cases | Removed defect | RustNN fix and current reachability |
| --- | ---: | --- | --- | --- |
| R1 | 1 / 1 | 49, TimeSformer | Reload had no `resample2d` output-shape branch. | Typed `Operation::Resample2d` now calls the builder's `resample2d_shape` path and `infer_resample2d_shape`. The case reaches comparison: generated N1, real PASS. |
| R2 | 16 / 21 | 0, 6, 8, 13-16, 23, 25-26, 29-30, 32-33, 36, 38, 40-41, 43-44, 50 | Reload lowercased `roundEven` to `roundeven` but matched `roundEven`; `clamp` was absent from the shape-preserving unary set. | Typed `Operation::RoundEven` and `Operation::Clamp` both use canonical `same_shape` inference. The former R2 cases now pass, reach comparison, or expose the later V1 cache-interface blocker; generated G1 cases remain blocked before reload. |
| R3 | 2 / 2 | 22, Janus image decoder; 37, DINOv3 | Reload had no `conv2d` output-shape branch. | Typed `Operation::Conv2d` now calls the builder's `conv2d_shape` path and `infer_conv2d_shape` with serialized options. Both cases pass in both modes. |
| R4 | 2 / 3 | 9, Jina reranker; 17 real, NLLB encoder; 34, Donut encoder | Reload lowercased `logicalNot` to `logicalnot` but matched only `logical_not`. | Typed `Operation::LogicalNot` now uses canonical `unary_element_wise_logical_shape` inference. Cases 9 and 17 real pass; case 34 reaches comparison and exposes N1 in both modes. Case 17 generated remains blocked earlier by G2. |
| R5 | 1 / 1 | 1, privacy filter | Reload lowercased `logicalAnd` to `logicaland` but matched only `logical_and`. | Typed `Operation::LogicalAnd` now uses canonical `element_wise_logical_shape` inference. The case reaches comparison: generated N1, real PASS. |

RustNN commit `7f07a5e1` (`Make empty shapes unambiguously scalar`) followed the unification by enforcing that every completed operand descriptor has a known shape and that `[]` means a rank-0 scalar. It strengthens the shared recorder/loader inference path and removes remaining placeholder ambiguity, but it did not add the operator dispatch that directly cleared R1-R5. This attribution is based on the code changes between the two recorded RustNN revisions; the full manifest was run at `7f07a5e1`, not bisected at the intermediate commit.


The deeper reachability exposes numerical differences that were not observable in the previous run. Eight generated cases and two real cases fail comparison. Seven mismatches occur only with generated weights, one only with real weights, and Donut encoder mismatches in both modes. These are validation failures, not tolerance-level passes.

### Result differences by case

| Cases | Generated | Real | Detail |
| --- | --- | --- | --- |
| 1, 6, 15, 26, 29, 41, 49 | Numerical mismatch | PASS | Generated-only semantic disagreement |
| 33 | PASS | Numerical mismatch | DETR real weights |
| 34 | Numerical mismatch | Numerical mismatch | Donut encoder; both modes |
| 7, 24, 31, 35, 39 | Generated preparation | Zero-sized input | Real weights reach the existing empty-buffer bug |
| 8, 25, 36, 40 | Generated preparation | Missing input descriptor | `encoder_hidden_states` absent from cached graph metadata |
| 32 | Generated preparation | Missing output descriptor | `present_0_encoder_key` absent from cached graph metadata |
| 17 | Generated preparation | PASS | Large tensor-valued `Constant` remains generator-only |
| 27 | PASS | PASS | Real Voxtral sidecar is now cache-complete |

All other cases either pass in both modes or retain the same first-blocker family recorded in the case ledger below. The current per-case outcomes that differ from that 2026-09-09 ledger are exactly the rows summarized above plus former R1-R5 entries that now pass.

### Current failure families

| Code | Generated | Real | Stage | Cause and next action |
| --- | ---: | ---: | --- | --- |
| N1 | 8 | 2 | Comparison | Native ORT and reloaded WebNN output differ beyond tolerance. Localize the first divergent operation, beginning with case 34 because both weight modes reproduce it. |
| V1 | 0 | 5 | Cached graph validation | Four cache-branch decoders lose the `encoder_hidden_states` input descriptor and one loses `present_0_encoder_key` output metadata. Reconcile optimized ONNX and serialized graph interfaces. |
| E1 | 4 | 4 | Export | Uint4 constants cannot be represented in the current Safetensors mapping. |
| G1 | 10 | 0 | Generated preparation | An initializer has no recorded consumers; keep generation fail-closed while extending consumer analysis. |
| G2 | 1 | 0 | Generated preparation | A large tensor-valued `Constant` has an ambiguous role. |
| I1 | 3 | 3 | Native ORT input | Deterministic token-type ID `2` exceeds a two-row embedding. |
| I2 | 2 | 7 | Native ORT input | Zero-sized past-key tensors receive one generated element. |
| I3 | 1 | 1 | Native ORT input | Generated `seqlens_k` is invalid for `GroupQueryAttention`. |
| O1 | 2 | 2 | Native ORT load | Both Chronos models feed a float `ConstantOfShape` result to `Gather` indices. |

Generated totals: 11 generated-model preparation, 4 export, 6 native-ORT input, 2 native-ORT model-load, 8 comparison failures, and 21 passes. Real totals: 4 export, 5 cached-graph interface, 11 native-ORT input, 2 native-ORT model-load, 2 comparison failures, and 28 passes.

### Cached timing and storage

| Run | Cases | Wall time |
| --- | ---: | ---: |
| Generated, warm | 52 | 7m 47.5s |
| Real, warm and cache-complete | 52 | 6m 47.2s |

After the sweeps, `.onnx-cache` occupied 76 GB and `.webnn-cache` occupied 75 GB. No download was needed or skipped in either completed result.

### Current suggested repair order

1. Investigate N1 first because the refactors now expose actual numerical disagreement. Start with Donut encoder, then DETR's real-only mismatch and the seven generated-only cases.
2. Reconcile cached WebNN input/output descriptors for the five V1 cache-branch failures.
3. Fix I2 zero-element input generation and add semantic pinned/generated values for I1/I3.
4. Design explicit Int4/Uint4 external-weight encoding for E1.
5. Extend generated-weight graph analysis for G1/G2 while retaining fail-closed behavior.
6. Verify or regenerate the two Chronos exports.

## Previous post-rebase baseline (2026-09-09)

- Sweep date: 2026-09-09
- onnx2webnn: `e5a5f88c` on `fkrall/cache-backed-validation`
- rustnn: `2783a191` on `fkrall/executable-webnn-reload`
- ORT: repository-local Linux x64 1.29.0 build
- Manifest: `tests/models/manifest.json` (52 cases, 45 unique ONNX files)
- Execution: one validation worker; ORT may use multiple CPU threads inside a case

The same `validate-models --selection all --jobs 1` command shown in the historical baseline below was used for both `--weights generated` and `--weights real`.

### Previous summary

| Weight mode | Pass | Functional fail | Download skipped | Result |
| --- | ---: | ---: | ---: | --- |
| Generated (`g4`) | 7 | 45 | 0 | Complete (52/52) |
| Real | 6 | 45 | 1 | Functional results for all cache-complete cases (51/52) |

Every case that reached output comparison passed. There were no numerical mismatches. The generated-only pass is Voxtral `embed_tokens_fp16.onnx`; its real external-data sidecar repeatedly timed out while reading the HTTP response, so the real result is unknown rather than failed.

The six passes in both modes are:

- FastVLM `embed_tokens_quantized.onnx`
- Janus `lm_head.onnx`
- Janus `gen_head.onnx`
- Janus `gen_img_embeds.onnx`
- Qwen2.5-VL `embed_tokens_quantized.onnx`
- Tiny RoFormer `model.onnx`

Voxtral `embed_tokens_fp16.onnx` additionally passes with generated weights.

### Rebase comparison

The preceding manifest had 63 cases. The rebased manifest has 52: 21 exact cases were retained, 31 were added, and 42 were removed. Identity includes the file, sorted dimension overrides, and pinned inputs. All 21 retained cases preserved their generated and real pass/fail status and first-blocker family. Therefore the change from 6/63 generated and 7/63 real passes to 7/52 generated and 6/51 evaluated real cases reflects manifest replacement and the skipped download, not a regression in retained cases.

Generated and real weights expose different first blockers in eleven functional cases: generated-model safeguards reject unused or ambiguous initializers before conversion, while real weights proceed to zero-sized-input or reload blockers. This remains a reachability difference; it did not produce a numerical mismatch.

### Cached timing and storage

| Run | Cases | Wall time | User + system CPU | Max RSS |
| --- | ---: | ---: | ---: | ---: |
| Generated, post-rebase cache fill | 52 | 16m 24.7s | 718.2s | 27.3 GB |
| Generated, warm | 52 | 7m 7.6s | 800.4s | 27.3 GB |
| Real, warm and cache-complete | 51 | 3m 24.4s | 988.3s | 27.3 GB |

The initial real migration required about 1h 32m 53s across a timed-out run and a resume, mostly because large Hugging Face transfers stalled and each retry restarted its `.part` file from byte zero. Chronos and Qwen eventually completed. Voxtral `embed_tokens_fp16.onnx_data` remained incomplete after five 300-second attempts and was excluded from the warm real timing. Stable SSH connectivity does not rule this out: the failures occurred while reading individual HTTP responses from the Hugging Face large-file path. The downloader currently has neither HTTP Range resume nor preservation of partial progress across retries.

At the end of the sweep `.onnx-cache` and `.webnn-cache` each occupied about 75 GB, with about 129 GiB free on the filesystem. Warm runs remain compute-bound because selected cases are still reconverted, overwritten, reloaded, and executed through both CPU ORT paths.

### Previous failure families

Each code is the first observed blocker. A graph may reveal later blockers once it is fixed.

| Code | Generated | Real | Stage | Likely owner | Cause and next action |
| --- | ---: | ---: | --- | --- | --- |
| R1 | 1 | 1 | Reload | RustNN | `resample2d` output shape is not inferred during JSON reload. Add the reload inference case using the existing shape helper. |
| R2 | 16 | 21 | Reload | RustNN | Quantization expansion first leaves `roundEven`/`clamp` results unresolved. Normalize the lowercased operation name and infer these shape-preserving unary operations. |
| R3 | 2 | 2 | Reload | RustNN | `conv2d` output shape is not inferred during JSON reload. Reuse normal builder inference with serialized options. |
| R4 | 2 | 3 | Reload | RustNN | `logicalNot` becomes `logicalnot`, but reload accepts a different spelling. Normalize operation names consistently. |
| R5 | 1 | 1 | Reload | RustNN | `logicalAnd` becomes `logicaland`, but reload accepts a different spelling. Normalize operation names consistently. |
| E1 | 4 | 4 | Export | RustNN serialization | Uint4 constants cannot be represented in the current Safetensors mapping. Define an explicit packed representation and metadata. |
| G1 | 10 | 0 | Generated model preparation | Generated fixture | An initializer has no recorded consumers. Extend graph/subgraph consumer analysis or prove it is prunable; keep generation fail-closed. |
| G2 | 1 | 0 | Generated model preparation | Generated fixture | A large tensor-valued `Constant` has an ambiguous role. Preserve or classify it rather than randomizing blindly. |
| I1 | 3 | 3 | Native ORT input | Validator/manifest | Deterministic integer input `2` exceeds a two-row token-type embedding. Pin semantic IDs or derive valid Gather bounds. |
| I2 | 2 | 7 | Native ORT input | Validator | Zero-sized past-key tensors receive one generated element because element count is forced to at least one. Preserve empty buffers. |
| I3 | 1 | 1 | Native ORT input | Validator/manifest | Arbitrary deterministic sequence metadata is invalid for `GroupQueryAttention` (`seqlens_k`). Generate semantic values or pin them. |
| O1 | 2 | 2 | Native ORT model load | Upstream model/export | Both Chronos ONNX files are rejected by native ORT because a float `ConstantOfShape` result feeds `Gather` indices. Confirm exporter/opset compatibility before judging WebNN conversion. |
| D1 | 0 | 1 | Download | Downloader | Voxtral's external-data response repeatedly stalled. Add resumable Range downloads and retain partial bytes between attempts. |

Generated totals: 22 reload, 4 export, 11 generated-model preparation, 8 native-ORT input/model-load failures, and 7 passes. Real totals: 28 reload, 4 export, 13 native-ORT input/model-load failures, 1 skipped download, and 6 passes.

### Previous case ledger

`PASS` means export, reload, both executions, and comparison completed. Other entries contain the first-blocker code above.

| # | Manifest case | Generated | Real |
| ---: | --- | --- | --- |
| 0 | `briaai--RMBG-1.4 :: model_quantized.onnx` | R2 | R2 |
| 1 | `openai--privacy-filter :: model_quantized.onnx` | R5 | R5 |
| 2 | `nomic-ai--nomic-embed-text-v1.5 :: model_quantized.onnx` | I1 | I1 |
| 3 | `mixedbread-ai--mxbai-embed-large-v1 :: model_quantized.onnx` | I1 | I1 |
| 4 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=64`, `past=0`) | E1 | E1 |
| 5 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=1`, `past=64`) | E1 | E1 |
| 6 | `distil-whisper--distil-large-v2 :: encoder_model_quantized.onnx` | R2 | R2 |
| 7 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 8 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | R2 |
| 9 | `jinaai--jina-reranker-v2-base-multilingual :: model_quantized.onnx` | R4 | R4 |
| 10 | `onnx-community--FastVLM-0.5B-ONNX :: embed_tokens_quantized.onnx` | PASS | PASS |
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | I2 | I2 |
| 12 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=1`, `past=64`) | I3 | I3 |
| 13 | `onnx-community--FastVLM-0.5B-ONNX :: vision_encoder_quantized.onnx` | R2 | R2 |
| 14 | `Marqo--marqo-fashionSigLIP :: text_model_quantized.onnx` | R2 | R2 |
| 15 | `Marqo--marqo-fashionSigLIP :: vision_model_quantized.onnx` | R2 | R2 |
| 16 | `AdamCodd--vit-base-nsfw-detector :: model_quantized.onnx` | R2 | R2 |
| 17 | `Xenova--nllb-200-distilled-600M :: encoder_model_quantized.onnx` | G2 | R4 |
| 18 | `onnx-community--Janus-Pro-1B-ONNX :: language_model_q4.onnx` | E1 | E1 |
| 19 | `onnx-community--Janus-Pro-1B-ONNX :: lm_head.onnx` | PASS | PASS |
| 20 | `onnx-community--Janus-Pro-1B-ONNX :: gen_head.onnx` | PASS | PASS |
| 21 | `onnx-community--Janus-Pro-1B-ONNX :: gen_img_embeds.onnx` | PASS | PASS |
| 22 | `onnx-community--Janus-Pro-1B-ONNX :: image_decode.onnx` | R3 | R3 |
| 23 | `Xenova--musicgen-small :: text_encoder_quantized.onnx` | R2 | R2 |
| 24 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 25 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | R2 |
| 26 | `Mozilla--distilvit :: encoder_model_quantized.onnx` | R2 | R2 |
| 27 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: embed_tokens_fp16.onnx` | PASS | D1 |
| 28 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: decoder_model_merged_q4.onnx` | E1 | E1 |
| 29 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: audio_encoder_quantized.onnx` | R2 | R2 |
| 30 | `Xenova--LaMini-Flan-T5-783M :: encoder_model_quantized.onnx` | R2 | R2 |
| 31 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 32 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | R2 |
| 33 | `Xenova--detr-resnet-50 :: model_quantized.onnx` | R2 | R2 |
| 34 | `Xenova--donut-base-finetuned-docvqa :: encoder_model_quantized.onnx` | R4 | R4 |
| 35 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 36 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | R2 |
| 37 | `onnx-community--dinov3-vits16-pretrain-lvd1689m-ONNX :: model.onnx` | R3 | R3 |
| 38 | `Xenova--distilbart-cnn-6-6 :: encoder_model_quantized.onnx` | R2 | R2 |
| 39 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | I2 |
| 40 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | R2 |
| 41 | `prithivMLmods--Common-Voice-Gender-Detection-ONNX :: model_quantized.onnx` | R2 | R2 |
| 42 | `Xenova--bert-base-multilingual-cased :: model_quantized.onnx` | I1 | I1 |
| 43 | `Xenova--distilbert-base-cased-distilled-squad :: model_quantized.onnx` | R2 | R2 |
| 44 | `onnx-community--vitpose-base-simple :: model_quantized.onnx` | R2 | R2 |
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` | O1 | O1 |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` | O1 | O1 |
| 47 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: embed_tokens_quantized.onnx` | PASS | PASS |
| 48 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: decoder_model_merged_quantized.onnx` | I2 | I2 |
| 49 | `onnx-community--timesformer-base-finetuned-k400 :: model_quantized.onnx` | R1 | R1 |
| 50 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model_quantized.onnx` | R2 | R2 |
| 51 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model.onnx` | PASS | PASS |

### Previous suggested repair order

1. Fix I2 zero-element input generation and add semantic pinned/generated values for I1/I3. These validator fixes expose later converter/reload behavior without changing model semantics.
2. Fix reload operation-name normalization and shape inference: R2, then R4/R5, `conv2d`, and `resample2d`. R2 currently blocks the largest group.
3. Design explicit Int4/Uint4 external-weight encoding for E1.
4. Extend generated-weight graph/subgraph analysis for G1/G2 while retaining fail-closed behavior.
5. Verify or regenerate the two Chronos exports accepted by the skeleton path but rejected by native ORT.
6. Add resumable HTTP Range downloads before relying on unattended real-weight sweeps.

## Historical pre-rebase baseline (63 cases)

### Baseline

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

### Summary

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

### Differences between generated and real weights

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

### Failure families

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

### Case ledger

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

### Suggested repair order

1. Fix M1 and I1 with manifest/input semantics; these are small validator setup changes and may unlock nine cases without converter work.
2. Fix RustNN reload inference in focused tests: operation-name normalization and `clamp`, then `conv2d`, then `resample2d`. Rerun the affected subsets after each fix because the reported operand is only the first blocker.
3. Design an explicit Int4/Uint4 external-weight encoding for `.webnn` caches before changing E1. Treating packed 4-bit data as an ordinary 8-bit tensor without metadata would be lossy or ambiguous.
4. Extend generated-weight graph analysis for G1/G2 while preserving its fail-closed rule; #32 is the proof that G2 can be a fixture-only false negative.
5. Add resumable downloads to make large real-weight retries cheaper. Rerun affected subsets in both modes after each functional fix and update this ledger.
