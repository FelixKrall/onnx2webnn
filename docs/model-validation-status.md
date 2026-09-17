# Full-model numerical validation status

> **Last edited:** `2026-09-17T11:08:08Z`<br>
> **Checkout:** `fkrall/cache-backed-validation` at `6f6ad8a`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

This document tracks the **latest recorded** full-model numerical-validation sweep and a concise
history of changes that materially altered coverage. Only the latest sweep keeps a complete
per-case ledger; older ledgers and duplicated historical failure tables are intentionally omitted.

The skeleton sweep establishes broad graph-construction coverage. Numerical validation additionally
exports the converted graph, reloads its `.webnn` and Safetensors artifacts, executes both the
original ONNX model and the reloaded graph on CPU ONNX Runtime, and compares matching outputs.

Proper dynamic cache-shape support is not currently available. Validation cases use fixed
dimension overrides and execute as independent snapshots, so they do not validate a complete
prefill-to-decode loop or growing KV-cache reuse. A decode artifact that
accepts a fixed past length `N` normally produces a cache of length `N + 1`, which cannot be fed
back into that same fixed-shape artifact. Repeated decoding therefore still requires proper dynamic
cache shapes, separately specialized artifacts, or a fixed-capacity cache with an explicit position.

### Runtime dtypes and comparison tolerances

WebNN graphs are not globally limited to floating point: RustNN records integer tensors and integer
operations where their operator contracts allow them. Matrix multiplication is different. The
portable WebNN `matmul` contract accepts matching `float16` or `float32` operands, and WebNN has no
fused equivalent of ORT's `com.microsoft.MatMulNBits`.

For q4 models, the original ONNX path gives native ORT packed Uint4 weights and a fused
`MatMulNBits` node. The reloaded path restores the same packed bytes, widens them for RustNN's
temporary ORT graph, applies `dequantizeLinear`, and runs ordinary matmul in the scale dtype. The
models in this manifest use Float32 scales and activations, so their reconstructed matmuls are
Float32. Different fused/decomposed accumulation orders are mathematically equivalent but not
bit-identical.

The validator therefore uses these output tolerances:

| Output/path | Comparison |
|-------------|------------|
| Float32 without `MatMulNBits` | `1e-5 + 1e-4 * abs(reference)` |
| Float32 from a model containing q4 `MatMulNBits` | `1e-3 + 1e-3 * abs(reference)` |
| Float32 from a model containing q8 `MatMulNBits` | `2e-3 + 2e-3 * abs(reference)` |
| Float16 | `1e-3 + 1e-2 * abs(reference)` |
| Integer and boolean | Exact |

The q4 Float32 envelope is based on a complete SmolLM2 output scan whose worst normalized
difference was `9.488e-4`. The q8 envelope is selected from each source graph's
`MatMulNBits.bits` attribute; it covers the measured Qwen output delta while remaining distinct
from q4. It does not make ORT's `accuracy_level=4` equivalent to WebNN:
level 4 permits internal Int8 activation quantization, while the WebNN lowering keeps Float32
activations. Voxtral uses level 4 on all 211 `MatMulNBits` nodes and remains blocked for this reason.

## Latest recorded sweep

- Generated sweep: 2026-09-12 at onnx2webnn `4926c3e`
- Real sweep: 2026-09-17 at onnx2webnn `6f6ad8a` plus the current Cast/Slice/comparison worktree
- RustNN: `65e76e67` plus the current Slice-backend worktree
- ORT: repository-local Linux x64 1.29.0 build
- Manifest: `tests/models/manifest.json` (52 cases, 45 unique ONNX files)
- Execution: one validation worker; ORT may use multiple CPU threads inside a case

These revisions identify the latest **tested** state for each weight mode. A newer checkout is not
considered the baseline for a mode until that mode has been rerun and this section is replaced.

```bash
ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights generated --jobs 1

ORT_DYLIB_PATH=../rustnn/target/onnxruntime/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 \
  target/release/onnx2webnn validate-models \
  --selection all --weights real --jobs 1
```

### Summary

| Weight mode | Pass | Fail | Download skipped | Result |
|-------------|-----:|-----:|-----------------:|--------|
| Generated (`g4`) | 21 | 31 | 0 | Complete (52/52) |
| Real | 48 | 4 | 0 | Complete (52/52) |

Twenty cases pass in both modes. Eight generated cases reach comparison; in the current real sweep,
one case remains a genuine numerical disagreement and one is the unsupported Voxtral execution
mode. The generated column remains the 2026-09-12 baseline; only the real sweep
was rerun for this change. Generated and real weights can expose different first blockers; a failure
before comparison is not evidence of a numerical mismatch.

The real-weight failures, exact affected cases, current diagnosis, and suggested ownership are
tracked in [Real-weight validation failures](real-weight-validation-failures.md).

### Current failure families

| Code | Generated | Real | Stage | Current cause / next action |
|------|----------:|-----:|-------|-----------------------------|
| N1 | 8 | 1 | Comparison | FastVLM prefill materially differs after repeated DynamicQuantizeLinear and decomposed GroupQueryAttention; its decode specialization passes. |
| E1 | 4 | 0 | Export | The retained generated baseline predates packed Int4/Uint4 serialization; the current real sweep confirms E1 is fixed. |
| G1 | 10 | 0 | Generated preparation | Initializer has no recorded consumers; extend consumer analysis while keeping generation fail-closed. |
| G2 | 1 | 0 | Generated preparation | Large tensor-valued Constant has an ambiguous role. |
| I1 | 3 | 0 | Native ORT input | Cleared for real weights by generating zero-valued `token_type_ids`; generated counts await a full rerun. |
| I2 | 2 | 0 | Native ORT input | Cleared for real weights by preserving zero-element buffers; generated counts await a full rerun. |
| I3 | 1 | 0 | Native ORT input | Cleared for real weights by generating an all-ones `attention_mask`; generated counts await a full rerun. |
| Q1 | 0 | 1 | Comparison | Unsupported execution semantics: Voxtral sets `MatMulNBits accuracy_level=4`, allowing native ORT to quantize activations to Int8; WebNN lowers to Float32 dequantize-plus-matmul. |
| O1 | 2 | 2 | Native ORT load | Upstream-blocked: both Chronos paths resolve to the same publisher artifact, which feeds a float ConstantOfShape result to Gather indices. |

Generated totals: 11 generated-model preparation, 4 export, 6 native-ORT input, 2 native-ORT
model-load, 8 comparison failures, and 21 passes. Real totals: no export, reload,
cached-interface, or native-ORT input failures, 2 native-ORT model-load failures, 1 numerical disagreement, 1 unsupported execution-mode
mismatch, and 48 passes.

### Current case ledger

`PASS` means export, reload, both executions, and numerical comparison completed. Any other value
is the current first-blocker code from the table above.

`WebNN If split` identifies merged decoders whose runtime `use_cache_branch` condition must be
pinned because WebNN has no `If` operation. Each prefill/decode entry produces its own independent
`.webnn`/Safetensors pair. Loading both graphs concurrently may duplicate their shared weights in
RAM and VRAM. This is distinct from ordinary repeated cases that differ only in fixed dimensions.

| # | Manifest case | Generated | Real | WebNN If split |
|---:|---------------|-----------|------|----------------|
| 0 | `briaai--RMBG-1.4 :: model_quantized.onnx` | PASS | PASS | — |
| 1 | `openai--privacy-filter :: model_quantized.onnx` | N1 | PASS | — |
| 2 | `nomic-ai--nomic-embed-text-v1.5 :: model_quantized.onnx` | I1 | PASS | — |
| 3 | `mixedbread-ai--mxbai-embed-large-v1 :: model_quantized.onnx` | I1 | PASS | — |
| 4 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=64`, `past=0`) | E1 | PASS | — |
| 5 | `HuggingFaceTB--SmolLM2-1.7B-Instruct :: model_q4.onnx` (`sequence=1`, `past=64`) | E1 | PASS | — |
| 6 | `distil-whisper--distil-large-v2 :: encoder_model_quantized.onnx` | N1 | PASS | — |
| 7 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | PASS | Prefill pair (1/2) |
| 8 | `distil-whisper--distil-large-v2 :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | PASS | Decode pair (2/2) |
| 9 | `jinaai--jina-reranker-v2-base-multilingual :: model_quantized.onnx` | PASS | PASS | — |
| 10 | `onnx-community--FastVLM-0.5B-ONNX :: embed_tokens_quantized.onnx` | PASS | PASS | — |
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | I2 | N1 | — |
| 12 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=1`, `past=64`) | I3 | PASS | — |
| 13 | `onnx-community--FastVLM-0.5B-ONNX :: vision_encoder_quantized.onnx` | PASS | PASS | — |
| 14 | `Marqo--marqo-fashionSigLIP :: text_model_quantized.onnx` | PASS | PASS | — |
| 15 | `Marqo--marqo-fashionSigLIP :: vision_model_quantized.onnx` | N1 | PASS | — |
| 16 | `AdamCodd--vit-base-nsfw-detector :: model_quantized.onnx` | PASS | PASS | — |
| 17 | `Xenova--nllb-200-distilled-600M :: encoder_model_quantized.onnx` | G2 | PASS | — |
| 18 | `onnx-community--Janus-Pro-1B-ONNX :: language_model_q4.onnx` | E1 | PASS | — |
| 19 | `onnx-community--Janus-Pro-1B-ONNX :: lm_head.onnx` | PASS | PASS | — |
| 20 | `onnx-community--Janus-Pro-1B-ONNX :: gen_head.onnx` | PASS | PASS | — |
| 21 | `onnx-community--Janus-Pro-1B-ONNX :: gen_img_embeds.onnx` | PASS | PASS | — |
| 22 | `onnx-community--Janus-Pro-1B-ONNX :: image_decode.onnx` | PASS | PASS | — |
| 23 | `Xenova--musicgen-small :: text_encoder_quantized.onnx` | PASS | PASS | — |
| 24 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | PASS | Prefill pair (1/2) |
| 25 | `Xenova--musicgen-small :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | PASS | Decode pair (2/2) |
| 26 | `Mozilla--distilvit :: encoder_model_quantized.onnx` | N1 | PASS | — |
| 27 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: embed_tokens_fp16.onnx` | PASS | PASS | — |
| 28 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: decoder_model_merged_q4.onnx` | E1 | Q1 | — |
| 29 | `onnx-community--Voxtral-Mini-3B-2507-ONNX :: audio_encoder_quantized.onnx` | N1 | PASS | — |
| 30 | `Xenova--LaMini-Flan-T5-783M :: encoder_model_quantized.onnx` | PASS | PASS | — |
| 31 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | PASS | Prefill pair (1/2) |
| 32 | `Xenova--LaMini-Flan-T5-783M :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | PASS | Decode pair (2/2) |
| 33 | `Xenova--detr-resnet-50 :: model_quantized.onnx` | PASS | PASS | — |
| 34 | `Xenova--donut-base-finetuned-docvqa :: encoder_model_quantized.onnx` | N1 | PASS | — |
| 35 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | PASS | Prefill pair (1/2) |
| 36 | `Xenova--donut-base-finetuned-docvqa :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | PASS | Decode pair (2/2) |
| 37 | `onnx-community--dinov3-vits16-pretrain-lvd1689m-ONNX :: model.onnx` | PASS | PASS | — |
| 38 | `Xenova--distilbart-cnn-6-6 :: encoder_model_quantized.onnx` | PASS | PASS | — |
| 39 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=0`) | G1 | PASS | Prefill pair (1/2) |
| 40 | `Xenova--distilbart-cnn-6-6 :: decoder_model_merged_quantized.onnx` (`cache=1`) | G1 | PASS | Decode pair (2/2) |
| 41 | `prithivMLmods--Common-Voice-Gender-Detection-ONNX :: model_quantized.onnx` | N1 | PASS | — |
| 42 | `Xenova--bert-base-multilingual-cased :: model_quantized.onnx` | I1 | PASS | — |
| 43 | `Xenova--distilbert-base-cased-distilled-squad :: model_quantized.onnx` | PASS | PASS | — |
| 44 | `onnx-community--vitpose-base-simple :: model_quantized.onnx` | PASS | PASS | — |
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` | O1 | O1 | — |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` | O1 | O1 | — |
| 47 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: embed_tokens_quantized.onnx` | PASS | PASS | — |
| 48 | `huggingworld--Qwen2.5-VL-3B-Instruct-ONNX :: decoder_model_merged_quantized.onnx` | I2 | PASS | — |
| 49 | `onnx-community--timesformer-base-finetuned-k400 :: model_quantized.onnx` | N1 | PASS | — |
| 50 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model_quantized.onnx` | PASS | PASS | — |
| 51 | `Xenova--tiny-random-RoFormerForMultipleChoice :: model.onnx` | PASS | PASS | — |

### Current timing and storage

| Run | Cases | Warm wall time |
|-----|------:|---------------:|
| Generated | 52 | 7m 47.5s |
| Real, cache-complete | 52 | 7m 13.1s |

At the end of the sweep, `.onnx-cache` occupied 76 GB and `.webnn-cache` 86 GB. The increase
is the newly exportable q4 artifacts. Neither completed run needed or skipped a download.

### Current repair order

1. Track the blocked FastVLM prefill disagreement while retaining its passing decode specialization.
2. Extend generated-weight role/consumer analysis for G1/G2 without weakening fail-closed behavior.
3. Monitor the upstream Chronos repository for corrected reference exports.

## Coverage change history

This history records only changes that explain coverage movement or establish that an apparent
movement was not a functional improvement. Detailed obsolete ledgers are available through Git.

| Date / tested revisions | Change | Comparable coverage effect |
|-------------------------|--------|----------------------------|
| 2026-09-17 — onnx2webnn `6f6ad8a` plus current Cast/Slice/comparison worktree, RustNN `65e76e67` plus Slice-backend worktree | Normalized ONNX numeric-to-Bool Cast through comparison, preserved positive Slice strides with WebNN extent semantics, and selected q4/q8 comparison envelopes from `MatMulNBits.bits`. | DETR, Donut encoder, and Qwen passed; real coverage rose **45 → 48** on the same 52 cases. FastVLM prefill, Voxtral level 4, and the two invalid Chronos artifacts remain blocked. |
| 2026-09-16 — onnx2webnn `24fdd50` plus current validator/manifest worktree, RustNN `7f07a5e1` plus packed-4-bit archive worktree | Measured complete q4 output error distributions and applied a `1e-3` absolute/relative Float32 envelope only to source graphs containing `MatMulNBits`. Classified Voxtral's `accuracy_level=4` Int8-activation execution mode as unsupported. | SmolLM2 prefill/decode and Janus passed; Voxtral remained Q1. Real passes rose **42 → 45** on the same 52 cases. |
| 2026-09-16 — onnx2webnn `24fdd50` plus current manifest/tests worktree, RustNN `7f07a5e1` plus packed-4-bit archive worktree | Stored packed Int4/Uint4 constants as versioned U8 Safetensors payloads while preserving logical dtype and shape in `.webnn`, then restored and executed them on reload. Marked the identical invalid Chronos publisher artifacts as upstream-blocked. | E1 was eliminated from the real sweep: all four q4 cases reached comparison and exposed N1. The aggregate remained **42/52** because those cases do not yet match numerically. |
| 2026-09-15 — onnx2webnn `ec5ba275` plus current validator worktree, RustNN `7f07a5e1` | Reconciled native ONNX interfaces with branch-specialized cached WebNN interfaces, dispatching only retained inputs and accepting omitted outputs only when native ORT proves they are empty. | Real passes rose **32 → 42** on the same 52 cases. All ten V1 cases passed numerically and no new blocker family appeared. |
| 2026-09-14 — onnx2webnn `aaa33e2` plus current validator worktree, RustNN `7f07a5e1` | Made standard `token_type_ids` zero, `attention_mask` one, and preserved zero-element input buffers without treating scalars as empty. | Real passes rose **28 → 32** on the same 52 cases. I1-I3 were eliminated: four cases passed, five exposed V1, and two exposed N1. |
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
