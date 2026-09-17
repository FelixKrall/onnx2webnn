# Real-weight validation failures

> **Last edited:** `2026-09-17T11:08:08Z`<br>
> **Checkout:** `fkrall/cache-backed-validation` at `6f6ad8a`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

This document is the current triage record for failures from full-model validation with publisher
weights. Generated-weight failures are intentionally out of scope. The summary and complete case
ledger remain in [Full-model numerical validation status](model-validation-status.md).

## Recorded run

- Date: 2026-09-17
- Tested executable: onnx2webnn `6f6ad8a` plus the current Cast/Slice/comparison worktree
- RustNN: `65e76e67` plus the current Slice-backend worktree
- Manifest: `tests/models/manifest.json` (52 cases)
- Runtime: CPU ONNX Runtime 1.29.0
- Cache state: complete; no model downloads or skips
- Result: 48 passed and 4 failed
- Warm wall time: 7m 13.1s, one validation worker

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
| N1 | 1 | Output comparison | Confirmed numerical disagreement | FastVLM prefill remains blocked; its decode specialization passes. |
| Q1 | 1 | Output comparison | Unsupported `MatMulNBits accuracy_level=4` execution semantics | Requires a WebNN/backend mechanism for Int8-quantized activations with packed q4 weights; do not relax tolerance to hide it. |
| O1 | 2 | Native ORT model load | The publisher ONNX is rejected before conversion can be compared | Blocked upstream pending a corrected publisher artifact. |

The real sweep has no deterministic-input or cached-interface failures. N1 proves that both
execution paths completed and returned materially different values. Q1 also reaches comparison,
but the source and reconstructed graphs intentionally execute different precision modes. O1 fails
in the reference model itself.

## N1: numerical disagreement

| # | Case | Comparison diagnostics |
|--:|------|------------------------|
| 11 | `onnx-community--FastVLM-0.5B-ONNX :: decoder_model_merged_quantized.onnx` (`sequence=64`, `past=0`) | `9,702,133 / 9,705,344` logits exceed tolerance; maximum absolute error `10.0572`, mean absolute error `0.760946`, RMSE `1.05325`. |

The source uses fused GroupQueryAttention while the WebNN path decomposes it, with repeated
DynamicQuantizeLinear steps amplifying drift across prefill. The sequence-1 decode specialization
passes, so only this fixed prefill specialization is recorded as blocked.

## Q1: unsupported `MatMulNBits accuracy_level=4`

Voxtral's decoder sets `accuracy_level=4` on all 211 `MatMulNBits` nodes. Native ORT may quantize
Float32 activations to Int8 internally in that mode. WebNN has no fused low-bit matmul operator, so
onnx2webnn emits packed Uint4 weights followed by `dequantizeLinear` and ordinary Float32 matmul.
The two paths therefore do not promise the same numerical algorithm.

This is not a packed-weight serialization problem. All 211 source weight payloads matched their
saved q4 tensors byte-for-byte by SHA-256; archive marker, U8 storage shape, nibble ordering, and
generated `0x88` zero points were also correct. Direct output analysis found mean absolute error
`0.0401`, maximum error `0.367`, and correlation `0.999502`. Changing only the source attributes
to `accuracy_level=0` reduced those figures to `9.66e-6`, `9.32e-5`, and `0.99999999997`.

The manifest marks case 28 as `blocked`, but `all` and `match` continue attempting it so a future
implementation becomes visible. It is not counted as numerically supported.

## O1: source ONNX rejected by native ORT

Both Chronos files fail during native ORT model loading. A `Gather` indices input is the float
output of `ConstantOfShape`; Gather indices must be integer.

| # | Case |
|--:|------|
| 45 | `kashif--chronos-2-onnx :: encoder_model.onnx` |
| 46 | `kashif--chronos-2-onnx :: decoder_model_merged.onnx` |

The two manifest paths resolve to the same invalid publisher artifact. They are explicitly marked
`blocked` with this reason but remain attempted by `all` and `match` selections, so a future
publisher correction will be visible. Until the source model runs in native ORT, these cases cannot
provide a numerical oracle and are not evidence for or against onnx2webnn correctness.

## Repair order

1. Track or further localize the blocked FastVLM prefill disagreement without weakening
   comparison tolerance.
2. Track Q1 until WebNN or a backend-specific fusion can express Int8 activations with packed q4 weights.
3. Monitor the upstream Chronos repository for corrected reference exports.

## Maintenance rule

This file describes only the latest complete real-weight run. Replace counts, cases, diagnoses,
and exact errors when a new run changes first blockers; do not append obsolete full failure tables.
Record revision-attributed coverage changes in the main status file's history section.
