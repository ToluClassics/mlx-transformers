# Support Contract

This document describes the evidence behind MLX Transformers 0.3.0 support
claims. The project provides Hugging Face-style model APIs implemented in MLX;
it is not a drop-in replacement for `transformers`, does not register with
`AutoModel`, and does not implement training.

## Status definitions

- **Verified** means a real safetensors checkpoint has passed strict loading
  and the listed inference path on Apple silicon, backed by bounded local
  regression tests.
- **Experimental** means the implementation is exported and has synthetic
  shape/behavior tests, but at least one important real-checkpoint path remains
  unverified.
- **Unsupported** means the behavior is deliberately rejected or outside the
  package contract.

Verification applies only to the listed tasks and configurations. It does not
imply that every checkpoint using the same architecture is compatible.

## Lifecycle priority

Support status and maintenance priority are separate:

- **Active** families receive compatibility, correctness, and evidence work:
  BERT, RoBERTa, XLM-RoBERTa, Llama, Phi/Phi-3, Qwen3, Qwen3-VL, Gemma 3, and
  M2M100/NLLB.
- **Maintenance-only** families remain importable for existing users but do
  not receive proactive feature-parity or performance work: OpenELM,
  Persimmon, and Fuyu.

Maintenance-only families retain bounded regression coverage so unrelated
changes do not silently break them. If keeping one compatible begins to impose
disproportionate cost or blocks the active support matrix, it may be deprecated
in release notes and removed from public exports in a subsequent release.
Security and fail-closed loading fixes still apply while a family remains
exported.

## Compatibility baseline

| Layer | Verified baseline |
| --- | --- |
| Hardware | Apple silicon with Metal |
| Operating system | macOS |
| Python | CI: 3.10 and 3.13; local validation: 3.12.9 |
| MLX | 0.31.0 and 0.32.0 |
| Transformers | 4.57.6 and 5.14.1 |
| Checkpoints | Safetensors, including validated shard indexes |
| Default dtype | float32 |
| Reduced precision | float16 or bfloat16 where the selected model path remains finite |

The package metadata bounds MLX to `<0.33` and Transformers to `<6`. New
upstream releases require compatibility evidence before those bounds move.

## Model matrix

| Family | Status | Evidence-backed scope |
| --- | --- | --- |
| BERT | Verified | Strict base/task-head loading and forward execution; MiniLM base-model parity at `1e-4` |
| Llama | Verified | Strict loading, forward, finite batched cached generation, and real-checkpoint parity |
| Phi | Experimental | Real Phi-2 strict load/forward; generation contract covered synthetically |
| Phi-3 | Verified | Real strict load/forward plus real 4-bit MLX checkpoint cached generation |
| Qwen3 | Verified | Real 0.6B strict load and batch-two generation parity with identical argmax |
| Gemma 3 | Verified | Real 1B text generation parity and real 4B image generation in float32/bfloat16 |
| M2M100/NLLB | Verified | Real distilled 600M encoder, decoder, and logits path |
| Qwen3-VL | Experimental | Real 2B text generation; image generation is synthetic-only |
| RoBERTa | Experimental | Base and task heads covered synthetically; no real family checkpoint gate |
| XLM-RoBERTa | Experimental | Base and task heads covered synthetically; no real family checkpoint gate |
| OpenELM | Experimental | Synthetic forward and finite generation only |
| Persimmon | Experimental | Synthetic forward and finite generation only |
| Fuyu | Experimental | Synthetic image/forward/generation only |

Gemma 3 4B multimodal inference produced non-finite values in float16 during
validation. Use float32 or `mx.bfloat16` for that path. Float16 is not a
verified Gemma 3 4B multimodal configuration.

Quantization structure and error paths have bounded tests. Real quantized
generation is currently verified only for the tested 4-bit Phi-3 checkpoint
with group size 64.

## Generation contract

Exported causal generators:

- require a finite `max_new_tokens` bound;
- yield one token ID per active batch item on every step;
- support left- or right-padded batches;
- stop and pad completed sequences independently using EOS/pad IDs;
- preserve caller-owned input mappings and arrays;
- support cached and uncached greedy or temperature sampling.

The legacy `max_length` argument remains a deprecated alias for generated-token
count. It does not use Hugging Face's total-sequence-length meaning and should
not be used in new code.

Beam search, top-k/top-p sampling, repetition penalties, and the complete
Transformers stopping-criteria API are not implemented.

## Loading contract

`from_pretrained` supports local checkpoint directories and Hugging Face Hub
model IDs. It validates safetensors shards and indexes, rejects missing or
unexpected model parameters, and supports explicit offline, cache, revision,
token, worker, and dtype controls.

PyTorch `.bin` checkpoints and remote model code are unsupported.
`trust_remote_code=True` is ignored with a warning rather than executing code
from a model repository.

## Explicitly unsupported

- non-Apple or non-Metal execution;
- training, optimizers, or trainer APIs;
- Transformers `AutoModel` registration or drop-in `GenerationMixin`
  compatibility;
- PyTorch `.bin` checkpoint loading;
- executing remote repository code;
- unbounded generation.

To promote an experimental path to verified, add a bounded test for the real
checkpoint, record the exact revision and environment, establish numerical or
behavioral acceptance criteria, and run it on Apple silicon.
