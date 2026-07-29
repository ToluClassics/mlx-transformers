# MLX Transformers Reproducibility Log

Validated protocol v1 results.

| Scenario | Chip | Memory | Model | Quantization | Prompt | Decode | TTFT | Prefill | Decode rate | Peak | Runs |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill-512-decode-64 | Apple M2 Max | 32 GB | Qwen/Qwen3-0.6B | none | 512 | 64 | 98.6 ms | 5192.9 tok/s | 65.1 tok/s | 4.10 GiB | 5 |
| prefill-512-decode-64 | Apple M2 Max | 32 GB | meta-llama/Llama-3.2-1B | none | 512 | 64 | 155.9 ms | 3284.2 tok/s | 50.2 tok/s | 5.66 GiB | 5 |
| prefill-512-decode-64 | Apple M2 Max | 32 GB | mlx-community/Phi-3-mini-4k-instruct-4bit | affine 4-bit/group-64 | 512 | 64 | 428.3 ms | 1195.4 tok/s | 61.8 tok/s | 3.43 GiB | 5 |
| short-decode-128 | Apple M2 Max | 32 GB | Qwen/Qwen3-0.6B | none | 64 | 128 | 24.1 ms | 2656.7 tok/s | 72.4 tok/s | 2.95 GiB | 5 |
| short-decode-128 | Apple M2 Max | 32 GB | meta-llama/Llama-3.2-1B | none | 64 | 128 | 28.6 ms | 2236.3 tok/s | 51.9 tok/s | 4.77 GiB | 5 |
| short-decode-128 | Apple M2 Max | 32 GB | mlx-community/Phi-3-mini-4k-instruct-4bit | affine 4-bit/group-64 | 64 | 128 | 63.8 ms | 1003.4 tok/s | 72.1 tok/s | 2.20 GiB | 5 |

Environment: macOS 26.0, Python 3.12.9, MLX 0.32.0, Transformers 5.14.1,
MLX Transformers commit `5883d5762ff324e0e61c9f507eef5d176e63146d`.

Checkpoint revisions:

- `Qwen/Qwen3-0.6B`: `c1899de289a04d12100db370d81485cdf75e47ca`
- `meta-llama/Llama-3.2-1B`: `4e20de362430cd3b72f300e6b0f18e50e7166e08`
- `mlx-community/Phi-3-mini-4k-instruct-4bit`:
  `5b3819ed6317784fb20eddeae9bed984f778d0d0`
