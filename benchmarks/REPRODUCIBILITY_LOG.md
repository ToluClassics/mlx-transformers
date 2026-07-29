# MLX Transformers Reproducibility Log

Validated protocol v1 results.

| Scenario | Chip | Memory | Model | Quantization | Prompt | Decode | TTFT | Prefill | Decode rate | Peak | Runs |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill-512-decode-64 | Apple M2 Max | 32 GB | mlx-community/Phi-3-mini-4k-instruct-4bit | affine 4-bit/group-64 | 512 | 64 | 428.8 ms | 1194.0 tok/s | 62.2 tok/s | 3.43 GiB | 5 |
| short-decode-128 | Apple M2 Max | 32 GB | mlx-community/Phi-3-mini-4k-instruct-4bit | affine 4-bit/group-64 | 64 | 128 | 63.6 ms | 1006.2 tok/s | 71.8 tok/s | 2.20 GiB | 5 |
