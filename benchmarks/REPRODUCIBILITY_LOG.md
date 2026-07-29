# MLX Transformers Reproducibility Log

Validated protocol v1 results. Each filename links to the complete machine-readable record.

| Scenario | Chip | Memory | Model | Quantization | Prompt | Decode | TTFT | Prefill | Decode rate | Peak | Runs | Result |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prefill-512-decode-64 | Apple M2 Max | 32 GB | mlx-community/Phi-3-mini-4k-instruct-4bit | affine 4-bit/group-64 | 512 | 64 | 428.8 ms | 1194.0 tok/s | 62.2 tok/s | 3.43 GiB | 5 | [JSON](reference-results/m2-max__phi3-4bit__prefill-512-decode-64__2026-07-28.json) |
| short-decode-128 | Apple M2 Max | 32 GB | mlx-community/Phi-3-mini-4k-instruct-4bit | affine 4-bit/group-64 | 64 | 128 | 63.6 ms | 1006.2 tok/s | 71.8 tok/s | 2.20 GiB | 5 | [JSON](reference-results/m2-max__phi3-4bit__short-decode-128__2026-07-28.json) |
