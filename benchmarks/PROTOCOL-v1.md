# Reproducible Benchmark Protocol v1

Protocol v1 measures deterministic single-sequence text generation on Apple
silicon. It is a performance protocol, not a model-quality evaluation.

## Built-in scenarios

| Scenario | Prefill tokens | Decode tokens | Temperature | Warmups | Runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short-decode-128` | 64 | 128 | 0 | 1 | 5 |
| `prefill-512-decode-64` | 512 | 64 | 0 | 1 | 5 |

The runner tokenizes the scenario's public source text without special tokens,
prepends the tokenizer's BOS token when present, repeats the resulting IDs, and
truncates them to the exact prefill length. EOS stopping is disabled so every
valid run produces the scenario's exact decode length.

## Reproducibility contract

Shareable results must:

- use an immutable 40-character Hugging Face revision;
- use a public `owner/model` checkpoint without remote model code;
- embed the complete scenario and its canonical SHA-256 checksum;
- record the package, Python, MLX, Transformers, macOS, chip, machine model,
  unified memory, quantization metadata, and MLX memory observations;
- record `--implementation-revision` when running an editable/source checkout;
- include every measured run plus mean, median, standard deviation, minimum,
  and maximum metrics;
- record matching generated-token checksums for every greedy measured run;
- contain no usernames, home-directory paths, cache paths, or credentials;
- pass `mlx-transformers-benchmark validate`.

The measured metrics are:

- time to first token (TTFT), which includes prefill and first-token sampling;
- prefill tokens per second, computed as prompt tokens divided by TTFT;
- decode tokens per second after the first generated token;
- end-to-end generation time;
- peak memory reported by the MLX allocator.

MLX allocator memory is not operating-system RSS. Compare results only when the
protocol version, scenario, model revision, quantization, and relevant software
versions are visible.

## Running and sharing

Run both scenarios with a pinned checkpoint revision:

```bash
mlx-transformers-benchmark run \
  --scenario short-decode-128 \
  --model mlx-community/Phi-3-mini-4k-instruct-4bit \
  --revision 5b3819ed6317784fb20eddeae9bed984f778d0d0 \
  --output phi3-short.json

mlx-transformers-benchmark run \
  --scenario prefill-512-decode-64 \
  --model mlx-community/Phi-3-mini-4k-instruct-4bit \
  --revision 5b3819ed6317784fb20eddeae9bed984f778d0d0 \
  --output phi3-prefill.json
```

Use `--local-files-only` to require an already-cached snapshot. Validate and
render the results:

```bash
mlx-transformers-benchmark validate phi3-short.json phi3-prefill.json
mlx-transformers-benchmark render \
  phi3-short.json phi3-prefill.json \
  --output REPRODUCIBILITY_LOG.md
```

Community submissions should include the original validated JSON. Markdown
tables are derived views and are not sufficient evidence on their own.
