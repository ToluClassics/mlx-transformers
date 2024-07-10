## Detailed Benchmark
Detailed runtime benchmark of model inferences, measured in milliseconds.

### Detailed Results
| Model                                   | mlx_gpu | mlx_cpu | torch_cpu | mlx_gpu/mlx_cpu speedup |
|---------------------------------------------|-------|-------|---------|-----------------------|
| bert-base-uncased / inputs_char_no=50   |   0.36 |   0.54 |  79.29 |    +51% |
| bert-base-uncased / inputs_char_no=100  |   0.41 |   0.59 |  41.17 |    +43% |
| bert-base-uncased / inputs_char_no=200  |   0.38 |   0.57 |  59.99 |    +48% |
| bert-base-uncased / inputs_char_no=500  |   0.41 |   0.58 |  74.36 |    +42% |
| bert-base-uncased / inputs_char_no=1000 |   0.40 |   0.57 | 125.44 |    +40% |
| roberta-base / inputs_char_no=50        |   0.36 |   0.54 |  42.24 |    +50% |
| roberta-base / inputs_char_no=100       |   0.37 |   0.53 |  38.99 |    +42% |
| roberta-base / inputs_char_no=200       |   0.38 |   0.55 |  45.73 |    +43% |
| roberta-base / inputs_char_no=500       |   0.40 |   0.58 |  59.64 |    +45% |
| roberta-base / inputs_char_no=1000      |   0.42 |   0.66 |  98.81 |    +55% |
| xlm-roberta-base / inputs_char_no=50    |   0.40 |   0.68 |  46.86 |    +70% |
| xlm-roberta-base / inputs_char_no=100   |   0.39 |   0.68 |  46.75 |    +72% |
| xlm-roberta-base / inputs_char_no=200   |   0.46 |   0.64 |  50.83 |    +38% |
| xlm-roberta-base / inputs_char_no=500   |   0.50 |   0.63 |  91.83 |    +25% |
| xlm-roberta-base / inputs_char_no=1000  |   0.40 |   0.68 | 116.08 |    +70% |

## Average Benchmark
Average runtime benchmark of model inferences, measured in milliseconds.

### Average Results
| Model             | mlx_gpu | mlx_cpu | torch_cpu | mlx_gpu/mlx_cpu speedup |
|-----------------------|-------|-------|---------|-----------------------|
| bert-base-uncased |   0.39 |   0.57 |  76.05 |    +45% |
| roberta-base      |   0.39 |   0.57 |  57.08 |    +47% |
| xlm-roberta-base  |   0.43 |   0.66 |  70.47 |    +53% |
