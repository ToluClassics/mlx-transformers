## Detailed Benchmark
Detailed runtime benchmark of model inferences, measured in milliseconds.

### Detailed Results
| Model                                                  | mlx_gpu | mlx_cpu | torch_cpu | mlx_gpu/mlx_cpu speedup |
|------------------------------------------------------------|-------|-------|---------|-----------------------|
| bert-base-uncased / inputs_char_no=50 / batch_size=1   |   0.37 |   0.57 |  84.74 |    +53% |
| bert-base-uncased / inputs_char_no=50 / batch_size=16  |   0.44 |   0.70 | 199.17 |    +60% |
| bert-base-uncased / inputs_char_no=50 / batch_size=32  |   0.84 |   0.94 | 385.70 |    +11% |
| bert-base-uncased / inputs_char_no=100 / batch_size=1  |   0.54 |   1.10 |  54.49 |   +103% |
| bert-base-uncased / inputs_char_no=100 / batch_size=16 |   0.42 |   0.65 | 309.32 |    +55% |
| bert-base-uncased / inputs_char_no=100 / batch_size=32 |   0.48 |   0.85 | 682.25 |    +76% |
| bert-base-uncased / inputs_char_no=200 / batch_size=1  |   0.42 |   0.78 |  71.76 |    +88% |
| bert-base-uncased / inputs_char_no=200 / batch_size=16 |   0.43 |   0.77 | 752.11 |    +76% |
| bert-base-uncased / inputs_char_no=200 / batch_size=32 |   0.42 |   0.76 | 1490.74 |    +79% |
| bert-base-uncased / inputs_char_no=500 / batch_size=1  |   0.41 |   0.77 | 119.86 |    +86% |
| bert-base-uncased / inputs_char_no=500 / batch_size=16 |   0.45 |   0.58 | 1835.18 |    +27% |
| bert-base-uncased / inputs_char_no=500 / batch_size=32 |   0.41 |   0.66 | 5791.45 |    +61% |
| roberta-base / inputs_char_no=50 / batch_size=1        |   0.54 |   0.84 |  57.48 |    +56% |
| roberta-base / inputs_char_no=50 / batch_size=16       |   0.41 |   0.73 | 269.02 |    +78% |
| roberta-base / inputs_char_no=50 / batch_size=32       |   0.41 |   0.62 | 550.03 |    +50% |
| roberta-base / inputs_char_no=100 / batch_size=1       |   0.49 |   0.64 |  59.81 |    +29% |
| roberta-base / inputs_char_no=100 / batch_size=16      |   0.42 |   0.59 | 465.86 |    +42% |
| roberta-base / inputs_char_no=100 / batch_size=32      |   0.44 |   0.63 | 990.06 |    +41% |
| roberta-base / inputs_char_no=200 / batch_size=1       |   0.42 |   0.68 |  88.49 |    +59% |
| roberta-base / inputs_char_no=200 / batch_size=16      |   0.46 |   0.62 | 1071.13 |    +35% |
| roberta-base / inputs_char_no=200 / batch_size=32      |   0.48 |   0.71 | 3201.12 |    +48% |
| roberta-base / inputs_char_no=500 / batch_size=1       |   0.47 |   1.65 | 180.33 |   +246% |
| roberta-base / inputs_char_no=500 / batch_size=16      |   0.43 |   0.66 | 4110.00 |    +51% |
| roberta-base / inputs_char_no=500 / batch_size=32      |   0.44 |   2.96 | 15630.43 |   +579% |
| xlm-roberta-base / inputs_char_no=50 / batch_size=1    |   0.43 |   0.79 |  37.87 |    +84% |
| xlm-roberta-base / inputs_char_no=50 / batch_size=16   |   0.51 |   0.61 | 196.84 |    +18% |
| xlm-roberta-base / inputs_char_no=50 / batch_size=32   |   0.52 |   0.72 | 384.63 |    +38% |
| xlm-roberta-base / inputs_char_no=100 / batch_size=1   |   0.43 |   0.77 |  50.78 |    +81% |
| xlm-roberta-base / inputs_char_no=100 / batch_size=16  |   0.65 |   0.80 | 411.10 |    +22% |
| xlm-roberta-base / inputs_char_no=100 / batch_size=32  |   0.53 |   0.87 | 751.44 |    +65% |
| xlm-roberta-base / inputs_char_no=200 / batch_size=1   |   0.45 |   0.93 |  77.99 |   +107% |
| xlm-roberta-base / inputs_char_no=200 / batch_size=16  |   0.57 |   1.09 | 706.65 |    +90% |
| xlm-roberta-base / inputs_char_no=200 / batch_size=32  |   0.45 |   0.98 | 1607.20 |   +120% |
| xlm-roberta-base / inputs_char_no=500 / batch_size=1   |   0.44 |   0.80 | 131.21 |    +80% |
| xlm-roberta-base / inputs_char_no=500 / batch_size=16  |   0.42 |   0.63 | 2483.16 |    +49% |
| xlm-roberta-base / inputs_char_no=500 / batch_size=32  |   0.56 |   0.79 | 5861.21 |    +41% |

## Average Benchmark of Input Lengths
Average runtime benchmark of model inferences with respect to input length, measured in milliseconds.

### Average Results wrt Input Length
| Model                                  | mlx_gpu | mlx_cpu | torch_cpu | mlx_gpu/mlx_cpu speedup |
|--------------------------------------------|-------|-------|---------|-----------------------|
| bert-base-uncased / inputs_char_no=50  |   0.55 |   0.74 | 223.20 |    +34% |
| bert-base-uncased / inputs_char_no=100 |   0.48 |   0.87 | 348.69 |    +80% |
| bert-base-uncased / inputs_char_no=200 |   0.42 |   0.77 | 771.53 |    +81% |
| bert-base-uncased / inputs_char_no=500 |   0.42 |   0.67 | 2582.16 |    +57% |
| roberta-base / inputs_char_no=50       |   0.45 |   0.73 | 292.18 |    +61% |
| roberta-base / inputs_char_no=100      |   0.45 |   0.62 | 505.24 |    +37% |
| roberta-base / inputs_char_no=200      |   0.45 |   0.67 | 1453.58 |    +47% |
| roberta-base / inputs_char_no=500      |   0.45 |   1.75 | 6640.25 |   +291% |
| xlm-roberta-base / inputs_char_no=50   |   0.49 |   0.70 | 206.45 |    +44% |
| xlm-roberta-base / inputs_char_no=100  |   0.54 |   0.82 | 404.44 |    +52% |
| xlm-roberta-base / inputs_char_no=200  |   0.49 |   1.00 | 797.28 |   +104% |
| xlm-roberta-base / inputs_char_no=500  |   0.47 |   0.74 | 2825.19 |    +56% |

## Average Benchmark of Batch Sizes
Average runtime benchmark of model inferences with respect to batch size, measured in milliseconds.

### Average Results wrt Batch Size
| Model                             | mlx_gpu | mlx_cpu | torch_cpu | mlx_gpu/mlx_cpu speedup |
|---------------------------------------|-------|-------|---------|-----------------------|
| bert-base-uncased / batch_size=1  |   0.43 |   0.81 |  82.71 |    +85% |
| bert-base-uncased / batch_size=16 |   0.43 |   0.67 | 773.95 |    +54% |
| bert-base-uncased / batch_size=32 |   0.54 |   0.80 | 2087.54 |    +49% |
| roberta-base / batch_size=1       |   0.48 |   0.95 |  96.53 |    +97% |
| roberta-base / batch_size=16      |   0.43 |   0.65 | 1479.00 |    +51% |
| roberta-base / batch_size=32      |   0.44 |   1.23 | 5092.91 |   +177% |
| xlm-roberta-base / batch_size=1   |   0.44 |   0.82 |  74.46 |    +88% |
| xlm-roberta-base / batch_size=16  |   0.54 |   0.78 | 949.44 |    +45% |
| xlm-roberta-base / batch_size=32  |   0.51 |   0.84 | 2151.12 |    +63% |

## Average Benchmark
Average runtime benchmark of model inferences, measured in milliseconds.

### Average Results
| Model             | mlx_gpu | mlx_cpu | torch_cpu | mlx_gpu/mlx_cpu speedup |
|-----------------------|-------|-------|---------|-----------------------|
| bert-base-uncased |   0.47 |   0.76 | 981.40 |    +62% |
| roberta-base      |   0.45 |   0.94 | 2222.81 |   +109% |
| xlm-roberta-base  |   0.50 |   0.82 | 1058.34 |    +64% |
