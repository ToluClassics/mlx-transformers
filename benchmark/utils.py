import io
import math
import random
import numpy as np
from collections import defaultdict



corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Natural language processing is a field of artificial intelligence.",
    "Machine learning models require a lot of data.",
    "Deep learning is a subset of machine learning.",
    "Supervised learning relies on labeled data for training.",
    "Unsupervised learning can discover hidden patterns in data.",
    "Reinforcement learning involves agents learning from their environment.",
    "Neural networks are inspired by the human brain.",
    "Convolutional neural networks are effective for image recognition.",
    "Recurrent neural networks are suitable for sequence data.",
    "Transfer learning allows models to leverage pre-trained knowledge.",
    "Hyperparameter tuning is crucial for optimizing model performance.",
    "Gradient descent is an optimization algorithm used in training.",
    "Overfitting occurs when a model performs well on training data but poorly on new data.",
    "Cross-validation helps to evaluate model performance more reliably.",
    "Data augmentation techniques can improve model robustness.",
    "The Turing Test assesses a machine's ability to exhibit intelligent behavior.",
    "Ethics in AI is a critical area of research and development.",
    "Generative adversarial networks can create realistic synthetic data."
]


def get_input_text(corpus, char_length):
    """
    Generates a input text with the required number of characters from the corpus.
    """
    selected_text = ""
    while len(selected_text) < char_length:
        selected_text += random.choice(corpus) + " "
    return selected_text[:char_length].strip()

def calculate_speedup(a, compared_to):
    percentage_difference = -((a - compared_to) / a)
    return percentage_difference * 100

def print_benchmark(times, backends, reduce_mean=False):
    output = io.StringIO()

    times = dict(times)

    if reduce_mean:
        new_times = defaultdict(lambda: defaultdict(list))
        for k, v in times.items():
            op = k.split("/")[0]
            for backend, runtime in v.items():
                new_times[op][backend].append(runtime)

        for k, v in new_times.items():
            for backend, runtimes in v.items():
                new_times[k][backend] = np.mean(new_times[k][backend])
        times = new_times

    # Column headers
    header_order = ["mlx_gpu", "mlx_gpu_compile", "mlx_cpu", "torch_mps", "torch_cpu", "torch_cuda"]
    headers = sorted(backends, key=lambda x: header_order.index(x))

    if "mlx_gpu_compile" in backends and "mlx_gpu" in backends:
        h = "mlx_gpu_compile/mlx_gpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu_compile"], compared_to=v["mlx_gpu"])

    if "torch_mps" in backends and "mlx_gpu" in backends:
        h = "mlx_gpu/torch_mps speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["torch_mps"])

    if "mlx_cpu" in backends and "mlx_gpu" in backends:
        h = "mlx_gpu/mlx_cpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["mlx_cpu"])

    if "torch_cpu" in backends and "torch_cuda" in backends:
        h = "torch_cuda/torch_cpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["torch_cuda"], compared_to=v["torch_cpu"])

    max_name_length = max(len(name) for name in times.keys())

    # Formatting the header row
    header_row = (
        "| Model" + " " * (max_name_length - 5) + " | " + " | ".join(headers) + " |"
    )
    header_line_parts = ["-" * (max_name_length + 6)] + [
        "-" * max(6, len(header)) for header in headers
    ]
    header_line = "|" + "|".join(header_line_parts) + "|"

    print(header_row)
    print(header_line)
    output.write(header_row + "\n")
    output.write(header_line + "\n")

    add_plus_symbol = (
        lambda x, rounding: f"{'+' if x > 0 else ''}{(int(x) if not math.isnan(x) else x) if rounding == 0 else round(x, rounding)}"
    )
    format_value = (
        lambda header: f"{add_plus_symbol(times[header], 0):>6}%"
        if "speedup" in header
        else f"{times[header]:>6.2f}"
    )

    for op, times in times.items():
        times_str = " | ".join(format_value(header) for header in headers)

        row = f"| {op.ljust(max_name_length)} | {times_str} |"
        print(row)
        output.write(row + "\n")

    return output.getvalue()
