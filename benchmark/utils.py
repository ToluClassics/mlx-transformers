import io
import math
import random
import numpy as np
from collections import defaultdict
from datasets import load_dataset



def get_dataset(dataset_name="castorini/wura"):
    """
    Load the dataset from Hugging Face
    """
    dataset = load_dataset(dataset_name, "eng", level="passage", verification_mode="no_checks")
    return dataset

def generate_inputs(char_length, batch_size, dataset=get_dataset()):
        inputs = []
        for i in range(batch_size):
            input_text = dataset["train"][i]['text']
            input_text = input_text[:char_length]  
            if len(input_text) < char_length:
                input_text = input_text.ljust(char_length)  
            inputs.append(input_text)
        return inputs

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

    if "mlx_cpu" in backends and "torch_cpu" in backends:
        h = "mlx_cpu/torch_cpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_cpu"], compared_to=v["torch_cpu"])

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
