import time
import numpy as np
import torch
import mlx.core as mx
import argparse
from collections import defaultdict
from utils import *
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaModel,
)

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mlx_transformers.models import BertForMaskedLM as MlxBertForMaskedLM
from src.mlx_transformers.models import RobertaModel as MlxRobertaModel
from src.mlx_transformers.models import XLMRobertaModel as MlxXLMRobertaModel


class Model:
    def __init__(self, model_name: str, model_class, tokenizer_class, config_class, custom_model_class):
        self.model_name = model_name
        self.config = config_class.from_pretrained(self.model_name)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name)
        self.hgf_model_class = model_class
        self.mlx_model_class = custom_model_class

    def load_hgf_model(self, device):
        return self.hgf_model_class.from_pretrained(self.model_name).to(device)
    
    def load_mlx_model(self):
        mlx_model = self.mlx_model_class(self.config)
        mlx_model.from_pretrained(self.model_name)
        return mlx_model
    
    def prepare_mlx_model_input(self, input_text):
        inputs_mlx = self.tokenizer(
            input_text, return_tensors="np", padding=True, truncation=True
        )
        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        return inputs_mlx
    
    def prepare_hgf_model_input(self, input_text, device):
        inputs_hgf = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        inputs_hgf = {key: v.to(device) for key, v in inputs_hgf.items()}
        return inputs_hgf
    
    def get_mlx_model_inference(self, model, inputs_mlx):
        return model(**inputs_mlx)

    def get_hgf_model_inference(self, model, inputs_hgf):
        return model(**inputs_hgf)


class Benchmark:
    def __init__(self, models, backends, corpus=corpus,  num_runs=5, input_lengths=[50, 100, 200, 500, 1000]):
        self.models = models
        self.backends = backends
        self.corpus=corpus
        self.num_runs = num_runs
        self.input_lengths = input_lengths

    def measure_inference_time(self, model, inputs, inference_func, backend):
        times = []
        for _ in range(self.num_runs):
            start_time = time.time()
            _ = inference_func(model, inputs)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        if backend in ["cuda", "mps"]:
            torch.cuda.empty_cache()
        
        return times

    def run_benchmark(self):
        detailed_results = []
        average_results = []

        for model_info in self.models:
            model_instance = Model(
                model_name=model_info['name'],
                model_class=model_info['hgf_class'],
                tokenizer_class=model_info['tokenizer_class'],
                config_class=model_info['config_class'],
                custom_model_class=model_info['mlx_class']
            )

            mlx_model = model_instance.load_mlx_model()
            for input_length in self.input_lengths:
                input_text = get_input_text(self.corpus, input_length)

                for backend in self.backends:
                    if backend == "mlx_cpu":
                        mx.set_default_device(mx.cpu)
                        inputs_mlx = model_instance.prepare_mlx_model_input(input_text)
                        mlx_inference_times = self.measure_inference_time(mlx_model, inputs_mlx, model_instance.get_mlx_model_inference, backend)
                    elif backend == "mlx_gpu":
                        mx.set_default_device(mx.gpu)
                        inputs_mlx = model_instance.prepare_mlx_model_input(input_text)
                        mlx_inference_times = self.measure_inference_time(mlx_model, inputs_mlx, model_instance.get_mlx_model_inference, backend)
                    elif backend == "mlx_gpu_compile":
                        mx.set_default(mx.gpu)
                        mx.compile()
                        inputs_mlx = model_instance.prepare_mlx_model_input(input_text)
                        mlx_model.compile()
                        mlx_inference_times = self.measure_inference_time(mlx_model, inputs_mlx, model_instance.get_mlx_model_inference, backend)
                    elif backend == "torch_cpu":
                        device = torch.device("cpu")
                        hgf_model = model_instance.load_hgf_model(device)
                        inputs_hgf = model_instance.prepare_hgf_model_input(input_text, device)
                        hgf_inference_times = self.measure_inference_time(hgf_model, inputs_hgf, model_instance.get_hgf_model_inference, backend)
                    elif backend == "torch_cuda":
                        if torch.cuda.is_available():
                            device = torch.device("cuda")
                            hgf_model = model_instance.load_hgf_model(device)
                            inputs_hgf = model_instance.prepare_hgf_model_input(input_text, device)
                            hgf_inference_times = self.measure_inference_time(hgf_model, inputs_hgf, model_instance.get_hgf_model_inference, backend)
                        else:
                            hgf_inference_times = [float('nan')] * self.num_runs
                    elif backend == "torch_mps":
                        if torch.backends.mps.is_available():
                            device = torch.device("mps")
                            hgf_model = model_instance.load_hgf_model(device)
                            inputs_hgf = model_instance.prepare_hgf_model_input(input_text, device)
                            hgf_inference_times = self.measure_inference_time(hgf_model, inputs_hgf, model_instance.get_hgf_model_inference, backend)
                        else:
                            hgf_inference_times = [float('nan')] * self.num_runs

                    result = {
                        'model': model_info['name'],
                        'backend': backend,
                        'input_length': input_length,
                        'average_time': np.mean(hgf_inference_times if 'torch' in backend else mlx_inference_times)
                    }

                    detailed_results.append(result)
        
        # Calculate average results
        for model_info in self.models:
            for backend in self.backends:
                average_time = np.mean([
                    res['average_time'] for res in detailed_results
                    if res['model'] == model_info['name'] and res['backend'] == backend
                ])
                average_results.append({
                    'model': model_info['name'],
                    'backend': backend,
                    'average_time': average_time
                })
        
        self.save_results(detailed_results, average_results)

    def save_results(self, detailed_results, average_results):
        backends = list(set(result['backend'] for result in detailed_results))
        
        detailed_times = defaultdict(dict)
        average_times = defaultdict(dict)

        for result in detailed_results:
            model_info = f"{result['model']} / inputs_char_no={result['input_length']}"
            detailed_times[model_info][result['backend']] = result['average_time']

        for result in average_results:
            model_info = result['model']
            average_times[model_info][result['backend']] = result['average_time']

        with open("benchmark/benchmark_results.md", "w") as f:
            f.write("## Detailed Benchmark\n")
            f.write("Detailed runtime benchmark of model inferences, measured in milliseconds.\n\n")
            
            f.write("### Detailed Results\n")
            output = print_benchmark(detailed_times, backends)
            f.write(output)

            f.write("\n## Average Benchmark\n")
            f.write("Average runtime benchmark of model inferences, measured in milliseconds.\n\n")
            
            f.write("### Average Results\n")
            output = print_benchmark(average_times, backends)
            f.write(output)


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inferences on different backends.")
    parser.add_argument("--backends", nargs="+", default=["mlx_cpu", "mlx_gpu", "torch_cpu"],
                        help="List of backends to benchmark on. E.g., --backends mlx_cpu mlx_gpu torch_cpu torch_cuda torch_mps")
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs for each benchmark')
    args = parser.parse_args()


    if "torch_mps" in args.backends:
        assert torch.backends.mps.is_available(), "MPS backend not available."
    if "torch_cuda" in args.backends:
        assert torch.cuda.is_available(), "CUDA device not found."


    models = [
        {
            'name': 'bert-base-uncased',
            'hgf_class': BertForMaskedLM,
            'tokenizer_class': BertTokenizer,
            'config_class': BertConfig,
            'mlx_class': MlxBertForMaskedLM,
        },
        {
            'name': 'roberta-base',
            'hgf_class': RobertaModel,
            'tokenizer_class': RobertaTokenizer,
            'config_class': RobertaConfig,
            'mlx_class': MlxRobertaModel,
        },
        {
            'name': 'xlm-roberta-base',
            'hgf_class': XLMRobertaModel,
            'tokenizer_class': XLMRobertaTokenizer,
            'config_class': XLMRobertaConfig,
            'mlx_class': MlxXLMRobertaModel,
        }
    ]

    benchmark = Benchmark(models=models, backends=args.backends, num_runs=args.num_runs)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
