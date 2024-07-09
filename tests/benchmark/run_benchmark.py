import time
import numpy as np
import mlx.core as mx
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForMaskedLM,
    RobertaModel,
    XLMRobertaModel,
)

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.mlx_transformers.models import BertForMaskedLM as MlxBertForMaskedLM
from src.mlx_transformers.models import RobertaModel as MlxRobertaModel
from src.mlx_transformers.models import XLMRobertaModel as MlxXLMRobertaModel

class Model:
    def __init__(self, model_name: str, model_class, tokenizer_class, config_class, custom_model_class, input_text: str):
        self.model_name = model_name
        self.config = config_class.from_pretrained(self.model_name)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name)
        self.hgf_model_class = model_class
        self.mlx_model_class = custom_model_class
        self.input_text = input_text

    def load_hgf_model(self):
        return self.hgf_model_class.from_pretrained(self.model_name)
    
    def load_mlx_model(self):
        mlx_model = self.mlx_model_class(self.config)
        mlx_model.from_pretrained(self.model_name)
        return mlx_model
    
    def prepare_mlx_model_input(self):
        inputs_mlx = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )
        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        return inputs_mlx
    
    def prepare_hgf_model_input(self):
        inputs_hgf = self.tokenizer(
            self.input_text, return_tensors="pt", padding=True, truncation=True
        )
        return inputs_hgf
    
    def get_mlx_model_inference(self, model, inputs_mlx):
        return model(**inputs_mlx)

    def get_hgf_model_inference(self, model, inputs_hgf):
        return model(**inputs_hgf)

class Benchmark:
    def __init__(self, models, num_runs=5):
        self.models = models
        self.num_runs = num_runs

    def measure_inference_time(self, model, inputs, inference_func):
        times = []
        for _ in range(self.num_runs):
            start_time = time.time()
            _ = inference_func(model, inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        return times

    def run_benchmark(self):
        results = []
        detailed_results = []
        for model_info in self.models:
            print(f"Running benchmark for {model_info['name']}")

            # Load models and inputs
            model_instance = Model(
                model_name=model_info['name'],
                model_class=model_info['hgf_class'],
                tokenizer_class=model_info['tokenizer_class'],
                config_class=model_info['config_class'],
                custom_model_class=model_info['mlx_class'],
                input_text=model_info['input_text']
            )

            hgf_model = model_instance.load_hgf_model()
            mlx_model = model_instance.load_mlx_model()

            inputs_hgf = model_instance.prepare_hgf_model_input()
            inputs_mlx = model_instance.prepare_mlx_model_input()

            # Measure inference times
            hgf_inference_times = self.measure_inference_time(hgf_model, inputs_hgf, model_instance.get_hgf_model_inference)
            mlx_inference_times = self.measure_inference_time(mlx_model, inputs_mlx, model_instance.get_mlx_model_inference)

            hgf_average_time = np.mean(hgf_inference_times)
            mlx_average_time = np.mean(mlx_inference_times)

            result = {
                'model': model_info['name'],
                'hgf_inference_time': hgf_average_time,
                'mlx_inference_time': mlx_average_time
            }

            detailed_result = {
                'model': model_info['name'],
                'hgf_inference_times': hgf_inference_times,
                'mlx_inference_times': mlx_inference_times
            }

            results.append(result)
            detailed_results.append(detailed_result)
            self.print_results(result)

        self.save_results(results, detailed_results)

    def print_results(self, result):
        print(f"Model: {result['model']}")
        print(f"Torch Inference Time: {result['hgf_inference_time']:.6f} seconds")
        print(f"MLX Inference Time: {result['mlx_inference_time']:.6f} seconds")
        print("-" * 50)

    def save_results(self, results, detailed_results):
        with open("benchmark_results.txt", "w") as f:
            f.write("Average Inference Times\n")
            f.write("=" * 50 + "\n")
            for result in results:
                f.write(f"Model: {result['model']}\n")
                f.write(f"Torch Inference Time: {result['hgf_inference_time']:.6f} seconds\n")
                f.write(f"MLX Inference Time: {result['mlx_inference_time']:.6f} seconds\n")
                f.write("-" * 50 + "\n")

            f.write("\nDetailed Inference Times\n")
            f.write("=" * 50 + "\n")
            for detailed_result in detailed_results:
                f.write(f"Model: {detailed_result['model']}\n")
                f.write("Torch Inference Times:\n")
                for i, time in enumerate(detailed_result['hgf_inference_times']):
                    f.write(f"  Run {i+1}: {time:.6f} seconds\n")
                f.write("MLX Inference Times:\n")
                for i, time in enumerate(detailed_result['mlx_inference_times']):
                    f.write(f"  Run {i+1}: {time:.6f} seconds\n")
                f.write("-" * 50 + "\n")

if __name__ == "__main__":
    models = [
        {
            'name': 'bert-base-uncased',
            'hgf_class': BertForMaskedLM,
            'tokenizer_class': AutoTokenizer,
            'config_class': AutoConfig,
            'mlx_class': MlxBertForMaskedLM,
            'input_text': 'This is a sample input text for BERT model.'
        },
        {
            'name': 'roberta-base',
            'hgf_class': RobertaModel,
            'tokenizer_class': AutoTokenizer,
            'config_class': AutoConfig,
            'mlx_class': MlxRobertaModel,
            'input_text': 'This is a sample input text for RoBERTa model.'
        },
        {
            'name': 'xlm-roberta-base',
            'hgf_class': XLMRobertaModel,
            'tokenizer_class': AutoTokenizer,
            'config_class': AutoConfig,
            'mlx_class': MlxXLMRobertaModel,
            'input_text': 'This is a sample input text for XLM-RoBERTa model.'
        }
    ]

    benchmark = Benchmark(models, num_runs=10)  # Set num_runs to 10 for this example
    benchmark.run_benchmark()
