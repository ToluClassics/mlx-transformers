import tempfile
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from src.mlx_transformers.lora import (
    LoraConfig,
    LoRALinear,
    apply_lora,
    has_lora_layers,
    load_lora_adapters,
    save_lora_adapters,
)


class TinySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinySelfAttention()
        self.mlp = nn.Linear(4, 4, bias=False)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [TinyLayer(), TinyLayer()]

    def __call__(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer.self_attn.o_proj(
                layer.self_attn.q_proj(hidden) + layer.self_attn.v_proj(hidden)
            )
            hidden = layer.mlp(hidden)
        return hidden


class TinyLoraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = TinyBackbone()

    def __call__(self, x):
        return self.model(x)


class TestLora(unittest.TestCase):
    def test_apply_lora_replaces_last_layers(self):
        model = TinyLoraModel()

        apply_lora(model, LoraConfig(num_layers=1))

        self.assertFalse(isinstance(model.model.layers[0].self_attn.q_proj, LoRALinear))
        self.assertTrue(isinstance(model.model.layers[1].self_attn.q_proj, LoRALinear))
        self.assertTrue(isinstance(model.model.layers[1].self_attn.v_proj, LoRALinear))
        self.assertFalse(isinstance(model.model.layers[1].self_attn.o_proj, LoRALinear))
        self.assertTrue(has_lora_layers(model))

    def test_save_and_load_lora_adapters(self):
        model = TinyLoraModel()
        config = LoraConfig(num_layers=1, r=4, alpha=8.0)
        apply_lora(model, config)

        lora_module = model.model.layers[1].self_attn.q_proj
        lora_module.lora_a = mx.ones_like(lora_module.lora_a)
        lora_module.lora_b = mx.ones_like(lora_module.lora_b)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_lora_adapters(model, tmpdir, config)

            reloaded = TinyLoraModel()
            load_lora_adapters(reloaded, tmpdir)

            reloaded_module = reloaded.model.layers[1].self_attn.q_proj
            self.assertTrue(isinstance(reloaded_module, LoRALinear))
            self.assertTrue(
                np.array_equal(
                    np.array(reloaded_module.lora_a), np.array(lora_module.lora_a)
                )
            )
            self.assertTrue(
                np.array_equal(
                    np.array(reloaded_module.lora_b), np.array(lora_module.lora_b)
                )
            )


if __name__ == "__main__":
    unittest.main()
