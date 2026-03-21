import unittest
import torch

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.mlx_transformers.models.cache import DynamicCache
from src.mlx_transformers.models import LlamaForCausalLM as MlxLlamaForCausalLM
from src.mlx_transformers.models.llama import LlamaModel


def load_hgf_model(model_name: str) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    return model


class TestMlxLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "meta-llama/Llama-3.2-1B"
        config = LlamaConfig.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxLlamaForCausalLM(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "Who is the prince of Wales?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array

        hgf_model = load_hgf_model(self.model_name)
        hgf_inputs = self.tokenizer(
            self.input_text, return_tensors="pt", truncation=True
        )

        hgf_outputs = hgf_model(**hgf_inputs, use_cache=True)
        mlx_logits = np.array(outputs.logits)
        hgf_logits = hgf_outputs.logits.detach().float().numpy()

        self.assertTrue(
            np.allclose(
                mlx_logits,
                hgf_logits,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.array_equal(mlx_logits.argmax(axis=-1), hgf_logits.argmax(axis=-1))
        )


class TestMlxLlamaLocalBehavior(unittest.TestCase):
    def _tiny_config(self, attn_implementation: str = "eager") -> LlamaConfig:
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        config._attn_implementation = attn_implementation
        config.use_cache = True
        return config

    def test_model_requires_exactly_one_of_input_ids_or_inputs_embeds(self):
        model = LlamaModel(self._tiny_config())
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        with self.assertRaisesRegex(
            ValueError,
            "You cannot specify both input_ids and inputs_embeds at the same time",
        ):
            model(input_ids=input_ids, inputs_embeds=inputs_embeds)

        with self.assertRaisesRegex(
            ValueError, "You have to specify either input_ids or inputs_embeds"
        ):
            model()

    def test_causal_lm_accepts_inputs_embeds(self):
        model = MlxLlamaForCausalLM(self._tiny_config())
        model.eval()
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (1, 4, 64))

    def test_causal_lm_computes_loss(self):
        model = MlxLlamaForCausalLM(self._tiny_config())
        model.eval()
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        outputs = model(input_ids=input_ids, labels=labels)

        self.assertIsNotNone(outputs.loss)

    def test_sdpa_matches_eager_on_same_weights(self):
        eager_model = MlxLlamaForCausalLM(self._tiny_config("eager"))
        sdpa_model = MlxLlamaForCausalLM(self._tiny_config("sdpa"))
        sdpa_model.update(eager_model.parameters())
        eager_model.eval()
        sdpa_model.eval()

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)

        eager_outputs = eager_model(input_ids=input_ids, attention_mask=attention_mask)
        sdpa_outputs = sdpa_model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertTrue(
            np.allclose(
                np.array(eager_outputs.logits),
                np.array(sdpa_outputs.logits),
                atol=1e-4,
            )
        )

    def test_dynamic_cache_object_is_reused(self):
        model = LlamaModel(self._tiny_config())
        model.eval()
        cache = DynamicCache()

        first_input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        second_input_ids = mx.array([[5]], dtype=mx.int32)
        first_attention_mask = mx.ones_like(first_input_ids)
        second_attention_mask = mx.ones((1, 5), dtype=mx.int32)

        model(
            input_ids=first_input_ids,
            attention_mask=first_attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        self.assertEqual(cache.get_seq_length(), 4)

        model(
            input_ids=second_input_ids,
            attention_mask=second_attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        self.assertEqual(cache.get_seq_length(), 5)

    def test_generate_respects_max_length(self):
        model = MlxLlamaForCausalLM(self._tiny_config())
        model.eval()
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)

        tokens = list(
            model.generate(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                max_length=3,
                temp=0.0,
            )
        )

        self.assertEqual(len(tokens), 3)
