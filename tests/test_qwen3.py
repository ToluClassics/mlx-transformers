import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from src.mlx_transformers.models import Qwen3ForCausalLM as MlxQwen3ForCausalLM
from src.mlx_transformers.models.qwen3 import Qwen3Model
from src.mlx_transformers.models.cache import DynamicCache


class TestMlxQwen3LocalBehavior(unittest.TestCase):
    def _tiny_config(
        self,
        attn_implementation: str = "eager",
        use_sliding_window: bool = False,
    ):
        layer_type = "sliding_attention" if use_sliding_window else "full_attention"
        return SimpleNamespace(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            attention_bias=False,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_parameters=None,
            rope_scaling=None,
            rope_theta=10000.0,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            _attn_implementation=attn_implementation,
            layer_types=[layer_type, layer_type],
        )

    def test_model_requires_exactly_one_of_input_ids_or_inputs_embeds(self):
        model = Qwen3Model(self._tiny_config())
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
        model = MlxQwen3ForCausalLM(self._tiny_config())
        model.eval()
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (1, 4, 64))

    def test_causal_lm_computes_loss(self):
        model = MlxQwen3ForCausalLM(self._tiny_config())
        model.eval()
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        outputs = model(input_ids=input_ids, labels=labels)

        self.assertIsNotNone(outputs.loss)

    def test_sdpa_matches_eager_on_same_weights(self):
        eager_model = MlxQwen3ForCausalLM(self._tiny_config("eager"))
        sdpa_model = MlxQwen3ForCausalLM(self._tiny_config("sdpa"))
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
        model = Qwen3Model(self._tiny_config())
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
        model = MlxQwen3ForCausalLM(self._tiny_config())
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

    def test_sliding_window_attention_is_not_implemented(self):
        with self.assertRaisesRegex(
            NotImplementedError, "sliding-window attention is not implemented"
        ):
            MlxQwen3ForCausalLM(self._tiny_config(use_sliding_window=True))
