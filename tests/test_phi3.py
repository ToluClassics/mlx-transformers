import unittest

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Phi3Config

from src.mlx_transformers.models import Phi3Model as MlxPhi3Model
from src.mlx_transformers.models import Phi3ForCausalLM as MlxPhi3ForCausalLM


def load_hgf_model(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


class TestMlxPhi3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "microsoft/Phi-3-mini-4k-instruct"
        config = AutoConfig.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPhi3ForCausalLM(config)
        cls.model.from_pretrained(
            cls.model_name,
            trust_remote_code=True,
        )

        cls.input_text = "Hey, are you conscious? Can you talk to me?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array


class TestMlxPhi3LocalBehavior(unittest.TestCase):
    def _tiny_config(self, attn_implementation: str = "eager"):
        config = Phi3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            original_max_position_embeddings=32,
            rope_theta=10000.0,
            rope_scaling=None,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = attn_implementation
        config.use_cache = True
        return config

    def test_model_requires_exactly_one_of_input_ids_or_inputs_embeds(self):
        model = MlxPhi3Model(self._tiny_config())
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

    def test_causal_lm_computes_loss(self):
        model = MlxPhi3ForCausalLM(self._tiny_config())
        model.eval()
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        outputs = model(input_ids=input_ids, labels=labels)

        self.assertIsNotNone(outputs.loss)

    def test_sdpa_matches_eager_on_same_weights(self):
        eager_model = MlxPhi3ForCausalLM(self._tiny_config("eager"))
        sdpa_model = MlxPhi3ForCausalLM(self._tiny_config("sdpa"))
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

    def test_generate_respects_max_length(self):
        model = MlxPhi3ForCausalLM(self._tiny_config())
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
