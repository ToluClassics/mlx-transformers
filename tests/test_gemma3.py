import unittest
from types import MethodType, SimpleNamespace

import mlx.core as mx

from src.mlx_transformers.models import (
    Gemma3ForConditionalGeneration as MlxGemma3ForConditionalGeneration,
)
from src.mlx_transformers.models.gemma3 import Gemma3Model


class TestMlxGemma3LocalBehavior(unittest.TestCase):
    def _tiny_config(self):
        text_config = SimpleNamespace(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=6,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
            attention_bias=False,
            attention_dropout=0.0,
            query_pre_attn_scalar=6,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
            final_logit_softcapping=None,
            attn_logit_softcapping=None,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
        )
        vision_config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_channels=3,
            image_size=4,
            patch_size=2,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
        )
        return SimpleNamespace(
            text_config=text_config,
            vision_config=vision_config,
            mm_tokens_per_image=4,
            image_token_index=99,
            tie_word_embeddings=True,
        )

    def test_causal_lm_accepts_image_inputs(self):
        model = MlxGemma3ForConditionalGeneration(self._tiny_config())
        model.eval()

        input_ids = mx.array([[1, 99, 99, 99, 99, 2]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        token_type_ids = mx.array([[0, 1, 1, 1, 1, 0]], dtype=mx.int32)
        pixel_values = mx.zeros((1, 3, 4, 4), dtype=mx.float32)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        self.assertEqual(outputs.logits.shape, (1, 6, 64))

    def test_placeholder_mask_detects_mismatch(self):
        config = self._tiny_config()
        model = Gemma3Model(config)
        input_ids = mx.array([[1, 99, 99, 2]], dtype=mx.int32)
        inputs_embeds = model.get_input_embeddings()(
            mx.array([[1, 0, 0, 2]], dtype=mx.int32)
        )

        with self.assertRaisesRegex(
            ValueError, "Image features and image tokens do not match"
        ):
            model.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=mx.zeros((1, 4, 24), dtype=mx.float32),
            )

    def test_loader_flattens_patch_embed_conv_weight(self):
        model = MlxGemma3ForConditionalGeneration(self._tiny_config())
        loaded = {}
        patch_embed_weight_key = (
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight"
        )
        patch_embed_bias_key = (
            "model.vision_tower.vision_model.embeddings.patch_embedding.bias"
        )

        def fake_load_weights(self, items, strict=False):
            loaded.update(dict(items))

        model.load_weights = MethodType(fake_load_weights, model)
        model._apply_pretrained_tensors(
            {
                patch_embed_weight_key: mx.zeros((8, 3, 2, 2)),
                patch_embed_bias_key: mx.zeros((8,)),
            }
        )

        self.assertEqual(
            loaded[patch_embed_weight_key].shape,
            (8, 12),
        )
