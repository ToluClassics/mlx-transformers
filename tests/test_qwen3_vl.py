import unittest
from types import MethodType, SimpleNamespace

import mlx.core as mx

from src.mlx_transformers.models import (
    Qwen3VLForConditionalGeneration as MlxQwen3VLForConditionalGeneration,
)
from src.mlx_transformers.models.qwen3_vl import Qwen3VLModel


class TestMlxQwen3VLLocalBehavior(unittest.TestCase):
    def _tiny_config(self):
        text_config = SimpleNamespace(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=6,
            max_position_embeddings=64,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            attention_bias=False,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 1],
            },
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
        )
        vision_config = SimpleNamespace(
            depth=2,
            hidden_size=8,
            hidden_act="gelu_pytorch_tanh",
            intermediate_size=16,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=1,
            temporal_patch_size=1,
            out_hidden_size=24,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
            initializer_range=0.02,
        )
        return SimpleNamespace(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=10,
            video_token_id=11,
            vision_start_token_id=12,
            vision_end_token_id=13,
            tie_word_embeddings=True,
        )

    def test_causal_lm_accepts_image_inputs(self):
        model = MlxQwen3VLForConditionalGeneration(self._tiny_config())
        model.eval()

        input_ids = mx.array([[1, 10, 10, 2]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        mm_token_type_ids = mx.array([[0, 1, 1, 0]], dtype=mx.int32)
        pixel_values = mx.zeros((2, 12), dtype=mx.float32)
        image_grid_thw = mx.array([[1, 1, 2]], dtype=mx.int32)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        self.assertEqual(outputs.logits.shape, (1, 4, 64))

    def test_compute_3d_position_ids_requires_mm_token_type_ids(self):
        config = self._tiny_config()
        model = Qwen3VLModel(config)
        input_ids = mx.array([[1, 10, 10, 2]], dtype=mx.int32)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_grid_thw = mx.array([[1, 1, 2]], dtype=mx.int32)

        with self.assertRaisesRegex(ValueError, "mm_token_type_ids"):
            model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
            )

    def test_placeholder_mask_detects_mismatch(self):
        config = self._tiny_config()
        model = Qwen3VLModel(config)
        input_ids = mx.array([[1, 10, 10, 2]], dtype=mx.int32)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        with self.assertRaisesRegex(
            ValueError, "Image features and image tokens do not match"
        ):
            model.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=mx.zeros((1, 24), dtype=mx.float32),
            )

    def test_loader_flattens_patch_embed_conv_weight(self):
        model = MlxQwen3VLForConditionalGeneration(self._tiny_config())
        loaded = {}

        def fake_load_weights(self, items, strict=False):
            loaded.update(dict(items))

        model.load_weights = MethodType(fake_load_weights, model)
        model._apply_pretrained_tensors(
            {
                "model.visual.patch_embed.proj.weight": mx.zeros((8, 3, 1, 2, 2)),
                "model.visual.patch_embed.proj.bias": mx.zeros((8,)),
            }
        )

        self.assertEqual(
            loaded["model.visual.patch_embed.proj.weight"].shape,
            (8, 12),
        )
