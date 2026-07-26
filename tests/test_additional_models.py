import unittest
from types import SimpleNamespace

import mlx.core as mx
from transformers import (
    FuyuConfig,
    M2M100Config,
    PersimmonConfig,
    RobertaConfig,
    XLMRobertaConfig,
)

from src.mlx_transformers.models import (
    FuyuForCausalLM,
    M2M100ForConditionalGeneration,
    OpenELMForCausalLM,
    PersimmonForCausalLM,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
)


def tiny_persimmon_config():
    return PersimmonConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def tiny_openelm_config():
    return SimpleNamespace(
        activation_fn_name="silu",
        ffn_dim_divisor=8,
        ffn_multipliers=[2.0, 2.0],
        ffn_with_glu=True,
        head_dim=4,
        max_context_length=32,
        model_dim=16,
        normalize_qk_projections=True,
        num_kv_heads=[2, 2],
        num_query_heads=[4, 4],
        num_transformer_layers=2,
        output_attentions=False,
        output_hidden_states=False,
        pad_token_id=0,
        eos_token_id=2,
        rope_freq_constant=10000,
        rope_max_length=32,
        share_input_output_layers=False,
        use_cache=True,
        use_return_dict=True,
        vocab_size=64,
    )


class TestAdditionalCausalFamilies(unittest.TestCase):
    def _assert_finite_batched_generation(self, model):
        model.eval()
        inputs = {
            "input_ids": mx.array([[1, 3, 4], [0, 5, 6]], dtype=mx.int32),
            "attention_mask": mx.array([[1, 1, 1], [0, 1, 1]], dtype=mx.int32),
        }

        tokens = list(
            model.generate(
                inputs,
                max_new_tokens=2,
                temp=0.0,
                eos_token_id=[],
            )
        )

        self.assertEqual(len(tokens), 2)
        self.assertTrue(all(token.shape == (2,) for token in tokens))
        self.assertEqual(inputs["input_ids"].shape, (2, 3))
        self.assertEqual(inputs["attention_mask"].shape, (2, 3))

    def test_openelm_generation_contract(self):
        self._assert_finite_batched_generation(
            OpenELMForCausalLM(tiny_openelm_config())
        )

    def test_persimmon_generation_contract(self):
        self._assert_finite_batched_generation(
            PersimmonForCausalLM(tiny_persimmon_config())
        )

    def test_fuyu_image_forward_and_generation_contract(self):
        text_config = tiny_persimmon_config()
        config = FuyuConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            image_size=4,
            patch_size=2,
            num_channels=3,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            image_token_id=10,
            text_config=text_config.to_dict(),
        )
        model = FuyuForCausalLM(config)
        model.eval()
        inputs = {
            "input_ids": mx.array([[1, 10, 3]], dtype=mx.int32),
            "attention_mask": mx.ones((1, 3), dtype=mx.int32),
            "image_patches": mx.ones((1, 1, 1, 12), dtype=mx.float32),
            "image_patches_indices": mx.array([[-1, 0, -1]], dtype=mx.int32),
        }

        output = model(**inputs, use_cache=True)
        self.assertEqual(output.logits.shape, (1, 3, 64))

        tokens = list(
            model.generate(
                inputs,
                max_new_tokens=2,
                temp=0.0,
                eos_token_id=[],
            )
        )
        self.assertEqual(len(tokens), 2)
        self.assertTrue(all(token.shape == (1,) for token in tokens))
        self.assertEqual(inputs["input_ids"].shape, (1, 3))


class TestM2M100Family(unittest.TestCase):
    def test_encoder_decoder_translation_path(self):
        config = M2M100Config(
            vocab_size=64,
            max_position_embeddings=32,
            encoder_layers=2,
            encoder_ffn_dim=32,
            encoder_attention_heads=4,
            decoder_layers=2,
            decoder_ffn_dim=32,
            decoder_attention_heads=4,
            d_model=16,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
        )
        model = M2M100ForConditionalGeneration(config)
        model.eval()
        input_ids = mx.array([[3, 4, 5], [6, 7, 1]], dtype=mx.int32)
        attention_mask = mx.array([[1, 1, 1], [1, 1, 0]], dtype=mx.int32)
        decoder_input_ids = mx.array([[2, 8], [2, 9]], dtype=mx.int32)
        decoder_attention_mask = mx.ones_like(decoder_input_ids)

        encoder_output = model.encode(input_ids, attention_mask)
        decoder_output = model.decode(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_output,
            attention_mask,
            None,
        )
        logits = model.lm_head(decoder_output)
        mx.eval(logits)

        self.assertEqual(encoder_output.shape, (2, 3, 16))
        self.assertEqual(decoder_output.shape, (2, 2, 16))
        self.assertEqual(logits.shape, (2, 2, 64))


class TestRobertaFamilies(unittest.TestCase):
    def _assert_family(
        self,
        config_class,
        model_class,
        sequence_class,
        token_class,
        question_answering_class,
    ):
        config = config_class(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            num_labels=3,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        input_ids = mx.array([[0, 5, 6, 2], [0, 7, 8, 2]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)

        base_output = model_class(config)(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = sequence_class(config)(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_output = token_class(config)(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        qa_config_dict = config.to_dict()
        qa_config_dict.pop("id2label", None)
        qa_config_dict.pop("label2id", None)
        qa_config_dict["num_labels"] = 2
        qa_config = config_class(**qa_config_dict)
        qa_output = question_answering_class(qa_config)(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        mx.eval(
            base_output.last_hidden_state,
            sequence_output.logits,
            token_output.logits,
            qa_output.start_logits,
            qa_output.end_logits,
        )

        self.assertEqual(base_output.last_hidden_state.shape, (2, 4, 16))
        self.assertEqual(sequence_output.logits.shape, (2, 3))
        self.assertEqual(token_output.logits.shape, (2, 4, 3))
        self.assertEqual(qa_output.start_logits.shape, (2, 4))
        self.assertEqual(qa_output.end_logits.shape, (2, 4))

    def test_roberta_exported_heads(self):
        self._assert_family(
            RobertaConfig,
            RobertaModel,
            RobertaForSequenceClassification,
            RobertaForTokenClassification,
            RobertaForQuestionAnswering,
        )

    def test_xlm_roberta_exported_heads(self):
        self._assert_family(
            XLMRobertaConfig,
            XLMRobertaModel,
            XLMRobertaForSequenceClassification,
            XLMRobertaForTokenClassification,
            XLMRobertaForQuestionAnswering,
        )
