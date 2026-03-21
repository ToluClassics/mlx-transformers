import warnings
import unittest

import mlx.core as mx
import numpy as np
from transformers.utils import logging as transformers_logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
)

from src.mlx_transformers.models import BertModel as MlxBertModel
from src.mlx_transformers.models import (
    BertForMaskedLM as MlxBertForMaskedLM,
)
from src.mlx_transformers.models import (
    BertForQuestionAnswering as MlxBertForQuestionAnswering,
)
from src.mlx_transformers.models import (
    BertForSequenceClassification as MlxBertForSequenceClassification,
)
from src.mlx_transformers.models import (
    BertForTokenClassification as MlxBertForTokenClassification,
)
from src.mlx_transformers.models.bert import BertEmbeddings, BertSelfOutput


def load_hgf_model(model_name: str, hgf_model_class):
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        model = hgf_model_class.from_pretrained(model_name)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
    return model


def load_tokenizer(tokenizer_class, model_name: str):
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*WordPiece\.__init__ will not create from files anymore.*",
            category=DeprecationWarning,
        )
        try:
            return tokenizer_class.from_pretrained(model_name)
        finally:
            transformers_logging.set_verbosity(previous_verbosity)


class TestMlxBert(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        cls.config = BertConfig.from_pretrained(cls.model_name)
        cls.tokenizer = load_tokenizer(BertTokenizer, cls.model_name)
        cls.hgf_model_class = BertModel

        cls.model = MlxBertModel(cls.config)
        cls.model.from_pretrained(cls.model_name, revision="main")

        cls.input_text = "Hello, my dog is cute"

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)
        outputs_mlx = np.array(outputs_mlx.last_hidden_state)

        inputs_hgf = self.tokenizer(
            self.input_text, return_tensors="pt", padding=True, truncation=True
        )
        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.last_hidden_state.detach().numpy()

        self.assertTrue(np.allclose(outputs_mlx, outputs_hgf, atol=1e-4))


class TestMlxBertForSequenceClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "textattack/bert-base-uncased-yelp-polarity"
        cls.config = AutoConfig.from_pretrained(cls.model_name)
        cls.tokenizer = load_tokenizer(AutoTokenizer, cls.model_name)

        cls.hgf_model_class = BertForSequenceClassification
        cls.model = MlxBertForSequenceClassification(cls.config)
        cls.model.from_pretrained(cls.model_name, revision="refs/pr/1")

        cls.input_text = "Hello, my dog is cute"

    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.logits, mx.array)

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}

        outputs_mlx = self.model(**inputs_mlx)
        outputs_mlx = np.array(outputs_mlx.logits)
        predicted_class_id = outputs_mlx.argmax().item()
        mlx_label = self.model.config.id2label[predicted_class_id]

        inputs_hgf = self.tokenizer(
            self.input_text, return_tensors="pt", padding=True, truncation=True
        )
        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)

        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.logits

        predicted_class_id = outputs_hgf.argmax().item()
        hgf_label = hgf_model.config.id2label[predicted_class_id]

        self.assertEqual(mlx_label, hgf_label)


class TestMlxBertForTokenClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "dslim/bert-base-NER"
        cls.config = AutoConfig.from_pretrained(cls.model_name)
        cls.tokenizer = load_tokenizer(AutoTokenizer, cls.model_name)

        cls.hgf_model_class = BertForTokenClassification
        cls.model = MlxBertForTokenClassification(cls.config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "HuggingFace is a company based in Paris and New York"

    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_text,
            return_tensors="np",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.logits, mx.array)

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_text,
            return_tensors="np",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)
        outputs_mlx = np.array(outputs_mlx.logits)
        mlx_predicted_token_class_ids = outputs_mlx.argmax(-1)
        mlx_predicted_tokens_classes = [
            self.model.config.id2label[t.item()]
            for t in mlx_predicted_token_class_ids[0]
        ]

        inputs_hgf = self.tokenizer(
            self.input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.logits

        hgf_predicted_token_class_ids = outputs_hgf.argmax(-1)
        hgf_predicted_tokens_classes = [
            hgf_model.config.id2label[t.item()]
            for t in hgf_predicted_token_class_ids[0]
        ]

        self.assertEqual(mlx_predicted_tokens_classes, hgf_predicted_tokens_classes)
        self.assertTrue(
            np.allclose(np.array(outputs_mlx), outputs_hgf.detach().numpy(), atol=1e-4)
        )


class TestMlxBertForQuestionAnswering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "deepset/bert-base-cased-squad2"
        cls.tokenizer = load_tokenizer(AutoTokenizer, cls.model_name)
        cls.config = AutoConfig.from_pretrained(cls.model_name)

        cls.hgf_model_class = BertForQuestionAnswering
        cls.model = MlxBertForQuestionAnswering(cls.config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_question = "Who was Jim Henson?"
        cls.input_text = "Jim Henson was a nice puppet"

    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_question, self.input_text, return_tensors="np"
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.start_logits, mx.array)
        self.assertIsInstance(outputs.end_logits, mx.array)

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_question, self.input_text, return_tensors="np"
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)

        mlx_answer_start_index = outputs_mlx.start_logits.argmax().item()
        mlx_answer_end_index = outputs_mlx.end_logits.argmax().item()
        mlx_predict_answer_tokens = inputs_mlx["input_ids"].tolist()
        mlx_predict_answer_tokens = mlx_predict_answer_tokens[0][
            mlx_answer_start_index : mlx_answer_end_index + 1
        ]
        mlx_answer = self.tokenizer.decode(
            mlx_predict_answer_tokens, skip_special_tokens=True
        )

        inputs_hgf = self.tokenizer(
            self.input_question, self.input_text, return_tensors="pt"
        )

        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)

        hgf_answer_start_index = outputs_hgf.start_logits.argmax()
        hgf_answer_end_index = outputs_hgf.end_logits.argmax()
        hgf_predict_answer_tokens = inputs_hgf.input_ids[
            0, hgf_answer_start_index : hgf_answer_end_index + 1
        ]
        hgf_answer = self.tokenizer.decode(
            hgf_predict_answer_tokens, skip_special_tokens=True
        )

        self.assertEqual(mlx_answer, hgf_answer)
        self.assertTrue(
            np.allclose(
                np.array(outputs_mlx.start_logits),
                outputs_hgf.start_logits.detach().numpy(),
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                np.array(outputs_mlx.end_logits),
                outputs_hgf.end_logits.detach().numpy(),
                atol=1e-4,
            )
        )


class TestMlxBertLocalBehavior(unittest.TestCase):
    def test_model_rejects_decoder_mode(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            is_decoder=True,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "Decoder mode is not implemented"
        ):
            MlxBertModel(config)

    def test_model_rejects_cross_attention_config(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            add_cross_attention=True,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "Cross-attention is not implemented"
        ):
            MlxBertModel(config)

    def test_model_requires_exactly_one_of_input_ids_or_inputs_embeds(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        model = MlxBertModel(config)
        model.eval()

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

    def test_model_accepts_inputs_embeds(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        model = MlxBertModel(config)
        model.eval()

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 4, 8))

    def test_model_rejects_use_cache(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        model = MlxBertModel(config)
        model.eval()

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        with self.assertRaisesRegex(
            NotImplementedError, "use_cache is not implemented"
        ):
            model(input_ids=input_ids, use_cache=True)

    def test_layer_rejects_cross_attention_inputs(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        layer = MlxBertModel(config).encoder.layer[0]
        hidden_states = mx.ones((1, 4, 8))
        encoder_hidden_states = mx.ones((1, 4, 8))

        with self.assertRaisesRegex(
            NotImplementedError, "Cross-attention is not implemented"
        ):
            layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

    def test_layer_rejects_past_key_value(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        layer = MlxBertModel(config).encoder.layer[0]
        hidden_states = mx.ones((1, 4, 8))
        past_key_value = ((mx.ones((1, 2, 4, 4)), mx.ones((1, 2, 4, 4))),)

        with self.assertRaisesRegex(NotImplementedError, "KV cache is not implemented"):
            layer(hidden_states, past_key_value=past_key_value)

    def test_embeddings_dropout_respects_train_eval_mode(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.5,
        )
        embeddings = BertEmbeddings(config)
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 16, dtype=mx.int32)

        embeddings.train()
        train_output = np.array(embeddings(input_ids=input_ids, position_ids=None))

        embeddings.eval()
        eval_output = np.array(embeddings(input_ids=input_ids, position_ids=None))

        self.assertFalse(np.allclose(train_output, eval_output))

    def test_self_output_dropout_respects_train_eval_mode(self):
        config = BertConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.5,
        )
        layer = BertSelfOutput(config)
        hidden_states = mx.ones((16, 8, 8))
        input_tensor = mx.zeros((16, 8, 8))

        layer.train()
        train_output = np.array(layer(hidden_states, input_tensor))

        layer.eval()
        eval_output = np.array(layer(hidden_states, input_tensor))

        self.assertFalse(np.allclose(train_output, eval_output))

    def test_sequence_classification_multilabel_loss_uses_logits(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=2,
        )
        config.problem_type = "multi_label_classification"
        model = MlxBertForSequenceClassification(config)
        model.eval()

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        labels = mx.array([[1.0, 0.0]])

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        logits = outputs.logits
        expected = mx.mean(
            mx.maximum(logits, 0.0)
            - logits * labels
            + mx.log1p(mx.exp(-mx.abs(logits)))
        )

        self.assertIsNotNone(outputs.loss)
        self.assertTrue(np.allclose(np.array(outputs.loss), np.array(expected)))

    def test_sequence_classification_accepts_inputs_embeds(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=2,
        )
        model = MlxBertForSequenceClassification(config)
        model.eval()

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        inputs_embeds = model.bert.get_input_embeddings()(input_ids)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (1, 2))

    def test_masked_lm_rejects_use_cache(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        masked_lm = MlxBertForMaskedLM(config)
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        with self.assertRaisesRegex(
            NotImplementedError, "use_cache is not implemented for BertForMaskedLM"
        ):
            masked_lm(input_ids=input_ids, use_cache=True)

    def test_question_answering_loss_path_runs_with_labels(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=2,
        )
        model = MlxBertForQuestionAnswering(config)
        model.eval()

        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        start_positions = mx.array([1], dtype=mx.int32)
        end_positions = mx.array([2], dtype=mx.int32)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.start_logits.shape, (1, 4))
        self.assertEqual(outputs.end_logits.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
