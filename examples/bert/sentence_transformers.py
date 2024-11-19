import mlx.core as mx
import numpy as np

from transformers import AutoConfig, AutoTokenizer
from mlx_transformers.models import BertModel as MLXBertModel


def _mean_pooling(last_hidden_state: mx.array, attention_mask: mx.array):
    token_embeddings = last_hidden_state
    input_mask_expanded = mx.expand_dims(attention_mask, -1)
    input_mask_expanded = mx.broadcast_to(
        input_mask_expanded, token_embeddings.shape
    ).astype(mx.float32)
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = mx.clip(input_mask_expanded.sum(axis=1), 1e-9, None)
    return sum_embeddings / sum_mask


sentences = ["This is an example sentence", "Each sentence is converted"]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
config = AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model = MLXBertModel(config)
model.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

inputs = tokenizer(sentences, return_tensors="np", padding=True, truncation=True)
inputs = {key: mx.array(v) for key, v in inputs.items()}

outputs = model(**inputs)

sentence_embeddings = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

print(sentence_embeddings)
