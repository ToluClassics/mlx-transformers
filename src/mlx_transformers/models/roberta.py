"""PyTorch RoBERTa model."""

import logging
import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import RobertaConfig

from .base import MlxPretrainedMixin
from .modelling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
)
from .utils import ACT2FN, get_extended_attention_mask

logger = logging.getLogger(__name__)


def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    mask = (input_ids != padding_idx).astype(mx.int16)

    incremental_indices = (
        mx.cumsum(mask, axis=1).astype(input_ids.dtype) + past_key_values_length
    ) * mask
    return incremental_indices + padding_idx


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        token_type_ids: mx.array = None,
        inputs_embeds: mx.array = None,
    ) -> mx.array:
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        input_shape[1]

        if token_type_ids is None:
            token_type_ids = mx.zeros(input_shape, dtype="int32")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_ids = token_type_ids.astype(mx.int32)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        return self.LayerNorm(embeddings)

    def create_position_ids_from_inputs_embeds(self, inputs_embeds: mx.array):
        input_shape = inputs_embeds.shape[:-1]
        seq_length = input_shape[1]

        position_ids = mx.arange(
            self.padding_idx + 1,
            seq_length + self.padding_idx + 1,
            dtype=inputs_embeds.dtype,
        )
        return mx.expand_dims(position_ids, axis=0).broadcast_to(input_shape)


class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.train = False

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        bsz, seq_len = x.shape[:2]
        return x.reshape(
            bsz, seq_len, self.num_attention_heads, self.attention_head_size
        ).transpose(0, 2, 1, 3)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mx.array]:
        mixed_query_layer = self.query(hidden_states)
        B, L, D = mixed_query_layer.shape

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = query_layer @ key_layer.transpose(0, 1, 3, 2)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in
            # RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = mx.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.train:
            attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs @ value_layer

        context_layer = context_layer.transpose(0, 2, 1, 3)

        context_layer = context_layer.reshape(B, L, -1)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.train = config.train if hasattr(config, "train") else False

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RobertaSelfAttention(
            config, position_embedding_type=position_embedding_type
        )
        self.output = RobertaSelfOutput(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mx.array]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)

        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        head_mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[Tuple[mx.array]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mx.array]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs

        return outputs


class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.train = config.train if hasattr(config, "train") else False

        self.layer = [RobertaLayer(config) for _ in range(config.num_hidden_layers)]
        self.gradient_checkpointing = False

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mx.array], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    None,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()  # TODO: fix tanh

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)

        pooled_output = mx.tanh(pooled_output)
        return pooled_output


class RobertaModel(nn.Module, MlxPretrainedMixin):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__()

        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[List], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = input_ids.shape

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = mx.ones(((batch_size, seq_length + 0)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = mx.zeros(input_shape)

        extended_attention_mask: mx.array = get_extended_attention_mask(
            attention_mask, input_shape
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = mx.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mx.array], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == mx.array):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.losses.mse_loss
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.losses.cross_entropy
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.losses.binary_cross_entropy
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForTokenClassification(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mx.array], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = nn.losses.cross_entropy
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForQuestionAnswering(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        start_positions: Optional[mx.array] = None,
        end_positions: Optional[mx.array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mx.array], QuestionAnsweringModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        splits = logits.split(2, axis=-1)
        start_logits, end_logits = splits[0], splits[1]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs,
            # we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.losses.cross_entropy
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
