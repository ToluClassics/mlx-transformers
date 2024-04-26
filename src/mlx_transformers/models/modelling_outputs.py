import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    cross_attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions:
    last_hidden_state: mx.array = None
    pooler_output: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    cross_attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class SequenceClassifierOutput:
    loss: Optional[mx.array] = None
    logits: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class SequenceClassifierOutputWithPast:
    loss: Optional[mx.array] = None
    logits: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class Seq2SeqSequenceClassifierOutput:
    loss: Optional[mx.array] = None
    logits: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    decoder_hidden_states: Optional[Tuple[mx.array, ...]] = None
    decoder_attentions: Optional[Tuple[mx.array, ...]] = None
    cross_attentions: Optional[Tuple[mx.array, ...]] = None
    encoder_last_hidden_state: Optional[mx.array] = None
    encoder_hidden_states: Optional[Tuple[mx.array, ...]] = None
    encoder_attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class TokenClassifierOutput:
    loss: Optional[mx.array] = None
    logits: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class QuestionAnsweringModelOutput:
    loss: Optional[mx.array] = None
    start_logits: mx.array = None
    end_logits: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class CausalLMOutputWithPast:
    loss: Optional[mx.array] = None
    logits: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class MaskedLMOutput:

    loss: Optional[mx.array] = None
    logits: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class CLIPVisionModelOutput:
    image_embeds: Optional[mx.array] = None
    last_hidden_state: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class CLIPTextModelOutput:
    text_embeds: Optional[mx.array] = None
    last_hidden_state: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class BaseModelOutputWithPooling:
    last_hidden_state: mx.array = None
    pooler_output: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class CLIPOutput:
    loss: Optional[mx.array] = None
    logits_per_image: mx.array = None
    logits_per_text: mx.array = None
    text_embeds: mx.array = None
    image_embeds: mx.array = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["text_model_output", "vision_model_output"]
                else getattr(self, k).to_tuple()
            )
            for k in self.keys()
        )


@dataclass
class BaseModelOutput:
    last_hidden_state: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class ImageClassifierOutput:

    loss: Optional[mx.array] = None
    logits: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
