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
class SequenceClassifierOutput:
    loss: Optional[mx.array] = None
    logits: mx.array = None
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
