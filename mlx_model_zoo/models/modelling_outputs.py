from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx

@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    cross_attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions:
    last_hidden_state: mx.array= None
    pooler_output: mx.array= None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    cross_attentions: Optional[Tuple[mx.array, ...]] = None