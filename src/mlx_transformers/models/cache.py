import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


# Adapted from: https://github.com/huggingface/transformers/blob/e74d793a3c3c0bc9bf3fb94bb31dd16934b1b0db/src/transformers/cache_utils.py#L14
@dataclass
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: mx.array,
        value_states: mx.array,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mx.array, mx.array]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError(
            "Make sure to implement `get_seq_length` in a subclass."
        )

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError(
            "Make sure to implement `get_max_length` in a subclass."
        )

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41."
            "Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated.
    This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer.
    The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[mx.array] = []
        self.value_cache: List[mx.array] = []
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[mx.array]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g.
        `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g.
        `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g.
         `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: mx.array,
        value_states: mx.array,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`mx.array`):
                The new key states to cache.
            value_states (`mx.array`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = mx.concatenate(
                [self.key_cache[layer_idx], key_states], axis=-2
            )
            self.value_cache[layer_idx] = mx.concatenate(
                [self.value_cache[layer_idx], value_states], axis=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states.
        A layer index can be optionally passed.
        """
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states.
        DynamicCache does not have a maximum length.
        """
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[mx.array], Tuple[mx.array]]:
        """
        Converts the `DynamicCache` instance into the its equivalent in the legacy cache format.
        """
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
