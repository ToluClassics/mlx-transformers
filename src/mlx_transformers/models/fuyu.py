import copy
from typing import Optional, Tuple, Union, List, Dict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import FuyuConfig

from .persimmon import PersimmonForCausalLM
from .base import MlxPretrainedMixin
from .modelling_outputs import CausalLMOutputWithPast


class FuyuForCausalLM(nn.Module, MlxPretrainedMixin):
    def __init__(self, config: FuyuConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.language_model = PersimmonForCausalLM(config.text_config)

        self.vision_embed_tokens = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels,
            config.hidden_size,
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def gather_continuous_embeddings(
        self,
        word_embeddings: mx.array,
        continuous_embeddings: List[mx.array],
        image_patch_input_indices: mx.array,
    ) -> mx.array:
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} "
                "and {word_embeddings.shape[0]=}"
            )

        output_embeddings = copy.deepcopy(word_embeddings)
        for batch_idx in range(word_embeddings.shape[0]):
            # First, find the positions of all the non-negative values in
            # image_patch_input_indices, those are the
            # positions in word_embeddings that we want to replace with content
            # from continuous_embeddings.

            dst_indices = np.nonzero(
                np.array(image_patch_input_indices)[batch_idx] >= 0
            )[0]
            dst_indices = mx.array(dst_indices)
            # Next look up those indices in image_patch_input_indices to find the
            # indices in continuous_embeddings that we
            # want to use to replace the values in word_embeddings.
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have
            # fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings"
                    f"{continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} "
                    f"in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[
                batch_idx
            ][src_indices]
        return output_embeddings

    def __call__(
        self,
        input_ids: mx.array = None,
        # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches: mx.array = None,
        image_patches_indices: mx.array = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[List[mx.array]] = None,
        inputs_embeds: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[mx.array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import FuyuProcessor, FuyuForCausalLM
        >>> from PIL import Image
        >>> import requests

        >>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        >>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

        >>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Generate a coco-style caption.\n"

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=7)
        >>> generation_text = processor.batch_decode(generated_ids[:, -7:], skip_special_tokens=True)
        >>> print(generation_text[0])
        A blue bus parked on the side of a road.
        ```
        """

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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_is or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = mx.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
            )
            position_ids = mx.expand_dims(position_ids, 0)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            if image_patches is not None and past_key_values is None:
                patch_embeddings = [
                    self.vision_embed_tokens(
                        patch.astype(self.vision_embed_tokens.weight.dtype)
                    ).squeeze(0)
                    for patch in image_patches
                ]
                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                )
            input_ids = None

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_patches=None,
        image_patches_indices=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            attention_mask = mx.array(attention_mask)
            position_ids = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
            position_ids, attention_mask = (
                np.array(position_ids),
                np.array(attention_mask),
            )
            position_ids = np.ma.array(
                data=position_ids, mask=attention_mask == 0
            ).filled(1)
            position_ids = mx.array(position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_patches_indices": image_patches_indices
                if past_key_values is None
                else None,
                "image_patches": image_patches if past_key_values is None else None,
            }
        )
        return model_inputs

    def generate(self, inputs: Dict, **kwargs):
        temp = kwargs.get("temp", 1.0)

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        # Process the prompt
        use_cache = kwargs.get("use_cache", True)
        inputs = self.prepare_inputs_for_generation(**inputs, use_cache=use_cache)

        output = self(**inputs)

        next_token_logits = output.logits[:, -1, :]
        next_token = sample(next_token_logits)

        while True:
            # Update the prompt
            next_token = mx.expand_dims(next_token, axis=0)

            inputs["input_ids"] = next_token
            inputs["attention_mask"] = mx.concatenate(
                [mx.array(inputs["attention_mask"]), mx.ones_like(next_token)], axis=-1
            )

            past_key_values = output.past_key_values
            inputs = self.language_model.prepare_inputs_for_generation(
                input_ids=inputs["input_ids"],
                past_key_values=past_key_values,
                attention_mask=inputs["attention_mask"],
                inputs_embeds=None,
            )

            output = self.language_model(**inputs)

            next_token_logits = output.logits[:, -1, :]

            next_token = sample(next_token_logits)

            yield next_token
