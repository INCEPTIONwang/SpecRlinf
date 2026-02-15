# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig as OpenVLAOFTConfig,
)
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction as OpenVLAOFTForActionPrediction,
)
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    NormalizationType,
)
from transformers.generation import TopKLogitsWarper

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)


class OpenVLAOFTForRLActionPrediction(OpenVLAOFTForActionPrediction, BasePolicy):
    def __init__(
        self,
        config: OpenVLAOFTConfig,
        action_dim,
        num_action_chunks,
        add_value_head,
        max_prompt_length,
    ) -> None:
        super().__init__(config)

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks

        self.unnorm_key = config.unnorm_key
        if (
            self.unnorm_key not in self.norm_stats
            and f"{self.unnorm_key}_no_noops" in self.norm_stats
        ):
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        assert self.unnorm_key in self.norm_stats, (
            f"Action un-norm key {self.unnorm_key} not found in VLA `norm_stats`!"
        )

        if add_value_head:
            self.hidden_size = self.config.hidden_size
            output_dim = (
                1 if self.config.value_type == "chunk_level" else self.num_action_chunks
            )
            self.value_head = ValueHead(
                input_dim=self.hidden_size,
                hidden_sizes=(512, 128),
                output_dim=output_dim,
                activation="gelu",
                bias_last=False,
            )

        self.max_prompt_length = max_prompt_length

    def _build_embedding(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        action_horizon: int,
        cond_action_mask: Optional[torch.Tensor] = None,
    ):
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.shape[1] == attention_mask.shape[1]

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        n_patch_tokens = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm label & mask & embedding
        action_horizon = int(action_horizon)
        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * action_horizon :] = (
            True  # [B, L + act + 1], [many x 0; act x 1; 0]
        )
        if cond_action_mask is not None:
            if cond_action_mask.ndim == 3:
                cond_action_mask = cond_action_mask.reshape(cond_action_mask.shape[0], -1)
            cond_action_mask = cond_action_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
            # Unmask conditional action tokens so they can condition the model.
            all_actions_mask[:, -self.action_dim * action_horizon :] &= ~cond_action_mask

        input_embeddings = self.get_input_embeddings()(input_ids)  # [B, L + act + 1, D]
        input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))

        # vision
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, None, use_film=False
        )
        # [B, 256 * num_images, D]
        assert projected_patch_embeddings.shape[1] == n_patch_tokens

        # multimodal embeddings
        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0], -1, *projected_patch_embeddings.shape[2:]
        )
        multimodal_embeddings, multimodal_attention_mask = (
            self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
        )
        assert (
            multimodal_embeddings.shape[1]
            == input_embeddings.shape[1] + projected_patch_embeddings.shape[1]
        )
        assert (
            multimodal_attention_mask.shape[1]
            == attention_mask.shape[1] + projected_patch_embeddings.shape[1]
        )

        return multimodal_embeddings, multimodal_attention_mask

    def _get_action_stats(self) -> dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def _prepare_input_for_action_prediction(
        self,
        input_ids,
        attention_mask,
        action_horizon: int,
        cond_action_tokens: Optional[torch.Tensor] = None,
        cond_action_mask: Optional[torch.Tensor] = None,
    ):
        """Prepares input for action prediction by adding necessary tokens.

        Optionally fills action placeholder tokens with conditional action tokens.
        """
        # Add (ACTION_DIM * ACTION_HORIZON) placeholder tokens to input_ids to simulate action tokens
        action_horizon = int(action_horizon)
        placeholder_action_token_ids = torch.ones(
            (input_ids.shape[0], self.action_dim * action_horizon),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Optionally inject conditional action tokens into the placeholder positions.
        if cond_action_tokens is not None:
            if cond_action_tokens.ndim == 3:
                cond_action_tokens = cond_action_tokens.reshape(
                    cond_action_tokens.shape[0], -1
                )
            cond_action_tokens = cond_action_tokens.to(
                device=input_ids.device, dtype=input_ids.dtype
            )
            action_slice = input_ids[
                :, -self.action_dim * action_horizon :
            ]  # without STOP token yet
            if cond_action_mask is not None:
                if cond_action_mask.ndim == 3:
                    cond_action_mask = cond_action_mask.reshape(
                        cond_action_mask.shape[0], -1
                    )
                cond_action_mask = cond_action_mask.to(
                    device=input_ids.device, dtype=torch.bool
                )
                action_slice[cond_action_mask] = cond_action_tokens[cond_action_mask]
            else:
                action_slice[:] = cond_action_tokens
            input_ids[
                :, -self.action_dim * action_horizon :
            ] = action_slice

        # Keep STOP token to mirror training-time sequence layout for decoder-only causal LM.
        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype)
            * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones(
                (
                    attention_mask.shape[0],
                    input_ids.shape[-1] - attention_mask.shape[-1],
                )
            )
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["max"]),
                np.array(action_norm_stats["min"]),
            )
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        action_dim = normalized_actions.shape[-1]
        repeat_factor = action_dim // action_high.shape[0]
        action_high = action_high.repeat(repeat_factor)
        action_low = action_low.repeat(repeat_factor)
        mask = mask * repeat_factor

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8)
            + action_low,
            normalized_actions,
        )

        return actions

    def _tokens_to_actions(self, token_ids: torch.Tensor) -> np.ndarray:
        """Convert action token ids to unnormalized continuous actions."""
        if token_ids.ndim == 3:
            batch_size, num_chunks, action_dim = token_ids.shape
            token_ids = token_ids.reshape(batch_size, -1)
        else:
            batch_size = token_ids.shape[0]
            action_dim = self.action_dim
            num_chunks = int(token_ids.shape[1] // action_dim)
        token_ids_np = token_ids.detach().cpu().numpy()
        discretized_actions = self.vocab_size - token_ids_np
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        )
        normalized_actions = normalized_actions.reshape(-1, action_dim)
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(batch_size, num_chunks, action_dim)
        return actions

    def _forward_action_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        n_prompt_tokens: int,
        n_patches: int,
        *,
        action_horizon: int,
        cond_action_tokens: Optional[torch.Tensor] = None,
        cond_action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_horizon = int(action_horizon)
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids,
            attention_mask,
            action_horizon=action_horizon,
            cond_action_tokens=cond_action_tokens,
            cond_action_mask=cond_action_mask,
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1]
        assert torch.all(
            attention_mask[:, -1 - self.action_dim * action_horizon :] == 1
        )

        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids,
            attention_mask,
            pixel_values,
            action_horizon=action_horizon,
            cond_action_mask=cond_action_mask,
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + self.action_dim * action_horizon,
            :,
        ]
        last_hidden_states = outputs.hidden_states[-1][
            :, -self.action_dim * action_horizon - 1 : -1
        ]

        return logits_tensor, last_hidden_states

    def _select_batch_tensor(
        self, tensor: torch.Tensor, batch_indices: list[int] | None
    ) -> torch.Tensor:
        if batch_indices is None:
            return tensor
        index = torch.as_tensor(batch_indices, device=tensor.device, dtype=torch.long)
        return tensor.index_select(0, index)

    def _select_legacy_cache_batch(self, cache: Any, index: torch.Tensor) -> Any:
        if torch.is_tensor(cache):
            if cache.ndim == 0:
                return cache
            return cache.index_select(0, index.to(cache.device))
        if isinstance(cache, tuple):
            return tuple(self._select_legacy_cache_batch(v, index) for v in cache)
        if isinstance(cache, list):
            return [self._select_legacy_cache_batch(v, index) for v in cache]
        return cache

    def _select_past_key_values(self, past_key_values: Any, batch_indices: list[int]) -> Any:
        if past_key_values is None:
            return None
        index = torch.as_tensor(batch_indices, dtype=torch.long)
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy_cache = past_key_values.to_legacy_cache()
            selected_legacy = self._select_legacy_cache_batch(legacy_cache, index)
            cache_cls = past_key_values.__class__
            if hasattr(cache_cls, "from_legacy_cache"):
                return cache_cls.from_legacy_cache(selected_legacy)
            return selected_legacy
        return self._select_legacy_cache_batch(past_key_values, index)

    def _normalize_past_key_values(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        if isinstance(past_key_values, list):
            past_key_values = tuple(past_key_values)
        if isinstance(past_key_values, tuple):
            try:
                from transformers.cache_utils import DynamicCache

                return DynamicCache.from_legacy_cache(past_key_values)
            except Exception:
                return past_key_values
        return past_key_values

    def _clone_past_key_values(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        try:
            # Prefer native cache clone to preserve internal cache metadata.
            return copy.deepcopy(past_key_values)
        except Exception:
            pass
        try:
            legacy_cache = past_key_values.to_legacy_cache()
            cache_cls = past_key_values.__class__
            if hasattr(cache_cls, "from_legacy_cache"):
                return cache_cls.from_legacy_cache(legacy_cache)
            return legacy_cache
        except Exception:
            return past_key_values
        if isinstance(past_key_values, list):
            return tuple(past_key_values)
        return past_key_values

    def _ensure_llama_eager_attention(self) -> None:
        if getattr(self, "_llama_attention_forced_eager", False):
            return

        llm_model = getattr(self.language_model, "model", None)
        layers = getattr(llm_model, "layers", None)
        if layers is None:
            self._llama_attention_forced_eager = True
            return

        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
        except Exception:
            self._llama_attention_forced_eager = True
            return

        replaced = 0
        for idx, layer in enumerate(layers):
            old_attn = getattr(layer, "self_attn", None)
            if old_attn is None:
                continue
            if type(old_attn).__name__ != "LlamaSdpaAttention":
                continue

            layer_idx = getattr(old_attn, "layer_idx", idx)
            new_attn = LlamaAttention(config=llm_model.config, layer_idx=layer_idx)
            new_attn.load_state_dict(old_attn.state_dict(), strict=True)
            param = next(old_attn.parameters())
            new_attn.to(device=param.device, dtype=param.dtype)
            layer.self_attn = new_attn
            replaced += 1

        if replaced > 0:
            lm_config = getattr(self.language_model, "config", None)
            if lm_config is not None and hasattr(lm_config, "_attn_implementation"):
                lm_config._attn_implementation = "eager"
            model_config = getattr(llm_model, "config", None)
            if model_config is not None and hasattr(model_config, "_attn_implementation"):
                model_config._attn_implementation = "eager"
        try:
            first_attn_name = type(getattr(layers[0], "self_attn", None)).__name__
            self._append_spec_log(
                f"openvlaoft_force_eager replaced={int(replaced)} first_attn={first_attn_name}"
            )
        except Exception:
            pass
        self._llama_attention_forced_eager = True

    def _build_prefix_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
    ) -> dict[str, Any]:
        input_embeddings = self.get_input_embeddings()(input_ids)
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, None, use_film=False
        )
        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0], -1, *projected_patch_embeddings.shape[2:]
        )
        prefix_embeddings, prefix_attention_mask = self._build_multimodal_attention(
            input_embeddings,
            projected_patch_embeddings,
            attention_mask,
        )
        prefix_position_ids = prefix_attention_mask.cumsum(dim=1) - 1

        outputs = self.language_model(
            input_ids=None,
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embeddings,
            labels=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        if outputs.past_key_values is None:
            raise RuntimeError("OpenVLA-OFT prefix prefill did not return past_key_values; cannot reuse prefix cache.")

        return {
            "past_key_values": outputs.past_key_values,
            "attention_mask": prefix_attention_mask,
            "last_position_ids": prefix_position_ids[:, -1],
            "last_logits": outputs.logits[:, -1, :],
            "last_hidden": outputs.hidden_states[-1][:, -1, :],
        }

    def _build_suffix_action_embeddings(
        self,
        *,
        batch_size: int,
        action_horizon: int,
        device: torch.device,
        cond_action_tokens: Optional[torch.Tensor] = None,
        cond_action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_len = int(action_horizon) * int(self.action_dim)
        suffix_token_ids = torch.ones(
            (batch_size, action_len),
            device=device,
            dtype=torch.long,
        )
        all_actions_mask = torch.ones(
            (batch_size, action_len),
            device=device,
            dtype=torch.bool,
        )

        cond_mask_flat = None
        if cond_action_mask is not None:
            cond_mask_flat = cond_action_mask.reshape(batch_size, -1).to(
                device=device, dtype=torch.bool
            )

        if cond_action_tokens is not None:
            cond_tokens_flat = cond_action_tokens.reshape(batch_size, -1).to(
                device=device, dtype=torch.long
            )
            if cond_mask_flat is not None:
                suffix_token_ids[cond_mask_flat] = cond_tokens_flat[cond_mask_flat]
                all_actions_mask &= ~cond_mask_flat
            else:
                suffix_token_ids = cond_tokens_flat
                all_actions_mask.zero_()
        elif cond_mask_flat is not None:
            all_actions_mask &= ~cond_mask_flat

        suffix_embeddings = self.get_input_embeddings()(suffix_token_ids)
        suffix_embeddings = suffix_embeddings * (~all_actions_mask.unsqueeze(-1))
        return suffix_embeddings

    def _build_suffix_attention_mask_4d(
        self,
        *,
        prefix_attention_mask: torch.Tensor,
        suffix_attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = int(prefix_attention_mask.shape[0])
        prefix_len = int(prefix_attention_mask.shape[1])
        suffix_len = int(suffix_attention_mask.shape[1])
        device = prefix_attention_mask.device

        prefix_valid = prefix_attention_mask.to(dtype=torch.bool)
        suffix_valid = suffix_attention_mask.to(dtype=torch.bool)

        prefix_allow = prefix_valid[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_causal = torch.tril(
            torch.ones((suffix_len, suffix_len), device=device, dtype=torch.bool)
        )
        suffix_allow = (
            suffix_causal.unsqueeze(0)
            & suffix_valid[:, None, :]
            & suffix_valid[:, :, None]
        )
        allow_mask = torch.cat([prefix_allow, suffix_allow], dim=2)

        additive_mask = torch.zeros(
            (batch_size, 1, suffix_len, prefix_len + suffix_len),
            device=device,
            dtype=dtype,
        )
        additive_mask = additive_mask.masked_fill(
            ~allow_mask.unsqueeze(1), torch.finfo(dtype).min
        )
        return additive_mask

    def _forward_action_logits_with_prefix_cache(
        self,
        *,
        prefix_cache: dict[str, Any],
        action_horizon: int,
        batch_indices: list[int] | None = None,
        cond_action_tokens: Optional[torch.Tensor] = None,
        cond_action_mask: Optional[torch.Tensor] = None,
        force_eager_attention: bool = True,
        use_cache_position: bool = True,
        use_cache_output: bool = False,
        cond_prefix_via_kv: bool = False,
        group_by_prefix_position: bool = True,
        group_by_cond_prefix: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_horizon = int(action_horizon)
        suffix_len = action_horizon * int(self.action_dim)

        prefix_attention_mask = self._select_batch_tensor(
            prefix_cache["attention_mask"], batch_indices
        )
        prefix_last_position_ids = self._select_batch_tensor(
            prefix_cache["last_position_ids"], batch_indices
        )
        prefix_last_logits = self._select_batch_tensor(
            prefix_cache["last_logits"], batch_indices
        )
        prefix_last_hidden = self._select_batch_tensor(
            prefix_cache["last_hidden"], batch_indices
        )

        batch_size = int(prefix_attention_mask.shape[0])
        if group_by_prefix_position and batch_size > 1:
            unique_last_pos = torch.unique(prefix_last_position_ids)
            if int(unique_last_pos.numel()) > 1:
                if batch_indices is None:
                    base_batch_indices = list(
                        range(int(prefix_cache["attention_mask"].shape[0]))
                    )
                else:
                    base_batch_indices = [int(i) for i in batch_indices]

                merged_logits: torch.Tensor | None = None
                merged_hidden: torch.Tensor | None = None
                group_sizes: list[int] = []
                for pos_value in unique_last_pos.tolist():
                    local_idx = (prefix_last_position_ids == pos_value).nonzero(
                        as_tuple=False
                    ).squeeze(-1)
                    if local_idx.numel() == 0:
                        continue
                    local_idx = local_idx.to(dtype=torch.long)
                    local_idx_list = [int(i) for i in local_idx.tolist()]
                    sub_batch_indices = [base_batch_indices[i] for i in local_idx_list]
                    sub_cond_tokens = (
                        cond_action_tokens.index_select(0, local_idx)
                        if cond_action_tokens is not None
                        else None
                    )
                    sub_cond_mask = (
                        cond_action_mask.index_select(0, local_idx)
                        if cond_action_mask is not None
                        else None
                    )
                    sub_logits, sub_hidden = self._forward_action_logits_with_prefix_cache(
                        prefix_cache=prefix_cache,
                        action_horizon=action_horizon,
                        batch_indices=sub_batch_indices,
                        cond_action_tokens=sub_cond_tokens,
                        cond_action_mask=sub_cond_mask,
                        force_eager_attention=force_eager_attention,
                        use_cache_position=use_cache_position,
                        use_cache_output=use_cache_output,
                        cond_prefix_via_kv=cond_prefix_via_kv,
                        group_by_prefix_position=False,
                        group_by_cond_prefix=group_by_cond_prefix,
                    )
                    if merged_logits is None:
                        merged_logits = torch.empty(
                            (batch_size, sub_logits.shape[1], sub_logits.shape[2]),
                            device=sub_logits.device,
                            dtype=sub_logits.dtype,
                        )
                        merged_hidden = torch.empty(
                            (batch_size, sub_hidden.shape[1], sub_hidden.shape[2]),
                            device=sub_hidden.device,
                            dtype=sub_hidden.dtype,
                        )
                    merged_logits.index_copy_(0, local_idx.to(sub_logits.device), sub_logits)
                    merged_hidden.index_copy_(0, local_idx.to(sub_hidden.device), sub_hidden)
                    group_sizes.append(int(local_idx.numel()))

                if merged_logits is None or merged_hidden is None:
                    raise RuntimeError("Failed to build grouped KV forward outputs.")
                try:
                    self._append_spec_log(
                        "openvlaoft_suffix_group "
                        f"groups={int(len(group_sizes))} "
                        f"group_sizes={group_sizes} "
                        f"unique_prefix_last_pos={int(unique_last_pos.numel())}"
                    )
                except Exception:
                    pass
                return merged_logits, merged_hidden

        cond_tokens_flat = None
        cond_mask_flat = None
        cond_prefix_lens = None
        cond_prefix_valid = False
        if cond_action_tokens is not None and cond_action_mask is not None:
            cond_tokens_flat = cond_action_tokens.reshape(batch_size, -1).to(
                device=prefix_attention_mask.device, dtype=torch.long
            )
            cond_mask_flat = cond_action_mask.reshape(batch_size, -1).to(
                device=prefix_attention_mask.device, dtype=torch.bool
            )
            if int(cond_mask_flat.shape[1]) != int(suffix_len):
                raise ValueError(
                    "cond_action_mask length mismatch with suffix length: "
                    f"mask_len={int(cond_mask_flat.shape[1])}, suffix_len={int(suffix_len)}"
                )
            cond_prefix_lens = cond_mask_flat.sum(dim=1).to(dtype=torch.long)
            token_positions = torch.arange(
                suffix_len, device=cond_mask_flat.device, dtype=torch.long
            ).unsqueeze(0)
            expected_prefix_mask = token_positions < cond_prefix_lens.unsqueeze(1)
            cond_prefix_valid = bool(torch.equal(cond_mask_flat, expected_prefix_mask))
            if cond_prefix_via_kv and not cond_prefix_valid:
                try:
                    self._append_spec_log(
                        "openvlaoft_cond_prefix_kv_fallback "
                        "reason=non_prefix_mask"
                    )
                except Exception:
                    pass

        if (
            cond_prefix_via_kv
            and cond_prefix_valid
            and cond_prefix_lens is not None
            and group_by_cond_prefix
            and batch_size > 1
        ):
            unique_cond_prefix_lens = torch.unique(cond_prefix_lens)
            if int(unique_cond_prefix_lens.numel()) > 1:
                if batch_indices is None:
                    base_batch_indices = list(
                        range(int(prefix_cache["attention_mask"].shape[0]))
                    )
                else:
                    base_batch_indices = [int(i) for i in batch_indices]

                merged_logits: torch.Tensor | None = None
                merged_hidden: torch.Tensor | None = None
                group_sizes: list[int] = []
                group_cond_lens: list[int] = []
                for cond_len in unique_cond_prefix_lens.tolist():
                    local_idx = (cond_prefix_lens == cond_len).nonzero(as_tuple=False).squeeze(
                        -1
                    )
                    if local_idx.numel() == 0:
                        continue
                    local_idx = local_idx.to(dtype=torch.long)
                    local_idx_list = [int(i) for i in local_idx.tolist()]
                    sub_batch_indices = [base_batch_indices[i] for i in local_idx_list]
                    sub_cond_tokens = (
                        cond_action_tokens.index_select(0, local_idx)
                        if cond_action_tokens is not None
                        else None
                    )
                    sub_cond_mask = (
                        cond_action_mask.index_select(0, local_idx)
                        if cond_action_mask is not None
                        else None
                    )
                    sub_logits, sub_hidden = self._forward_action_logits_with_prefix_cache(
                        prefix_cache=prefix_cache,
                        action_horizon=action_horizon,
                        batch_indices=sub_batch_indices,
                        cond_action_tokens=sub_cond_tokens,
                        cond_action_mask=sub_cond_mask,
                        force_eager_attention=force_eager_attention,
                        use_cache_position=use_cache_position,
                        use_cache_output=use_cache_output,
                        cond_prefix_via_kv=cond_prefix_via_kv,
                        group_by_prefix_position=False,
                        group_by_cond_prefix=False,
                    )
                    if merged_logits is None:
                        merged_logits = torch.empty(
                            (batch_size, sub_logits.shape[1], sub_logits.shape[2]),
                            device=sub_logits.device,
                            dtype=sub_logits.dtype,
                        )
                        merged_hidden = torch.empty(
                            (batch_size, sub_hidden.shape[1], sub_hidden.shape[2]),
                            device=sub_hidden.device,
                            dtype=sub_hidden.dtype,
                        )
                    merged_logits.index_copy_(0, local_idx.to(sub_logits.device), sub_logits)
                    merged_hidden.index_copy_(0, local_idx.to(sub_hidden.device), sub_hidden)
                    group_sizes.append(int(local_idx.numel()))
                    group_cond_lens.append(int(cond_len))

                if merged_logits is None or merged_hidden is None:
                    raise RuntimeError("Failed to build grouped KV-condition outputs.")
                try:
                    self._append_spec_log(
                        "openvlaoft_suffix_group_cond "
                        f"groups={int(len(group_sizes))} "
                        f"group_sizes={group_sizes} "
                        f"cond_prefix_lens={group_cond_lens}"
                    )
                except Exception:
                    pass
                return merged_logits, merged_hidden

        if batch_indices is None:
            past_key_values = prefix_cache["past_key_values"]
        else:
            past_key_values = self._select_past_key_values(
                prefix_cache["past_key_values"], batch_indices
            )
        past_key_values = self._normalize_past_key_values(past_key_values)
        past_key_values = self._clone_past_key_values(past_key_values)

        base_prefix_last_logits = prefix_last_logits
        base_prefix_last_hidden = prefix_last_hidden
        base_prefix_cache_len = int(prefix_attention_mask.shape[1])
        cond_prefix_len = 0
        cond_logits = None
        cond_hidden = None

        use_cond_prefix_kv = (
            cond_prefix_via_kv
            and cond_tokens_flat is not None
            and cond_mask_flat is not None
            and cond_prefix_valid
            and cond_prefix_lens is not None
        )
        if use_cond_prefix_kv:
            unique_cond_prefix_lens = torch.unique(cond_prefix_lens)
            if int(unique_cond_prefix_lens.numel()) != 1:
                use_cond_prefix_kv = False
            else:
                cond_prefix_len = int(unique_cond_prefix_lens[0].item())

        if use_cond_prefix_kv and cond_prefix_len > 0:
            cond_prefix_token_ids = cond_tokens_flat[:, :cond_prefix_len]
            cond_prefix_embeddings = self.get_input_embeddings()(cond_prefix_token_ids)
            cond_prefix_attention_mask = torch.ones(
                (batch_size, cond_prefix_len),
                device=prefix_attention_mask.device,
                dtype=prefix_attention_mask.dtype,
            )
            cond_full_attention_mask = torch.cat(
                [prefix_attention_mask, cond_prefix_attention_mask], dim=1
            )
            cond_position_offsets = torch.arange(
                cond_prefix_len,
                device=prefix_last_position_ids.device,
                dtype=prefix_last_position_ids.dtype,
            ).unsqueeze(0)
            cond_position_ids = (
                prefix_last_position_ids.unsqueeze(1) + 1 + cond_position_offsets
            )
            cond_cache_position = torch.arange(
                base_prefix_cache_len,
                base_prefix_cache_len + cond_prefix_len,
                device=prefix_attention_mask.device,
                dtype=torch.long,
            )
            cond_outputs = self.language_model(
                input_ids=None,
                attention_mask=cond_full_attention_mask,
                position_ids=cond_position_ids,
                cache_position=cond_cache_position if use_cache_position else None,
                past_key_values=past_key_values,
                inputs_embeds=cond_prefix_embeddings,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if cond_outputs.past_key_values is None:
                raise RuntimeError(
                    "KV prefix conditioning forward did not return past_key_values."
                )
            cond_logits = cond_outputs.logits
            cond_hidden = cond_outputs.hidden_states[-1]
            past_key_values = self._normalize_past_key_values(cond_outputs.past_key_values)
            past_key_values = self._clone_past_key_values(past_key_values)
            prefix_attention_mask = cond_full_attention_mask
            prefix_last_position_ids = cond_position_ids[:, -1]
            prefix_last_logits = cond_logits[:, -1, :]
            prefix_last_hidden = cond_hidden[:, -1, :]

        suffix_decode_len = int(suffix_len - cond_prefix_len) if use_cond_prefix_kv else int(suffix_len)
        if suffix_decode_len < 0:
            raise RuntimeError(
                f"Invalid suffix decode length: total={int(suffix_len)} cond_prefix={int(cond_prefix_len)}"
            )

        suffix_logits = None
        suffix_hidden = None
        if suffix_decode_len > 0:
            if use_cond_prefix_kv:
                embed_dim = int(self.get_input_embeddings().weight.shape[1])
                suffix_embeddings = torch.zeros(
                    (batch_size, suffix_decode_len, embed_dim),
                    device=prefix_attention_mask.device,
                    dtype=self.get_input_embeddings().weight.dtype,
                )
            else:
                suffix_embeddings = self._build_suffix_action_embeddings(
                    batch_size=batch_size,
                    action_horizon=action_horizon,
                    device=prefix_attention_mask.device,
                    cond_action_tokens=cond_action_tokens,
                    cond_action_mask=cond_action_mask,
                )

            suffix_attention_mask = torch.ones(
                (batch_size, suffix_decode_len),
                device=prefix_attention_mask.device,
                dtype=prefix_attention_mask.dtype,
            )
            full_attention_mask = torch.cat(
                [prefix_attention_mask, suffix_attention_mask], dim=1
            )
            full_attention_mask_4d = self._build_suffix_attention_mask_4d(
                prefix_attention_mask=prefix_attention_mask,
                suffix_attention_mask=suffix_attention_mask,
                dtype=suffix_embeddings.dtype,
            )
            prefix_cache_len = int(prefix_attention_mask.shape[1])
            cache_position = torch.arange(
                prefix_cache_len,
                prefix_cache_len + suffix_decode_len,
                device=prefix_attention_mask.device,
                dtype=torch.long,
            )
            position_offsets = torch.arange(
                suffix_decode_len,
                device=prefix_last_position_ids.device,
                dtype=prefix_last_position_ids.dtype,
            ).unsqueeze(0)
            suffix_position_ids = prefix_last_position_ids.unsqueeze(1) + 1 + position_offsets

        if force_eager_attention:
            self._ensure_llama_eager_attention()
        try:
            layers = getattr(getattr(self.language_model, "model", None), "layers", None)
            first_attn_name = (
                type(getattr(layers[0], "self_attn", None)).__name__
                if layers is not None and len(layers) > 0
                else "unknown"
            )
            self._append_spec_log(
                "openvlaoft_suffix_forward "
                f"attn={first_attn_name} "
                f"force_eager={int(force_eager_attention)} "
                f"use_cache_pos={int(use_cache_position)} "
                f"use_cache_out={int(use_cache_output)} "
                f"cond_mode={'kv_prefix' if use_cond_prefix_kv else 'embedding'} "
                f"cond_prefix_len={int(cond_prefix_len)} "
                f"prefix_len={int(prefix_attention_mask.shape[1])} "
                f"suffix_len={int(suffix_decode_len)} "
                f"suffix_pos_start=({int(suffix_position_ids[:, 0].min().item()) if suffix_decode_len > 0 else -1},{int(suffix_position_ids[:, 0].max().item()) if suffix_decode_len > 0 else -1}) "
                f"cache_pos_start={int(cache_position[0].item()) if suffix_decode_len > 0 else -1} "
                f"pos_cache_delta=({int((cache_position[0] - suffix_position_ids[:, 0]).min().item()) if suffix_decode_len > 0 else 0},{int((cache_position[0] - suffix_position_ids[:, 0]).max().item()) if suffix_decode_len > 0 else 0}) "
                f"mask2d_shape={tuple(full_attention_mask.shape) if suffix_decode_len > 0 else tuple(prefix_attention_mask.shape)} "
                f"mask4d_shape={tuple(full_attention_mask_4d.shape) if suffix_decode_len > 0 else (batch_size, 1, 0, int(prefix_attention_mask.shape[1]))} "
                f"cache={type(past_key_values).__name__} "
                f"cache_pos=({int(cache_position[0]) if suffix_decode_len > 0 else -1},{int(cache_position[-1]) if suffix_decode_len > 0 else -1})"
            )
        except Exception:
            pass
        if suffix_decode_len > 0:
            suffix_outputs = self.language_model(
                input_ids=None,
                attention_mask=full_attention_mask,
                position_ids=suffix_position_ids,
                cache_position=cache_position if use_cache_position else None,
                past_key_values=past_key_values,
                inputs_embeds=suffix_embeddings,
                labels=None,
                use_cache=use_cache_output,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            suffix_logits = suffix_outputs.logits
            suffix_hidden = suffix_outputs.hidden_states[-1]

        if use_cond_prefix_kv:
            logits_parts: list[torch.Tensor] = []
            hidden_parts: list[torch.Tensor] = []
            if cond_prefix_len > 0:
                if cond_logits is None or cond_hidden is None:
                    raise RuntimeError("Missing conditioned prefix outputs in KV mode.")
                logits_parts.append(
                    torch.cat(
                        [base_prefix_last_logits.unsqueeze(1), cond_logits[:, :-1, :]],
                        dim=1,
                    )
                )
                hidden_parts.append(
                    torch.cat(
                        [base_prefix_last_hidden.unsqueeze(1), cond_hidden[:, :-1, :]],
                        dim=1,
                    )
                )
            if suffix_decode_len > 0:
                if suffix_logits is None or suffix_hidden is None:
                    raise RuntimeError("Missing suffix outputs in KV mode.")
                logits_parts.append(
                    torch.cat([prefix_last_logits.unsqueeze(1), suffix_logits[:, :-1, :]], dim=1)
                )
                hidden_parts.append(
                    torch.cat([prefix_last_hidden.unsqueeze(1), suffix_hidden[:, :-1, :]], dim=1)
                )
            if not logits_parts or not hidden_parts:
                raise RuntimeError("Empty logits/hidden parts in KV prefix mode.")
            logits_tensor = (
                logits_parts[0]
                if len(logits_parts) == 1
                else torch.cat(logits_parts, dim=1)
            )
            last_hidden_states = (
                hidden_parts[0]
                if len(hidden_parts) == 1
                else torch.cat(hidden_parts, dim=1)
            )
        else:
            if suffix_logits is None or suffix_hidden is None:
                raise RuntimeError("Missing suffix outputs in embedding mode.")
            logits_tensor = torch.cat(
                [prefix_last_logits.unsqueeze(1), suffix_logits[:, :-1, :]], dim=1
            )
            if suffix_len > 1:
                last_hidden_states = torch.cat(
                    [prefix_last_hidden.unsqueeze(1), suffix_hidden[:, :-1, :]], dim=1
                )
            else:
                last_hidden_states = prefix_last_hidden.unsqueeze(1)

        if int(logits_tensor.shape[1]) != int(suffix_len):
            raise RuntimeError(
                "Invalid logits length from prefix-cache forward: "
                f"expected={int(suffix_len)} got={int(logits_tensor.shape[1])}"
            )

        return logits_tensor, last_hidden_states

    def _get_spec_log_path(self) -> str | None:
        path = getattr(self, "spec_log_path", None)
        if not path:
            path = getattr(self.config, "spec_log_path", None)
        if path:
            return str(path)
        return None

    def _append_spec_log(self, line: str):
        path = self._get_spec_log_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            return

    def _log_kv_consistency(
        self,
        *,
        kv_logits_tensor: torch.Tensor,
        kv_hidden_states: torch.Tensor,
        ref_logits_tensor: torch.Tensor,
        ref_hidden_states: torch.Tensor,
        mode: str,
        batch_size: int,
        draft_horizon: int,
        compare_mode: str,
    ) -> None:
        action_vocab_start = int(self.vocab_size - self.config.n_action_bins)
        kv_logits = kv_logits_tensor[..., action_vocab_start : self.vocab_size].float()
        ref_logits = ref_logits_tensor[..., action_vocab_start : self.vocab_size].float()
        logits_abs_diff = (kv_logits - ref_logits).abs()
        logits_max_abs = float(logits_abs_diff.max().item())
        logits_mean_abs = float(logits_abs_diff.mean().item())
        argmax_eq = float(
            (kv_logits.argmax(dim=-1) == ref_logits.argmax(dim=-1))
            .float()
            .mean()
            .item()
        )
        hidden_abs_diff = (kv_hidden_states.float() - ref_hidden_states.float()).abs()
        hidden_max_abs = float(hidden_abs_diff.max().item())
        hidden_mean_abs = float(hidden_abs_diff.mean().item())

        # Decompose where mismatch comes from: first action token vs autoregressive suffix.
        first_logits_max_abs = float(logits_abs_diff[:, :1, :].max().item())
        first_argmax_eq = float(
            (kv_logits[:, :1, :].argmax(dim=-1) == ref_logits[:, :1, :].argmax(dim=-1))
            .float()
            .mean()
            .item()
        )
        if kv_logits.shape[1] > 1:
            tail_logits_abs_diff = logits_abs_diff[:, 1:, :]
            tail_logits_max_abs = float(tail_logits_abs_diff.max().item())
            tail_argmax_eq = float(
                (
                    kv_logits[:, 1:, :].argmax(dim=-1)
                    == ref_logits[:, 1:, :].argmax(dim=-1)
                )
                .float()
                .mean()
                .item()
            )
        else:
            tail_logits_max_abs = 0.0
            tail_argmax_eq = 1.0

        self._append_spec_log(
            "openvlaoft_kv_consistency "
            f"mode={mode} batch={int(batch_size)} "
            f"draft_horizon={int(draft_horizon)} "
            f"compare_mode={compare_mode} "
            f"logits_max_abs={logits_max_abs:.6e} logits_mean_abs={logits_mean_abs:.6e} "
            f"logits_argmax_eq={argmax_eq:.6f} "
            f"first_logits_max_abs={first_logits_max_abs:.6e} first_argmax_eq={first_argmax_eq:.6f} "
            f"tail_logits_max_abs={tail_logits_max_abs:.6e} tail_argmax_eq={tail_argmax_eq:.6f} "
            f"hidden_max_abs={hidden_max_abs:.6e} hidden_mean_abs={hidden_mean_abs:.6e}"
        )

    def _resolve_horizons(
        self,
        *,
        enable_speculative: bool,
        mode: str,
        action_horizon_override: int | None = None,
        eval_action_horizon_override: int | None = None,
        spec_action_horizon_override: int | None = None,
    ) -> tuple[int, int]:
        max_horizon = int(self.num_action_chunks)
        mode = str(mode).lower()

        # Keep training path unchanged: rollout/advantage buffers assume full num_action_chunks.
        if mode == "train":
            return max_horizon, max_horizon

        action_horizon = (
            action_horizon_override
            if action_horizon_override is not None
            else getattr(self.config, "action_horizon", None)
        )
        if action_horizon is None:
            action_horizon = max_horizon
        action_horizon = int(action_horizon)
        if action_horizon < 1 or action_horizon > max_horizon:
            raise ValueError(
                f"action_horizon must be in [1, {max_horizon}], got {action_horizon}"
            )

        if enable_speculative:
            spec_action_horizon = (
                spec_action_horizon_override
                if spec_action_horizon_override is not None
                else getattr(self.config, "spec_action_horizon", None)
            )
            if spec_action_horizon is None:
                spec_action_horizon = action_horizon
            spec_action_horizon = int(spec_action_horizon)
            if spec_action_horizon < 1 or spec_action_horizon > max_horizon:
                raise ValueError(
                    f"spec_action_horizon must be in [1, {max_horizon}], got {spec_action_horizon}"
                )
            draft_horizon = spec_action_horizon
        else:
            draft_horizon = action_horizon

        exec_horizon = draft_horizon
        if str(mode).lower() == "eval" and not enable_speculative:
            eval_action_horizon = (
                eval_action_horizon_override
                if eval_action_horizon_override is not None
                else getattr(self.config, "eval_action_horizon", None)
            )
            if eval_action_horizon is not None:
                eval_action_horizon = int(eval_action_horizon)
                if eval_action_horizon < 1:
                    raise ValueError(
                        f"eval_action_horizon must be >= 1, got {eval_action_horizon}"
                    )
                if eval_action_horizon > draft_horizon:
                    raise ValueError(
                        "eval_action_horizon must be <= action_horizon when speculative decoding is disabled. "
                        f"got eval_action_horizon={eval_action_horizon}, action_horizon={draft_horizon}"
                    )
                exec_horizon = eval_action_horizon

        return int(draft_horizon), int(exec_horizon)

    def _speculative_verify_tokens(
        self,
        *,
        draft_tokens: torch.Tensor,
        draft_logprobs: torch.Tensor,
        prefix_cache: dict[str, Any],
        verify_horizon: int,
        spec_chunk_size: int,
        spec_verify_batch_size: int | None = None,
        verify_conf_enabled: bool = True,
        verify_seq_enabled: bool = True,
        spec_debug: bool = False,
        spec_debug_max_mismatch_logs: int = 3,
        force_eager_attention: bool = True,
        use_cache_output: bool = False,
        cond_prefix_via_kv: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run speculative + sequential verification on draft action tokens."""
        batch_size, total_horizon, action_dim = draft_tokens.shape
        horizon = int(verify_horizon)
        if horizon < 1 or horizon > int(total_horizon):
            raise ValueError(
                f"verify_horizon must be in [1, {int(total_horizon)}], got {horizon}"
            )
        conf_enabled = bool(verify_conf_enabled)
        seq_enabled = bool(verify_seq_enabled)
        if not conf_enabled and not seq_enabled:
            raise ValueError("At least one of spec_verify_conf or spec_verify_seq must be True")

        device = draft_tokens.device
        draft_tokens_prefix = draft_tokens[:, :horizon]
        draft_logprobs_prefix = draft_logprobs[:, :horizon]
        draft_conf = draft_logprobs_prefix.mean(dim=2).detach().cpu().numpy()

        accepted_conf = torch.zeros((batch_size, horizon), dtype=torch.bool, device=device)
        accepted_seq = torch.zeros((batch_size, horizon), dtype=torch.bool, device=device)
        accepted_conf[:, 0] = True
        accepted_seq[:, 0] = True
        accepted_rank_conf = torch.zeros((batch_size,), dtype=torch.long, device=device)
        accepted_rank_seq = torch.zeros((batch_size,), dtype=torch.long, device=device)

        conf_active = torch.full(
            (batch_size,), bool(conf_enabled), dtype=torch.bool, device=device
        )
        seq_active = torch.full(
            (batch_size,), bool(seq_enabled), dtype=torch.bool, device=device
        )
        fail_pos_conf = [None] * batch_size
        fail_pos_seq = [None] * batch_size
        fail_action_conf = [None] * batch_size
        fail_action_seq = [None] * batch_size
        conf_stop_prefix_len = [None] * batch_size
        seq_stop_prefix_len = [None] * batch_size
        mismatch_log_count = [0 for _ in range(batch_size)]

        spec_debug_max_mismatch_logs = int(spec_debug_max_mismatch_logs)
        if spec_debug_max_mismatch_logs < 0:
            spec_debug_max_mismatch_logs = 0

        def _prefix_len(mask: torch.Tensor) -> int:
            length = 1
            for t in range(1, horizon):
                if not bool(mask[t].item()):
                    break
                length += 1
            return length

        def _token_vec_to_action(token_vec: torch.Tensor) -> np.ndarray:
            token_ids = (
                token_vec.detach()
                .to(device="cpu", dtype=torch.long)
                .numpy()
                .reshape(1, -1)
            )
            discretized_actions = self.vocab_size - token_ids
            discretized_actions = np.clip(
                discretized_actions - 1,
                a_min=0,
                a_max=self.bin_centers.shape[0] - 1,
            )
            normalized_actions = self.bin_centers[discretized_actions].astype(np.float32)
            actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
            return np.asarray(actions.reshape(-1), dtype=np.float32)

        def _maybe_log_mismatch(
            *,
            b: int,
            kind: str,
            chunk_start: int,
            pos: int,
            rank: int,
            pred: torch.Tensor,
            draft: torch.Tensor,
        ) -> None:
            if not spec_debug or spec_debug_max_mismatch_logs <= 0:
                return
            if mismatch_log_count[b] >= spec_debug_max_mismatch_logs:
                return

            pred_cpu = pred.detach().to(device="cpu", dtype=torch.long)
            draft_cpu = draft.detach().to(device="cpu", dtype=torch.long)
            token_diff = (pred_cpu - draft_cpu).to(dtype=torch.long)

            pred_action = _token_vec_to_action(pred_cpu)
            draft_action = _token_vec_to_action(draft_cpu)
            action_diff = pred_action - draft_action
            abs_action_diff = np.abs(action_diff)

            mismatch_log_count[b] += 1
            conf_pos = (
                float(draft_conf[b, pos])
                if 0 <= pos < int(draft_conf.shape[1])
                else float("nan")
            )

            self._append_spec_log(
                "openvlaoft_spec_mismatch "
                f"env={int(b)} kind={kind} chunk_start={int(chunk_start)} "
                f"pos={int(pos)} rank={int(rank)} conf={conf_pos:.6f} "
                f"pred_token={pred_cpu.tolist()} draft_token={draft_cpu.tolist()} "
                f"token_diff={token_diff.tolist()} "
                f"pred_action={np.round(pred_action, 4).tolist()} "
                f"draft_action={np.round(draft_action, 4).tolist()} "
                f"action_diff={np.round(action_diff, 4).tolist()} "
                f"action_abs_max={float(np.max(abs_action_diff)) if abs_action_diff.size else float('nan'):.6f} "
                f"action_l1={float(np.mean(abs_action_diff)) if abs_action_diff.size else float('nan'):.6f} "
                f"action_l2={float(np.linalg.norm(action_diff)) if action_diff.size else float('nan'):.6f} "
                f"log_idx={int(mismatch_log_count[b])}/{int(spec_debug_max_mismatch_logs)}"
            )

        final_tokens = draft_tokens.clone()

        chunk_size = int(spec_chunk_size)
        if chunk_size <= 0:
            return final_tokens, {}
        if horizon % chunk_size != 0:
            raise ValueError(
                f"spec_chunk_size must divide action horizon: h={horizon} chunk={chunk_size}"
            )

        for start in range(1, horizon, chunk_size):
            end = min(start + chunk_size, horizon)

            order_conf_list = [None] * batch_size
            order_seq_list = [None] * batch_size
            task_indices_conf: list[list[int]] = [[] for _ in range(batch_size)]
            task_indices_seq: list[list[int]] = [[] for _ in range(batch_size)]
            task_env_idx: list[int] = []
            task_cond_tokens: list[torch.Tensor] = []
            task_cond_mask: list[torch.Tensor] = []

            for b in range(batch_size):
                if not conf_active[b] and not seq_active[b]:
                    continue
                pos = np.arange(start, end, dtype=np.int64)
                order_seq = pos
                order_seq_list[b] = order_seq
                if conf_active[b]:
                    conf_vals = draft_conf[b, pos]
                    order_conf = pos[np.lexsort((pos, -conf_vals))]
                else:
                    order_conf = np.array([], dtype=np.int64)
                order_conf_list[b] = order_conf

                if conf_active[b]:
                    for i in range(int(order_conf.shape[0])):
                        fixed = accepted_conf[b].clone()
                        if i > 0:
                            fixed[order_conf[:i]] = True
                        cond_mask = torch.zeros(
                            (total_horizon, action_dim), dtype=torch.bool, device=device
                        )
                        cond_mask[:horizon] = fixed[:, None].repeat(1, action_dim)
                        task_indices_conf[b].append(len(task_env_idx))
                        task_env_idx.append(b)
                        task_cond_tokens.append(draft_tokens[b])
                        task_cond_mask.append(cond_mask)

                if seq_active[b]:
                    for i in range(int(order_seq.shape[0])):
                        fixed = accepted_seq[b].clone()
                        if i > 0:
                            fixed[order_seq[:i]] = True
                        cond_mask = torch.zeros(
                            (total_horizon, action_dim), dtype=torch.bool, device=device
                        )
                        cond_mask[:horizon] = fixed[:, None].repeat(1, action_dim)
                        task_indices_seq[b].append(len(task_env_idx))
                        task_env_idx.append(b)
                        task_cond_tokens.append(draft_tokens[b])
                        task_cond_mask.append(cond_mask)

            if not task_env_idx:
                continue

            num_tasks = len(task_env_idx)
            verify_batch_size = spec_verify_batch_size
            if verify_batch_size is None:
                verify_batch_size = num_tasks
            verify_batch_size = int(verify_batch_size)
            if verify_batch_size <= 0:
                verify_batch_size = num_tasks
            verify_tokens_batches: list[torch.Tensor] = []

            for bs_start in range(0, num_tasks, verify_batch_size):
                bs_end = min(bs_start + verify_batch_size, num_tasks)
                sub_env_idx = task_env_idx[bs_start:bs_end]
                cond_action_tokens = torch.stack(task_cond_tokens[bs_start:bs_end], dim=0)
                cond_action_mask = torch.stack(task_cond_mask[bs_start:bs_end], dim=0)

                logits_tensor, _last_hidden_states = self._forward_action_logits_with_prefix_cache(
                    prefix_cache=prefix_cache,
                    action_horizon=total_horizon,
                    batch_indices=sub_env_idx,
                    cond_action_tokens=cond_action_tokens,
                    cond_action_mask=cond_action_mask,
                    force_eager_attention=force_eager_attention,
                    use_cache_output=use_cache_output,
                    cond_prefix_via_kv=cond_prefix_via_kv,
                )

                logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
                logits_tensor[..., self.vocab_size :] = -torch.inf
                idxs = logits_tensor.argmax(dim=-1)  # [sub_tasks, act]
                verify_tokens_batches.append(idxs.reshape(-1, total_horizon, action_dim))

            verify_tokens = torch.cat(verify_tokens_batches, dim=0)

            for b in range(batch_size):
                if conf_active[b] and task_indices_conf[b]:
                    order_conf = order_conf_list[b]
                    verify_tokens_conf = verify_tokens[task_indices_conf[b]]
                    for i in range(int(order_conf.shape[0])):
                        pos = int(order_conf[i])
                        pred = verify_tokens_conf[i, pos]
                        draft = draft_tokens_prefix[b, pos]
                        if torch.equal(pred, draft):
                            accepted_conf[b, pos] = True
                            accepted_rank_conf[b] += 1
                            continue
                        _maybe_log_mismatch(
                            b=b,
                            kind="conf",
                            chunk_start=start,
                            pos=pos,
                            rank=i,
                            pred=pred,
                            draft=draft,
                        )
                        if fail_pos_conf[b] is None:
                            fail_pos_conf[b] = pos
                            fail_action_conf[b] = pred.detach().cpu()
                        conf_active[b] = False
                        conf_stop_prefix_len[b] = _prefix_len(accepted_conf[b])
                        break

                if seq_active[b] and task_indices_seq[b]:
                    order_seq = order_seq_list[b]
                    verify_tokens_seq = verify_tokens[task_indices_seq[b]]
                    for i in range(int(order_seq.shape[0])):
                        pos = int(order_seq[i])
                        pred = verify_tokens_seq[i, pos]
                        draft = draft_tokens_prefix[b, pos]
                        if torch.equal(pred, draft):
                            accepted_seq[b, pos] = True
                            accepted_rank_seq[b] += 1
                            continue
                        _maybe_log_mismatch(
                            b=b,
                            kind="seq",
                            chunk_start=start,
                            pos=pos,
                            rank=i,
                            pred=pred,
                            draft=draft,
                        )
                        if fail_pos_seq[b] is None:
                            fail_pos_seq[b] = pos
                            fail_action_seq[b] = pred.detach().cpu()
                        seq_active[b] = False
                        seq_stop_prefix_len[b] = _prefix_len(accepted_seq[b])
                        break

                if (
                    seq_active[b]
                    and conf_stop_prefix_len[b] is not None
                    and _prefix_len(accepted_seq[b]) >= int(conf_stop_prefix_len[b])
                ):
                    seq_active[b] = False
                if (
                    conf_active[b]
                    and seq_stop_prefix_len[b] is not None
                    and _prefix_len(accepted_conf[b]) >= int(seq_stop_prefix_len[b])
                ):
                    conf_active[b] = False

        accepted_prefix_len_conf_raw = [
            _prefix_len(accepted_conf[b]) for b in range(batch_size)
        ]
        accepted_prefix_len_seq_raw = [
            _prefix_len(accepted_seq[b]) for b in range(batch_size)
        ]
        accepted_prefix_len_conf = list(accepted_prefix_len_conf_raw)
        accepted_prefix_len_seq = list(accepted_prefix_len_seq_raw)
        if not conf_enabled:
            accepted_prefix_len_conf = list(accepted_prefix_len_seq)
            accepted_rank_conf = accepted_rank_seq.clone()
            fail_pos_conf = [None for _ in range(batch_size)]
            fail_action_conf = [None for _ in range(batch_size)]
        if not seq_enabled:
            accepted_prefix_len_seq = list(accepted_prefix_len_conf)
            accepted_rank_seq = accepted_rank_conf.clone()
            fail_pos_seq = [None for _ in range(batch_size)]
            fail_action_seq = [None for _ in range(batch_size)]

        if conf_enabled and seq_enabled:
            accepted_prefix_len = [
                int(min(accepted_prefix_len_conf[b], accepted_prefix_len_seq[b]))
                for b in range(batch_size)
            ]
        elif conf_enabled:
            accepted_prefix_len = [int(v) for v in accepted_prefix_len_conf]
        else:
            accepted_prefix_len = [int(v) for v in accepted_prefix_len_seq]

        append_pos_list = [-1] * batch_size
        for b in range(batch_size):
            append_action = None
            append_pos = None
            if conf_enabled and not seq_enabled:
                if (
                    fail_pos_conf[b] is not None
                    and int(fail_pos_conf[b]) == accepted_prefix_len[b]
                ):
                    append_action = fail_action_conf[b]
                    append_pos = int(fail_pos_conf[b])
            elif seq_enabled and not conf_enabled:
                if (
                    fail_pos_seq[b] is not None
                    and int(fail_pos_seq[b]) == accepted_prefix_len[b]
                ):
                    append_action = fail_action_seq[b]
                    append_pos = int(fail_pos_seq[b])
            elif accepted_prefix_len_conf[b] < accepted_prefix_len_seq[b]:
                if (
                    fail_pos_conf[b] is not None
                    and int(fail_pos_conf[b]) == accepted_prefix_len[b]
                ):
                    append_action = fail_action_conf[b]
                    append_pos = int(fail_pos_conf[b])
            elif accepted_prefix_len_seq[b] < accepted_prefix_len_conf[b]:
                if (
                    fail_pos_seq[b] is not None
                    and int(fail_pos_seq[b]) == accepted_prefix_len[b]
                ):
                    append_action = fail_action_seq[b]
                    append_pos = int(fail_pos_seq[b])
            else:
                if (
                    fail_pos_seq[b] is not None
                    and int(fail_pos_seq[b]) == accepted_prefix_len[b]
                ):
                    append_action = fail_action_seq[b]
                    append_pos = int(fail_pos_seq[b])
                elif (
                    fail_pos_conf[b] is not None
                    and int(fail_pos_conf[b]) == accepted_prefix_len[b]
                ):
                    append_action = fail_action_conf[b]
                    append_pos = int(fail_pos_conf[b])

            if append_action is not None and append_pos is not None:
                final_tokens[b, append_pos] = append_action.to(device=device)
                append_pos_list[b] = int(append_pos)

        def _to_pos_list(values: list[Any]) -> list[int]:
            out: list[int] = []
            for v in values:
                out.append(int(v) if v is not None else -1)
            return out

        verify_batch_size = spec_verify_batch_size
        if verify_batch_size is None:
            verify_batch_size = batch_size
        info: dict[str, Any] = {
            "accepted_prefix_len": accepted_prefix_len,
            "accepted_prefix_len_conf": accepted_prefix_len_conf,
            "accepted_prefix_len_seq": accepted_prefix_len_seq,
            "accepted_rank_conf": accepted_rank_conf.detach().cpu().tolist(),
            "accepted_rank_seq": accepted_rank_seq.detach().cpu().tolist(),
            "spec_chunk_size": int(chunk_size),
            "append_pos": append_pos_list,
            "fail_pos_conf": _to_pos_list(fail_pos_conf),
            "fail_pos_seq": _to_pos_list(fail_pos_seq),
            "verify_horizon": int(horizon),
            "verify_batch_size": int(verify_batch_size),
            "mismatch_log_count": [int(v) for v in mismatch_log_count],
            "spec_debug_max_mismatch_logs": int(spec_debug_max_mismatch_logs),
            "spec_verify_conf": bool(conf_enabled),
            "spec_verify_seq": bool(seq_enabled),
        }

        return final_tokens, info

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")
        mode = str(kwargs.pop("mode", "eval")).lower()
        if mode not in {"train", "eval"}:
            mode = "eval"

        spec_chunk_size = kwargs.pop("spec_chunk_size", None)
        if spec_chunk_size is None:
            spec_chunk_size = getattr(self.config, "spec_chunk_size", None)
        if spec_chunk_size is not None:
            spec_chunk_size = int(spec_chunk_size)
            if spec_chunk_size <= 0:
                spec_chunk_size = None

        enable_speculative = kwargs.pop("enable_speculative", None)
        if enable_speculative is None:
            enable_speculative = getattr(self.config, "enable_speculative", None)
        if enable_speculative is None:
            enable_speculative = spec_chunk_size is not None
        else:
            enable_speculative = bool(enable_speculative)
        if not enable_speculative:
            spec_chunk_size = None

        spec_debug = kwargs.pop("spec_debug", None)
        if spec_debug is None:
            spec_debug = getattr(self.config, "spec_debug", False)
        spec_debug = bool(spec_debug)
        spec_compare_full_forward = kwargs.pop("spec_compare_full_forward", None)
        if spec_compare_full_forward is None:
            spec_compare_full_forward = getattr(
                self.config, "spec_compare_full_forward", False
            )
        spec_compare_full_forward = bool(spec_compare_full_forward)
        spec_compare_force_eager = kwargs.pop("spec_compare_force_eager", None)
        if spec_compare_force_eager is None:
            spec_compare_force_eager = getattr(
                self.config, "spec_compare_force_eager", True
            )
        spec_compare_force_eager = bool(spec_compare_force_eager)
        spec_compare_alt_cache_position = kwargs.pop(
            "spec_compare_alt_cache_position", None
        )
        if spec_compare_alt_cache_position is None:
            spec_compare_alt_cache_position = getattr(
                self.config, "spec_compare_alt_cache_position", True
            )
        spec_compare_alt_cache_position = bool(spec_compare_alt_cache_position)
        spec_force_eager_attention = kwargs.pop("spec_force_eager_attention", None)
        if spec_force_eager_attention is None:
            spec_force_eager_attention = getattr(
                self.config, "spec_force_eager_attention", True
            )
        spec_force_eager_attention = bool(spec_force_eager_attention)
        spec_kv_use_cache_output = kwargs.pop("spec_kv_use_cache_output", None)
        if spec_kv_use_cache_output is None:
            spec_kv_use_cache_output = getattr(
                self.config, "spec_kv_use_cache_output", False
            )
        spec_kv_use_cache_output = bool(spec_kv_use_cache_output)
        spec_debug_max_mismatch_logs = kwargs.pop("spec_debug_max_mismatch_logs", None)
        if spec_debug_max_mismatch_logs is None:
            spec_debug_max_mismatch_logs = getattr(
                self.config, "spec_debug_max_mismatch_logs", 3
            )
        spec_debug_max_mismatch_logs = int(spec_debug_max_mismatch_logs)
        if spec_debug_max_mismatch_logs < 0:
            spec_debug_max_mismatch_logs = 0
        spec_verify_batch_size = kwargs.pop("spec_verify_batch_size", None)
        if spec_verify_batch_size is None:
            spec_verify_batch_size = getattr(self.config, "spec_verify_batch_size", None)
        if spec_verify_batch_size is not None:
            spec_verify_batch_size = int(spec_verify_batch_size)
            if spec_verify_batch_size <= 0:
                spec_verify_batch_size = None
        spec_verify_conf = kwargs.pop("spec_verify_conf", None)
        if spec_verify_conf is None:
            spec_verify_conf = getattr(self.config, "spec_verify_conf", True)
        spec_verify_conf = bool(spec_verify_conf)
        spec_verify_seq = kwargs.pop("spec_verify_seq", None)
        if spec_verify_seq is None:
            spec_verify_seq = getattr(self.config, "spec_verify_seq", True)
        spec_verify_seq = bool(spec_verify_seq)
        if not spec_verify_conf and not spec_verify_seq:
            raise ValueError("At least one of spec_verify_conf or spec_verify_seq must be True")
        spec_cond_prefix_via_kv = kwargs.pop("spec_cond_prefix_via_kv", None)
        if spec_cond_prefix_via_kv is None:
            spec_cond_prefix_via_kv = getattr(self.config, "spec_cond_prefix_via_kv", False)
        spec_cond_prefix_via_kv = bool(spec_cond_prefix_via_kv)

        action_horizon_override = kwargs.pop("action_horizon", None)
        eval_action_horizon_override = kwargs.pop("eval_action_horizon", None)
        spec_action_horizon_override = kwargs.pop("spec_action_horizon", None)
        draft_horizon, exec_horizon = self._resolve_horizons(
            enable_speculative=enable_speculative,
            mode=mode,
            action_horizon_override=action_horizon_override,
            eval_action_horizon_override=eval_action_horizon_override,
            spec_action_horizon_override=spec_action_horizon_override,
        )

        if enable_speculative and spec_chunk_size is None:
            spec_chunk_size = int(draft_horizon)
        if spec_chunk_size is not None:
            if spec_chunk_size > draft_horizon:
                raise ValueError(
                    f"spec_chunk_size must be <= spec_action_horizon/action_horizon. got chunk={spec_chunk_size}, horizon={draft_horizon}"
                )
            if draft_horizon % int(spec_chunk_size) != 0:
                raise ValueError(
                    f"spec_chunk_size must divide action horizon: h={draft_horizon} chunk={int(spec_chunk_size)}"
                )

        if spec_compare_full_forward and spec_compare_force_eager:
            # Compare mode should use one attention backend to avoid backend-induced noise.
            self._ensure_llama_eager_attention()

        if env_obs is not None:
            task_descriptions = [
                f"In: What action should the robot take to {t.lower()}?\nOut: "
                for t in env_obs["task_descriptions"]
            ]
            if env_obs["main_images"].ndim == 4:
                env_obs["main_images"] = env_obs["main_images"].unsqueeze(1)
            assert env_obs["main_images"].ndim == 5

            all_images = [
                env_obs["main_images"].permute(0, 1, 4, 2, 3)
            ]  # [B, 1, H, W, C] -> [B, 1, C, H, W]
            if self.vision_backbone.get_num_images_in_input() > 1:
                if env_obs["wrist_images"].ndim == 4:
                    env_obs["wrist_images"] = env_obs["wrist_images"].unsqueeze(1)
                assert env_obs["wrist_images"].ndim == 5
                wrist_imgs = env_obs["wrist_images"].permute(
                    0, 1, 4, 2, 3
                )  # [B, N_IMG, H, W, C] -> [B, N_IMG, C, H, W]
                all_images.extend(
                    [wrist_imgs[:, i] for i in range(wrist_imgs.shape[1])]
                )

            max_length = self.max_prompt_length
            device = next(self.parameters()).device
            precision = next(self.parameters()).dtype

            primary_image = all_images.pop(0)
            images = {"images": primary_image}
            inputs = self.input_processor(
                text=task_descriptions,
                images=images,
                proprio_states=env_obs["states"],
                padding="max_length",
                max_length=max_length,
            )

            if all_images:
                all_wrist_inputs = [
                    self.input_processor(
                        text=task_descriptions,
                        images={"images": wrist_image.unsqueeze(1)},
                        proprio_states=env_obs["states"],
                        padding="max_length",
                        max_length=max_length,
                    )
                    for wrist_image in all_images
                ]

                # Concatenate all images
                primary_pixel_values = inputs["pixel_values"]
                all_wrist_pixel_values = [
                    wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
                ]
                inputs["pixel_values"] = torch.cat(
                    [primary_pixel_values] + all_wrist_pixel_values, dim=1
                )

            input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = inputs["attention_mask"].to(
                device=device, dtype=torch.bool
            )
            pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)

            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(B, N * C, H, W)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        # assert first token is 1
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        prefix_cache: dict[str, Any] | None = None
        if spec_chunk_size is not None:
            if spec_force_eager_attention:
                # Keep prefix prefill and suffix decode on the same backend.
                self._ensure_llama_eager_attention()
            prefix_cache = self._build_prefix_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            if spec_debug:
                prefix_valid_tokens = prefix_cache["attention_mask"].to(torch.long).sum(dim=1)
                prefix_last_pos = prefix_cache["last_position_ids"].to(torch.long)
                prefix_len = int(prefix_cache["attention_mask"].shape[1])
                prefix_pos_gap = prefix_len - (prefix_last_pos + 1)
                self._append_spec_log(
                    "openvlaoft_prefix_cache "
                    f"mode={mode} batch={int(input_ids.shape[0])} "
                    f"prefix_tokens={int(prefix_cache['attention_mask'].shape[1])} "
                    f"prefix_valid=({int(prefix_valid_tokens.min().item())},{int(prefix_valid_tokens.max().item())}) "
                    f"prefix_last_pos=({int(prefix_last_pos.min().item())},{int(prefix_last_pos.max().item())}) "
                    f"prefix_pos_gap=({int(prefix_pos_gap.min().item())},{int(prefix_pos_gap.max().item())}) "
                    f"draft_horizon={int(draft_horizon)}"
                )

            logits_tensor, last_hidden_states = self._forward_action_logits_with_prefix_cache(
                prefix_cache=prefix_cache,
                action_horizon=int(draft_horizon),
                force_eager_attention=spec_force_eager_attention,
                use_cache_output=spec_kv_use_cache_output,
            )
            if spec_compare_full_forward:
                n_prompt_tokens = int(input_ids.shape[1] - 1)
                n_patches = int(
                    self.vision_backbone.get_num_patches()
                    * self.vision_backbone.get_num_images_in_input()
                )
                ref_logits_tensor, ref_last_hidden_states = self._forward_action_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    n_prompt_tokens=n_prompt_tokens,
                    n_patches=n_patches,
                    action_horizon=int(draft_horizon),
                )
                if spec_debug:
                    self._log_kv_consistency(
                        kv_logits_tensor=logits_tensor,
                        kv_hidden_states=last_hidden_states,
                        ref_logits_tensor=ref_logits_tensor,
                        ref_hidden_states=ref_last_hidden_states,
                        mode=mode,
                        batch_size=int(input_ids.shape[0]),
                        draft_horizon=int(draft_horizon),
                        compare_mode="spec",
                    )
                if spec_compare_alt_cache_position:
                    alt_logits_tensor, alt_last_hidden_states = (
                        self._forward_action_logits_with_prefix_cache(
                            prefix_cache=prefix_cache,
                            action_horizon=int(draft_horizon),
                            force_eager_attention=spec_force_eager_attention,
                            use_cache_position=False,
                            use_cache_output=spec_kv_use_cache_output,
                        )
                    )
                    if spec_debug:
                        self._log_kv_consistency(
                            kv_logits_tensor=alt_logits_tensor,
                            kv_hidden_states=alt_last_hidden_states,
                            ref_logits_tensor=ref_logits_tensor,
                            ref_hidden_states=ref_last_hidden_states,
                            mode=mode,
                            batch_size=int(input_ids.shape[0]),
                            draft_horizon=int(draft_horizon),
                            compare_mode="spec_alt_nocachepos",
                        )
        else:
            n_prompt_tokens = int(input_ids.shape[1] - 1)
            n_patches = int(
                self.vision_backbone.get_num_patches()
                * self.vision_backbone.get_num_images_in_input()
            )
            logits_tensor, last_hidden_states = self._forward_action_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                n_prompt_tokens=n_prompt_tokens,
                n_patches=n_patches,
                action_horizon=int(draft_horizon),
            )
            if spec_compare_full_forward:
                prefix_cache = self._build_prefix_cache(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
                if spec_debug:
                    prefix_valid_tokens = prefix_cache["attention_mask"].to(torch.long).sum(dim=1)
                    prefix_last_pos = prefix_cache["last_position_ids"].to(torch.long)
                    prefix_len = int(prefix_cache["attention_mask"].shape[1])
                    prefix_pos_gap = prefix_len - (prefix_last_pos + 1)
                    self._append_spec_log(
                        "openvlaoft_prefix_cache_compare_only "
                        f"mode={mode} batch={int(input_ids.shape[0])} "
                        f"prefix_tokens={int(prefix_cache['attention_mask'].shape[1])} "
                        f"prefix_valid=({int(prefix_valid_tokens.min().item())},{int(prefix_valid_tokens.max().item())}) "
                        f"prefix_last_pos=({int(prefix_last_pos.min().item())},{int(prefix_last_pos.max().item())}) "
                        f"prefix_pos_gap=({int(prefix_pos_gap.min().item())},{int(prefix_pos_gap.max().item())}) "
                        f"draft_horizon={int(draft_horizon)}"
                    )
                kv_logits_tensor, kv_last_hidden_states = (
                    self._forward_action_logits_with_prefix_cache(
                        prefix_cache=prefix_cache,
                        action_horizon=int(draft_horizon),
                        force_eager_attention=spec_force_eager_attention,
                        use_cache_output=spec_kv_use_cache_output,
                    )
                )
                if spec_debug:
                    self._log_kv_consistency(
                        kv_logits_tensor=kv_logits_tensor,
                        kv_hidden_states=kv_last_hidden_states,
                        ref_logits_tensor=logits_tensor,
                        ref_hidden_states=last_hidden_states,
                        mode=mode,
                        batch_size=int(input_ids.shape[0]),
                        draft_horizon=int(draft_horizon),
                        compare_mode="non_spec_double_forward",
                    )
                if spec_compare_alt_cache_position:
                    alt_logits_tensor, alt_last_hidden_states = (
                        self._forward_action_logits_with_prefix_cache(
                            prefix_cache=prefix_cache,
                            action_horizon=int(draft_horizon),
                            force_eager_attention=spec_force_eager_attention,
                            use_cache_position=False,
                            use_cache_output=spec_kv_use_cache_output,
                        )
                    )
                    if spec_debug:
                        self._log_kv_consistency(
                            kv_logits_tensor=alt_logits_tensor,
                            kv_hidden_states=alt_last_hidden_states,
                            ref_logits_tensor=logits_tensor,
                            ref_hidden_states=last_hidden_states,
                            mode=mode,
                            batch_size=int(input_ids.shape[0]),
                            draft_horizon=int(draft_horizon),
                            compare_mode="non_spec_double_forward_alt_nocachepos",
                        )

        logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        logits_tensor[..., self.vocab_size :] = -torch.inf

        if do_sample:
            processed_logits_tensor = logits_tensor / kwargs["temperature"]
            top_k = min(
                kwargs["top_k"], processed_logits_tensor.size(-1)
            )  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)
            processed_logprob_tensor = F.log_softmax(
                processed_logits_tensor, dim=-1
            )  # [B, act, vocab_size + 64]

            probs_tensor = torch.exp(
                processed_logprob_tensor
            )  # [B, act, vocab_size + 64]
            probs_flat = probs_tensor.view(
                -1, processed_logprob_tensor.shape[-1]
            )  # [B * act, vocab_size + 64]

            sample_flat = torch.multinomial(
                probs_flat, num_samples=1, replacement=True
            )  # [B * act, 1]
            idxs = sample_flat.view(
                processed_logprob_tensor.shape[0], processed_logprob_tensor.shape[1]
            )  # [B, act]
        else:
            processed_logits_tensor = logits_tensor
            idxs = processed_logits_tensor.argmax(dim=-1)  # [B, act]

        assert torch.all(
            idxs >= self.vocab_size - self.config.n_action_bins
        ) and torch.all(idxs < self.vocab_size)

        action_tokens_full = idxs.reshape(-1, int(draft_horizon), self.action_dim)
        spec_info = None
        if spec_chunk_size is not None:
            if prefix_cache is None:
                raise RuntimeError("prefix_cache is required for speculative verification.")
            draft_logprobs_full = compute_logprobs_from_logits(
                logits=processed_logits_tensor, target=idxs
            ).reshape(-1, int(draft_horizon), self.action_dim)
            action_tokens_full, spec_info = self._speculative_verify_tokens(
                draft_tokens=action_tokens_full,
                draft_logprobs=draft_logprobs_full,
                prefix_cache=prefix_cache,
                verify_horizon=int(draft_horizon),
                spec_chunk_size=int(spec_chunk_size),
                spec_verify_batch_size=spec_verify_batch_size,
                verify_conf_enabled=spec_verify_conf,
                verify_seq_enabled=spec_verify_seq,
                spec_debug=spec_debug,
                spec_debug_max_mismatch_logs=spec_debug_max_mismatch_logs,
                force_eager_attention=spec_force_eager_attention,
                use_cache_output=spec_kv_use_cache_output,
                cond_prefix_via_kv=spec_cond_prefix_via_kv,
            )

        actions_full = self._tokens_to_actions(action_tokens_full)

        action_logits = processed_logits_tensor
        action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size :] = -torch.inf

        final_token_flat = action_tokens_full.reshape(action_tokens_full.shape[0], -1)
        chunk_logprobs = compute_logprobs_from_logits(
            logits=action_logits, target=final_token_flat
        )

        if hasattr(self, "value_head") and calculate_values:
            hidden_features = last_hidden_states[
                :, -self.action_dim * int(draft_horizon)
            ]  # [batch_size, hidden_dim]

            chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions_full = actions_full.reshape(-1, int(draft_horizon), self.action_dim)
        if mode == "eval":
            output_horizon = int(draft_horizon) if spec_chunk_size is not None else int(exec_horizon)
        else:
            output_horizon = int(draft_horizon)
        chunk_actions = chunk_actions_full[:, :output_horizon, :]

        chunk_action_tokens = action_tokens_full

        forward_inputs["action_tokens"] = chunk_action_tokens

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if spec_chunk_size is not None and isinstance(spec_info, dict):
            batch_size = int(chunk_actions_full.shape[0])
            spec_info_list: list[dict[str, Any]] = []
            accepted_prefix = spec_info.get("accepted_prefix_len", [])
            accepted_conf = spec_info.get("accepted_prefix_len_conf", [])
            accepted_seq = spec_info.get("accepted_prefix_len_seq", [])
            accepted_rank_conf = spec_info.get("accepted_rank_conf", [])
            accepted_rank_seq = spec_info.get("accepted_rank_seq", [])
            append_pos = spec_info.get("append_pos", [])
            fail_pos_conf = spec_info.get("fail_pos_conf", [])
            fail_pos_seq = spec_info.get("fail_pos_seq", [])
            mismatch_log_count = spec_info.get("mismatch_log_count", [])
            mismatch_log_cap = int(
                spec_info.get("spec_debug_max_mismatch_logs", spec_debug_max_mismatch_logs)
            )
            spec_verify_conf_flag = bool(spec_info.get("spec_verify_conf", spec_verify_conf))
            spec_verify_seq_flag = bool(spec_info.get("spec_verify_seq", spec_verify_seq))
            for b in range(batch_size):
                prefix_len = int(accepted_prefix[b]) if b < len(accepted_prefix) else 1
                prefix_len = max(1, min(prefix_len, int(draft_horizon)))
                append_idx = int(append_pos[b]) if b < len(append_pos) else -1
                exec_len = int(prefix_len + 1) if append_idx == prefix_len else int(prefix_len)
                exec_len = max(1, min(exec_len, int(draft_horizon)))

                fail_conf = int(fail_pos_conf[b]) if b < len(fail_pos_conf) else -1
                fail_seq = int(fail_pos_seq[b]) if b < len(fail_pos_seq) else -1
                reject = None
                if fail_conf >= 0 or fail_seq >= 0:
                    if fail_seq < 0 or (fail_conf >= 0 and fail_conf <= fail_seq):
                        reject = {"kind": "conf", "pos": int(fail_conf)}
                    else:
                        reject = {"kind": "seq", "pos": int(fail_seq)}

                info = {
                    "accepted_prefix_len": prefix_len,
                    "accepted_exec_len": exec_len,
                    "accepted_prefix_len_conf": int(accepted_conf[b]) if b < len(accepted_conf) else prefix_len,
                    "accepted_prefix_len_seq": int(accepted_seq[b]) if b < len(accepted_seq) else prefix_len,
                    "accepted_rank_conf": int(accepted_rank_conf[b]) if b < len(accepted_rank_conf) else 0,
                    "accepted_rank_seq": int(accepted_rank_seq[b]) if b < len(accepted_rank_seq) else 0,
                    "spec_chunk_size": int(spec_chunk_size),
                    "spec_verify_batch_size": int(spec_info.get("verify_batch_size", spec_verify_batch_size or batch_size)),
                    "spec_verify_conf": bool(spec_verify_conf_flag),
                    "spec_verify_seq": bool(spec_verify_seq_flag),
                    "append_pos": append_idx,
                    "spec_mismatch_log_count": int(mismatch_log_count[b]) if b < len(mismatch_log_count) else 0,
                    "spec_debug_max_mismatch_logs": mismatch_log_cap,
                    "accepted_actions": np.asarray(chunk_actions_full[b, :exec_len], dtype=np.float32),
                }
                if reject is not None:
                    info["reject"] = reject
                spec_info_list.append(info)

                if spec_debug:
                    self._append_spec_log(
                        "openvlaoft_spec_debug "
                        f"mode={mode} env={b} draft_horizon={int(draft_horizon)} "
                        f"exec_len={exec_len} prefix_len={prefix_len} "
                        f"prefix_conf={info['accepted_prefix_len_conf']} "
                        f"prefix_seq={info['accepted_prefix_len_seq']} "
                        f"append_pos={append_idx} fail_conf_pos={fail_conf} fail_seq_pos={fail_seq} "
                        f"verify_conf={int(bool(info['spec_verify_conf']))} verify_seq={int(bool(info['spec_verify_seq']))} "
                        f"verify_bs={info['spec_verify_batch_size']} "
                        f"mismatch_logs={info['spec_mismatch_log_count']}/{info['spec_debug_max_mismatch_logs']}"
                    )

            result["spec_info"] = spec_info_list

        return chunk_actions, result

    def preprocess_for_train(self, data):
        # action-token: [bsz, chunk-step, action-dim] -> [bsz, chunk-step x action-dim]
        for key in ["action_tokens"]:
            value = data[key]
            data[key] = value.reshape(
                value.shape[0],
                self.action_dim * self.num_action_chunks,
                *value.shape[3:],
            )
        return data

    def setup_config_and_processor(self, model_config, input_processor):
        self.vocab_size = (
            model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        )
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_scale = 1.0

        self.input_processor = input_processor

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: bool = False,
        data: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: Optional[bool] = None,
    ):
        if data is not None:
            data = self.preprocess_for_train(data)
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            pixel_values = data["pixel_values"]

            action_tokens = data["action_tokens"]

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        attention_mask = attention_mask.to(torch.long)
        # llm inputs
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids,
            attention_mask,
            action_horizon=self.num_action_chunks,
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
        assert torch.all(
            input_ids[:, -self.action_dim * self.num_action_chunks - 2] == 29871
        )
        assert torch.all(
            attention_mask[:, -2 - self.action_dim * self.num_action_chunks :] == 1
        )  # [B, L + act + 1]

        # multimodal
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids,
            attention_mask,
            pixel_values,
            action_horizon=self.num_action_chunks,
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        if compute_values:
            output_hidden_states = True

        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        if compute_logprobs:
            logits = outputs.logits[
                :, -self.action_dim * self.num_action_chunks - 1 : -1
            ]  # [B, action-dim, vocab-size]

            processed_logits_tensor = logits / data["temperature"]
            top_k = min(data["top_k"], processed_logits_tensor.size(-1))  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            action_logits = processed_logits_tensor
            action_logits[
                ..., : self.vocab_size - self.config.n_action_bins
            ] = -torch.inf
            action_logits[..., self.vocab_size :] = -torch.inf

            logprobs = compute_logprobs_from_logits(
                logits=action_logits, target=action_tokens
            )

            entropy = None
            if compute_entropy:
                entropy = compute_entropy_from_logits(logits=action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[
                :, -self.action_dim * self.num_action_chunks - 1
            ]  # [batch_size, hidden_dim]
            values = self.value_head(hidden_features)
        else:
            values = None

        result = {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

        return result
