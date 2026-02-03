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
        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * self.num_action_chunks :] = (
            True  # [B, L + act + 1], [many x 0; act x 1; 0]
        )
        if cond_action_mask is not None:
            if cond_action_mask.ndim == 3:
                cond_action_mask = cond_action_mask.reshape(cond_action_mask.shape[0], -1)
            cond_action_mask = cond_action_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
            # Unmask conditional action tokens so they can condition the model.
            all_actions_mask[:, -self.action_dim * self.num_action_chunks :] &= ~cond_action_mask

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
        cond_action_tokens: Optional[torch.Tensor] = None,
        cond_action_mask: Optional[torch.Tensor] = None,
    ):
        """Prepares input for action prediction by adding necessary tokens.

        Optionally fills action placeholder tokens with conditional action tokens.
        """
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = torch.ones(
            (input_ids.shape[0], self.action_dim * self.num_action_chunks),
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
                :, -self.action_dim * self.num_action_chunks :
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
                :, -self.action_dim * self.num_action_chunks :
            ] = action_slice

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
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
        cond_action_tokens: Optional[torch.Tensor] = None,
        cond_action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids,
            attention_mask,
            cond_action_tokens=cond_action_tokens,
            cond_action_mask=cond_action_mask,
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1]
        assert torch.all(
            attention_mask[:, -1 - self.action_dim * self.num_action_chunks :] == 1
        )

        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values, cond_action_mask=cond_action_mask
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
            + self.action_dim * self.num_action_chunks,
            :,
        ]
        last_hidden_states = outputs.hidden_states[-1][
            :, -self.action_dim * self.num_action_chunks - 1 : -1
        ]

        return logits_tensor, last_hidden_states

    def _speculative_verify_tokens(
        self,
        *,
        draft_tokens: torch.Tensor,
        draft_logprobs: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        n_prompt_tokens: int,
        n_patches: int,
        spec_chunk_size: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run speculative + sequential verification on draft action tokens."""
        batch_size, horizon, action_dim = draft_tokens.shape
        device = draft_tokens.device

        draft_conf = draft_logprobs.mean(dim=2).detach().cpu().numpy()

        accepted_conf = torch.zeros((batch_size, horizon), dtype=torch.bool, device=device)
        accepted_seq = torch.zeros((batch_size, horizon), dtype=torch.bool, device=device)
        accepted_conf[:, 0] = True
        accepted_seq[:, 0] = True
        accepted_rank_conf = torch.zeros((batch_size,), dtype=torch.long, device=device)
        accepted_rank_seq = torch.zeros((batch_size,), dtype=torch.long, device=device)

        conf_active = torch.ones((batch_size,), dtype=torch.bool, device=device)
        seq_active = torch.ones((batch_size,), dtype=torch.bool, device=device)
        fail_pos_conf = [None] * batch_size
        fail_pos_seq = [None] * batch_size
        fail_action_conf = [None] * batch_size
        fail_action_seq = [None] * batch_size
        conf_stop_prefix_len = [None] * batch_size
        seq_stop_prefix_len = [None] * batch_size

        def _prefix_len(mask: torch.Tensor) -> int:
            length = 1
            for t in range(1, horizon):
                if not bool(mask[t].item()):
                    break
                length += 1
            return length

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
                        cond_mask = fixed[:, None].repeat(1, action_dim)
                        task_indices_conf[b].append(len(task_env_idx))
                        task_env_idx.append(b)
                        task_cond_tokens.append(draft_tokens[b])
                        task_cond_mask.append(cond_mask)

                if seq_active[b]:
                    for i in range(int(order_seq.shape[0])):
                        fixed = accepted_seq[b].clone()
                        if i > 0:
                            fixed[order_seq[:i]] = True
                        cond_mask = fixed[:, None].repeat(1, action_dim)
                        task_indices_seq[b].append(len(task_env_idx))
                        task_env_idx.append(b)
                        task_cond_tokens.append(draft_tokens[b])
                        task_cond_mask.append(cond_mask)

            if not task_env_idx:
                continue

            task_input_ids = input_ids[task_env_idx]
            task_attention_mask = attention_mask[task_env_idx]
            task_pixel_values = pixel_values[task_env_idx]
            cond_action_tokens = torch.stack(task_cond_tokens, dim=0)
            cond_action_mask = torch.stack(task_cond_mask, dim=0)

            logits_tensor, _last_hidden_states = self._forward_action_logits(
                task_input_ids,
                task_attention_mask,
                task_pixel_values,
                n_prompt_tokens,
                n_patches,
                cond_action_tokens=cond_action_tokens,
                cond_action_mask=cond_action_mask,
            )

            logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
            logits_tensor[..., self.vocab_size :] = -torch.inf
            idxs = logits_tensor.argmax(dim=-1)  # [num_tasks, act]
            verify_tokens = idxs.reshape(-1, horizon, action_dim)

            for b in range(batch_size):
                if conf_active[b] and task_indices_conf[b]:
                    order_conf = order_conf_list[b]
                    verify_tokens_conf = verify_tokens[task_indices_conf[b]]
                    for i in range(int(order_conf.shape[0])):
                        pos = int(order_conf[i])
                        pred = verify_tokens_conf[i, pos]
                        draft = draft_tokens[b, pos]
                        if torch.equal(pred, draft):
                            accepted_conf[b, pos] = True
                            accepted_rank_conf[b] += 1
                            continue
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
                        draft = draft_tokens[b, pos]
                        if torch.equal(pred, draft):
                            accepted_seq[b, pos] = True
                            accepted_rank_seq[b] += 1
                            continue
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

        accepted_prefix_len_conf = [
            _prefix_len(accepted_conf[b]) for b in range(batch_size)
        ]
        accepted_prefix_len_seq = [
            _prefix_len(accepted_seq[b]) for b in range(batch_size)
        ]
        accepted_prefix_len = [
            int(min(accepted_prefix_len_conf[b], accepted_prefix_len_seq[b]))
            for b in range(batch_size)
        ]

        append_pos_list = [-1] * batch_size
        for b in range(batch_size):
            append_action = None
            append_pos = None
            if accepted_prefix_len_conf[b] < accepted_prefix_len_seq[b]:
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

        info: dict[str, Any] = {
            "accepted_prefix_len": accepted_prefix_len,
            "accepted_prefix_len_conf": accepted_prefix_len_conf,
            "accepted_prefix_len_seq": accepted_prefix_len_seq,
            "accepted_rank_conf": accepted_rank_conf.detach().cpu().tolist(),
            "accepted_rank_seq": accepted_rank_seq.detach().cpu().tolist(),
            "spec_chunk_size": int(chunk_size),
            "append_pos": append_pos_list,
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
        spec_chunk_size = kwargs.pop("spec_chunk_size", None)
        if spec_chunk_size is None:
            spec_chunk_size = getattr(self.config, "spec_chunk_size", None)
        if spec_chunk_size is not None:
            spec_chunk_size = int(spec_chunk_size)
            if spec_chunk_size <= 0:
                spec_chunk_size = None
        spec_enabled = kwargs.pop("spec_enabled", None)
        if spec_enabled is None:
            spec_enabled = getattr(self.config, "spec_enabled", None)
        if spec_enabled is None:
            spec_enabled = spec_chunk_size is not None
        else:
            spec_enabled = bool(spec_enabled)
        if not spec_enabled:
            spec_chunk_size = None

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

        n_prompt_tokens = input_ids.shape[-1] - 1
        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        n_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        logits_tensor, last_hidden_states = self._forward_action_logits(
            input_ids, attention_mask, pixel_values, n_prompt_tokens, n_patches
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

        # assert torch.all(idxs >= 0) and torch.all(idxs < self.config.n_action_bins)
        # generated_ids = idxs + (self.vocab_size - self.config.n_action_bins)
        assert torch.all(
            idxs >= self.vocab_size - self.config.n_action_bins
        ) and torch.all(idxs < self.vocab_size)

        action_tokens = idxs.reshape(-1, self.num_action_chunks, self.action_dim)
        if spec_chunk_size is not None:
            draft_logprobs = compute_logprobs_from_logits(
                logits=processed_logits_tensor, target=idxs
            ).reshape(-1, self.num_action_chunks, self.action_dim)
            final_tokens, spec_info = self._speculative_verify_tokens(
                draft_tokens=action_tokens,
                draft_logprobs=draft_logprobs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                n_prompt_tokens=n_prompt_tokens,
                n_patches=n_patches,
                spec_chunk_size=spec_chunk_size,
            )
            action_tokens = final_tokens
        actions = self._tokens_to_actions(action_tokens)
        actions = actions.reshape(idxs.shape)

        action_logits = processed_logits_tensor
        action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size :] = -torch.inf

        final_token_flat = action_tokens.reshape(action_tokens.shape[0], -1)
        chunk_logprobs = compute_logprobs_from_logits(
            logits=action_logits, target=final_token_flat
        )

        if hasattr(self, "value_head") and calculate_values:
            hidden_features = last_hidden_states[
                :, -self.action_dim * self.num_action_chunks
            ]  # [batch_size, hidden_dim]

            chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
        chunk_action_tokens = action_tokens

        forward_inputs["action_tokens"] = chunk_action_tokens

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if spec_chunk_size is not None:
            result["spec_info"] = spec_info

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
            input_ids, attention_mask
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
            input_ids, attention_mask, pixel_values
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
