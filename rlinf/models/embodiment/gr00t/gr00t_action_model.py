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

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from torch.distributions import Normal
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.gr00t.utils import (
    squeeze_dict_values,
    unsqueeze_dict_values,
)
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


class FlowMatchingActionHeadForRLActionPrediction(FlowmatchingActionHead):
    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
        rl_head_config: dict[str, Any],
        output_action_chunks: int,
        valid_action_dim: int,
    ):
        super().__init__(config)
        self.action_chunk = output_action_chunks
        self.rl_config = rl_head_config
        self.padding_value = rl_head_config.padding_value
        self.valid_action_dim = valid_action_dim

        if self.rl_config.use_vlm_value:
            proj_width = 2048
        else:
            proj_width = 3584

        if self.rl_config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if self.rl_config.noise_method == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.hidden_size,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

    def get_logprob_norm(self, sample, mu, sigma):
        if self.rl_config.safe_get_logprob:
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
        else:
            # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
            return log_prob

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: int,
        state_features: torch.Tensor,
        idx: Optional[int | torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        Pay attention: The time notation of gr00t is different from openpi.
        In gr00t, the time is from 0 to 1, while in openpi, the time is from 1 to 0.
        """
        # expand the shape
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.rl_config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.rl_config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.rl_config.noise_level).to(device)

        # velocity prediction
        t_cont = idx / float(denoise_steps)
        timesteps_tensor = (
            (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
        )
        action_features = self.action_encoder(x_t, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        model_output = model_output[:, -self.action_horizon :]

        # ode/sde sampling
        v_t = self.action_decoder(model_output, embodiment_id)

        timesteps = torch.linspace(
            0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
        )
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        # Emphasize: In Gr00t, x0: noise, x1: data.
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = (
                t_input + delta
            )  # notice the plus here, it's different from openpi.
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.rl_config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.rl_config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = (torch.ones_like(t_input) - (t_input + delta)) * cos_term
                x1_weight = t_input + delta
                x_t_std = (1 - (t_input + delta)) * sin_term
            elif self.rl_config.noise_method == "reinflow":
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = self.reinflow_explore_noise_net(model_output)
            else:
                raise ValueError(f"Invalid noise method: {self.rl_config.noise_method}")
        # In eval, this equals to x_t_mean = x_t + v*dt(dt>0).
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []

        if self.rl_config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        # determine the denoise step for the logprob calculation
        if mode == "train":
            if self.rl_config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.rl_config.noise_method == "flow_sde":
                    if self.rl_config.ignore_last:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 2)] * num_steps
                        )
                    else:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 1)] * num_steps
                        )
                elif self.rl_config.noise_method == "flow_cps":
                    # the last denoising step of the flow-cps is deterministic
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
                elif self.rl_config.noise_method == "reinflow":
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        # Run denoising steps.
        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="train",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )
            else:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="eval",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.valid_action_dim
        ]
        if compute_values:
            values = self.get_value(vl_embs, state_features)
            values = values[:, None]
        else:
            values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

        return BatchFeature(
            data={"action_pred": x_0}
        ), {  # this is for gr00t validity check
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains,
        denoise_inds,
        compute_values=True,
    ):
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]

        chains_log_probs = []

        if self.rl_config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
                compute_values=compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values:
            chains_values = self.get_value(vl_embs, state_features)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros(
                (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
            )  # (B, 1)
        return chains_log_probs, chains_values

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def get_value(self, vl_embs, state_features):
        # TODO: add value vlm mode param
        bsize = vl_embs.shape[0]
        mask_length = vl_embs.shape[1]
        if self.rl_config.value_vlm_mode == "mean_token":
            prefix_mask = [True] * mask_length
        elif self.rl_config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (mask_length - 1) + [True] * 1
        elif self.rl_config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (mask_length - 1)
        vl_embs_value = vl_embs[:, prefix_mask, :]
        vl_embs_value = vl_embs_value.mean(dim=1, keepdim=False)
        # vl_embs_value = vl_embs_value.to(dtype=torch.float32)
        state_features_value = state_features.reshape(bsize, -1)
        if self.rl_config.use_vlm_value:
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)
        values_vlm = self.value_head(value_embs)[:, 0]
        return values_vlm


class GR00T_N1_5_ForRLActionPrediction(GR00T_N1_5, BasePolicy):
    """
    GR00T_N1_5 model for reinforcement learning action prediction.
    It's a combination of the Gr00tPolicy and GR00T_N1_5 model.

    Notes:
        - Device is handled by huggingface worker.
        - EmbodimentTag determines the state encoder and action head to use.
          we use "new_embodiment" reserved by gr00t.

    """

    _no_split_modules = [
        "Eagle2_5_VLForConditionalGeneration",
        "FlowMatchingActionHeadForRLActionPrediction",
        "TimestepEncoder",
        "TimestepEmbedding",
        "ValueHead",
    ]

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        rl_head_config: dict[str, Any],
        local_model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
        obs_converter_type: str = "libero",
        output_action_chunks: int = 1,
        action_horizon: Optional[int] = None,
        eval_action_horizon: Optional[int] = None,
        enable_speculative: bool = False,
        spec_batch_size: int = 8,
        spec_action_horizon: Optional[int] = None,
        spec_diffusion_num_steps: Optional[int] = None,
        spec_chunk_size: int = 5,
        rollout_segment: bool = False,
        rollout_segment_size: int = 5,
        spec_conf_alpha: float = 0.8,
        spec_conf_eps: float = 1e-6,
        spec_delta_threshold: float = 0.1,
        spec_delta_thresholds: Optional[list[float]] = None,
        spec_debug: bool = False,
        spec_profile_timing: bool = False,
        spec_verify_conf: bool = True,
        spec_verify_seq: bool = True,
    ):
        super().__init__(config, local_model_path)

        self.padding_value = rl_head_config.padding_value
        self._modality_config = modality_config  # ModalityConfig(delta_indices=[0], modality_keys=['video.ego_view'])
        self._modality_transform = modality_transform
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks
        self.model_path = Path(local_model_path)
        self._action_horizon_cfg = action_horizon
        self._eval_action_horizon_cfg = eval_action_horizon
        self._enable_speculative = bool(enable_speculative)
        self._spec_batch_size_cfg = int(spec_batch_size)
        self._spec_action_horizon_cfg = spec_action_horizon
        self._spec_diffusion_num_steps_cfg = spec_diffusion_num_steps
        self._spec_chunk_size_cfg = int(spec_chunk_size)
        self._rollout_segment_cfg = bool(rollout_segment)
        self._rollout_segment_size_cfg = int(rollout_segment_size)
        self._spec_conf_alpha_cfg = float(spec_conf_alpha)
        self._spec_conf_eps_cfg = float(spec_conf_eps)
        self._spec_delta_threshold_cfg = float(spec_delta_threshold)
        self._spec_delta_thresholds_cfg = spec_delta_thresholds
        self._spec_debug_cfg = bool(spec_debug)
        self._spec_profile_timing_cfg = bool(spec_profile_timing)
        self._spec_verify_conf_cfg = bool(spec_verify_conf)
        self._spec_verify_seq_cfg = bool(spec_verify_seq)

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        if denoising_steps is not None:
            if hasattr(self, "action_head") and hasattr(
                self.action_head, "num_inference_timesteps"
            ):
                self.action_head.num_inference_timesteps = denoising_steps

        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION[obs_converter_type]
        self._load_metadata(self.model_path / "experiment_cfg")

        # The param loading is after construction in from_pretrained(), so it should be safe to to so.
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowMatchingActionHeadForRLActionPrediction(
            action_head_cfg, rl_head_config, output_action_chunks, self.valid_action_dim
        )

    def eval(self):
        self._modality_transform.eval()
        super().eval()

    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        normalized_input = {
            "state": data["state"],
            "state_mask": data["state_mask"],
            "eagle_input_ids": data["eagle_input_ids"],
            "eagle_attention_mask": data["eagle_attention_mask"],
            "eagle_pixel_values": data["eagle_pixel_values"].reshape(
                -1, *data["eagle_pixel_values"].shape[2:]
            ),
            "eagle_image_sizes": data["eagle_image_sizes"].reshape(
                -1, *data["eagle_image_sizes"].shape[2:]
            ),
            "embodiment_id": data["embodiment_id"],
        }
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)

        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        log_probs, value_t = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            : self.action_head.action_chunk,
            : self.valid_action_dim,
        ]
        # post process
        if self.action_head.rl_config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = data["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = data["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize),
                denoise_inds[:, 0],
                : self.action_head.action_chunk,
                : self.valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": value_t,
            "entropy": None,
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        return_obs: bool = True,
        **kwargs,
    ):
        if mode == "eval":
            if self._enable_speculative:
                return self._predict_action_batch_spec(env_obs, return_obs=return_obs)
            return self._predict_action_batch_full(env_obs, return_obs=return_obs)

        normalized_input, is_batch = self._prepare_normalized_input(env_obs)
        normalized_action, result = self._get_rl_action(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=self.output_action_chunks
        )
        return raw_action, result

    def _prepare_normalized_input(self, env_obs) -> tuple[dict[str, Any], bool]:
        env_obs_local = dict(env_obs)
        env_obs_local["states"] = env_obs_local["states"].to(torch.bfloat16).cpu().float()

        observations = self.obs_convert_fn(env_obs_local)
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)

        normalized_input = self.apply_transforms(obs_copy)
        for key in normalized_input:
            if torch.is_tensor(normalized_input[key]) and normalized_input[key].dtype == torch.float32:
                normalized_input[key] = normalized_input[key].to(torch.bfloat16)

        normalized_input["eagle_input_ids"] = torch.nn.functional.pad(
            normalized_input["eagle_input_ids"],
            pad=(0, self.padding_value - normalized_input["eagle_input_ids"].shape[-1]),
            mode="constant",
            value=0,
        )
        normalized_input["eagle_attention_mask"] = torch.nn.functional.pad(
            normalized_input["eagle_attention_mask"],
            pad=(
                0,
                self.padding_value - normalized_input["eagle_attention_mask"].shape[-1],
            ),
            mode="constant",
            value=0,
        )
        return normalized_input, is_batch

    def _slice_env_obs(self, env_obs, env_idx: int) -> dict[str, Any]:
        sliced = {}
        for key, value in env_obs.items():
            if torch.is_tensor(value):
                sliced[key] = value[env_idx : env_idx + 1]
            elif isinstance(value, list):
                sliced[key] = [value[env_idx]]
            elif isinstance(value, dict):
                sliced[key] = {k: v[env_idx : env_idx + 1] for k, v in value.items()}
            else:
                sliced[key] = value
        return sliced

    def _repeat_normalized_input(
        self, normalized_input: dict[str, Any], batch_size: int
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in normalized_input.items():
            if not torch.is_tensor(value):
                out[key] = value
                continue
            if int(value.shape[0]) == int(batch_size):
                out[key] = value
                continue
            # GR00T image tensors are flattened as [B * num_images, ...].
            # For single-env speculative decode, this becomes [num_images, ...] (e.g., 2),
            # so we need to repeat by env-batch rather than requiring leading dim == 1.
            if key in {"eagle_pixel_values", "eagle_image_sizes"}:
                image_nums = int(getattr(self, "image_nums", 1))
                if image_nums > 0 and int(value.shape[0]) % image_nums == 0:
                    value_batch = int(value.shape[0]) // image_nums
                    if value_batch == int(batch_size):
                        out[key] = value
                        continue
                    if value_batch == 1:
                        out[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
                        continue
            if int(value.shape[0]) != 1:
                raise ValueError(
                    f"Expected tensor batch 1 or {int(batch_size)} for key={key}, got {int(value.shape[0])}"
                )
            out[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
        return out

    def _model_action_horizon(self) -> int:
        action_horizon = getattr(self.action_head, "action_horizon", None)
        if action_horizon is None:
            cfg = getattr(self.action_head, "config", None)
            action_horizon = getattr(cfg, "action_horizon", None)
        if action_horizon is None:
            cfg = getattr(self, "config", None)
            action_head_cfg = getattr(cfg, "action_head_cfg", None)
            if isinstance(action_head_cfg, dict):
                action_horizon = action_head_cfg.get("action_horizon", None)
        if action_horizon is None:
            raise ValueError("Cannot resolve GR00T action horizon from model/action_head config.")
        return int(action_horizon)

    def _action_horizon(self) -> int:
        model_horizon = self._model_action_horizon()
        action_horizon = self._action_horizon_cfg
        if action_horizon is None:
            action_horizon = int(self.output_action_chunks)
        action_horizon = int(action_horizon)
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
        if action_horizon > model_horizon:
            raise ValueError(
                f"action_horizon={action_horizon} must be <= model action horizon={model_horizon}"
            )
        return action_horizon

    def _eval_action_horizon(self) -> int:
        action_horizon = self._action_horizon()
        eval_horizon = self._eval_action_horizon_cfg
        if eval_horizon is None:
            return action_horizon
        eval_horizon = int(eval_horizon)
        if eval_horizon < 1:
            raise ValueError(f"eval_action_horizon must be >= 1, got {eval_horizon}")
        if eval_horizon > action_horizon:
            raise ValueError(
                f"eval_action_horizon={eval_horizon} must be <= action_horizon={action_horizon}"
            )
        return eval_horizon

    def _spec_action_horizon(self) -> int:
        model_horizon = self._model_action_horizon()
        action_horizon = self._action_horizon()
        spec_action_horizon = self._spec_action_horizon_cfg
        if spec_action_horizon is None:
            spec_action_horizon = action_horizon
        spec_action_horizon = int(spec_action_horizon)
        if spec_action_horizon < 1:
            raise ValueError(
                f"spec_action_horizon must be >= 1, got {spec_action_horizon}"
            )
        if spec_action_horizon > model_horizon:
            raise ValueError(
                f"spec_action_horizon={spec_action_horizon} must be <= model action horizon={model_horizon}"
            )
        return spec_action_horizon

    def _spec_diffusion_steps(self) -> int:
        diffusion_steps = self._spec_diffusion_num_steps_cfg
        if diffusion_steps is None:
            diffusion_steps = int(self.action_head.num_inference_timesteps)
        diffusion_steps = int(diffusion_steps)
        if diffusion_steps < 1:
            raise ValueError(
                f"spec_diffusion_num_steps must be >= 1, got {diffusion_steps}"
            )
        return diffusion_steps

    def _spec_batch_size(self) -> int:
        batch_size = int(self._spec_batch_size_cfg)
        if batch_size < 1:
            raise ValueError(f"spec_batch_size must be >= 1, got {batch_size}")
        return batch_size

    def _spec_verify_conf_enabled(self) -> bool:
        return bool(self._spec_verify_conf_cfg)

    def _spec_verify_seq_enabled(self) -> bool:
        return bool(self._spec_verify_seq_cfg)

    def _spec_delta_thresholds(self) -> np.ndarray:
        if self._spec_delta_thresholds_cfg is None:
            return np.repeat(np.float32(self._spec_delta_threshold_cfg), 6)
        if isinstance(self._spec_delta_thresholds_cfg, str):
            raw = self._spec_delta_thresholds_cfg.strip()
            parts = [p for p in raw.replace(",", " ").split() if p]
            vals = np.asarray([float(p) for p in parts], dtype=np.float32)
        else:
            vals = np.asarray(self._spec_delta_thresholds_cfg, dtype=np.float32)
        if vals.size == 1:
            vals = np.repeat(vals, 6)
        return vals

    def _spec_conf_dim(self, action_dim: int) -> int:
        if action_dim >= 7:
            return 6
        if action_dim > 1:
            return action_dim - 1
        return action_dim

    def _get_spec_log_path(self) -> str | None:
        path = getattr(self, "spec_log_path", None)
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

    def _normalized_to_exec_actions(
        self, normalized_action: torch.Tensor, *, chunk_size: int
    ) -> np.ndarray:
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=int(chunk_size)
        )
        return np.asarray(raw_action)

    def _build_spec_prefill(
        self,
        normalized_input: dict[str, Any],
        *,
        profile_timing: bool = False,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        timing_info: dict[str, float] = {}
        state_device = (
            normalized_input["state"].device
            if torch.is_tensor(normalized_input.get("state", None))
            else torch.device("cpu")
        )
        timing_enabled = bool(profile_timing) and state_device.type == "cuda"

        if timing_enabled:
            torch.cuda.synchronize(device=state_device)
            t0 = time.perf_counter()

        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self.action_head.process_backbone_output(backbone_outputs)
        prefill = {
            "vl_embs": backbone_outputs.backbone_features,
            "embodiment_id": action_inputs.embodiment_id,
            "state_features": self.action_head.state_encoder(
                action_inputs.state, action_inputs.embodiment_id
            ),
        }

        if timing_enabled:
            prefill_device = (
                prefill["vl_embs"].device
                if torch.is_tensor(prefill.get("vl_embs", None))
                else state_device
            )
            torch.cuda.synchronize(device=prefill_device)
            t1 = time.perf_counter()
            timing_info["prefill_ms"] = float((t1 - t0) * 1000.0)

        return prefill, timing_info

    def _repeat_spec_prefill(
        self, prefill: dict[str, Any], batch_size: int
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in prefill.items():
            if not torch.is_tensor(value):
                out[key] = value
                continue
            if value.ndim == 0:
                out[key] = value
                continue
            if int(value.shape[0]) == int(batch_size):
                out[key] = value
                continue
            if int(value.shape[0]) != 1:
                raise ValueError(
                    f"Expected prefill batch 1 or {int(batch_size)} for key={key}, got {int(value.shape[0])}"
                )
            out[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
        return out

    def _spec_sample_actions(
        self,
        normalized_input: dict[str, Any] | None,
        *,
        num_steps: int,
        fixed_actions: torch.Tensor | None = None,
        fixed_action_mask: torch.Tensor | None = None,
        profile_timing: bool = False,
        prefill: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        num_steps = int(num_steps)
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        timing_info: dict[str, float] = {}
        if prefill is None:
            if normalized_input is None:
                raise ValueError("normalized_input cannot be None when prefill is None")
            prefill, prefill_timing = self._build_spec_prefill(
                normalized_input,
                profile_timing=bool(profile_timing),
            )
            timing_info.update(prefill_timing)

        vl_embs = prefill["vl_embs"]
        embodiment_id = prefill["embodiment_id"]
        state_features = prefill["state_features"]
        state_device = vl_embs.device if torch.is_tensor(vl_embs) else torch.device("cpu")
        timing_enabled = bool(profile_timing) and state_device.type == "cuda"

        batch_size = int(vl_embs.shape[0])
        device = vl_embs.device
        model_horizon = self._model_action_horizon()
        model_action_dim = int(getattr(self.action_head.config, "action_dim", self.valid_action_dim))
        noise = self.action_head.sample_noise(
            (batch_size, model_horizon, model_action_dim), device
        )
        x_t = noise

        if fixed_actions is None:
            fixed_actions = torch.zeros_like(noise)
        else:
            if fixed_actions.ndim == 2:
                fixed_actions = fixed_actions.unsqueeze(0).expand(batch_size, -1, -1)
            fixed_actions = fixed_actions.to(device=device, dtype=noise.dtype)
            if int(fixed_actions.shape[1]) < model_horizon:
                raise ValueError(
                    "fixed_actions horizon is smaller than model action horizon: "
                    f"{int(fixed_actions.shape[1])} < {model_horizon}"
                )
            if int(fixed_actions.shape[1]) > model_horizon:
                fixed_actions = fixed_actions[:, :model_horizon, :]
            if int(fixed_actions.shape[2]) != model_action_dim:
                raise ValueError(
                    "fixed_actions action_dim mismatch: "
                    f"{int(fixed_actions.shape[2])} != {model_action_dim}"
                )

        if fixed_action_mask is None:
            fixed_action_mask = torch.zeros(
                (batch_size, model_horizon), dtype=torch.bool, device=device
            )
        else:
            if fixed_action_mask.ndim == 1:
                fixed_action_mask = fixed_action_mask.unsqueeze(0).expand(batch_size, -1)
            fixed_action_mask = fixed_action_mask.to(device=device, dtype=torch.bool)
            if int(fixed_action_mask.shape[1]) < model_horizon:
                raise ValueError(
                    "fixed_action_mask horizon is smaller than model action horizon: "
                    f"{int(fixed_action_mask.shape[1])} < {model_horizon}"
                )
            if int(fixed_action_mask.shape[1]) > model_horizon:
                fixed_action_mask = fixed_action_mask[:, :model_horizon]
        if fixed_action_mask.ndim == 2:
            fixed_action_mask = fixed_action_mask.unsqueeze(-1)
        if fixed_action_mask.ndim != 3:
            raise ValueError(
                f"Expected fixed_action_mask ndim 2 or 3, got {fixed_action_mask.ndim}"
            )

        timesteps = torch.linspace(
            0.0, 1.0, num_steps + 1, device=device, dtype=vl_embs.dtype
        )

        if timing_enabled:
            torch.cuda.synchronize(device=device)
            t2 = time.perf_counter()

        for idx in range(num_steps):
            t_input = timesteps[idx]
            t_next = timesteps[idx + 1]

            fixed_x_t = (1.0 - t_input) * noise + t_input * fixed_actions
            x_t = torch.where(fixed_action_mask, fixed_x_t, x_t)

            x_t_mean, x_t_std = self.action_head.sample_mean_var_val(
                vl_embs=vl_embs,
                denoise_steps=num_steps,
                x_t=x_t,
                embodiment_id=embodiment_id,
                state_features=state_features,
                idx=idx,
                mode="eval",
                compute_values=False,
            )
            x_t = x_t_mean + self.action_head.sample_noise(x_t.shape, device) * x_t_std

            fixed_x_t_next = (1.0 - t_next) * noise + t_next * fixed_actions
            x_t = torch.where(fixed_action_mask, fixed_x_t_next, x_t)

        if timing_enabled:
            torch.cuda.synchronize(device=device)
            t3 = time.perf_counter()
            timing_info["diffusion_ms_full"] = float((t3 - t2) * 1000.0)

        return x_t, timing_info

    def _compute_confidence(
        self, actions: np.ndarray, *, alpha: float, eps: float, conf_dim: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions_3d = np.asarray(actions)
        if actions_3d.ndim == 2:
            actions_3d = actions_3d[None, ...]
        use_dim = min(int(conf_dim), int(actions_3d.shape[-1]))
        u = np.cumsum(actions_3d[:, :, :use_dim], axis=1)
        mu = np.mean(u, axis=0)
        var = np.var(u, axis=0)
        d2 = np.sum((u - mu) ** 2 / (var + float(eps)), axis=-1)
        conf = np.exp(-0.5 * float(alpha) * d2)
        tau = np.median(conf, axis=0)
        high = conf >= tau[None, :]
        conf_stats = {
            "mu_abs_mean": float(np.mean(np.abs(mu))) if mu.size else float("nan"),
            "var_mean": float(np.mean(var)) if var.size else float("nan"),
            "var_max": float(np.max(var)) if var.size else float("nan"),
            "conf_mean": float(np.mean(conf)) if conf.size else float("nan"),
            "conf_std": float(np.std(conf)) if conf.size else float("nan"),
        }
        return conf.astype(np.float32), tau.astype(np.float32), high, conf_stats

    def _select_draft(
        self, actions: np.ndarray, *, alpha: float, eps: float, conf_dim: int
    ) -> dict[str, Any]:
        conf, tau, high, conf_stats = self._compute_confidence(
            actions, alpha=alpha, eps=eps, conf_dim=conf_dim
        )
        count = np.sum(high, axis=1).astype(np.int32)
        sum_conf = np.sum(conf * high, axis=1).astype(np.float32)
        best_count = int(np.max(count))
        candidates = np.flatnonzero(count == best_count)
        best = (
            int(candidates[0])
            if candidates.size == 1
            else int(candidates[np.argmax(sum_conf[candidates])])
        )
        return {
            "conf": conf,
            "tau": tau,
            "high": high,
            "count": count,
            "sum": sum_conf,
            "best": best,
            "conf_stats": conf_stats,
        }

    def _action_match(
        self, pred: np.ndarray, draft: np.ndarray, *, delta_thresholds: np.ndarray
    ) -> bool:
        pred_1d = np.asarray(pred).reshape(-1)
        draft_1d = np.asarray(draft).reshape(-1)
        compare_dim = self._spec_conf_dim(int(pred_1d.shape[-1]))
        if int(delta_thresholds.size) < int(compare_dim):
            raise ValueError(
                f"delta_thresholds must have at least {compare_dim} dims, got {int(delta_thresholds.size)}"
            )
        thr = np.asarray(delta_thresholds, dtype=np.float32).reshape(-1)[:compare_dim]
        return bool(
            np.all(np.abs(pred_1d[:compare_dim] - draft_1d[:compare_dim]) < thr)
        )

    def _predict_action_batch_full(
        self, env_obs, *, return_obs: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        normalized_input, is_batch = self._prepare_normalized_input(env_obs)
        action_horizon = self._action_horizon()
        eval_action_horizon = self._eval_action_horizon()
        diffusion_num_steps = int(self.action_head.num_inference_timesteps)
        normalized_action, _timing_info = self._spec_sample_actions(
            normalized_input,
            num_steps=diffusion_num_steps,
            profile_timing=bool(self._spec_profile_timing_cfg),
        )
        raw_action = self._normalized_to_exec_actions(
            normalized_action, chunk_size=int(action_horizon)
        )
        if raw_action.ndim == 3 and int(raw_action.shape[1]) > int(eval_action_horizon):
            raw_action = raw_action[:, : int(eval_action_horizon), :]
        if not is_batch and raw_action.ndim >= 3:
            raw_action = raw_action[0]
        result: dict[str, Any] = {}
        if return_obs:
            result["forward_inputs"] = None
        return raw_action, result

    def _spec_decode_single(self, env_obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        normalized_input, _is_batch = self._prepare_normalized_input(env_obs)
        chunk_size = int(self._spec_chunk_size_cfg)
        action_horizon = self._spec_action_horizon()
        if chunk_size <= 0:
            chunk_size = int(action_horizon)
        if action_horizon % int(chunk_size) != 0:
            raise ValueError(
                f"spec_chunk_size must divide spec_action_horizon: h={action_horizon} chunk={chunk_size}"
            )
        chunk, info = self._speculative_decode_chunk(
            normalized_input,
            batch_size=int(self._spec_batch_size()),
            action_horizon=int(action_horizon),
            diffusion_num_steps=int(self._spec_diffusion_steps()),
            conf_alpha=float(self._spec_conf_alpha_cfg),
            conf_eps=float(self._spec_conf_eps_cfg),
            delta_thresholds=self._spec_delta_thresholds(),
            spec_debug=bool(self._spec_debug_cfg),
            spec_chunk_size=int(chunk_size),
            timing_enabled=bool(self._spec_profile_timing_cfg),
            output_len=int(action_horizon),
            verify_conf_enabled=self._spec_verify_conf_enabled(),
            verify_seq_enabled=self._spec_verify_seq_enabled(),
        )
        return chunk, info

    def _predict_action_batch_spec(
        self, env_obs, *, return_obs: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        env_batch = int(env_obs["states"].shape[0])
        actions_out: list[np.ndarray] = []
        spec_infos: list[dict[str, Any]] = []
        for env_idx in range(env_batch):
            single_env_obs = self._slice_env_obs(env_obs, env_idx)
            action_chunk, spec_info = self._spec_decode_single(single_env_obs)
            actions_out.append(np.asarray(action_chunk))
            spec_infos.append(spec_info)
        actions = np.stack(actions_out, axis=0)
        result: dict[str, Any] = {"spec_info": spec_infos}
        if return_obs:
            result["forward_inputs"] = None
        return actions, result

    def _speculative_decode_chunk(
        self,
        normalized_input: dict[str, Any],
        *,
        batch_size: int,
        action_horizon: int,
        diffusion_num_steps: int,
        conf_alpha: float,
        conf_eps: float,
        delta_thresholds: np.ndarray,
        spec_debug: bool,
        spec_chunk_size: int,
        timing_enabled: bool,
        output_len: int,
        verify_conf_enabled: bool,
        verify_seq_enabled: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prefill_base, prefill_timing = self._build_spec_prefill(
            normalized_input,
            profile_timing=bool(timing_enabled),
        )
        prefill_batch = self._repeat_spec_prefill(prefill_base, int(batch_size))
        prefill_cache: dict[int, dict[str, Any]] = {int(batch_size): prefill_batch}

        def _get_prefill_for_batch(size: int) -> dict[str, Any]:
            key = int(size)
            cached = prefill_cache.get(key, None)
            if cached is not None:
                return cached
            expanded = self._repeat_spec_prefill(prefill_base, key)
            prefill_cache[key] = expanded
            return expanded

        actions_raw_t, timing_info = self._spec_sample_actions(
            None,
            num_steps=int(diffusion_num_steps),
            profile_timing=bool(timing_enabled),
            prefill=prefill_batch,
        )
        timing_info.update(prefill_timing)
        actions_exec = self._normalized_to_exec_actions(
            actions_raw_t, chunk_size=int(action_horizon)
        )
        actions_raw_full = (
            actions_raw_t[:, : int(action_horizon)].detach().float().cpu().numpy()
        )

        conf_enabled = bool(verify_conf_enabled)
        seq_enabled = bool(verify_seq_enabled)
        if not conf_enabled and not seq_enabled:
            raise ValueError("At least one of spec_verify_conf or spec_verify_seq must be True")

        conf_dim = self._spec_conf_dim(int(actions_exec.shape[-1]))
        selection = self._select_draft(
            actions_exec, alpha=float(conf_alpha), eps=float(conf_eps), conf_dim=conf_dim
        )
        conf = np.asarray(selection["conf"])
        conf_stats = selection.get("conf_stats", {})
        selected_path = int(selection["best"])

        b, h, d_exec = actions_exec.shape
        d_model = int(actions_raw_full.shape[2])
        greedy_chain_exec = np.zeros((h, d_exec), dtype=actions_exec.dtype)
        greedy_chain_exec[0] = actions_exec[selected_path, 0]
        greedy_chain_raw_full = np.zeros((h, d_model), dtype=actions_raw_full.dtype)
        greedy_chain_raw_full[0] = actions_raw_full[selected_path, 0]
        for t in range(1, h):
            greedy_chain_exec[t] = actions_exec[selected_path, t]
            greedy_chain_raw_full[t] = actions_raw_full[selected_path, t]

        draft_conf_per_t = conf[selected_path].astype(np.float32)
        chunk_size = int(spec_chunk_size)
        if chunk_size <= 0:
            chunk_size = int(h)
        if h % chunk_size != 0:
            raise ValueError(f"spec_chunk_size must divide action horizon: h={h} chunk={chunk_size}")

        chunk_orders: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for start in range(0, h, chunk_size):
            end = min(start + chunk_size, h)
            pos = np.arange(start, end, dtype=np.int64)
            order_conf = pos[np.lexsort((pos, -draft_conf_per_t[pos]))]
            order_seq = pos
            chunk_orders.append((order_conf, order_seq, int(start), int(end)))

        def _pending_order(order: np.ndarray, accepted_positions: set[int]) -> np.ndarray:
            if order.size == 0:
                return order
            pending = [int(pos) for pos in order if int(pos) not in accepted_positions]
            if not pending:
                return np.zeros((0,), dtype=np.int64)
            return np.asarray(pending, dtype=np.int64)

        def _prefix_len(accepted_positions: set[int]) -> int:
            accepted_prefix_len = 1
            for t in range(1, h):
                if t not in accepted_positions:
                    break
                accepted_prefix_len += 1
            return accepted_prefix_len

        def _run_verification_chunk(
            *,
            kind: str,
            verify_order: np.ndarray,
            verify_actions_compare_slice: np.ndarray,
            verify_actions_exec_slice: np.ndarray,
            accepted_positions: set[int],
            accepted_rank: int,
            chunk_start: int,
        ) -> tuple[set[int], int, bool, int | None, np.ndarray | None, dict[str, Any] | None]:
            for i in range(int(verify_order.shape[0])):
                pos = int(verify_order[i])
                pred_vec = np.asarray(verify_actions_exec_slice[i, pos])
                draft_vec = np.asarray(greedy_chain_exec[pos])
                space_label = "execution"
                if self._action_match(pred_vec, draft_vec, delta_thresholds=delta_thresholds):
                    accepted_positions.add(pos)
                    accepted_rank += 1
                    continue

                pred_1d = np.asarray(pred_vec).reshape(-1)
                draft_1d = np.asarray(draft_vec).reshape(-1)
                compare_dim = self._spec_conf_dim(int(pred_1d.shape[0]))
                diff = pred_1d[:compare_dim] - draft_1d[:compare_dim]
                abs_diff = np.abs(diff)
                over = abs_diff >= np.asarray(
                    delta_thresholds[:compare_dim], dtype=np.float32
                )
                conf_pos = (
                    float(draft_conf_per_t[pos])
                    if 0 <= pos < int(draft_conf_per_t.shape[0])
                    else float("nan")
                )
                delta_list = np.round(
                    np.asarray(delta_thresholds[:compare_dim], dtype=np.float32), 3
                ).tolist()
                pred_list = np.round(pred_1d[:compare_dim], 3).tolist()
                draft_list = np.round(draft_1d[:compare_dim], 3).tolist()
                abs_diff_list = np.round(abs_diff, 3).tolist()
                reject_detail = {
                    "kind": kind,
                    "pos": int(pos),
                    "rank": int(i),
                    "conf": conf_pos,
                    "space": space_label,
                    "delta": delta_list,
                    "pred": pred_list,
                    "draft": draft_list,
                    "abs_diff": abs_diff_list,
                    "abs_diff_max": float(np.max(abs_diff)) if abs_diff.size else float("nan"),
                    "over_dims": over.astype(np.int8).tolist(),
                    "chunk_start": int(chunk_start),
                }
                if spec_debug:
                    self._append_spec_log(
                        "spec_reject "
                        "kind={kind} pos={pos} rank={rank} conf={conf:.4f} "
                        "space={space} chunk_start={chunk_start} delta={delta} abs_diff_max={abs_diff_max:.4f} "
                        "over_dims={over_dims} pred={pred} draft={draft}".format(
                            kind=kind,
                            pos=int(pos),
                            rank=int(i),
                            conf=conf_pos,
                            space=space_label,
                            chunk_start=int(chunk_start),
                            delta=delta_list,
                            abs_diff_max=float(np.max(abs_diff))
                            if abs_diff.size
                            else float("nan"),
                            over_dims=over.astype(np.int8).tolist(),
                            pred=pred_list,
                            draft=draft_list,
                        )
                    )
                pred_exec = np.asarray(verify_actions_exec_slice[i, pos])
                return (
                    accepted_positions,
                    accepted_rank,
                    False,
                    pos,
                    pred_exec,
                    reject_detail,
                )
            return accepted_positions, accepted_rank, True, None, None, None

        accepted_positions_conf: set[int] = {0} if conf_enabled else set(range(h))
        accepted_positions_seq: set[int] = {0} if seq_enabled else set(range(h))
        accepted_rank_conf = 0 if conf_enabled else max(0, h - 1)
        accepted_rank_seq = 0 if seq_enabled else max(0, h - 1)
        fail_pos_conf: int | None = None
        fail_action_conf: np.ndarray | None = None
        fail_detail_conf: dict[str, Any] | None = None
        fail_pos_seq: int | None = None
        fail_action_seq: np.ndarray | None = None
        fail_detail_seq: dict[str, Any] | None = None
        conf_active = conf_enabled
        seq_active = seq_enabled
        conf_stop_prefix_len: int | None = None
        seq_stop_prefix_len: int | None = None

        model_horizon = self._model_action_horizon()
        for order_conf_full, order_seq_full, chunk_start, chunk_end in chunk_orders:
            verify_h = int(chunk_end)
            order_conf = (
                _pending_order(order_conf_full, accepted_positions_conf)
                if conf_active
                else np.zeros((0,), dtype=np.int64)
            )
            order_seq = (
                _pending_order(order_seq_full, accepted_positions_seq)
                if seq_active
                else np.zeros((0,), dtype=np.int64)
            )
            k_conf = int(order_conf.shape[0])
            k_seq = int(order_seq.shape[0])
            k_total = k_conf + k_seq
            if k_total == 0:
                continue

            fixed_actions_batch = np.zeros(
                (k_total, model_horizon, d_model), dtype=greedy_chain_raw_full.dtype
            )
            fixed_action_mask_batch = np.zeros((k_total, model_horizon), dtype=np.bool_)

            if conf_active:
                pos_fixed_global = np.asarray(
                    sorted(p for p in accepted_positions_conf if p < int(verify_h)),
                    dtype=np.int64,
                )
                for i in range(k_conf):
                    if pos_fixed_global.size:
                        fixed_actions_batch[i, pos_fixed_global] = greedy_chain_raw_full[pos_fixed_global]
                        fixed_action_mask_batch[i, pos_fixed_global] = True
                    if i > 0:
                        pos_prev = order_conf[:i]
                        fixed_actions_batch[i, pos_prev] = greedy_chain_raw_full[pos_prev]
                        fixed_action_mask_batch[i, pos_prev] = True

            if seq_active:
                pos_fixed_global = np.asarray(
                    sorted(p for p in accepted_positions_seq if p < int(verify_h)),
                    dtype=np.int64,
                )
                for i in range(k_seq):
                    j = k_conf + i
                    if pos_fixed_global.size:
                        fixed_actions_batch[j, pos_fixed_global] = greedy_chain_raw_full[pos_fixed_global]
                        fixed_action_mask_batch[j, pos_fixed_global] = True
                    if i > 0:
                        pos_prev = order_seq[:i]
                        fixed_actions_batch[j, pos_prev] = greedy_chain_raw_full[pos_prev]
                        fixed_action_mask_batch[j, pos_prev] = True

            fixed_actions_t = torch.as_tensor(
                fixed_actions_batch, device=actions_raw_t.device
            )
            fixed_action_mask_t = torch.as_tensor(
                fixed_action_mask_batch, device=actions_raw_t.device
            )
            prefill_verify = _get_prefill_for_batch(k_total)
            verify_actions_raw_t, _verify_timing = self._spec_sample_actions(
                None,
                num_steps=int(diffusion_num_steps),
                fixed_actions=fixed_actions_t,
                fixed_action_mask=fixed_action_mask_t,
                profile_timing=False,
                prefill=prefill_verify,
            )
            verify_actions = self._normalized_to_exec_actions(
                verify_actions_raw_t, chunk_size=int(verify_h)
            )
            verify_actions_raw_np = (
                verify_actions_raw_t[:, : int(verify_h)].detach().float().cpu().numpy()
            )

            if conf_active:
                (
                    accepted_positions_conf,
                    accepted_rank_conf,
                    conf_active,
                    conf_fail_pos,
                    conf_fail_action,
                    conf_fail_detail,
                ) = _run_verification_chunk(
                    kind="conf",
                    verify_order=order_conf,
                    verify_actions_compare_slice=verify_actions_raw_np[:k_conf],
                    verify_actions_exec_slice=verify_actions[:k_conf],
                    accepted_positions=accepted_positions_conf,
                    accepted_rank=accepted_rank_conf,
                    chunk_start=chunk_start,
                )
                if not conf_active and fail_pos_conf is None:
                    fail_pos_conf = conf_fail_pos
                    fail_action_conf = conf_fail_action
                    fail_detail_conf = conf_fail_detail
                if not conf_active:
                    conf_stop_prefix_len = _prefix_len(accepted_positions_conf)

            if seq_active:
                (
                    accepted_positions_seq,
                    accepted_rank_seq,
                    seq_active,
                    seq_fail_pos,
                    seq_fail_action,
                    seq_fail_detail,
                ) = _run_verification_chunk(
                    kind="seq",
                    verify_order=order_seq,
                    verify_actions_compare_slice=verify_actions_raw_np[k_conf:],
                    verify_actions_exec_slice=verify_actions[k_conf:],
                    accepted_positions=accepted_positions_seq,
                    accepted_rank=accepted_rank_seq,
                    chunk_start=chunk_start,
                )
                if not seq_active and fail_pos_seq is None:
                    fail_pos_seq = seq_fail_pos
                    fail_action_seq = seq_fail_action
                    fail_detail_seq = seq_fail_detail
                if not seq_active:
                    seq_stop_prefix_len = _prefix_len(accepted_positions_seq)

            if (
                seq_active
                and conf_stop_prefix_len is not None
                and _prefix_len(accepted_positions_seq) >= int(conf_stop_prefix_len)
            ):
                seq_active = False
            if (
                conf_active
                and seq_stop_prefix_len is not None
                and _prefix_len(accepted_positions_conf) >= int(seq_stop_prefix_len)
            ):
                conf_active = False
            if not conf_active and not seq_active:
                break

        accepted_prefix_len_conf = _prefix_len(accepted_positions_conf)
        accepted_prefix_len_seq = _prefix_len(accepted_positions_seq)
        accepted_prefix_len = int(min(accepted_prefix_len_conf, accepted_prefix_len_seq))

        chunk = greedy_chain_exec[:accepted_prefix_len]
        append_action = None
        append_pos = None
        if accepted_prefix_len_conf < accepted_prefix_len_seq:
            if fail_pos_conf is not None and int(fail_pos_conf) == accepted_prefix_len:
                append_action = fail_action_conf
                append_pos = int(fail_pos_conf)
        elif accepted_prefix_len_seq < accepted_prefix_len_conf:
            if fail_pos_seq is not None and int(fail_pos_seq) == accepted_prefix_len:
                append_action = fail_action_seq
                append_pos = int(fail_pos_seq)
        else:
            if fail_pos_seq is not None and int(fail_pos_seq) == accepted_prefix_len:
                append_action = fail_action_seq
                append_pos = int(fail_pos_seq)
            elif fail_pos_conf is not None and int(fail_pos_conf) == accepted_prefix_len:
                append_action = fail_action_conf
                append_pos = int(fail_pos_conf)
        if append_action is not None:
            append_action = np.asarray(append_action)
            chunk = np.concatenate([np.asarray(chunk), append_action[None, ...]], axis=0)

        accepted_actions = np.asarray(chunk)
        accepted_exec_len = (
            int(accepted_actions.shape[0]) if accepted_actions.ndim > 0 else 0
        )
        if output_len > 0:
            if chunk.shape[0] < int(output_len):
                pad_end = min(int(output_len), int(greedy_chain_exec.shape[0]))
                if chunk.shape[0] < pad_end:
                    chunk = np.concatenate(
                        [chunk, greedy_chain_exec[chunk.shape[0] : pad_end]], axis=0
                    )
            if chunk.shape[0] < int(output_len):
                last = chunk[-1] if chunk.size else greedy_chain_exec[0]
                pad = np.repeat(
                    last[None, ...], int(output_len) - int(chunk.shape[0]), axis=0
                )
                chunk = np.concatenate([chunk, pad], axis=0)
            if chunk.shape[0] > int(output_len):
                chunk = chunk[: int(output_len)]

        info: dict[str, Any] = {
            "accepted_prefix_len": int(accepted_prefix_len),
            "accepted_prefix_len_conf": int(accepted_prefix_len_conf),
            "accepted_prefix_len_seq": int(accepted_prefix_len_seq),
            "accepted_rank": int(min(accepted_rank_conf, accepted_rank_seq)),
            "accepted_rank_conf": int(accepted_rank_conf),
            "accepted_rank_seq": int(accepted_rank_seq),
            "selected_path": int(selected_path),
            "spec_chunk_size": int(chunk_size),
            "accepted_actions": accepted_actions,
            "accepted_exec_len": int(accepted_exec_len),
            "conf_stats": conf_stats,
            "spec_verify_conf": bool(conf_enabled),
            "spec_verify_seq": bool(seq_enabled),
        }
        reject_detail = None
        if accepted_prefix_len_conf < accepted_prefix_len_seq:
            reject_detail = fail_detail_conf
        elif accepted_prefix_len_seq < accepted_prefix_len_conf:
            reject_detail = fail_detail_seq
        else:
            reject_detail = fail_detail_seq or fail_detail_conf
        if reject_detail is not None:
            info["reject"] = reject_detail
        if append_pos is not None:
            info["append_pos"] = int(append_pos)
        if timing_enabled:
            info.update(timing_info)
        if spec_debug:
            self._append_spec_log(
                "gr00t_spec_debug "
                f"draft_horizon={int(action_horizon)} exec_len={int(accepted_exec_len)} "
                f"prefix_len={int(accepted_prefix_len)} prefix_conf={int(accepted_prefix_len_conf)} "
                f"prefix_seq={int(accepted_prefix_len_seq)} append_pos={int(append_pos) if append_pos is not None else -1}"
            )
        return chunk, info

    def apply_transforms(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def _get_rl_action(self, normalized_input: dict[str, Any]) -> torch.Tensor:
        # We expand get_action() and replace action head inference with RL inference.
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
            backbone_outputs, action_inputs
        )
        actions = rlinf_outputs["actions"]
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        actions = actions.float()

        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **normalized_input,
        }
        bsize = normalized_input["state"].shape[0]
        forward_inputs["eagle_pixel_values"] = normalized_input[
            "eagle_pixel_values"
        ].reshape(
            bsize, self.image_nums, *normalized_input["eagle_pixel_values"].shape[1:]
        )
        forward_inputs["eagle_image_sizes"] = normalized_input[
            "eagle_image_sizes"
        ].reshape(
            bsize, self.image_nums, *normalized_input["eagle_image_sizes"].shape[1:]
        )

        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        return actions, result

    def _get_action_from_normalized_input(
        self, normalized_input: dict[str, Any]
    ) -> torch.Tensor:
        # Set up autocast context if needed
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=self.compute_dtype),
        ):
            model_pred = self.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(
        self, normalized_action: torch.Tensor
    ) -> dict[str, Any]:
        # The transform unapply path converts tensors to numpy; numpy does not support bfloat16.
        # Always convert to float32 before unapplying normalization.
        return self.unapply_transforms({"action": normalized_action.float().cpu()})

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

        # calculate real intput action dim for rl learning.
        valid_action_dim = 0
        for v in metadata.modalities.action.values():
            valid_action_dim += v.shape[0]
        self.valid_action_dim = valid_action_dim

        self.image_nums = len(metadata.modalities.video.keys())
