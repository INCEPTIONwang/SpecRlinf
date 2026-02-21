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

import logging
import os
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token
    # speculative decoding (eval-only)
    enable_speculative: bool = False
    spec_batch_size: int = 8
    spec_action_horizon: int | None = None
    spec_diffusion_num_steps: int | None = None
    spec_chunk_size: int = 5
    spec_rollout_segment_size: int = 5
    rollout_segment: bool | None = None
    rollout_segment_size: int | None = None
    spec_conf_alpha: float = 0.8
    spec_conf_eps: float = 1e-6
    spec_delta_threshold: float = 0.1
    spec_delta_thresholds: list[float] | None = None
    spec_debug: bool = False
    spec_log_conf_stats: bool = True
    spec_profile_timing: bool = False
    spec_verify_conf: bool = True
    spec_verify_seq: bool = True
    eval_action_horizon: int | None = None


class OpenPi0ForRLActionPrediction(PI0Pytorch, BasePolicy):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep with dynamic action_horizon support."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            att_masks += [1]

        action_len = int(noisy_actions.shape[1])
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        att_masks += [1] + ([0] * (action_len - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name == "pi05_maniskill":
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
            self.value_head = self.value_head.to(
                dtype=self.action_out_proj.weight.dtype
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )
            self.noise_head = self.noise_head.to(
                dtype=self.action_out_proj.weight.dtype
            )

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def _tensor_to_numpy(self, x):
        """Convert tensor to numpy, handling BFloat16/Float16 conversion."""
        if torch.is_tensor(x):
            x_cpu = x.detach().cpu()
            # BFloat16 and Float16 are not supported by numpy, convert to float32
            if x_cpu.dtype in (torch.bfloat16, torch.float16):
                x_cpu = x_cpu.float()
            return np.asarray(x_cpu)
        return x

    def _tensor_to_numpy_single(self, x, index):
        """Convert single tensor element to numpy, handling BFloat16/Float16 conversion."""
        if torch.is_tensor(x):
            x_cpu = x[index].detach().cpu()
            # BFloat16 and Float16 are not supported by numpy, convert to float32
            if x_cpu.dtype in (torch.bfloat16, torch.float16):
                x_cpu = x_cpu.float()
            return np.asarray(x_cpu)
        return x[index]

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy (Convert BFloat16/Float16 to float32 for numpy compatibility)
        inputs = jax.tree.map(self._tensor_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs, *, limit_chunk: bool = True):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: self._tensor_to_numpy_single(x, i), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        if limit_chunk:
            outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return super().forward(observation, actions)

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        # input transform
        observation = self.input_transform(data, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation - ensure float32 to prevent BFloat16 conversion issues
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            state = env_obs["states"]
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed_obs["observation/state"] = state
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        return_obs=True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if mode == "eval":
            if getattr(self.config, "enable_speculative", False):
                return self._predict_action_batch_spec(env_obs=env_obs, return_obs=return_obs)
            if self._rollout_segment_enabled():
                return self._predict_action_batch_segment(env_obs=env_obs, return_obs=return_obs)
            return self._predict_action_batch_full(env_obs=env_obs, return_obs=return_obs)
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "observation/image": env_obs["main_images"],
            "observation/state": env_obs["states"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        if env_obs["wrist_images"] is not None:
            forward_inputs["observation/wrist_image"] = env_obs["wrist_images"]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def _predict_action_batch_full(self, env_obs, return_obs=True) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(observation, mode="eval", compute_values=False)
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}, limit_chunk=False
        )["actions"].numpy()
        eval_horizon = self._eval_action_horizon()
        if actions.shape[1] > int(eval_horizon):
            actions = actions[:, : int(eval_horizon)]
        result: dict[str, Any] = {}
        if return_obs:
            result["forward_inputs"] = None
        return actions, result

    def _predict_action_batch_segment(self, env_obs, return_obs=True) -> tuple[np.ndarray, dict[str, Any]]:
        env_batch = env_obs["states"].shape[0]
        actions_out = []
        for env_idx in range(env_batch):
            single_env_obs = self._slice_env_obs(env_obs, env_idx)
            action_seq = self._segment_decode_single(single_env_obs)
            actions_out.append(action_seq)
        actions = np.stack(actions_out, axis=0)
        eval_horizon = self._eval_action_horizon()
        if actions.shape[1] > int(eval_horizon):
            actions = actions[:, : int(eval_horizon)]
        result: dict[str, Any] = {}
        if return_obs:
            result["forward_inputs"] = None
        return actions, result

    def _predict_action_batch_spec(self, env_obs, return_obs=True) -> tuple[np.ndarray, dict[str, Any]]:
        env_batch = env_obs["states"].shape[0]
        actions_out = []
        spec_infos = []
        for env_idx in range(env_batch):
            single_env_obs = self._slice_env_obs(env_obs, env_idx)
            action_chunk, spec_info = self._spec_decode_single(single_env_obs)
            actions_out.append(action_chunk)
            spec_infos.append(spec_info)
        actions = np.stack(actions_out, axis=0)
        result = {"spec_info": spec_infos}
        if return_obs:
            result["forward_inputs"] = None
        return actions, result

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

    def _spec_decode_single(self, env_obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        action_horizon = self._spec_action_horizon()
        diffusion_num_steps = self._spec_diffusion_steps()
        batch_size = self._spec_batch_size()
        spec_chunk_size = int(self.config.spec_chunk_size)
        rollout_segment_size = self._rollout_segment_size()
        conf_alpha = float(self.config.spec_conf_alpha)
        conf_eps = float(self.config.spec_conf_eps)
        chunked_draft = self._rollout_segment_enabled()
        spec_debug = bool(self.config.spec_debug)
        timing_enabled = bool(self.config.spec_profile_timing)
        verify_conf_enabled = self._spec_verify_conf_enabled()
        verify_seq_enabled = self._spec_verify_seq_enabled()

        action_chunk = int(self.config.action_chunk)
        if action_horizon < action_chunk:
            logger.warning(
                "spec_action_horizon (%d) < action_chunk (%d); spec will generate shorter sequences.",
                action_horizon,
                action_chunk,
            )
            if spec_debug:
                self._append_spec_log(
                    f"spec_action_horizon {action_horizon} < action_chunk {action_chunk}; using shorter sequences"
                )

        delta_thresholds = self._spec_delta_thresholds()
        chunk, info = self._speculative_decode_chunk(
            observation,
            batch_size=batch_size,
            action_horizon=action_horizon,
            diffusion_num_steps=diffusion_num_steps,
            conf_alpha=conf_alpha,
            conf_eps=conf_eps,
            delta_thresholds=delta_thresholds,
            spec_debug=spec_debug,
            spec_chunk_size=spec_chunk_size,
            rollout_segment_size=rollout_segment_size,
            chunked_draft=chunked_draft,
            timing_enabled=timing_enabled,
            output_len=action_horizon,
            verify_conf_enabled=verify_conf_enabled,
            verify_seq_enabled=verify_seq_enabled,
        )
        return chunk, info

    def _segment_decode_single(self, env_obs: dict[str, Any]) -> np.ndarray:
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        action_horizon = int(self.config.action_horizon)
        diffusion_num_steps = int(self.config.num_steps)
        rollout_segment_size = self._rollout_segment_size()
        timing_enabled = bool(self.config.spec_profile_timing)

        actions_exec, _actions_raw_full, _prefix_cache, _timing_info = self._generate_chunked_draft(
            observation,
            batch_size=1,
            diffusion_num_steps=diffusion_num_steps,
            action_horizon=action_horizon,
            rollout_segment_size=rollout_segment_size,
            chunked=True,
            timing_enabled=timing_enabled,
        )
        return np.asarray(actions_exec[0])

    def _spec_action_horizon(self) -> int:
        action_horizon = getattr(self.config, "spec_action_horizon", None)
        if action_horizon is None:
            action_horizon = int(self.config.action_horizon)
        action_horizon = int(action_horizon)
        if action_horizon > int(self.config.action_horizon):
            raise ValueError(
                f"spec_action_horizon={action_horizon} must be <= model action_horizon={int(self.config.action_horizon)}"
            )
        return action_horizon

    def _eval_action_horizon(self) -> int:
        action_horizon = int(self.config.action_horizon)
        eval_horizon = getattr(self.config, "eval_action_horizon", None)
        if eval_horizon is None:
            return action_horizon
        eval_horizon = int(eval_horizon)
        if eval_horizon < 1:
            raise ValueError(f"eval_action_horizon must be >= 1, got {eval_horizon}")
        if eval_horizon > action_horizon:
            raise ValueError(
                f"eval_action_horizon={eval_horizon} must be <= model action_horizon={action_horizon}"
            )
        return eval_horizon

    def _spec_diffusion_steps(self) -> int:
        diffusion_num_steps = getattr(self.config, "spec_diffusion_num_steps", None)
        if diffusion_num_steps is None:
            diffusion_num_steps = int(self.config.num_steps)
        diffusion_num_steps = int(diffusion_num_steps)
        if diffusion_num_steps < 1:
            raise ValueError(f"spec_diffusion_num_steps must be >= 1, got {diffusion_num_steps}")
        return diffusion_num_steps

    def _rollout_segment_enabled(self) -> bool:
        value = getattr(self.config, "rollout_segment", None)
        if value is None:
            return False
        return bool(value)

    def _rollout_segment_size(self) -> int:
        value = getattr(self.config, "rollout_segment_size", None)
        if value is None:
            value = getattr(self.config, "spec_rollout_segment_size", None)
        if value is None:
            value = 5
        size = int(value)
        if size < 1:
            raise ValueError(f"rollout_segment_size must be >= 1, got {size}")
        return size

    def _spec_batch_size(self) -> int:
        batch_size = int(getattr(self.config, "spec_batch_size", 1))
        if batch_size < 1:
            raise ValueError(f"spec_batch_size must be >= 1, got {batch_size}")
        return batch_size

    def _spec_verify_conf_enabled(self) -> bool:
        return bool(getattr(self.config, "spec_verify_conf", True))

    def _spec_verify_seq_enabled(self) -> bool:
        return bool(getattr(self.config, "spec_verify_seq", True))

    def _spec_delta_thresholds(self) -> np.ndarray:
        value = getattr(self.config, "spec_delta_thresholds", None)
        if value is None:
            return np.repeat(np.float32(self.config.spec_delta_threshold), 6)
        if isinstance(value, str):
            raw = value.strip()
            parts = [p for p in raw.replace(",", " ").split() if p]
            vals = np.asarray([float(p) for p in parts], dtype=np.float32)
        else:
            vals = np.asarray(value, dtype=np.float32)
        if vals.size == 1:
            vals = np.repeat(vals, 6)
        return vals

    def _spec_conf_dim(self, action_dim: int) -> int:
        if action_dim >= 7:
            return 6
        if action_dim > 1:
            return action_dim - 1
        return action_dim

    def _spec_compare_dim(self, pred_dim: int, draft_dim: int) -> int:
        """Resolve comparison dimension for speculative verification.

        Use environment action dimensionality (if available) so thresholds for
        env-space actions (e.g. metaworld 4-dim -> compare first 3 dims) work
        even when model-space vectors are wider.
        """
        env_dim = int(getattr(self.config, "action_env_dim", min(pred_dim, draft_dim)))
        compare_dim = self._spec_conf_dim(env_dim)
        return max(0, min(int(compare_dim), int(pred_dim), int(draft_dim)))

    def _take_tree_first_batch(self, value):
        if value is None:
            return None
        if hasattr(value, "batch_split"):
            batch_size = self._cache_batch_size(value)
            if batch_size is not None and batch_size > 1:
                try:
                    split = value.batch_split(full_batch_size=int(batch_size), split_size=1)
                    if split:
                        return split[0]
                except Exception:
                    pass
            if batch_size is not None and batch_size <= 1:
                return value
            legacy_cache = self._legacy_cache_from_cache(value)
            if legacy_cache is not None:
                try:
                    from transformers.cache_utils import DynamicCache

                    return DynamicCache.from_legacy_cache(self._legacy_cache_take_first(legacy_cache))
                except Exception:
                    return self._legacy_cache_take_first(legacy_cache)
            return value
        legacy_cache = self._legacy_cache_from_cache(value)
        if legacy_cache is not None:
            try:
                from transformers.cache_utils import DynamicCache

                return DynamicCache.from_legacy_cache(self._legacy_cache_take_first(legacy_cache))
            except Exception:
                return self._legacy_cache_take_first(legacy_cache)
        if torch.is_tensor(value):
            return value[:1]
        if isinstance(value, (list, tuple)):
            items = [self._take_tree_first_batch(v) for v in value]
            return type(value)(items)
        if isinstance(value, dict):
            return {k: self._take_tree_first_batch(v) for k, v in value.items()}
        return value

    def _observation_first_batch(self, observation: _model.Observation) -> _model.Observation:
        return _model.Observation(
            images=self._take_tree_first_batch(observation.images),
            image_masks=self._take_tree_first_batch(observation.image_masks),
            state=self._take_tree_first_batch(observation.state),
            tokenized_prompt=self._take_tree_first_batch(observation.tokenized_prompt),
            tokenized_prompt_mask=self._take_tree_first_batch(observation.tokenized_prompt_mask),
            token_ar_mask=self._take_tree_first_batch(observation.token_ar_mask),
            token_loss_mask=self._take_tree_first_batch(observation.token_loss_mask),
        )

    def _cache_batch_size(self, cache) -> int | None:
        key_cache = getattr(cache, "key_cache", None)
        if isinstance(key_cache, list) and key_cache:
            return int(key_cache[0].shape[0])
        value_cache = getattr(cache, "value_cache", None)
        if isinstance(value_cache, list) and value_cache:
            return int(value_cache[0].shape[0])
        return None

    def _legacy_cache_from_cache(self, cache):
        if isinstance(cache, tuple):
            return cache
        if hasattr(cache, "to_legacy_cache"):
            try:
                return cache.to_legacy_cache()
            except Exception:
                return None
        return None

    def _legacy_cache_batch_size(self, legacy_cache) -> int | None:
        if not legacy_cache:
            return None
        key = legacy_cache[0][0]
        return int(key.shape[0])

    def _legacy_cache_take_first(self, legacy_cache):
        return tuple((k[:1], v[:1]) for (k, v) in legacy_cache)

    def _legacy_cache_repeat(self, legacy_cache, batch_size: int):
        return tuple(
            (
                k.repeat_interleave(int(batch_size), dim=0),
                v.repeat_interleave(int(batch_size), dim=0),
            )
            for (k, v) in legacy_cache
        )

    def _normalize_past_key_values(self, past_key_values):
        if past_key_values is None:
            return None
        if isinstance(past_key_values, tuple):
            try:
                from transformers.cache_utils import DynamicCache

                return DynamicCache.from_legacy_cache(past_key_values)
            except Exception:
                return past_key_values
        return past_key_values

    def _expand_tree_batch(self, value, batch_size: int):
        if value is None:
            return None
        if hasattr(value, "batch_repeat_interleave"):
            current = self._cache_batch_size(value)
            if current is None or int(current) == int(batch_size):
                return value
            if int(current) == 1:
                legacy_cache = self._legacy_cache_from_cache(value)
                if legacy_cache is not None:
                    try:
                        from transformers.cache_utils import DynamicCache

                        return DynamicCache.from_legacy_cache(
                            self._legacy_cache_repeat(legacy_cache, batch_size)
                        )
                    except Exception:
                        return self._legacy_cache_repeat(legacy_cache, batch_size)
                try:
                    import copy

                    value_copy = copy.deepcopy(value)
                    value_copy.batch_repeat_interleave(int(batch_size))
                    return value_copy
                except Exception as exc:
                    raise ValueError(
                        f"Expected cache batch 1 or {int(batch_size)}, got {int(current)}"
                    ) from exc
            legacy_cache = self._legacy_cache_from_cache(value)
            if legacy_cache is not None:
                current_legacy = self._legacy_cache_batch_size(legacy_cache)
                if current_legacy == 1:
                    try:
                        from transformers.cache_utils import DynamicCache

                        return DynamicCache.from_legacy_cache(
                            self._legacy_cache_repeat(legacy_cache, batch_size)
                        )
                    except Exception:
                        return self._legacy_cache_repeat(legacy_cache, batch_size)
            raise ValueError(
                f"Expected cache batch 1 or {int(batch_size)}, got {int(current)}"
            )
        legacy_cache = self._legacy_cache_from_cache(value)
        if legacy_cache is not None:
            current_legacy = self._legacy_cache_batch_size(legacy_cache)
            if current_legacy is None or int(current_legacy) == int(batch_size):
                return self._normalize_past_key_values(legacy_cache)
            if int(current_legacy) == 1:
                try:
                    from transformers.cache_utils import DynamicCache

                    return DynamicCache.from_legacy_cache(
                        self._legacy_cache_repeat(legacy_cache, batch_size)
                    )
                except Exception:
                    return self._legacy_cache_repeat(legacy_cache, batch_size)
            raise ValueError(
                f"Expected cache batch 1 or {int(batch_size)}, got {int(current_legacy)}"
            )
        if torch.is_tensor(value):
            if int(value.shape[0]) == int(batch_size):
                return value
            if int(value.shape[0]) == 1:
                return value.repeat(batch_size, *([1] * (value.ndim - 1)))
            raise ValueError(f"Expected batch 1 or {batch_size}, got {int(value.shape[0])}")
        if isinstance(value, (list, tuple)):
            items = [self._expand_tree_batch(v, batch_size) for v in value]
            return type(value)(items)
        if isinstance(value, dict):
            return {k: self._expand_tree_batch(v, batch_size) for k, v in value.items()}
        return value

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

    def _compute_prefix_cache(
        self,
        observation: _model.Observation,
        *,
        profile_timing: bool = False,
    ) -> tuple[Any, Any]:
        observation_single = self._observation_first_batch(observation)
        device = observation_single.state.device
        timing_enabled = bool(profile_timing) and device.type == "cuda"
        if timing_enabled:
            torch.cuda.synchronize(device=device)
            t0 = time.perf_counter()
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(
            observation_single, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (_, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        past_key_values = self._normalize_past_key_values(past_key_values)
        if timing_enabled:
            torch.cuda.synchronize(device=device)
            t1 = time.perf_counter()
            self.last_timing_prefill_ms = float((t1 - t0) * 1000.0)
        return (
            self._take_tree_first_batch(prefix_pad_masks),
            self._take_tree_first_batch(past_key_values),
        )

    def _spec_sample_actions(
        self,
        observation: _model.Observation,
        *,
        num_steps: int,
        action_horizon: int | None = None,
        fixed_actions: torch.Tensor | None = None,
        fixed_action_mask: torch.Tensor | None = None,
        prefix_cache=None,
        capture_prefix_cache: bool = False,
        profile_timing: bool = False,
    ) -> torch.Tensor:
        bsize = observation.state.shape[0]
        horizon = int(self.config.action_horizon) if action_horizon is None else int(action_horizon)
        if horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {horizon}")
        if horizon > int(self.config.action_horizon):
            raise ValueError(
                f"action_horizon={horizon} must be <= model action_horizon={int(self.config.action_horizon)}"
            )

        device = observation.state.device
        actions_shape = (bsize, horizon, self.config.action_dim)
        noise = self.sample_noise(actions_shape, device)

        if fixed_actions is None:
            fixed_actions = torch.zeros_like(noise)
        else:
            if fixed_actions.ndim == 2:
                fixed_actions = fixed_actions[None, ...].expand(bsize, -1, -1)
            if int(fixed_actions.shape[1]) < horizon:
                raise ValueError(
                    f"fixed_actions.shape[1]={int(fixed_actions.shape[1])} is smaller than action_horizon={horizon}"
                )
            if int(fixed_actions.shape[1]) > horizon:
                fixed_actions = fixed_actions[:, :horizon, :]

        if fixed_action_mask is None:
            fixed_action_mask = torch.zeros((bsize, horizon), dtype=torch.bool, device=noise.device)
        else:
            if fixed_action_mask.ndim == 1:
                fixed_action_mask = fixed_action_mask[None, ...].expand(bsize, -1)
            if int(fixed_action_mask.shape[1]) < horizon:
                raise ValueError(
                    f"fixed_action_mask.shape[1]={int(fixed_action_mask.shape[1])} is smaller than action_horizon={horizon}"
                )
            if int(fixed_action_mask.shape[1]) > horizon:
                fixed_action_mask = fixed_action_mask[:, :horizon, ...]
        if fixed_action_mask.ndim == 2:
            fixed_action_mask = fixed_action_mask[:, :, None]
        elif fixed_action_mask.ndim != 3:
            raise ValueError(f"Expected fixed_action_mask ndim 2 or 3, got {fixed_action_mask.ndim}")

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        timing_enabled = bool(profile_timing) and device.type == "cuda"
        use_prefix_cache = False
        if prefix_cache is not None:
            prefix_pad_masks, past_key_values = prefix_cache
            past_key_values = self._normalize_past_key_values(past_key_values)
            try:
                prefix_pad_masks = self._expand_tree_batch(prefix_pad_masks, bsize)
                past_key_values = self._expand_tree_batch(past_key_values, bsize)
                use_prefix_cache = True
            except ValueError as exc:
                if getattr(self.config, "spec_debug", False):
                    self._append_spec_log(f"spec_prefix_cache_skip reason={exc}")
                use_prefix_cache = False
        if not use_prefix_cache and capture_prefix_cache:
            prefix_cache = self._compute_prefix_cache(observation, profile_timing=profile_timing)
            self.last_prefix_cache = prefix_cache
            prefix_pad_masks, past_key_values = prefix_cache
            past_key_values = self._normalize_past_key_values(past_key_values)
            prefix_pad_masks = self._expand_tree_batch(prefix_pad_masks, bsize)
            past_key_values = self._expand_tree_batch(past_key_values, bsize)
            use_prefix_cache = True
        if not use_prefix_cache:
            if timing_enabled:
                torch.cuda.synchronize(device=device)
                t0 = time.perf_counter()
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            (_, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            if capture_prefix_cache:
                past_key_values = self._normalize_past_key_values(past_key_values)
                self.last_prefix_cache = (
                    self._take_tree_first_batch(prefix_pad_masks),
                    self._take_tree_first_batch(past_key_values),
                )
            if timing_enabled:
                torch.cuda.synchronize(device=device)
                t1 = time.perf_counter()
                self.last_timing_prefill_ms = float((t1 - t0) * 1000.0)

        x_t = noise
        timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        if timing_enabled:
            torch.cuda.synchronize(device=device)
            t2 = time.perf_counter()
        for idx in range(int(num_steps)):
            t_input = timesteps[idx]
            t_next = timesteps[idx + 1]

            time_expanded = t_input.expand(bsize)[:, None, None]
            fixed_x_t = time_expanded * noise + (1.0 - time_expanded) * fixed_actions
            x_t = torch.where(fixed_action_mask, fixed_x_t, x_t)

            x_t_mean, x_t_std, _value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                "eval",
                num_steps,
                compute_values=False,
            )
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std

            time_expanded_next = t_next.expand(bsize)[:, None, None]
            fixed_x_t_next = time_expanded_next * noise + (1.0 - time_expanded_next) * fixed_actions
            x_t = torch.where(fixed_action_mask, fixed_x_t_next, x_t)
        if timing_enabled:
            torch.cuda.synchronize(device=device)
            t3 = time.perf_counter()
            self.last_timing_diffusion_ms = float((t3 - t2) * 1000.0)
        return x_t

    def _repeat_tensor(self, value, batch_size: int):
        if value is None:
            return None
        if torch.is_tensor(value):
            if int(value.shape[0]) == int(batch_size):
                return value
            if int(value.shape[0]) == 1:
                return value.repeat(batch_size, *([1] * (value.ndim - 1)))
            raise ValueError(f"Expected batch 1 or {batch_size}, got {int(value.shape[0])}")
        value = np.asarray(value)
        if int(value.shape[0]) == int(batch_size):
            return value
        if int(value.shape[0]) == 1:
            return np.repeat(value, batch_size, axis=0)
        raise ValueError(f"Expected batch 1 or {batch_size}, got {int(value.shape[0])}")

    def _repeat_observation(self, observation: _model.Observation, batch_size: int) -> _model.Observation:
        images = {k: self._repeat_tensor(v, batch_size) for k, v in observation.images.items()}
        image_masks = {k: self._repeat_tensor(v, batch_size) for k, v in observation.image_masks.items()}
        state = self._repeat_tensor(observation.state, batch_size)
        tokenized_prompt = self._repeat_tensor(observation.tokenized_prompt, batch_size)
        tokenized_prompt_mask = self._repeat_tensor(observation.tokenized_prompt_mask, batch_size)
        token_ar_mask = self._repeat_tensor(observation.token_ar_mask, batch_size)
        token_loss_mask = self._repeat_tensor(observation.token_loss_mask, batch_size)
        return _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        )

    def _apply_output_transform(
        self, actions: torch.Tensor, state: torch.Tensor, *, limit_chunk: bool
    ) -> np.ndarray:
        outputs = self.output_transform({"actions": actions, "state": state}, limit_chunk=limit_chunk)
        return outputs["actions"].numpy()

    def _compute_confidence(
        self, actions: np.ndarray, *, alpha: float, eps: float, conf_dim: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions_3d = np.asarray(actions)
        if actions_3d.ndim == 2:
            actions_3d = actions_3d[None, ...]
        if actions_3d.ndim != 3:
            raise ValueError(f"Expected actions to have shape (B,H,D) or (H,D), got {actions_3d.shape}")
        use_dim = min(conf_dim, actions_3d.shape[-1])
        u = np.cumsum(actions_3d[:, :, :use_dim], axis=1)
        mu = np.mean(u, axis=0)
        var = np.var(u, axis=0)
        d2 = np.sum((u - mu) ** 2 / (var + eps), axis=-1)
        conf_cont = np.exp(-0.5 * d2)
        conf = conf_cont
        tau = np.median(conf, axis=0)
        high = conf >= tau[None, :]
        mu_abs = np.abs(mu)
        conf_stats = {
            "u_abs_mean": float(np.mean(np.abs(u))) if u.size else float("nan"),
            "u_abs_max": float(np.max(np.abs(u))) if u.size else float("nan"),
            "mu_abs_mean": float(np.mean(mu_abs)) if mu_abs.size else float("nan"),
            "mu_abs_max": float(np.max(mu_abs)) if mu_abs.size else float("nan"),
            "mu_abs_mean_dim": (
                np.mean(mu_abs, axis=0).astype(np.float32).tolist()
                if mu_abs.ndim == 2 and mu_abs.shape[1] > 0
                else []
            ),
            "mu_abs_max_dim": (
                np.max(mu_abs, axis=0).astype(np.float32).tolist()
                if mu_abs.ndim == 2 and mu_abs.shape[1] > 0
                else []
            ),
            "var_mean": float(np.mean(var)) if var.size else float("nan"),
            "var_max": float(np.max(var)) if var.size else float("nan"),
            "var_min": float(np.min(var)) if var.size else float("nan"),
            "var_mean_dim": (
                np.mean(var, axis=0).astype(np.float32).tolist()
                if var.ndim == 2 and var.shape[1] > 0
                else []
            ),
            "var_max_dim": (
                np.max(var, axis=0).astype(np.float32).tolist()
                if var.ndim == 2 and var.shape[1] > 0
                else []
            ),
            "var_min_dim": (
                np.min(var, axis=0).astype(np.float32).tolist()
                if var.ndim == 2 and var.shape[1] > 0
                else []
            ),
            "conf_mean": float(np.mean(conf)) if conf.size else float("nan"),
            "conf_std": float(np.std(conf)) if conf.size else float("nan"),
            "tau_mean": float(np.mean(tau)) if tau.size else float("nan"),
            "alpha": float(alpha),
            "eps": float(eps),
        }
        return conf.astype(np.float32), tau.astype(np.float32), high, conf_stats

    def _select_draft(self, actions: np.ndarray, *, alpha: float, eps: float, conf_dim: int) -> dict[str, Any]:
        conf, tau, high, conf_stats = self._compute_confidence(
            actions, alpha=alpha, eps=eps, conf_dim=conf_dim
        )
        count = np.sum(high, axis=1).astype(np.int32)
        sum_conf = np.sum(conf * high, axis=1).astype(np.float32)
        best_count = int(np.max(count))
        candidates = np.flatnonzero(count == best_count)
        best = int(candidates[0]) if candidates.size == 1 else int(candidates[np.argmax(sum_conf[candidates])])
        return {
            "conf": conf,
            "tau": tau,
            "high": high,
            "count": count,
            "sum": sum_conf,
            "best": best,
            "conf_stats": conf_stats,
        }

    def _action_match(self, pred: np.ndarray, draft: np.ndarray, *, delta_thresholds: np.ndarray) -> bool:
        pred_1d = np.asarray(pred).reshape(-1)
        draft_1d = np.asarray(draft).reshape(-1)
        compare_dim = self._spec_compare_dim(int(pred_1d.shape[-1]), int(draft_1d.shape[-1]))
        if delta_thresholds.size < compare_dim:
            raise ValueError(
                f"delta_thresholds must have at least {compare_dim} elements, got {delta_thresholds.size}"
            )
        thr = np.asarray(delta_thresholds, dtype=np.float32).reshape(-1)[:compare_dim]
        return bool(np.all(np.abs(pred_1d[:compare_dim] - draft_1d[:compare_dim]) < thr))

    def _generate_chunked_draft(
        self,
        observation: _model.Observation,
        *,
        batch_size: int,
        diffusion_num_steps: int,
        action_horizon: int,
        rollout_segment_size: int,
        chunked: bool,
        timing_enabled: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, Any] | None, dict[str, float]]:
        total_horizon = int(action_horizon)
        obs_b = self._repeat_observation(observation, batch_size)
        _ = rollout_segment_size
        _ = chunked

        actions_full_raw = self._spec_sample_actions(
            obs_b,
            num_steps=int(diffusion_num_steps),
            action_horizon=int(total_horizon),
            capture_prefix_cache=True,
            profile_timing=bool(timing_enabled),
        )
        prefix_cache = getattr(self, "last_prefix_cache", None)
        actions_full_exec = self._apply_output_transform(
            actions_full_raw, obs_b.state, limit_chunk=False
        )

        timing_info: dict[str, float] = {}
        if timing_enabled:
            timing_info["prefill_ms"] = float(
                getattr(self, "last_timing_prefill_ms", 0.0)
            )
            timing_info["diffusion_ms_full"] = float(
                getattr(self, "last_timing_diffusion_ms", 0.0)
            )

        return (
            actions_full_exec,
            actions_full_raw.detach().cpu().numpy(),
            prefix_cache,
            timing_info,
        )

    def _speculative_decode_chunk(
        self,
        observation: _model.Observation,
        *,
        batch_size: int,
        action_horizon: int,
        diffusion_num_steps: int,
        conf_alpha: float,
        conf_eps: float,
        delta_thresholds: np.ndarray,
        spec_debug: bool,
        spec_chunk_size: int,
        rollout_segment_size: int,
        chunked_draft: bool,
        timing_enabled: bool,
        output_len: int,
        verify_conf_enabled: bool,
        verify_seq_enabled: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        actions_exec, actions_raw_full, prefix_cache, timing_info = self._generate_chunked_draft(
            observation,
            batch_size=int(batch_size),
            diffusion_num_steps=int(diffusion_num_steps),
            action_horizon=int(action_horizon),
            rollout_segment_size=int(rollout_segment_size),
            chunked=bool(chunked_draft),
            timing_enabled=bool(timing_enabled),
        )

        conf_enabled = bool(verify_conf_enabled)
        seq_enabled = bool(verify_seq_enabled)
        if not conf_enabled and not seq_enabled:
            raise ValueError("At least one of spec_verify_conf or spec_verify_seq must be True")

        conf_dim = self._spec_conf_dim(int(actions_exec.shape[-1]))
        selection = self._select_draft(actions_exec, alpha=float(conf_alpha), eps=float(conf_eps), conf_dim=conf_dim)
        conf = np.asarray(selection["conf"])
        conf_stats = selection.get("conf_stats", {})
        log_conf_stats = bool(getattr(self.config, "spec_log_conf_stats", True))
        selected_path = int(selection["best"])
        best_first_idx = selected_path

        b, h, d_exec = actions_exec.shape
        d_model = int(actions_raw_full.shape[2])
        greedy_chain_exec = np.zeros((h, d_exec), dtype=actions_exec.dtype)
        greedy_chain_exec[0] = actions_exec[best_first_idx, 0]
        greedy_chain_raw_full = np.zeros((h, d_model), dtype=actions_raw_full.dtype)
        greedy_chain_raw_full[0] = actions_raw_full[best_first_idx, 0]
        for t in range(1, h):
            greedy_chain_exec[t] = actions_exec[selected_path, t]
            greedy_chain_raw_full[t] = actions_raw_full[selected_path, t]

        draft_conf_per_t = conf[selected_path].astype(np.float32)
        chunk_size = int(spec_chunk_size)
        if chunk_size <= 0:
            chunk_size = int(h)
        if h % chunk_size != 0:
            raise ValueError(f"spec_chunk_size must divide action horizon: h={h} chunk={chunk_size}")

        chunk_summaries: list[dict[str, Any]] = []
        chunk_orders: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for start in range(0, h, chunk_size):
            end = min(start + chunk_size, h)
            pos = np.arange(start, end, dtype=np.int64)
            order_conf = pos[np.lexsort((pos, -draft_conf_per_t[pos]))]
            order_seq = pos
            chunk_summaries.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "top_pos": int(order_conf[0]) if order_conf.size else None,
                    "top_conf": float(draft_conf_per_t[int(order_conf[0])]) if order_conf.size else None,
                }
            )
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
                compare_dim = self._spec_compare_dim(int(pred_1d.shape[0]), int(draft_1d.shape[0]))
                diff = pred_1d[:compare_dim] - draft_1d[:compare_dim]
                abs_diff = np.abs(diff)
                over = abs_diff >= np.asarray(delta_thresholds[:compare_dim], dtype=np.float32)
                conf_pos = float(draft_conf_per_t[pos]) if 0 <= pos < int(draft_conf_per_t.shape[0]) else float("nan")
                reject_detail = {
                    "kind": kind,
                    "pos": int(pos),
                    "rank": int(i),
                    "conf": conf_pos,
                    "abs_diff_max": float(np.max(abs_diff)) if abs_diff.size else float("nan"),
                    "over_dims": over.astype(np.int8).tolist(),
                    "space": space_label,
                    "chunk_start": int(chunk_start),
                }
                if spec_debug:
                    delta_list = np.round(
                        np.asarray(delta_thresholds[:compare_dim], dtype=np.float32), 3
                    ).tolist()
                    pred_list = np.round(pred_1d[:compare_dim], 3).tolist()
                    draft_list = np.round(draft_1d[:compare_dim], 3).tolist()
                    self._append_spec_log(
                        "spec_reject kind={kind} pos={pos} rank={rank} conf={conf:.4f} "
                        "space={space} chunk_start={chunk_start} delta={delta} abs_diff_max={abs_diff_max:.4f} "
                        "over_dims={over_dims} pred={pred} draft={draft}".format(
                            kind=kind,
                            pos=pos,
                            rank=i,
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
                return accepted_positions, accepted_rank, False, pos, pred_exec, reject_detail
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

            fixed_actions_batch = np.zeros((k_total, verify_h, d_model), dtype=greedy_chain_raw_full.dtype)
            fixed_action_mask_batch = np.zeros((k_total, verify_h), dtype=np.bool_)

            if conf_active:
                pos_fixed_global = np.asarray(
                    sorted(p for p in accepted_positions_conf if p < int(verify_h)), dtype=np.int64
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
                    sorted(p for p in accepted_positions_seq if p < int(verify_h)), dtype=np.int64
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

            obs_verify = self._repeat_observation(observation, k_total)
            fixed_actions_t = torch.as_tensor(fixed_actions_batch, device=observation.state.device)
            fixed_action_mask_t = torch.as_tensor(fixed_action_mask_batch, device=observation.state.device)
            verify_actions_raw = self._spec_sample_actions(
                obs_verify,
                num_steps=int(diffusion_num_steps),
                action_horizon=int(verify_h),
                fixed_actions=fixed_actions_t,
                fixed_action_mask=fixed_action_mask_t,
                prefix_cache=prefix_cache,
                profile_timing=bool(timing_enabled),
            )
            verify_actions = self._apply_output_transform(
                verify_actions_raw, obs_verify.state, limit_chunk=False
            )
            verify_actions_raw_np = verify_actions_raw.detach().float().cpu().numpy()

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
        accepted_exec_len = int(accepted_actions.shape[0]) if accepted_actions.ndim > 0 else 0
        if output_len > 0:
            if chunk.shape[0] < int(output_len):
                pad_end = min(int(output_len), int(greedy_chain_exec.shape[0]))
                if chunk.shape[0] < pad_end:
                    chunk = np.concatenate([chunk, greedy_chain_exec[chunk.shape[0] : pad_end]], axis=0)
            if chunk.shape[0] < int(output_len):
                last = chunk[-1] if chunk.size else greedy_chain_exec[0]
                pad = np.repeat(last[None, ...], int(output_len) - int(chunk.shape[0]), axis=0)
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
            "best_first_idx": int(best_first_idx),
            "selected_path": selected_path,
            "spec_chunk_size": int(chunk_size),
            "verify_chunks": chunk_summaries,
            "accepted_actions": accepted_actions,
            "accepted_exec_len": int(accepted_exec_len),
            "conf_stats": conf_stats if log_conf_stats and isinstance(conf_stats, dict) else None,
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
        return chunk, info

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )  # [bs,n_action_steps,max_action_dim]
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if not self.use_vlm_value:
                chains_values.append(value_t)
        if self.use_vlm_value:
            chains_values.append(self.get_value_from_vlm(prefix_output))
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
