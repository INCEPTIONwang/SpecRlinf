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
import gc
import os
from collections import deque
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path
            spec_log_path = None
            log_dir = self.cfg.runner.logger.get("log_path", None)
            if log_dir:
                spec_log_path = os.path.join(str(log_dir), "spec_debug.log")
                if rollout_model_config.get("openpi") is None:
                    rollout_model_config.openpi = {}
                rollout_model_config.openpi.spec_log_path = spec_log_path

        self.hf_model = get_model(rollout_model_config)
        self._spec_log_path = spec_log_path
        if spec_log_path:
            setattr(self.hf_model, "spec_log_path", spec_log_path)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def _append_spec_log(self, line: str):
        path = getattr(self, "_spec_log_path", None)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            return

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def update_intervene_actions(self, env_output, forward_inputs):
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            else:
                raise NotImplementedError(f"{forward_inputs.keys()=}")
        return forward_inputs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]  # save actions

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )
                    actions, result = self.predict(extracted_obs)
                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=last_forward_inputs[stage_id],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)
                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )
                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                with self.worker_timer():
                    actions, result = self.predict(extracted_obs)
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = int(self.cfg.env.eval.max_steps_per_rollout_epoch)
        spec_accept_lengths = [None for _ in range(self.num_pipeline_stages)]
        spec_reject_counts = [None for _ in range(self.num_pipeline_stages)]
        spec_last_reject = [None for _ in range(self.num_pipeline_stages)]
        spec_total_accept = 0
        spec_total_chunks = 0
        action_plans = [None for _ in range(self.num_pipeline_stages)]
        pbar = tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        )

        def _init_spec_buffers(stage_id: int, num_envs: int):
            if spec_accept_lengths[stage_id] is not None and len(
                spec_accept_lengths[stage_id]
            ) == int(num_envs):
                return
            spec_accept_lengths[stage_id] = [[] for _ in range(num_envs)]
            spec_reject_counts[stage_id] = [
                {"conf": 0, "seq": 0} for _ in range(num_envs)
            ]
            spec_last_reject[stage_id] = [None for _ in range(num_envs)]

        def _record_spec_info(
            stage_id: int,
            spec_info_list: list[dict[str, Any]],
            env_indices: list[int] | None,
            num_envs: int,
        ):
            nonlocal spec_total_accept, spec_total_chunks
            if not isinstance(spec_info_list, list) or not spec_info_list:
                return
            _init_spec_buffers(stage_id, num_envs)
            if env_indices is None:
                env_indices = list(range(len(spec_info_list)))
            for local_idx, info in enumerate(spec_info_list):
                if not info:
                    continue
                env_idx = int(env_indices[local_idx])
                accept_len = info.get("accepted_prefix_len")
                if accept_len is not None:
                    spec_accept_lengths[stage_id][env_idx].append(int(accept_len))
                    spec_total_accept += int(accept_len)
                    spec_total_chunks += 1
                reject = info.get("reject")
                if isinstance(reject, dict):
                    kind = reject.get("kind")
                    if kind in ("conf", "seq"):
                        spec_reject_counts[stage_id][env_idx][kind] += 1
                    spec_last_reject[stage_id][env_idx] = reject

        def _ensure_action_plans(stage_id: int, num_envs: int):
            if action_plans[stage_id] is not None and len(
                action_plans[stage_id]
            ) == int(num_envs):
                return
            action_plans[stage_id] = [deque() for _ in range(num_envs)]

        def _slice_env_obs(env_obs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
            sliced: dict[str, Any] = {}
            for key, value in env_obs.items():
                if torch.is_tensor(value):
                    sliced[key] = value[indices]
                elif isinstance(value, list):
                    sliced[key] = [value[i] for i in indices]
                elif isinstance(value, dict):
                    sliced[key] = _slice_env_obs(value, indices)
                else:
                    sliced[key] = value
            return sliced

        def _clear_plans_on_done(stage_id: int, env_output: dict[str, Any]):
            if action_plans[stage_id] is None:
                return
            dones = env_output.get("dones")
            if dones is None:
                return
            if torch.is_tensor(dones):
                done_mask = dones[:, -1] if dones.ndim > 1 else dones
                done_mask = done_mask.bool().cpu().numpy()
            else:
                done_arr = np.asarray(dones)
                done_mask = done_arr[:, -1] if done_arr.ndim > 1 else done_arr
                done_mask = done_mask.astype(bool)
            for env_idx in np.flatnonzero(done_mask):
                action_plans[stage_id][int(env_idx)].clear()

        def _log_eval_info(stage_id: int, eval_info: list[dict[str, Any] | None]):
            if not isinstance(eval_info, list) or not eval_info or pbar.disable:
                return
            items = [info for info in eval_info if info]
            for env_idx, info in enumerate(eval_info):
                if not info:
                    continue
                eval_line = (
                    "metaworld_eval episode={episode} task_id={task_id} task={task} "
                    "desc={desc} difficulty={difficulty} trial_id={trial_id} "
                    "success={success} return={return:.4f} episode_len={episode_len}".format(
                        **info
                    )
                )
                self._append_spec_log(eval_line)
                if spec_accept_lengths[stage_id] is None:
                    continue
                accept_list = spec_accept_lengths[stage_id][env_idx]
                if accept_list:
                    avg_accept = float(sum(accept_list)) / float(len(accept_list))
                    counts = spec_reject_counts[stage_id][env_idx]
                    global_avg = (
                        float(spec_total_accept) / float(spec_total_chunks)
                        if spec_total_chunks > 0
                        else float("nan")
                    )
                    last_reject = spec_last_reject[stage_id][env_idx]
                    reject_msg = "none"
                    if isinstance(last_reject, dict):
                        reject_kind = last_reject.get("kind", "unknown")
                        reject_pos = last_reject.get("pos", -1)
                        reject_max = last_reject.get("abs_diff_max", float("nan"))
                        reject_over = last_reject.get("over_dims")
                        reject_msg = (
                            f"{reject_kind}@{reject_pos} max_diff={reject_max:.4f} "
                            f"over={reject_over}"
                        )
                    spec_line = (
                        "metaworld_spec episode={episode} accepted_len_avg={avg:.3f} "
                        "chunks={chunks} reject_conf={rej_conf} reject_seq={rej_seq} "
                        "global_avg={global_avg:.3f} last_reject={last_reject}".format(
                            episode=int(info["episode"]),
                            avg=avg_accept,
                            chunks=len(accept_list),
                            rej_conf=counts.get("conf", 0),
                            rej_seq=counts.get("seq", 0),
                            global_avg=global_avg,
                            last_reject=reject_msg,
                        )
                    )
                    self._append_spec_log(spec_line)
                    spec_accept_lengths[stage_id][env_idx] = []
                    spec_reject_counts[stage_id][env_idx] = {"conf": 0, "seq": 0}
                    spec_last_reject[stage_id][env_idx] = None
            if items:
                last = items[-1]
                pbar.set_postfix(
                    task=last["task"],
                    success=last["success"],
                    diff=last["difficulty"],
                )

        for _ in pbar:
            action_plans = [None for _ in range(self.num_pipeline_stages)]
            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    eval_info = env_output.get("eval_info")
                    _log_eval_info(stage_id, eval_info)
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    num_envs = 0
                    for value in extracted_obs.values():
                        if torch.is_tensor(value):
                            num_envs = int(value.shape[0])
                            break
                        if isinstance(value, list):
                            num_envs = len(value)
                            break
                    _ensure_action_plans(stage_id, num_envs)
                    _clear_plans_on_done(stage_id, env_output)

                    need_plan = [
                        idx
                        for idx, plan in enumerate(action_plans[stage_id])
                        if not plan
                    ]
                    actions_new = None
                    spec_info_list = None
                    if need_plan:
                        sliced_obs = _slice_env_obs(extracted_obs, need_plan)
                        actions_new, result = self.predict(sliced_obs, mode="eval")
                        if torch.is_tensor(actions_new):
                            actions_new = actions_new.cpu().numpy()
                        spec_info_list = (
                            result.get("spec_info") if isinstance(result, dict) else None
                        )
                        if isinstance(spec_info_list, list):
                            _record_spec_info(
                                stage_id, spec_info_list, need_plan, num_envs
                            )
                        for local_idx, env_idx in enumerate(need_plan):
                            plan_actions = None
                            if isinstance(spec_info_list, list) and local_idx < len(
                                spec_info_list
                            ):
                                info = spec_info_list[local_idx]
                                if isinstance(info, dict):
                                    accepted_actions = info.get("accepted_actions")
                                    if accepted_actions is not None:
                                        plan_actions = np.asarray(accepted_actions)
                            if plan_actions is None and actions_new is not None:
                                plan_actions = np.asarray(actions_new[local_idx])
                            if plan_actions is None:
                                continue
                            if plan_actions.ndim == 1:
                                plan_actions = plan_actions[None, ...]
                            for step_action in plan_actions:
                                action_plans[stage_id][env_idx].append(
                                    np.asarray(step_action, dtype=np.float32)
                                )

                    step_actions = []
                    for env_idx in range(num_envs):
                        plan = action_plans[stage_id][env_idx]
                        if plan:
                            step_actions.append(
                                np.asarray(plan.popleft(), dtype=np.float32)
                            )
                        else:
                            step_actions.append(
                                np.zeros(
                                    (self.cfg.actor.model.action_dim,),
                                    dtype=np.float32,
                                )
                            )
                    actions_to_send = np.stack(step_actions, axis=0)[:, None, :]
                    self.send_chunk_actions(output_channel, actions_to_send, mode="eval")
            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel, mode="eval")
                eval_info = env_output.get("eval_info")
                _log_eval_info(stage_id, eval_info)
                _clear_plans_on_done(stage_id, env_output)
        pbar.close()

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        # send rollout_batch to actor
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
