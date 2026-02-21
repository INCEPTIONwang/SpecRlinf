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

import typing

import torch

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.envs.metaworld import MetaWorldBenchmark

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics_list_for_agg = eval_metrics_list
        if self._is_metaworld_eval():
            self._log_metaworld_eval_details(eval_metrics_list)
            eval_metrics_list_for_agg = [
                {k: v for k, v in metrics.items() if k not in {"task_id", "trial_id"}}
                for metrics in eval_metrics_list
            ]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list_for_agg)
        return eval_metrics

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()

    def _is_metaworld_eval(self) -> bool:
        env_cfg = getattr(self.cfg, "env", None)
        if env_cfg is None:
            return False
        eval_cfg = getattr(env_cfg, "eval", None)
        if eval_cfg is None:
            return False
        return getattr(eval_cfg, "env_type", None) == "metaworld"

    def _log_metaworld_eval_details(self, eval_metrics_list):
        if not eval_metrics_list:
            return
        if "task_id" not in eval_metrics_list[0]:
            return

        metrics = {}
        for key in eval_metrics_list[0].keys():
            values = [m[key] for m in eval_metrics_list if key in m]
            if not values:
                continue
            if torch.is_tensor(values[0]):
                metrics[key] = torch.concat(values, dim=0).cpu()

        task_ids = metrics.get("task_id")
        if task_ids is None:
            return
        task_ids_np = task_ids.numpy().astype(int)
        trial_ids = metrics.get("trial_id")
        trial_ids_np = trial_ids.numpy().astype(int) if trial_ids is not None else None

        success = metrics.get("success_once", metrics.get("success_at_end"))
        if success is None:
            return
        success_np = success.numpy()
        success_flags = success_np > 0.5 if success_np.dtype != bool else success_np

        returns = metrics.get("return")
        returns_np = returns.numpy() if returns is not None else None
        episode_len = metrics.get("episode_len")
        episode_len_np = episode_len.numpy().astype(int) if episode_len is not None else None

        task_suite_name = self.cfg.env.eval.task_suite_name
        benchmark = MetaWorldBenchmark(task_suite_name)
        env_names = benchmark.get_env_names()
        difficulty_map = benchmark.get_task_difficulty_map()

        total_episodes = int(task_ids_np.shape[0])
        task_stats = {}
        diff_stats = {}
        total_success = 0

        for idx, task_id in enumerate(task_ids_np):
            env_name = env_names[task_id] if 0 <= task_id < len(env_names) else f"task_{task_id}"
            difficulty = difficulty_map.get(env_name, "unknown")
            success_flag = bool(success_flags[idx])
            total_success += int(success_flag)

            task_entry = task_stats.setdefault(task_id, {"success": 0, "total": 0})
            task_entry["success"] += int(success_flag)
            task_entry["total"] += 1

            diff_entry = diff_stats.setdefault(difficulty, {"success": 0, "total": 0})
            diff_entry["success"] += int(success_flag)
            diff_entry["total"] += 1

            trial_id = int(trial_ids_np[idx]) if trial_ids_np is not None else -1
            return_val = float(returns_np[idx]) if returns_np is not None else float("nan")
            episode_len_val = int(episode_len_np[idx]) if episode_len_np is not None else -1
            self.logger.info(
                "metaworld_eval episode=%d/%d task_id=%d task=%s difficulty=%s trial_id=%d success=%s return=%.4f episode_len=%d",
                idx + 1,
                total_episodes,
                task_id,
                env_name,
                difficulty,
                trial_id,
                success_flag,
                return_val,
                episode_len_val,
            )

        for task_id in sorted(task_stats.keys()):
            env_name = env_names[task_id] if 0 <= task_id < len(env_names) else f"task_{task_id}"
            difficulty = difficulty_map.get(env_name, "unknown")
            entry = task_stats[task_id]
            rate = (entry["success"] / entry["total"] * 100.0) if entry["total"] > 0 else 0.0
            self.logger.info(
                "metaworld_task_summary task_id=%d task=%s difficulty=%s success_rate=%.1f%% successes=%d episodes=%d",
                task_id,
                env_name,
                difficulty,
                rate,
                entry["success"],
                entry["total"],
            )

        for difficulty in ["easy", "medium", "hard", "very_hard", "unknown"]:
            if difficulty not in diff_stats:
                continue
            entry = diff_stats[difficulty]
            rate = (entry["success"] / entry["total"] * 100.0) if entry["total"] > 0 else 0.0
            label = "Very Hard" if difficulty == "very_hard" else difficulty.title()
            self.logger.info(
                "metaworld_difficulty_summary difficulty=%s success_rate=%.1f%% successes=%d episodes=%d",
                label,
                rate,
                entry["success"],
                entry["total"],
            )

        overall_rate = (total_success / total_episodes * 100.0) if total_episodes > 0 else 0.0
        self.logger.info(
            "metaworld_difficulty_summary difficulty=Avg success_rate=%.1f%% successes=%d episodes=%d",
            overall_rate,
            total_success,
            total_episodes,
        )
