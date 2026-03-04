#!/usr/bin/env bash
set -euo pipefail

# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-5base
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10base
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10ar
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10sp
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15base
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15sp
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15ar
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10sp
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15sp

# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-5base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-6base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-7base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-8base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-9base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-10base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-11base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-12base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-13base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-14base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-15base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-15sp

bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openpi_pi05-10sp_time

# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-1base
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-2base
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-3base
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-4base
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-5base
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-6base
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-7base
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-8base


# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-8sp \
#     actor.model.openpi.spec_verify_conf=true \
#     actor.model.openpi.spec_verify_seq=true
# bash examples/embodiment/eval_embodiment.sh maniskill_ppo_openpi_pi05-8sp \
#     actor.model.openpi.spec_verify_conf=false \
#     actor.model.openpi.spec_verify_seq=true

# source /home/feng/data/wxh/SpecRlinf/gr00t-libero/bin/activate

# bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_gr00t-1base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_gr00t-2base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_gr00t-3base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_gr00t-4base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_gr00t-5base
# bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=true \
#     actor.model.spec_verify_seq=true
# bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=false \
#     actor.model.spec_verify_seq=true

# bash examples/embodiment/eval_embodiment.sh libero_object_ppo_gr00t-1base
# bash examples/embodiment/eval_embodiment.sh libero_object_ppo_gr00t-2base
# bash examples/embodiment/eval_embodiment.sh libero_object_ppo_gr00t-3base
# bash examples/embodiment/eval_embodiment.sh libero_object_ppo_gr00t-4base
# bash examples/embodiment/eval_embodiment.sh libero_object_ppo_gr00t-5base
# bash examples/embodiment/eval_embodiment.sh libero_object_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=true \
#     actor.model.spec_verify_seq=true
# bash examples/embodiment/eval_embodiment.sh libero_object_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=false \
#     actor.model.spec_verify_seq=true


# bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_gr00t-1base
# bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_gr00t-2base
# bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_gr00t-3base
# bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_gr00t-4base
# bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_gr00t-5base
# bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=true \
#     actor.model.spec_verify_seq=true
# bash examples/embodiment/eval_embodiment.sh libero_goal_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=false \
#     actor.model.spec_verify_seq=true


# bash examples/embodiment/eval_embodiment.sh libero_10_ppo_gr00t-1base
# bash examples/embodiment/eval_embodiment.sh libero_10_ppo_gr00t-2base
# bash examples/embodiment/eval_embodiment.sh libero_10_ppo_gr00t-3base
# bash examples/embodiment/eval_embodiment.sh libero_10_ppo_gr00t-4base
# bash examples/embodiment/eval_embodiment.sh libero_10_ppo_gr00t-5base
# bash examples/embodiment/eval_embodiment.sh libero_10_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=true \
#     actor.model.spec_verify_seq=true
# bash examples/embodiment/eval_embodiment.sh libero_10_ppo_gr00t-5sp \
#     actor.model.spec_verify_conf=false \
#     actor.model.spec_verify_seq=true
