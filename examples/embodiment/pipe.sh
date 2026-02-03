#!/usr/bin/env bash
set -euo pipefail

# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-5base
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10base
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10ar
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10sp
# bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15base
bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15sp
bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15ar
bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-10sp
bash examples/embodiment/eval_embodiment.sh metaworld_50_ppo_openpi_pi05-15sp