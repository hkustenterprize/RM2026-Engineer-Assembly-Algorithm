#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PUBLIC_ROOT}"

PATH=/usr/bin:${PATH} colcon build --symlink-install \
  --base-paths \
    runtime/interfaces \
    runtime/core \
    runtime/sim \
    runtime/host \
  --packages-select \
    arm_exchange_interfaces \
    arm_exchange_core \
    mujoco_engine \
    arm_exchange_sim \
    arm_exchange_host \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
