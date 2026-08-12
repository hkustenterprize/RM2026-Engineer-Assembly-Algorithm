#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/setup_env.sh"
set -u
cd "${PUBLIC_ROOT}"

# Keep unrelated user-site pytest plugins out of the ROS workspace test run.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test \
  --packages-select \
    arm_exchange_interfaces \
    arm_exchange_core \
    mujoco_engine \
    arm_exchange_sim \
    arm_exchange_host \
  --event-handlers console_direct+

colcon test-result --verbose
