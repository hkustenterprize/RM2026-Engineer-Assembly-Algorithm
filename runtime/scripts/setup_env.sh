#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "setup_env.sh must be sourced:"
  echo "  source runtime/scripts/setup_env.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_SETUP="${PUBLIC_ROOT}/install/setup.bash"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f "/opt/ros/humble/setup.bash" ]]; then
    source "/opt/ros/humble/setup.bash"
  elif [[ -f "${HOME}/ros2_humble/install/setup.bash" ]]; then
    source "${HOME}/ros2_humble/install/setup.bash"
  fi
fi

if [[ ! -f "${INSTALL_SETUP}" ]]; then
  echo "ROS 2 install setup not found: ${INSTALL_SETUP}"
  echo "Run: runtime/scripts/build.sh"
  return 1
fi

source "${INSTALL_SETUP}"
