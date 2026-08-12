import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("arm_exchange_sim")
    default_config_path = os.path.join(pkg_share, "config", "simulation_config.yaml")

    sim_config_path_arg = DeclareLaunchArgument(
        "sim_config_path",
        default_value=default_config_path,
        description="Path to the arm exchange MuJoCo simulation config.",
    )

    mujoco_simulator_node = Node(
        package="mujoco_engine",
        executable="simulator",
        name="arm_exchange_mujoco_simulator",
        output="screen",
        emulate_tty=True,
        parameters=[{"config_path": LaunchConfiguration("sim_config_path")}],
    )

    return LaunchDescription([sim_config_path_arg, mujoco_simulator_node])
