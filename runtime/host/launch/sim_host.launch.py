from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _host_nodes(
    camera_view,
    enable_perception,
    enable_operator_input,
    keyboard_device,
    mouse_device,
):
    nodes = [
        Node(
            package="arm_exchange_host",
            executable="perception_node",
            name="arm_exchange_perception",
            output="screen",
            emulate_tty=True,
            condition=IfCondition(enable_perception),
            parameters=[{"show_window": camera_view}],
        ),
        Node(
            package="arm_exchange_host",
            executable="planning_node",
            name="arm_exchange_planning",
            output="screen",
            emulate_tty=True,
        ),
        Node(
            package="arm_exchange_host",
            executable="task_node",
            name="arm_exchange_task",
            output="screen",
            emulate_tty=True,
        ),
    ]
    nodes.append(
        Node(
            package="arm_exchange_sim",
            executable="operator_input",
            name="operator_input",
            output="screen",
            emulate_tty=True,
            condition=IfCondition(enable_operator_input),
            parameters=[
                {
                    "keyboard_device": keyboard_device,
                    "mouse_device": mouse_device,
                }
            ],
        )
    )
    return nodes


def generate_launch_description():
    sim_share = get_package_share_directory("arm_exchange_sim")
    camera_view_arg = DeclareLaunchArgument(
        "camera_view",
        default_value="true",
        description="Show OpenCV perception overlay window in the host stack.",
    )
    enable_perception_arg = DeclareLaunchArgument(
        "enable_perception",
        default_value="false",
        description="Start the model-based perception node; requires downloaded checkpoints.",
    )
    enable_operator_input_arg = DeclareLaunchArgument(
        "enable_operator_input",
        default_value="false",
        description="Start the Linux event-device operator input node.",
    )
    keyboard_device_arg = DeclareLaunchArgument(
        "keyboard_device",
        default_value="",
        description="Linux keyboard event device used by operator_input.",
    )
    mouse_device_arg = DeclareLaunchArgument(
        "mouse_device",
        default_value="",
        description="Optional Linux mouse event device used by operator_input.",
    )
    sim_config_arg = DeclareLaunchArgument(
        "sim_config_path",
        default_value=sim_share + "/config/simulation_config.yaml",
        description="Path to the arm exchange simulation config.",
    )
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            sim_share + "/launch/arm_exchange_sim.launch.py"
        ),
        launch_arguments={"sim_config_path": LaunchConfiguration("sim_config_path")}.items(),
    )
    camera_view = LaunchConfiguration("camera_view")
    enable_perception = LaunchConfiguration("enable_perception")
    enable_operator_input = LaunchConfiguration("enable_operator_input")
    keyboard_device = LaunchConfiguration("keyboard_device")
    mouse_device = LaunchConfiguration("mouse_device")

    return LaunchDescription([
        camera_view_arg,
        enable_perception_arg,
        enable_operator_input_arg,
        keyboard_device_arg,
        mouse_device_arg,
        sim_config_arg,
        sim_launch,
        *_host_nodes(
            camera_view,
            enable_perception,
            enable_operator_input,
            keyboard_device,
            mouse_device,
        ),
    ])
