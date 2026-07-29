"""Launch the PS2-RL bridge against an already-running PX4 SITL + agent.

Start these first, in separate terminals:

    cd ~/PX4-Autopilot && make px4_sitl gz_x500
    MicroXRCEAgent udp4 -p 8888

Then:

    ros2 launch ps2rl_px4_bridge ps2rl_sitl.launch.py \
        ps2rl_path:=$HOME/PS2-RL \
        run_dir:=$HOME/PS2-RL/checkpoints/deployed_ps2/quadrotor_ps2_learned

Launch arguments override config/bridge.yaml, but only when actually given.
An unset argument leaves the YAML value alone — passing an empty string through
as an override would silently blank out a required parameter and kill the node
in its constructor, before it ever publishes a setpoint.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

OVERRIDABLE = ("run_dir", "ps2rl_path", "checkpoint", "jax_platform", "log_csv")


def _setup(context, *args, **kwargs):
    config_path = LaunchConfiguration("config").perform(context).strip()

    overrides = {}
    for name in OVERRIDABLE:
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            overrides[name] = value

    parameters = [config_path] if config_path else []
    if overrides:
        parameters.append(overrides)

    return [
        Node(
            package="ps2rl_px4_bridge",
            executable="ps2rl_bridge",
            name="ps2rl_px4_bridge",
            output="screen",
            emulate_tty=True,
            parameters=parameters,
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("ps2rl_px4_bridge")
    default_config = os.path.join(share, "config", "bridge.yaml")

    args = [DeclareLaunchArgument("config", default_value=default_config)]
    args += [DeclareLaunchArgument(name, default_value="") for name in OVERRIDABLE]

    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
