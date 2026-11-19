"""Launch the slow-track bridge (forward + loop) against an already-running PX4 SITL + agent.

Start these first, in separate terminals:

    MicroXRCEAgent udp4 -p 8888
    cd ~/PX4-Autopilot && make px4_sitl gz_x500

Then, first the PD controller, then the trained policy:

    ros2 launch ps2rl_px4_bridge ps2rl_slow_track.launch.py
    ros2 launch ps2rl_px4_bridge ps2rl_slow_track.launch.py controller:=policy run:=outputs/slow_track/slow_loop_s0 label:=policy_s0

Any argument below overrides the value in the config file, but only when it is given: an unset
argument leaves the YAML value alone. To clear a path pass ``run:=none``; an empty value means
"not given". Parameters that are not listed here are changed in the YAML (``config:=/path/to/copy.yaml``).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# argument -> what it is (shown by `ros2 launch ... --show-args`)
OVERRIDABLE = {
    "ps2rl_path": "PS2-RL checkout; relative paths below are taken from it",
    "controller": "pd (the base controller) or policy (a trained run)",
    "run": "policy: run directory of scripts/train_slow_track.py",
    "weights": "best or final",
    "reference_path": "reference .npz (pd; empty = the slow-loop reference)",
    "observer": "commanded, measured or off",
    "handover_height": "hand-over point above the take-off point [m]",
    "max_track_error": "abort when the tracking error exceeds this [m]",
    "tilt_abort_deg": "abort above this tilt [deg]",
    "end_hold_time": "hover after the reference [s], then land",
    "log_root": "each flight is recorded in <log_root>/<time>-<label>/",
    "label": "name of this flight in the record folder",
    "exit_when_done": "false: keep the node alive after the flight",
}


def _setup(context, *args, **kwargs):
    config_path = LaunchConfiguration("config").perform(context).strip()

    overrides = {}
    for name in OVERRIDABLE:
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            # passed as text: the node converts it to the type of the parameter's default
            overrides[name] = ParameterValue("" if value.lower() == "none" else value, value_type=str)

    parameters = [config_path] if config_path else []
    if overrides:
        parameters.append(overrides)

    return [
        Node(
            package="ps2rl_px4_bridge",
            executable="ps2rl_slow_track",
            name="ps2rl_slow_track",
            output="screen",
            emulate_tty=True,
            parameters=parameters,
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("ps2rl_px4_bridge")
    default_config = os.path.join(share, "config", "slow_track_x500.yaml")

    args = [DeclareLaunchArgument("config", default_value=default_config, description="parameter file")]
    args += [DeclareLaunchArgument(name, default_value="", description=text) for name, text in OVERRIDABLE.items()]

    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
