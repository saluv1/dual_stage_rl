"""Launch the obstacle-only demo bridge (online backup filter, virtual cylinders) against an already-running PX4 SITL + agent.

Start these first, in separate terminals:

    MicroXRCEAgent udp4 -p 8888
    cd ~/PX4-Autopilot && make px4_sitl gz_x500

Then, for example:

    ros2 launch ps2rl_px4_bridge ps2rl_cyl_online.launch.py course:=O3 label:=O3_planner
    ros2 launch ps2rl_px4_bridge ps2rl_cyl_online.launch.py course:=S1 label:=S1_planner
    ros2 launch ps2rl_px4_bridge ps2rl_cyl_online.launch.py course:=O3 nominal:=straight label:=O3_straight
    ros2 launch ps2rl_px4_bridge ps2rl_cyl_online.launch.py course:=O3 control_period:=0.02 process:=false io:=executor executor:=multi setpoint_rate:=50 outgoing_stamp:=px4   # as flown 2026-11-20 (13:46)

Any argument below overrides the value in the config file, but only when it is given: an unset argument leaves the
YAML value alone. To clear a text value pass ``obstacles:=none``; an empty value means "not given". Parameters that are
not listed here are changed in the YAML (``config:=/path/to/copy.yaml``).
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
    "backup": "exported backup folder (backup.json)",
    "course": "O1..O4 / S1..S4 from cyl_online_candidates.json, or custom",
    "obstacles": "custom cylinders 'x,y,r;x,y,r' [m] (x east of the take-off point)",
    "goal": "custom goal 'x,y' [m]",
    "nominal": "planner (heading search) or straight (no search: the filter alone goes round)",
    "search_margin": "planner: ray clearance from the cylinders' surfaces [m]",
    "cruise": "cruise speed [m/s]",
    "blends": "2, 3 or 5",
    "horizon": "backup rollout [s]",
    "period": "commands per filter decision",
    "control_period": "seconds per command (0.02 = 50 Hz, 0.04 = 25 Hz)",
    "process": "true: the controller in its own process",
    "worker_cpu": "process: pin the controller to this CPU core (-1: no)",
    "executor": "io executor: single or multi (ROS executor threads)",
    "io": "poll (no rclpy executor) or executor",
    "setpoint_rate": "offboard stream rate [Hz]",
    "outgoing_stamp": "receipt (0: PX4 stamps on receipt), node or px4",
    "bridge_cpu": "pin the bridge process to this CPU core (-1: no)",
    "backend": "numpy, numba or auto",
    "flight_height": "hand-over point and course height above the take-off point [m]",
    "policy_timeout": "longest controller phase [s]",
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
            executable="ps2rl_cyl_online",
            name="ps2rl_cyl_online",
            output="screen",
            emulate_tty=True,
            parameters=parameters,
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("ps2rl_px4_bridge")
    default_config = os.path.join(share, "config", "cyl_online_x500.yaml")

    args = [DeclareLaunchArgument("config", default_value=default_config, description="parameter file")]
    args += [DeclareLaunchArgument(name, default_value="", description=text) for name, text in OVERRIDABLE.items()]

    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
