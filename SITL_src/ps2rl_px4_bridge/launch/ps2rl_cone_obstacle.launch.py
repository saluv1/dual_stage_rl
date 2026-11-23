"""Launch the outside-cone bridge against an already-running PX4 SITL + agent (light airframe).

Start these first, in separate terminals:

    MicroXRCEAgent udp4 -p 8888
    cd ~/PX4-Autopilot && make px4_sitl gz_x500_powerloop

Then, e.g. the four comparison flights:

    ros2 launch ps2rl_px4_bridge ps2rl_cone_obstacle.launch.py label:=ps2_ue
    ros2 launch ps2rl_px4_bridge ps2rl_cone_obstacle.launch.py run:=outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0_rs0p01 label:=tracker_ue
    ros2 launch ps2rl_px4_bridge ps2rl_cone_obstacle.launch.py run:=outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0_rs0p01 filter:=none label:=tracker_none
    ros2 launch ps2rl_px4_bridge ps2rl_cone_obstacle.launch.py filter:=nominal label:=ps2_nominal

Each flight prints the measured roll/pitch rate response at the end ("effective delay"); if a
flight oscillates, fly again with ``predict_ahead:=`` about that value.

Any argument below overrides the value in the config file, but only when it is given: an unset
argument leaves the YAML value alone. The node reads each value in the type of its default, so nothing needs quoting beyond what the shell
needs for ``pad_offset:="0,0,0.1"``. To clear a path pass ``backup_from:=none``; an empty value means
"not given".
Parameters that are not listed here are changed in the YAML (``config:=/path/to/copy.yaml``).
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
    "run": "Phase-II run directory (the policy): the PS2 run or the stage-1 tracker",
    "weights": "best or final",
    "filter": "ue (UE-bCBF), nominal (plain bCBF) or none (policy alone; the funnel is virtual)",
    "backup_from": "chosen_backup.json of the Phase-I comparison (none: the run's own backup)",
    "observer": "measured, commanded or off",
    "predict_ahead": "delay compensation [s]: the controller acts on the state predicted this far ahead (0 = off)",
    "qp_solve_dtype": "float64 or float32",
    "pad_offset": "task anchor relative to where the vehicle stands before arming, x,y,z [m]",
    "policy_duration": "length of the policy phase [s]; 0 = one training episode",
    "funnel_abort_depth": "filtered flights: abort when inside the funnel by more than this [m]",
    "tilt_abort_deg": "abort above this tilt [deg]",
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
            executable="ps2rl_cone_obstacle",
            name="ps2rl_cone_obstacle",
            output="screen",
            emulate_tty=True,
            parameters=parameters,
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("ps2rl_px4_bridge")
    default_config = os.path.join(share, "config", "cone_obstacle_x500_powerloop.yaml")

    args = [DeclareLaunchArgument("config", default_value=default_config, description="parameter file")]
    args += [DeclareLaunchArgument(name, default_value="", description=text) for name, text in OVERRIDABLE.items()]

    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
