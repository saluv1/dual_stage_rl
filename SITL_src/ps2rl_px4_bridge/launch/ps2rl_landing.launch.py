"""Launch the filtered-landing bridge against an already-running PX4 SITL + agent.

Start these first, in separate terminals:

    MicroXRCEAgent udp4 -p 8888
    cd ~/PX4-Autopilot && make px4_sitl gz_x500

Then:

    ros2 launch ps2rl_px4_bridge ps2rl_landing.launch.py

Any argument below overrides the value in the config file, but only when it is given: an unset
argument leaves the YAML value alone. For example

    ros2 launch ps2rl_px4_bridge ps2rl_landing.launch.py target:=apex max_position_error:=0 label:=apex

The node reads each value in the type of its default (``use_filter:=false`` is a boolean,
``max_position_error:=0`` a float), so nothing needs quoting beyond what the shell needs for
``start_position:="-2,0,2"``. To clear a path or to ask for the default start, pass
``reference_path:=none`` / ``start_position:=auto``; an empty value means "not given".
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
    "backup": "landing Phase-I run directory (cone, floor, base set, backup policy)",
    "filter": "landing BCBF settings, KEY=VAL separated by ','",
    "use_filter": "false: the nominal controller alone (comparison flights)",
    "qp_solve_dtype": "float64 or float32",
    "nominal": "lqr_target or track_reference",
    "target": "target of lqr_target relative to the pad centre: pad_centre, apex or x,y,z [m]",
    "max_position_error": "cap on the position error the LQR sees [m]; 0 = off",
    "reference_path": "reference .npz for track_reference",
    "start_position": "start of the filtered descent relative to the pad centre, x,y,z [m]; auto = the reference's start",
    "pad_offset": "pad centre relative to where the vehicle stands before arming, x,y,z [m]",
    "policy_timeout": "longest filtered descent [s]",
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
            executable="ps2rl_landing",
            name="ps2rl_landing",
            output="screen",
            emulate_tty=True,
            parameters=parameters,
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("ps2rl_px4_bridge")
    default_config = os.path.join(share, "config", "landing_x500.yaml")

    args = [DeclareLaunchArgument("config", default_value=default_config, description="parameter file")]
    args += [DeclareLaunchArgument(name, default_value="", description=text) for name, text in OVERRIDABLE.items()]

    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
