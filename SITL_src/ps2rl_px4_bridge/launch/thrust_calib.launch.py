"""Run the thrust sweep. Use an empty world with vertical clearance."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("calib_altitude", default_value="15.0"),
            DeclareLaunchArgument("output_yaml", default_value="/tmp/thrust_fit.yaml"),
            Node(
                package="ps2rl_px4_bridge",
                executable="thrust_calib",
                name="ps2rl_thrust_calib",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "calib_altitude": LaunchConfiguration("calib_altitude"),
                        "output_yaml": LaunchConfiguration("output_yaml"),
                    }
                ],
            ),
        ]
    )
