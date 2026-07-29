from glob import glob
import os

from setuptools import find_packages, setup

package_name = "ps2rl_px4_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="PS2-RL policy bridge to PX4 offboard body-rate control.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ps2rl_bridge = ps2rl_px4_bridge.bridge_node:main",
            "thrust_calib = ps2rl_px4_bridge.thrust_calib_node:main",
        ],
    },
)
