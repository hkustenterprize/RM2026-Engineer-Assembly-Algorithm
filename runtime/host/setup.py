from glob import glob
import os

from setuptools import setup


package_name = "arm_exchange_host"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HKUST ENTERPRIZE",
    maintainer_email="zguobd@connect.ust.hk",
    description="Host-side ROS2 nodes for arm exchange.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = arm_exchange_host.perception_node:main",
            "planning_node = arm_exchange_host.planning_node:main",
            "task_node = arm_exchange_host.task_node:main",
        ]
    },
)
