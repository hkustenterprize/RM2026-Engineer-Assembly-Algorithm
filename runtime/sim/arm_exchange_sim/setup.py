from glob import glob
import os

from setuptools import find_packages, setup


package_name = "arm_exchange_sim"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        *[
            (os.path.join("share", package_name, root), [os.path.join(root, f) for f in files])
            for root, dirs, files in os.walk("model", followlinks=True)
            if files
        ],
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HKUST ENTERPRIZE",
    maintainer_email="zguobd@connect.ust.hk",
    description="MuJoCo ROS2 simulation environment for the arm exchange task.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "station_panel = arm_exchange_sim.station_panel:main",
            "operator_input = arm_exchange_sim.operator_input:main",
        ]
    },
)
