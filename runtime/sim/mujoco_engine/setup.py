from setuptools import find_packages, setup


package_name = "mujoco_engine"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HKUST ENTERPRIZE",
    maintainer_email="zguobd@connect.ust.hk",
    description="Reusable ROS 2 execution engine for configuration-driven MuJoCo simulations.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "simulator = mujoco_engine.simulator:main",
        ]
    },
)
