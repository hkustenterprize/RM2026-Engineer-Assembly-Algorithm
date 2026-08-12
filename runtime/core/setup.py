from setuptools import find_packages, setup


package_name = "arm_exchange_core"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    package_data={
        package_name: [
            "system_config.yaml",
            "system_config.example.yaml",
            "assets/collision/*.obj",
        ]
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="HKUST ENTERPRIZE",
    maintainer_email="zguobd@connect.ust.hk",
    description="Numerical perception, planning, and robot-model algorithms for arm exchange.",
    license="MIT",
    tests_require=["pytest"],
)
