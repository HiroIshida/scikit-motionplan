from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "imageio",
    "numpy",
    "matplotlib",
    "scipy",
    "tinyfk>=0.6.7",
    "robot_descriptions>=1.4.0",
    "scikit-robot[all]>=0.0.29",
    "open3d",
    "osqp",
    "ompl-thin>=0.0.16",
    "selcol>=0.0.3.2",
    "threadpoolctl",
    "types-PyYAML",
]

setup(
    name="scikit-motionplan",
    version="0.0.1",
    description="scipy based robot planning framework",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(include=["skmp*"]),
    package_data={
        "skmp": ["py.typed", "robot/fetch_coll_spheres.yaml", "robot/pr2_coll_spheres.yaml"]
    },
)
