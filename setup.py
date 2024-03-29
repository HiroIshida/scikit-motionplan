from setuptools import setup

setup_requires = []

install_requires = [
    "imageio",
    "numpy",
    "matplotlib",
    "scipy",
    "tinyfk>=0.6.3",
    "robot_descriptions>=1.4.0",
    "scikit-robot[all]>=0.0.29",
    "open3d",
    "osqp",
    "ompl-thin>=0.0.16",
    "selcol>=0.0.3.2",
    "threadpoolctl",
]

setup(
    name="scikit-motionplan",
    version="0.0.1",
    description="scipy based robot planning framework",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    package_data={"skmp": ["py.typed"]},
)
