from setuptools import setup

setup_requires = []

install_requires = [
    "numpy",
    "matplotlib",
    "scipy",
    "tinyfk",
    "robot_descriptions",
    "scikit-robot",
    "osqp",
    "ompl-thin>=0.0.5",
]

setup(
    name="scikit-motionplan",
    version="0.0.0",
    description="scipy based robot planning framework",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    extras_require={"experimental": ["selcol"]},
    package_data={"skmp": ["py.typed"]},
)
