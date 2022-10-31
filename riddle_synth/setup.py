from setuptools import find_packages, setup

setup(
    name="riddle_synth",
    author="arc-community",
    version="0.1",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "dev": [
            "black",
            "pytest",
        ],
    },
    packages=["riddle_synth"] + ["riddle_synth." + pkg for pkg in find_packages("riddle_script")],
)