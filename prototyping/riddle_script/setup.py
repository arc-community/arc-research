from setuptools import find_packages, setup

setup(
    name="riddle_script",
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
    packages=["riddle_script"] + ["riddle_script." + pkg for pkg in find_packages("riddle_script")],
)
