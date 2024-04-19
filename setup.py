import setuptools
from setuptools import find_packages, setup

description = "MLX transformers is a machine learning framework with similar Interface to Huggingface transformers using MLX core as backend."

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mlx-transformers",
    version="0.0.1",
    author="Ogundepo Odunayo",
    author_email="ogundepoodunayo@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ToluClassics/mlx-transformers",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
