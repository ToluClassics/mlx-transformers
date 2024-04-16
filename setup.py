import setuptools

description = """

"""

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

packages = setuptools.find_packages()
packages.remove('tests')

setuptools.setup(
    name="mlx-modelzoo",
    version="0.1.0",
    author="Ogundepo Odunayo",
    author_email="ogundepoodunayo@gmail.com",
    description="",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/ToluClassics/mlx-model-zoo",
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.10',
)
