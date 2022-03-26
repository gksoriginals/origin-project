from setuptools import setup, find_packages

requirements = [
    "flask",
    "requests",
    "tensorflow",
    "numpy",
    "pytest",
    "gunicorn"
]

setup(
    name="origin",
    version="0.0.1",
    include_package_data=True,
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
)