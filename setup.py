from setuptools import setup, find_packages

setup(
    name="pyconsolidation",
    version="1.0.0",
    author="Henri Fung",
    description="A Python-based toolkit for solving 3D bin packing and container loading problems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kaikwan/pyconsolidation.git",  # Replace with the actual repository URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "torch",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)