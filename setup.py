from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torchutils",
    version="0.1.0",
    author="supertimeo",
    author_email="fort.timeo38@gmail.com",
    description="A collection of useful utilities for PyTorch projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supertimeo/torchutils",
    packages=find_packages(include=['torchutils', 'torchutils.*']),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.0.0",
        "sentence-transformers>=2.0.0",
        "torchtext>=0.15.0",
        "torchmetrics>=1.0.0",
        "fastai>=2.7.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "kornia>=0.8.1",
        "pycocotools>=2.0.4",
        "pydantic>=2.5.2",
        "torchsummary>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
