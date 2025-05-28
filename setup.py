from setuptools import setup, find_packages

setup(
    name="torchutils",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
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
    entry_points={
        'console_scripts': [
            'torchutils = torchutils.__main__:main',
        ],
    },
)
