# torchutils

A collection of useful utilities for PyTorch projects.

## Installation

You can install torchutils using pip:

```bash
pip install torchutils
```

## Features

- Custom layers (LearnDepthwiseConv, CBAM, ResidualBlock)
- Custom activation functions
- Image transformations
- Utility functions for PyTorch

## Usage

Here's a basic example of using torchutils:

```python
import torchutils as tu

# Create a model using torchutils layers
model = tu.layer.CNN()

# Use custom activation functions
tu.activatefunc.GatedActivation()

# Apply image transformations
tu.transform.FourierReduct()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
