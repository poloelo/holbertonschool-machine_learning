# Classification Using Neural Networks

## Description

This project builds a binary image classifier from scratch with numpy.
It starts with a single neuron: its weights, bias and activated output,
then its forward propagation (sigmoid), its logistic regression cost,
the evaluation of its predictions, one pass of gradient descent, and
finally its training loop. The same steps are then repeated for a
neural network with one hidden layer, trained with backpropagation.

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- pycodestyle 2.11.1
- All files must be executable and end with a new line
- The first line of all files must be exactly `#!/usr/bin/env python3`
- All modules, classes and functions must have a documentation string
- Only `import numpy as np` is allowed (plus `matplotlib.pyplot` in tasks
  7 and 15), and no loops unless noted
- All matrix multiplications use `numpy.matmul`

## Files

| File | Description |
| --- | --- |
| `0-neuron.py` | `Neuron` with public `W`, `b` and `A` |
| `1-neuron.py` | `Neuron` with private attributes and getters |
| `2-neuron.py` | `Neuron.forward_prop`: sigmoid forward propagation |
| `3-neuron.py` | `Neuron.cost`: logistic regression cost |
| `4-neuron.py` | `Neuron.evaluate`: predicted labels and cost |
| `5-neuron.py` | `Neuron.gradient_descent`: one pass of gradient descent |
| `6-neuron.py` | `Neuron.train`: training loop |
| `7-neuron.py` | `Neuron.train` with `verbose`, `graph` and `step` |
| `8-neural_network.py` | `NeuralNetwork` with public attributes |
| `9-neural_network.py` | `NeuralNetwork` with private attributes and getters |
| `10-neural_network.py` | `NeuralNetwork.forward_prop` |
| `11-neural_network.py` | `NeuralNetwork.cost` |
| `12-neural_network.py` | `NeuralNetwork.evaluate` |
| `13-neural_network.py` | `NeuralNetwork.gradient_descent`: backpropagation |
| `14-neural_network.py` | `NeuralNetwork.train`: training loop |
| `15-neural_network.py` | `NeuralNetwork.train` with `verbose`, `graph` and `step` |

The main files expect the datasets (`Binary_Train.npz`, `Binary_Dev.npz`,
`MNIST.npz`) in a `../data` directory; they are not part of the repository.

## Author

Paul Gioria
