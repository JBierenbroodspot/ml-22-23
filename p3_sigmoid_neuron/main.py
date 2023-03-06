from __future__ import annotations

from typing import List
import operator
import math

from p1_perceptron import Perceptron, PerceptronLayer, PerceptronNetwork, is_activation_function


@is_activation_function("sigmoid")
def sigmoid_activation(inputs: List[float], weights: List[float], bias: float) -> float:
    """Calculates activation by applying the sigmoid function to a weighted sum.

    Args:
        inputs: Inputs to for each weight.
        weights: Weight for each input.
        bias: Gets added to the weighted sum.

    Returns:
        A value somewhere between 0 and 1. The more negative a number is the closer it gets to 0 and the larger the
        number is the closer it gets to 1.
    """
    weighted_sum: float = sum(map(operator.mul, inputs, weights)) + bias
    return 1 / (1 + (math.e ** (-weighted_sum)))


class Neuron(Perceptron):
    ...


class NeuronLayer(PerceptronLayer):
    children: List[Neuron]

    def __init__(self, neurons: List[Neuron]):
        self.children = neurons


class NeuronNetwork(PerceptronNetwork):
    layers: List[NeuronLayer]

    def __init__(self, layers: List[NeuronLayer]):
        self.layers = layers

    feed_forward = PerceptronNetwork.predict
    feed_forward.__doc__ = PerceptronNetwork.predict.__doc__
