"""This module contains the code of `p1_perceptron`, `p2_perceptron_learning_rule` and `p3_sigmoid_neuron`. This is done
because the modules were designed without knowledge of what the final product should look like so it has become a bit
cumbersome to work with.

The code is almost entirely duplicated from the packages names above apart from some minor quality-of-life changes.

A major change however is that a Perceptron will be an extension of a Neuron rather than the other way around.
"""
from __future__ import annotations

from typing import List

from ml.activation import step_activation, ActivationFunction


class Neuron:
    bias: float
    weights: List[float]
    activation_function: ActivationFunction
    eta: float

    def __init__(self, weights: List[float], bias: float, activation_function: ActivationFunction, eta: float):
        self.bias = bias
        self.weights = weights
        self.activation_function = activation_function
        self.eta = eta

    def __str__(self) -> str:
        return "".join((
            f"<{self.__class__.__name__} {{",
            f"activation: {str(self.activation_function)}, bias: {self.bias}, weights: {self.weights}}}>"
        ))

    def __repr__(self) -> str:
        return self.__str__()


class Perceptron(Neuron):
    """A special kind of Neuron which uses the step activation function."""

    def __init__(self, weights: List[float], bias: float):
        self.activation_function = step_activation

    def activate(self, inputs: List[float]) -> float:
        """Calculates activation using the given inputs.

        Args:
            inputs: Input numbers, amount of inputs should not exceed len(self.weights).

        Returns:
            The activation value.
        """
        return self.activation_function(inputs, self.weights, self.bias)

    def update(self, input_values: List[float], expected_value: float) -> Perceptron:
        """Update the weights and bias using the Perceptron learning rule.

        Args:
            input_values: Values to activate perceptron with.
            expected_value: Expected output after activating perceptron.

        Returns:
            self to make method chaining possible.
        """
        error: float = expected_value - self.activate(input_values)

        self.bias += self.eta * error
        self.weights = [weight + self.eta * error * input_value
                        for weight, input_value in zip(self.weights, input_values)]
        return self

    def update_multiple(self, input_matrix: List[List[float]], expected_values: List[float]) -> Perceptron:
        """Applies the learning rule to a list of inputs.

        Args:
            input_matrix: A list containing inputs for the perceptron.
            expected_values: A list containing the expected values for each input array.

        Returns:
            self.
        """
        for sample, expected_value in zip(input_matrix, expected_values):
            self.update(sample, expected_value)

        return self

    def loss(self, input_matrix: List[List[float]], expected_values: List[float]) -> float:
        """Calculates the MSE using a list of inputs.

        Args:
            input_matrix: A list containing values to use as input for the perceptron.
            expected_values: The values which are expected for each input list.

        Returns:
            The mean squared error.
        """
        def error_squared(inputs: List[float], target: float) -> float:
            """Calculates the square of the error given some inputs and a target.
            Should be a lambda function but my linter flake8 complains about using those.
            """
            return (self.activate(inputs)-target)**2

        return sum(map(error_squared, input_matrix, expected_values)) / len(expected_values)


class NeuronLayer:
    children: List[Neuron]

    def __init__(self, children: List[Neuron]):
        self.children = children

    def __str__(self) -> str:
        out_str: str = f"<{self.__class__.__name__} {{\n"

        for index, child in enumerate(self.children):
            out_str += f"\t{index}: {str(child)},\n"

        out_str += "}>"

        return out_str

    def __repr__(self) -> str:
        return self.__str__()

    def activate(self, input_values: List[float]) -> List[float]:
        """Activate every child within the layer using the input_values.

        Args:
            input_values: Float values used to calculate activation of the children.

        Returns:
            The output for every child within the layer.
        """
        return [child.activate(input_values) for child in self.children]


class NeuronNetwork:
    layers: List[NeuronLayer]

    def __init__(self, layers: List[NeuronLayer]):
        self.layers = layers

    def __str__(self) -> str:
        out_str: str = f"<{self.__class__.__name__} {{\n"

        for layer in self.layers:
            out_str += f"{str(layer)},\n"

        out_str += "}>"

        return out_str

    def __repr__(self) -> str:
        return self.__str__()

    def feed_forward(self, input_values: List[float]) -> List[float]:
        """Calculates the output of of multiple layers of neurons.

        The input_values are fed into the first layer and the result is then fed into the next layer until the last
        layer, of which the output values get returned as the prediction.

        Args:
            input_values: The initial values of which to predict the output of.

        Returns:
            The output of the last layer after it has been fed by the layer before that.
        """
        output_values: List[float] = input_values

        for layer in self.layers:
            output_values = layer.activate(output_values)

        return output_values
