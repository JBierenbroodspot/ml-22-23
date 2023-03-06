from __future__ import annotations

from typing import List

import p1_perceptron
from p1_perceptron import ActivationFunction


class Perceptron(p1_perceptron.Perceptron):
    eta: float

    def __init__(self, weights: List[float], bias: float, activation_function: ActivationFunction, eta: float):
        super().__init__(weights, bias, activation_function)
        self.eta = eta

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
            Should be a lambda function but my linter flake8 complains about.
            """
            return (self.activate(inputs)-target)**2

        return sum(map(error_squared, input_matrix, expected_values)) / len(expected_values)
