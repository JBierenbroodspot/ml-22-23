"""Contains an extension to p1_perceptron.Perceptron. This is done in such a way that the following should NOT be done:

from p1_perceptron import Perceptron
from p2_perceptron_learning_rule import Perceptron

Although this should also never be desired to do as p2 makes p1 obsolete.
"""
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

    def train(self, training_set: List[List[float]], expected_values: List[float],
              max_iterations: int = 500, loss_limit: float = 0.1) -> Perceptron:
        """Applies the learning rule until either the maximum amount of iterations has been reached or the loss() method
        has reached a lower bound.

        Args:
            training_set: Training data.
            expected_values: True values for training data.
            max_iterations: Maximum amount of training iterations. Defaults to 500.
            loss_limit: The lower MSE limit to stop improving the perceptron when reached. Defaults to 0.

        Returns:
            self
        """
        verbose_output: str
        iteration: int = 0

        while iteration < max_iterations and self.loss(training_set, expected_values) > loss_limit:
            self.update_multiple(training_set, expected_values)
            iteration += 1

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
