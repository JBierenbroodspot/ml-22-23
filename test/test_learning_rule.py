import unittest
import math
from typing import List

from p2_perceptron_learning_rule import Perceptron
from p1_perceptron import step_activation


class TestLearningRule(unittest.TestCase):
    perceptron: Perceptron

    def setUp(self):
        self.perceptron = Perceptron([.1, -.2, .8], .533, step_activation, 0.1)

    def test_has_eta(self):
        """Test if learning rate eta is present."""
        msg = "Perceptron does not have attribute eta."

        self.assertTrue(hasattr(self.perceptron, "eta"), msg)

    def test_update(self):
        """Test if the update() method returns the correct values."""
        input_values: List[float] = [38, 1, -.8]
        expected_output: float = 0
        expected_bias: float = 0.433
        expected_weights: List[float] = [-3.7, -0.3, 0.88]

        self.perceptron.update(input_values, expected_output)

        msg = f"Bias after update {self.perceptron.bias} does not equal expected value {expected_bias}"
        self.assertAlmostEquals(expected_bias, self.perceptron.bias, 4, msg)

        msg = f"Weights after update {self.perceptron.weights} does not equal expected values {expected_weights}"
        result: bool = all(math.isclose(self.perceptron.weights[i], expected_weights[i], rel_tol=1e-4) 
                           for i in range(len(input_values)))
        self.assertTrue(result, msg)

    def test_loss(self):
        """Test if the loss() method returns the correct values."""
        input_values: List[List[float]] = [
            [0, 1, 8],
            [1, 9, 1],
            [-1, -1, 1],
            [9000, -293, .00000004],
            [0, 0, 0],
        ]
        expected_values: List[float] = [1, 0, 1, 1, 0]
        expected_loss: float = 0.2
        loss: float = self.perceptron.loss(input_values, expected_values)

        msg = f"The actual MSE {loss} does not match expected MSE {expected_loss}"
        self.assertEquals(expected_loss, loss, msg)


if __name__ == "__main__":
    unittest.main()
