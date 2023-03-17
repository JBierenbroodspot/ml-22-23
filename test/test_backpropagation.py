import unittest

from p4_backpropagation import OutputNeuron, HiddenNeuron, NeuronNetwork

from ml.activation import sigmoid_activation


class TestOutputNeuron(unittest.TestCase):
    output_neuron: OutputNeuron

    def setUp(self):
        self.output_neuron = OutputNeuron([0.534, 0.799], -0.146, sigmoid_activation, 1)
        self.output_neuron.activate_and_set_all_deltas([0.391, 0.511], 1)

    def test_error(self):
        """Test if the error property works correctly."""
        error_value: float = self.output_neuron.error
        expected_vale: float = -0.091

        msg = f"The output of error: {error_value}, does not match the expected value {expected_vale}"

        self.assertAlmostEqual(error_value, expected_vale, 4, msg)

    def test_sigmoid_derivative(self):
        """Tests whether the sigmoid derivative works."""
        derivative_value: float = self.output_neuron.activation_derivative(0.616)
        expected_value: float = 0.2365

        msg = f"The output of sigmoid derivative: {derivative_value} does not match the expected value {expected_value}"

        self.assertAlmostEqual(derivative_value, expected_value, 4, msg)

    def test_gradient(self):
        """Tests whether the gradient value gets set correctly."""
        expected_values: list[float] = [-0.035581, -0.046501]

        for expected_val, input_val in zip(expected_values, self.output_neuron.inputs):
            gradient_value: float = self.output_neuron.get_gradient(input_val)
            msg = f"The gradient with input {input_val}: {gradient_value} does not match expected value {expected_val}"

            self.assertAlmostEqual(gradient_value, expected_val, 4, msg)

    def test_new_deltas(self):
        """Tests if new weights gets calculated correctly."""
        expected_values: list[float] = [-0.091, -0.035581, -0.046501]

        for expected_val, delta in zip(expected_values, self.output_neuron.deltas):
            msg = f"The new weight {delta} does not match the expected value {expected_val}"

            self.assertAlmostEqual(delta, expected_val, 4, msg)

    def test_update(self):
        """Test if the neurons get updated correctly."""
        expected_values: list[float] = [-0.055, 0.5695614834589499, 0.8454754937276813]
        self.output_neuron.update()

        msg = f"The updated bias: {self.output_neuron.bias} does not match expected bias {expected_values[0]}"
        self.assertAlmostEqual(self.output_neuron.bias, expected_values[0], 4, msg)

        for expected_val, weight in zip(expected_values[1:], self.output_neuron.weights):
            msg = f"The updated weight: {weight} does not match the expected weight {expected_val}"

            self.assertAlmostEqual(weight, expected_val, 4, msg)


class TestHiddenNeuron(unittest.TestCase):
    hidden_neuron: HiddenNeuron

    def setUp(self):
        ...


class TestNeuronNetwork(unittest.TestCase):
    network: NeuronNetwork

    def setUp(self):
        ...


if __name__ == "__main__":
    unittest.main()
