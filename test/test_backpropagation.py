import unittest

from p4_backpropagation import OutputNeuron, HiddenNeuron, NeuronNetwork

from ml.activation import sigmoid_activation


class TestOutputNeuron(unittest.TestCase):
    output_neuron: OutputNeuron

    def setUp(self):
        self.output_neuron = OutputNeuron([0.534, 0.799], -0.146, sigmoid_activation, 0.1)
        self.output_neuron.activate_and_set_new_weights([0.391, 0.511], 1)

    def test_error(self):
        """Test if the error property works correctly."""
        error_value: float = self.output_neuron.error
        expected_vale: float = -0.091

        msg = f"The output of error: {error_value}, does not match the expected value {expected_vale}"

        self.assertAlmostEqual(error_value, expected_vale, 4, msg)


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
