import unittest

from p4_backpropagation import OutputNeuron, HiddenNeuron, NeuronNetwork

from ml.activation import sigmoid_activation


class TestOutputNeuron(unittest.TestCase):
    output_neuron: OutputNeuron

    def setUp(self):
        self.output_neuron = OutputNeuron([0.534, 0.799], -0.146, sigmoid_activation, 0.1)

    def test_error(self):
        """Test if the error property works correctly."""
        error_value: float = self.output_neuron.get_error(1, [0.391, 0.511])
        expected_vale: float = -0.091

        msg = f"The output of error: {error_value}, does not match the expected value {expected_vale}"

        self.assertAlmostEqual(error_value, expected_vale, 4, msg)

    def test_gradient(self):
        """Test the gradient method."""
        ...

    def test_get_delta(self):
        """Test the get_delta method."""
        ...

    def test_has_new_weights(self):
        """Test if the attribute in which get_all_deltas stores the new weights."""
        ...

    def test_get_all_deltas(self):
        """Test the get_all_deltas method."""
        ...

    def test_update(self):
        """Test the update method."""
        ...


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
