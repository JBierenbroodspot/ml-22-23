import unittest

from p4_backpropagation import OutputNeuron, HiddenNeuron, NeuronNetwork

from ml.activation import sigmoid_activation


class TestOutputNeuron(unittest.TestCase):
    output_neuron: OutputNeuron

    def setUp(self):
        self.output_neuron = OutputNeuron([0], 0, sigmoid_activation, 0)

    def test_error(self):
        """Test if the error property works correctly."""
        ...

    def test_has__error(self):
        """Test if the attribute which the error property will modify exists."""
        msg = "OutputNeuron does not have the attribute `_error`."

        self.assertTrue(hasattr(self.out, "_error"), msg)

    def test_sigmoid_derivative(self):
        """Test the sigmoid derivative method."""
        ...

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
