import unittest

from p3_sigmoid_neuron import Neuron, NeuronNetwork, NeuronLayer, sigmoid_activation


class TestNeuron(unittest.TestCase):
    neuron: Neuron

    def setUp(self):
        self.neuron = Neuron([], 1, sigmoid_activation)

    def test_activation(self):
        """Test if the sigmoid activation function works correctly."""
        weights_and_biases = [
            ([1, 1], 1),
            ([-8, 2, 0], -1),
            ([100, 100, -1000], 0.234543),
            ([1], 1),
            ([-7, -7, -7, -7], 83),
        ]
        inputs_and_expected_values = [
            ([5, 1], 0.9991),
            ([2, 0, -20], 0.0),
            ([28, 1, 2], 1.0),
            ([1.3333], 0.9116),
            ([-7, -7, -7, -7], 1.0),
        ]

        for i in range(len(inputs_and_expected_values)):
            self.neuron.weights, self.neuron.bias = weights_and_biases[i]
            inputs, expected_value = inputs_and_expected_values[i]

            msg = f"neuron.activate({inputs}) does not match expected value {expected_value}"
            self.assertAlmostEquals(self.neuron.activate(inputs), expected_value, 4, msg)


class TestNeuronNetwork(unittest.TestCase):
    neuron_network: NeuronNetwork

    def setUp(self):
        layers = [
            NeuronLayer([
                Neuron([1, 1], -1, sigmoid_activation),
                Neuron([1, 1], -2, sigmoid_activation),
                Neuron([-1, -1], -1, sigmoid_activation),
            ]),
            NeuronLayer([
                Neuron([0.5, 1, .8], 1, sigmoid_activation),
                Neuron([.22, 2, .1], -1, sigmoid_activation),
            ]),
        ]
        self.neuron_network = NeuronNetwork(layers)

    def test_feed_forward(self):
        """Test if network predicts the correct outcome based on an input, using sigmoid activation."""
        input_values = [1, 0]
        expected_value = [0.834, 0.4157]

        prediction = self.neuron_network.feed_forward(input_values)

        msg = f"feed_forward() returns {prediction}, expected {expected_value}"
        for pred, expected in zip(prediction, expected_value):
            self.assertAlmostEqual(pred, expected, 3, msg)


if __name__ == "__main__":
    unittest.main()
