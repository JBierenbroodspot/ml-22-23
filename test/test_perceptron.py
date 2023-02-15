import unittest

from p1_perceptron import Perceptron, PerceptronLayer, PerceptronNetwork


class TestPerceptron(unittest.TestCase):
    perceptron: Perceptron

    def setUp(self):
        self.perceptron = Perceptron([0.0], 1)

    def test_has_bias(self):
        """Test if the Perceptron has the bias attribute."""

        msg = "Perceptron does not contain attribute 'bias'."

        self.assertTrue(hasattr(self.perceptron, "bias"), msg)

    def test_has_weights(self):
        """Test if the Perceptron has the weight attribute."""

        msg = "Perceptron does not contain attribute 'weight'."

        self.assertTrue(hasattr(self.perceptron, "weights"), msg)

    def test_bias(self):
        """Test if the bias attribute is equal to -threshold."""

        msg = "Attribute 'bias' does not equal '-threshold'."

        # key = threshold and value = expected bias
        thresholds = {20: -20, -10: 10}

        for threshold, expected_bias in thresholds.items():
            perceptron = Perceptron([], threshold)
            print(str(perceptron))

            self.assertEqual(perceptron.bias, expected_bias, msg)

    def test_has___str__(self):
        """Test if the Perceptron has a '__str__()' method."""

        msg = "Perceptron does not have a custom '__str__()' method."

        # Test if standard __str__() is not returned
        self.assertNotIn("object at", str(self.perceptron), msg)

    def test_activate_output(self):
        """Test if the Perceptron's activate() output is either 1 or 0."""

        msg = "The output of Perceptron.activate() is not either 1 or 0"
        perceptron_output: int

        arguments = (  # Each tuple exists of: weights, threshold, input
            ([1.0, 1.0, 1.0], 3, [1.0, 1.0, 1.0]),
            ([1.0, 1.0], 3, [1.0, 1.0]),
            ([1.0, .5], 0.5, [.0, 1.0]),
            ([1.0, .5], 0.5, [.0, .0]),
        )

        for argument in arguments:
            self.perceptron = Perceptron(argument[0], argument[1])
            perceptron_output = self.perceptron.activate(argument[2])

            self.assertIn(perceptron_output, (0, 1,), msg)

    def test_activate(self):
        """Tests if Perceptron's activate function calculates output correctly."""

        msg = "Perceptron.activate() does not return correct value"
        perceptron_output: int

        arguments = (  # Each tuple exists of: weights, threshold, input, expected output
            ([1.0, 1.0, 1.0], 3, [1.0, 1.0, 1.0], 1),
            ([1.0, 1.0], 3, [1.0, 1.0], 0),
            ([1.0, .5], 0.5, [.0, 1.0], 1),
            ([1.0, .5], 0.5, [.0, .0], 0),
        )

        for argument in arguments:
            self.perceptron = Perceptron(argument[0], argument[1])
            perceptron_output = self.perceptron.activate(argument[2])

            self.assertEqual(argument[3], perceptron_output, msg)


class TestPerceptronLayer(unittest.TestCase):
    def setUp(self):
        self.perceptron_layer = PerceptronLayer([])

    def test_has_perceptrons(self):
        """Test if the layer contains a attribute for assigning Perceptrons."""

        msg = "PerceptronLayer does not contain the attribute 'perceptrons'"

        self.assertTrue(hasattr(self.perceptron_layer, "perceptrons"), msg)

    def test_has___str__(self):
        """Test if the PerceptronLayer has a '__str__()' method."""

        msg = "PerceptronLayer does not have a custom '__str__()' method."

        # Test if standard __str__() is not returned
        self.assertNotIn("object at", str(self.perceptron_layer), msg)

    def test_activate_count(self):
        """Test if activate() returns the correct amount of outputs."""

        msg = "PerceptronLayer.activate() does not return the correct amount of outputs"

        for i in range(1, 11):
            perceptrons = list([Perceptron([0], 0) for j in range(1, i)])
            perceptron_layer = PerceptronLayer(perceptrons)

            self.assertEqual(len(perceptrons), len(perceptron_layer.activate([])), msg)

    def test_activate(self):
        """Test if activate() returns the correct values."""

        msg = "Perceptron.activate() does not return the correct values"

        perceptrons = [Perceptron([1, 1], 1)]
        perceptron_layer = PerceptronLayer(perceptrons)

        self.assertEqual(perceptron_layer.activate([0, 0]), [0], msg)
        self.assertEqual(perceptron_layer.activate([0, 1]), [1], msg)
        self.assertEqual(perceptron_layer.activate([1, 0]), [1], msg)
        self.assertEqual(perceptron_layer.activate([1, 1]), [1], msg)

        perceptrons = [Perceptron([1, 1], 1), Perceptron([1, 1], 2)]
        perceptron_layer = PerceptronLayer(perceptrons)

        self.assertEqual(perceptron_layer.activate([0, 0]), [0, 0], msg)
        self.assertEqual(perceptron_layer.activate([0, 1]), [1, 0], msg)
        self.assertEqual(perceptron_layer.activate([1, 0]), [1, 0], msg)
        self.assertEqual(perceptron_layer.activate([1, 1]), [1, 1], msg)


class TestPerceptronNetwork(unittest.TestCase):
    def setUp(self):
        layers = [
            PerceptronLayer([Perceptron([1, 1], 1), Perceptron([1, 1], 2), Perceptron([-1, -1], 1)]),
            PerceptronLayer([Perceptron([0.5, 1, .8], -1), Perceptron([.22, 2, .1], 1)]),
        ]
        self.perceptron_network = PerceptronNetwork(layers)

    def test_has_layers(self):
        """Test if the network has an attribute for assigning PerceptronLayers.
        """
        msg = "PerceptronNetwork does not contain attribute 'layers'"

        self.assertTrue(hasattr(self.perceptron_network, "layers"), msg)

    def test_has___str__(self):
        """Test if the PerceptronNetwork has a '__str__()' method."""

        msg = "PerceptronNetwork does not have a custom '__str__()' method."

        # Test if standard __str__() is not returned
        self.assertNotIn("object at", str(self.perceptron_network), msg)

    def test_predict(self):
        """Test if network predicts the correct outcome based on an input."""

        msg = "PerceptronNetwork.predict() returns an incorrect output."

        input_values = [1, 0]
        expected_value = [1, 0]

        self.assertEqual(self.perceptron_network.predict(input_values), expected_value, msg)


if __name__ == "__main__":
    unittest.main()
