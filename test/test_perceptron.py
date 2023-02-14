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

        self.assertIn(self.perceptron.activate([0.0]), (0, 1,), msg)


class TestPerceptronLayer(unittest.TestCase):
    def test_has_perceptrons(self):
        """Test if the layer contains a attribute for assigning Perceptrons."""

        msg = "PerceptronLayer does not contain the attribute 'perceptrons'"

        self.assertTrue(hasattr(Perceptron, "perceptrons"), msg)

    def test_has___str__(self):
        """Test if the PerceptronLayer has a '__str__()' method."""

        msg = "PerceptronLayer does not have a custom '__str__()' method."
        perceptron_layer = PerceptronLayer()

        # Test if standard __str__() is not returned
        self.assertNotIn("object at", str(perceptron_layer), msg)


class TestPerceptronNetwork(unittest.TestCase):
    def test_has_layers(self):
        """Test if the network has an attribute for assigning PerceptronLayers.
        """
        msg = "PerceptronNetwork does not contain attribute 'layers'"

        self.assertTrue(hasattr(PerceptronNetwork, "layers"), msg)

    def test_has___str__(self):
        """Test if the PerceptronNetwork has a '__str__()' method."""

        msg = "PerceptronNetwork does not have a custom '__str__()' method."
        perceptron_network = PerceptronNetwork()

        # Test if standard __str__() is not returned
        self.assertNotIn("object at", str(perceptron_network), msg)


if __name__ == "__main__":
    unittest.main()
