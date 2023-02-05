import unittest

from p1_perceptron import Perceptron


class TestPerceptron(unittest.TestCase):
    def test_has_bias(self):
        """Test if the Perceptron has the bias attribute."""

        msg = "Perceptron does not contain attribute 'bias'."

        self.assertTrue(hasattr(Perceptron, "bias"), msg)

    def test_has_weights(self):
        """Test if the Perceptron has the weight attribute."""

        msg = "Perceptron does not contain attribute 'weight'."

        self.assertTrue(hasattr(Perceptron, "weights"), msg)

    def test_bias(self):
        """Test if the bias attribute is equal to -threshold."""

        msg = "Attribute 'bias' does not equal '-threshold'."
        thresholds = {20: -20, -10: 10}  # key = threshold and value = expected bias

        for threshold, expected_bias in thresholds.items():
            perceptron = Perceptron(threshold)
            print(str(perceptron))

            self.assertEqual(perceptron.bias, expected_bias, msg)

    def test_has___str__(self):
        """Test if the Perceptron has a '__str__()' method."""

        msg = "Perceptron does not have a custom '__str__()' method."
        perceptron = Perceptron()

        self.assertNotIn("object at", str(perceptron), msg)


if __name__ == "__main__":
    unittest.main()
