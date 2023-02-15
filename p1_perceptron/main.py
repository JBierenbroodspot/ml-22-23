""""""


class Perceptron:
    bias: float
    weights: list[float]

    def __init__(self, weights: list[float], threshold: float):
        self.bias = -threshold
        self.weights = weights

    def __str__(self) -> str:
        return f"Perceptron with t={-self.bias}"

    def activate(self, inputs: list[float]) -> int:
        """Activation function for the perceptron.

        It calculates the sum of each input multiplied by the corresponding weight and adds the bias to the result.
        The result is then compared to check whether it is greater than or equal to 0 and returns the integer value.


        Args:
            inputs (list[float]): Inputs for the perceptron, amount of inputs should not exceed len(self.weights).

        Returns:
            int: 0 if the weighted sum is lesser than 0, otherwise 1.
        """
        output: int = sum([vector * self.weights[index] for index, vector in enumerate(inputs)])
        output += self.bias

        return int(output >= 0)


class PerceptronLayer:
    perceptrons: list[Perceptron]

    def __init__(self, perceptrons: list[Perceptron]):
        self.perceptrons = perceptrons

    def __str__(self) -> str:
        return f"PerceptronLayer with {len(self.perceptrons)} perceptrons"

    def activate(self, input_values: list[float]) -> list[float]:
        """Activate every perceptron within the layer using the input_values.

        Args:
            input_values: Float values used to calculate activation of the perceptrons.

        Returns:
            The output for every perceptron within the layer.
        """
        return [perceptron.activate(input_values) for perceptron in self.perceptrons]


class PerceptronNetwork:
    layers: list[PerceptronLayer]

    def __init__(self, layers: list[PerceptronLayer]):
        self.layers = layers

    def __str__(self) -> str:
        return f"PerceptronNetwork with {len(self.layers)} layers"

    def predict(self, input_values: list[float]) -> list[float]:
        """Calculates the output of of multiple layers of Perceptrons.

        The input_values are fed into the first layer and the result is then fed into the next layer until the last
        layer, of which the output values get returned as the prediction.

        The name of the method can be a bit misleading but it is called this way to mimic the `predict()` methods for
        models within the sk_learn library.

        Args:
            input_values: The initial values of which to predict the output of.

        Returns:
            The output of the last layer after it has been fed by the layer before that.
        """
        output_values: list[float] = input_values

        for layer in self.layers:
            output_values = layer.activate(output_values)

        return output_values


def main() -> None:
    pass


if __name__ == "__main__":
    main()
