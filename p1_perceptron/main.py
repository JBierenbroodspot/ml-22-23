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
    pass


class PerceptronNetwork:
    pass


def main() -> None:
    pass


if __name__ == "__main__":
    main()
