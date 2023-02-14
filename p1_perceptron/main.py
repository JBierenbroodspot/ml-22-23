""""""


class Perceptron:
    bias: float
    weights: list[float]

    def activate(self, inputs: list[float]) -> int:
        output: int = sum([
            vector * self.weights[index]
            for index, vector in enumerate(inputs)
        ]) + self.bias

        return int(output >= 0)


class PerceptronLayer:
    pass


class PerceptronNetwork:
    pass


def main() -> None:
    pass


if __name__ == "__main__":
    main()
