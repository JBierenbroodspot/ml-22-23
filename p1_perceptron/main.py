from typing import Callable
from functools import partial
from typing import List


class ActivationFunction(partial):
    """An extension to the partial class to make it possible to get a set name for a function which can be called using
    `str()`. This is technically not the intended use for partial as it is normally used to make partially applied
    functions if you're doing functional programming and here it's used as a class that behaves like a function.
    """
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()


def is_activation_function(name: str) -> ActivationFunction:
    """A decorator that (mis)uses a partially applied function to assign a name to the decorated function.

    Args:
        name: Name of the activation function. Will be displayed by calling `str()` on said function.
    """
    def inner(func: Callable[[List[float], List[float], float], float]) -> ActivationFunction:
        activation_function = ActivationFunction(func)
        activation_function.name = name

        return activation_function
    return inner


@is_activation_function(name="step")
def step_activation(inputs: List[float], weights: List[float], bias: float) -> float:
    """Step activation for a perceptron.

    Calculates the weighted sum and gives either 1 or 0 as activation value.

    Args:
        inputs: Inputs to for each weight.
        weights: Weight for each input.
        bias: Gets added to the weighted sum.

    Returns:
        1 if the weighted sum is larger than or equal to 0, 0 if not.
    """
    output: int = sum([input_value * weights[index] for index, input_value in enumerate(inputs)])
    output += bias

    # The float conversion is redundant as python converts integers to float implicitly but as the Zen of Python says:
    # "Explicit is better than implicit". It is also done to keep function signatures consistent over multiple
    # activation functions.
    return float(int(output >= 0))


class Perceptron:
    bias: float
    weights: List[float]
    activation_function: ActivationFunction

    def __init__(self, weights: List[float], bias: float, activation_function: ActivationFunction):
        self.bias = bias
        self.weights = weights
        self.activation_function = activation_function

    def __str__(self) -> str:
        return (
            f"<Perceptron {{activation: {str(self.activation_function)}, bias: {self.bias}, weights: {self.weights}}}>"
        )

    def activate(self, inputs: List[float]) -> float:
        """Calculates activation for the Perceptron using the given inputs.

        Args:
            inputs: Inputs for the perceptron, amount of inputs should not exceed len(self.weights).

        Returns:
            The activation value.
        """
        return self.activation_function(inputs, self.weights, self.bias)


class PerceptronLayer:
    perceptrons: List[Perceptron]

    def __init__(self, perceptrons: List[Perceptron]):
        self.perceptrons = perceptrons

    def __str__(self) -> str:
        out_str: str = "<PerceptronLayer {\n"

        for index, perceptron in enumerate(self.perceptrons):
            out_str += f"\t{index}: {str(perceptron)},\n"

        out_str += "}>"

        return out_str

    def activate(self, input_values: List[float]) -> List[float]:
        """Activate every perceptron within the layer using the input_values.

        Args:
            input_values: Float values used to calculate activation of the perceptrons.

        Returns:
            The output for every perceptron within the layer.
        """
        return [perceptron.activate(input_values) for perceptron in self.perceptrons]


class PerceptronNetwork:
    layers: List[PerceptronLayer]

    def __init__(self, layers: List[PerceptronLayer]):
        self.layers = layers

    def __str__(self) -> str:
        out_str: str = "<PerceptronNetwork {\n"

        for layer in self.layers:
            out_str += f"{str(layer)},\n"

        out_str += "}>"

        return out_str

    def predict(self, input_values: List[float]) -> List[float]:
        """Calculates the output of of multiple layers of Perceptrons.

        The input_values are fed into the first layer and the result is then fed into the next layer until the last
        layer, of which the output values get returned as the prediction.

        The name of the method can be a bit misleading as there's not much predicting to do in a deterministic model but
        it is called this way to mimic the `predict()` methods for models within the sk_learn library.

        Args:
            input_values: The initial values of which to predict the output of.

        Returns:
            The output of the last layer after it has been fed by the layer before that.
        """
        output_values: List[float] = input_values

        for layer in self.layers:
            output_values = layer.activate(output_values)

        return output_values
