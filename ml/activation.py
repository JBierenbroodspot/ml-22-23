"""This module contains activation functions. The code is duplicate and solely exists because the packages prefixed with
`p*_` have become a bit of a mess and segmented due to the way they have been created.

It also uses numpy rather than Python lists.
"""
from typing import Callable
from functools import partial

import numpy as np
from numpy.typing import NDArray


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
    def inner(func: Callable[[NDArray[float], NDArray[float], float], float]) -> ActivationFunction:
        activation_function = ActivationFunction(func)
        activation_function.name = name

        return activation_function
    return inner


@is_activation_function(name="step")
def step_activation(inputs: NDArray[float], weights: NDArray[float], bias: float) -> float:
    """Step activation function.

    Calculates the weighted sum and gives either 1 or 0 as activation value.

    Args:
        inputs: Inputs to for each weight.
        weights: Weight for each input.
        bias: Gets added to the weighted sum.

    Returns:
        1 if the weighted sum is larger than or equal to 0, 0 if not.
    """
    output: float = np.dot(inputs, weights) + bias

    # The float conversion is redundant as python converts integers to float implicitly but as the Zen of Python says:
    # "Explicit is better than implicit". It is also done to keep function signatures consistent over multiple
    # activation functions.
    return float(int(output >= 0))


@is_activation_function("sigmoid")
def sigmoid_activation(inputs: NDArray[float], weights: NDArray[float], bias: float) -> float:
    """Calculates activation by applying the sigmoid function to a weighted sum.

    Args:
        inputs: Inputs to for each weight.
        weights: Weight for each input.
        bias: Gets added to the weighted sum.

    Returns:
        A value somewhere between 0 and 1. The more negative a number is the closer it gets to 0 and the larger the
        number is the closer it gets to 1.
    """
    weighted_sum: float = np.dot(inputs, weights) + bias
    return 1 / (1 + (np.e ** (-weighted_sum)))


def sigmoid_derivative(output: float) -> float:
    """Gets the derivative of the sigmoid activation function.

    Args:
        output: The output value to calculate the derivative with.

    Returns:
        The derivate.
    """
    return output * (1 - output)
