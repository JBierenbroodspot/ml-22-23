from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ml.neurons import Neuron, NeuronNetwork
import ml.activation
from ml.activation import ActivationFunction


class OutputNeuron(Neuron):
    error: float
    activation_derivative: Callable[float, float]
    activation_output: float
    deltas: NDArray[float]
    inputs: NDArray[float]

    def __init__(
        self, weights: NDArray[float], bias: float, activation_function: ActivationFunction, eta: float,
        activation_derivative: Callable[float, float] = None
    ):
        super().__init__(weights, bias, activation_function, eta)

        if not activation_derivative:
            self.activation_derivative = self.get_activation_derivative()
        else:
            self.activation_derivative = activation_derivative

    def activate(self, inputs: NDArray[float]) -> OutputNeuron:
        """Calculates activation using the given inputs and caches that value.

        Args:
            inputs: Input numbers, amount of inputs should not exceed len(self.weights).

        Returns:
            Self with an updated `activation_output` attribute.
        """
        activation_value: float = super().activate(inputs)

        self.inputs = np.array(inputs)
        self.activation_output = activation_value

        return self

    def get_activation_derivative(self) -> Callable[float, float]:
        """Looks for the derivative of the activation function in `ml.activation` and gets it if it exists.

        This may seem a bit of an unconventional way to implement this feature and I am very aware of the restrictions,
        such as making this method very inflexible if an activation function is defined outside of the `ml.activation`
        module as it cannot be found. It was, however, fun to implement with that being the reason I opted to implement
        it this way.

        Raises:
            AttributeError: If the derivative is not found.

        Returns:
            The derivative function of the activation function.
        """
        derivative_name: str = f"{str(self.activation_function)}_derivative"

        if not hasattr(ml.activation, derivative_name):
            raise AttributeError(f"The derivative of {str(self.activation_function)} cannot be found in ml.activation")

        return getattr(ml.activation, derivative_name)

    def set_error(self, target: float) -> OutputNeuron:
        """Calculates the error of this Neuron.

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_all_deltas()` to make sure all attributes are recent.

        Args:
            target: The desired output value of this Neuron.

        Raises:
            AttributeError: If this Neuron has not been activated yet.

        Returns:
            Self with the `error` attribute set.
        """
        if not hasattr(self, "activation_output"):
            raise AttributeError(" ".join(
                "This Neuron has not been activated yet and does have an activation value. Try chaining the",
                "`activate()` method like this: `neuron.activate().set_error()`."
            ))

        error_value: float = self.activation_derivative(self.activation_output) * -(target - self.activation_output)

        self.error = error_value
        return self

    def get_gradient(self, prev_neuron_output: float) -> float:
        """Calculates the gradient given the output of a Neuron in the previous layer.

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_all_deltas()` to make sure all attributes are recent.

        Args:
            prev_neuron_output: The output of a Neuron in the previous layer.

        Raises:
            AttributeError: If the `error` attribute has not been set yet.

        Returns:
            The gradient between this Neuron and a single Neuron in the previous layer.
        """
        if not hasattr(self, "error"):
            raise AttributeError(" ".join(
                "This Neuron does not have an `error` attribute yet. Try chaining the",
                "`set_error()` method like this: `neuron.activate().set_error().get_gradient()`."
            ))

        return prev_neuron_output * self.error

    def get_delta(self, is_bias: bool = False, gradient: float = None) -> float:
        """Calculates the delta between this Neuron and a previous Neuron by multiplying the gradient and
        learning rate (eta) or the error and the learning rate in case of the weight being the bias.

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_all_deltas()` to make sure all attributes are recent.

        Args:
            is_bias: Whether this weight is the bias or not. Defaults to False.
            gradient: The gradient between a previous Neuron and this Neuron, should not be None if `is_bias` is False.

        Raises:
            AttributeError: If the `error` attribute has not been set yet.

        Returns:
            float: _description_
        """
        if not hasattr(self, "error"):
            raise AttributeError(" ".join(
                "This Neuron does not have an `error` attribute yet. Try chaining the",
                "`set_error()` method like this: `neuron.activate().set_error().get_delta()`."
            ))

        if is_bias:
            return self.eta * self.error

        return self.eta * gradient

    def set_all_deltas(self) -> OutputNeuron:
        """Stores what each delta should be in the `deltas` attribute, with `deltas[0]` being the delta for the new
        bias. The rest is stored in order of each weight with the `deltas[1]` belonging to the top weight and the bottom
        and `deltas[len(deltas) - 1]` to the bottom weight.

        Example:
        |    Hidden Layer                 |    Output Layer
        | Neuron 1 __ deltas[1] _____     |
        |                             \\  |
        | Neuron 2 -- deltas[2] -------->--- This Neuron (new bias = deltas[0])
        |                              /  |
        | Neuron 3 __ deltas[3] ______/   |

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_all_deltas()` to make sure all attributes are recent.

        Raises:
            AttributeError: If the `error` attribute has not been set yet.

        Returns:
            Self with the new weights stored.
        """
        if not hasattr(self, "error"):
            raise AttributeError(" ".join(
                    "This Neuron does not have an `error` attribute yet. Try chaining the",
                    "`set_error()` method like this: `neuron.activate().set_error().set_all_deltas()`."
                ))

        # Reserve some memory so we don't need to reallocate memory every time a new weight is set.
        self.deltas = np.zeros(len(self.weights) + 1)
        self.deltas[0] = self.get_delta(is_bias=True)

        for index, weight in enumerate(self.weights, start=1):
            gradient = self.get_gradient(self.inputs[index - 1])
            self.deltas[index] = self.get_delta(gradient=gradient)

        return self

    def update(self) -> OutputNeuron:
        """Applies the new weights stored in `deltas` and then deletes the attribute.

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_all_deltas()` to make sure all attributes are recent.

        Raises:
            AttributeError: If `deltas` has not been set.

        Returns:
            Self with updated weights and bias.
        """
        if not hasattr(self, "deltas"):
            raise AttributeError(" ".join(
                    "This Neuron does not have an `deltas` attribute yet. Try chaining the",
                    "`set_error()` method like this: `neuron.activate().set_error().set_all_deltas().update()`."
                ))

        self.bias -= self.deltas[0]
        self.weights -= self.deltas[1:]

        # Delete the new weights to prevent update from using outdated weights the next time the method is used.
        del self.deltas

        return self

    def activate_and_set_all_deltas(self, inputs: NDArray[float], target: float) -> OutputNeuron:
        """A short-hand method for:
        `self.activate(inputs).set_error(target).set_all_deltas(prev_neuron_outputs)`

        Because the class does not keep track of what attribute belongs to what value. Because of that it is possible
        to calculate the gradient with an `error` that does not belong to the current output value. Instead this method
        could be called like this:

        `hidden_neuron = HiddenNeuron(...)                # Initialize the Neuron`
        `hidden_neuron.activate_and_set_all_deltas(...)  # Set the error, gradient and deltas`

        `# Now you can retrieve the calculated values`
        `hidden_neuron.error`
        `hidden_neuron.get_gradient(...)`
        `hidden_neuron.deltas`

        `# And apply them after you're done with them`
        `hidden_neuron.update()`

        Args:
            inputs: Input values to activate this Neuron with.
            target: The ideal output value given the input values.

        Returns:
            Self with the `error` and `deltas` attributes set.
        """
        self.activate(inputs).set_error(target).set_all_deltas()

        return self


class HiddenNeuron(Neuron):
    ...


class NeuronNetwork(NeuronNetwork):
    ...
