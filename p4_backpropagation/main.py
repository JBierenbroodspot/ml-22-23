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
    new_weights: NDArray[float]

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
        short-hand method `activate_and_set_new_weights()` to make sure all attributes are recent.

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
        short-hand method `activate_and_set_new_weights()` to make sure all attributes are recent.

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

    def get_new_weight(self, is_bias: bool = False, gradient: float = None) -> float:
        """Calculates what a new weight between this Neuron and a previous Neuron should by multiplying the gradient and
        learning rate (eta) or the error and the learning rate in case of the weight being the bias.

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_new_weights()` to make sure all attributes are recent.

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
                "`set_error()` method like this: `neuron.activate().set_error().get_new_weight()`."
            ))

        if is_bias:
            return self.eta * self.error

        return self.eta * gradient

    def set_new_weights(self, prev_neuron_outputs: NDArray[float]) -> OutputNeuron:
        """Stores what each new weight should be in the `new_weights` attribute, with `new_weights[0]` being the new
        bias. The rest is stored in order of each weight with the top weight being `new_weights[1]` and the bottom
        weight being `new_weights[len(new_weights) - 1]`.

        Example:
        |    Hidden Layer                  |    Output Layer
        | Neuron 1 __ new_weights[1] ___   |
        |                               \\  |
        | Neuron 2 -- new_weights[2] ---->--- This Neuron (new bias = new_weights[0])
        |                               /  |
        | Neuron 3 __ new_weights[3] __/   |

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_new_weights()` to make sure all attributes are recent.

        Args:
            prev_neuron_outputs: Outputs of all Neurons in the previous layer.

        Raises:
            AttributeError: If the `error` attribute has not been set yet.

        Returns:
            Self with the new weights stored.
        """
        if not hasattr(self, "error"):
            raise AttributeError(" ".join(
                    "This Neuron does not have an `error` attribute yet. Try chaining the",
                    "`set_error()` method like this: `neuron.activate().set_error().set_new_weights()`."
                ))

        # Reserve some memory so we don't need to reallocate memory every time a new weight is set.
        self.new_weights = np.zeros(len(self.weights))
        self.new_weights[0] = np.array([self.get_new_weight(is_bias=True)])

        for index, weight in enumerate(self.weights, start=1):
            gradient = self.get_gradient(prev_neuron_outputs[index])
            self.new_weights[index] = self.get_new_weight(gradient=gradient)

        return self

    def update(self) -> OutputNeuron:
        """Applies the new weights stored in `new_weights` and then deletes the attribute.

        Can be unsafe as the Neuron does not care if the values contained within are up-to-date. Prefer using the
        short-hand method `activate_and_set_new_weights()` to make sure all attributes are recent.

        Raises:
            AttributeError: If `new_weights` has not been set.

        Returns:
            Self with updated weights and bias.
        """
        if not hasattr(self, "new_weights"):
            raise AttributeError(" ".join(
                    "This Neuron does not have an `new_weights` attribute yet. Try chaining the",
                    "`set_error()` method like this: `neuron.activate().set_error().set_new_weights().update()`."
                ))

        self.bias -= self.new_weights[0]
        self.weights -= self.new_weights[1:]

        # Delete the new weights to prevent update from using outdated weights the next time the method is used.
        del self.new_weights

        return self

    def activate_and_set_new_weights(
        self, inputs: NDArray[float], target: float, prev_neuron_outputs: NDArray[float]
    ) -> OutputNeuron:
        """A short-hand method for:
        `self.activate(inputs).set_error(target).set_new_weights(prev_neuron_outputs)`

        Because the class does not keep track of what attribute belongs to what value. Because of that it is possible
        to calculate the gradient with an `error` that does not belong to the current output value. Instead this method
        could be called like this:

        `hidden_neuron = HiddenNeuron(...)                # Initialize the Neuron`
        `hidden_neuron.activate_and_set_new_weights(...)  # Set the error, gradient and new_weights`

        `# Now you can retrieve the calculated values`
        `hidden_neuron.error`
        `hidden_neuron.get_gradient(...)`
        `hidden_neuron.new_weights`

        `# And apply them after you're done with them`
        `hidden_neuron.update()`

        Args:
            inputs: Input values to activate this Neuron with.
            target: The ideal output value given the input values.
            prev_neuron_outputs: The output values of the Neurons in the previous layer.

        Returns:
            Self with the `error` and `new_weights` attributes set.
        """
        self.activate(inputs).set_error(target).set_new_weights(prev_neuron_outputs)

        return self


class HiddenNeuron(Neuron):
    ...


class NeuronNetwork(NeuronNetwork):
    ...
