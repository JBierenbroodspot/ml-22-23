from typing import Callable

from numpy.typing import NDArray

from ml.neurons import Neuron, NeuronNetwork
import ml.activation
from ml.activation import ActivationFunction


class OutputNeuron(Neuron):
    error: float
    activation_derivative: Callable[float, float]

    def __init__(self, weights: NDArray[float], bias: float, activation_function: ActivationFunction, eta: float):
        super().__init__(weights, bias, activation_function, eta)
        self.activation_derivative = self.get_activation_derivative()

    def get_activation_derivative(self) -> Callable[float, float]:
        derivative_name: str = f"{str(self.activation_function)}_derivative"

        if not hasattr(ml.activation, derivative_name):
            raise AttributeError(f"The derivative of {str(self.activation_function)} cannot be found in ml.activation")

        return getattr(ml.activation, derivative_name)

    def get_error(self, target: float, inputs: NDArray[float]) -> float:
        output_value: float = self.activate(inputs)
        error_value: float = self.activation_derivative(output_value) * -(target - output_value)

        self.error = error_value
        return error_value

    def get_gradient(
        self, prev_neuron_output: float, use_stored: bool = False, curr_neuron_target: float = None, 
        inputs: NDArray[float] = None
    ) -> float:
        if use_stored:
            return prev_neuron_output * self.error

        return self.get_error(curr_neuron_target, inputs)


class HiddenNeuron(Neuron):
    ...


class NeuronNetwork(NeuronNetwork):
    ...
