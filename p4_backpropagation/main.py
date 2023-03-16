from p3_sigmoid_neuron import Neuron, NeuronNetwork, sigmoid_activation


class OutputNeuron(Neuron):
    activation_function = sigmoid_activation


class HiddenNeuron(Neuron):
    activation_function = sigmoid_activation


class NeuronNetwork(NeuronNetwork):
    ...
