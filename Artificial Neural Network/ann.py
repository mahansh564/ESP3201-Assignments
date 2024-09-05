import random
import math

random.seed(42)
learning_rate = 0.001
optimizer = 1.0  # Set a clip value for gradient clipping


def sigmoid(x):
    return 1 / (1 + math.exp(-max(min(x, 50), -50)))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def tanh_derivative(x):
    return 1 - x**2

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def linear(x):
    return x

def softmax(x):
    exps = [math.exp(i) for i in x]
    sum_of_exps = sum(exps)
    return [i / sum_of_exps for i in exps]

class Neuron:
    def __init__(self, num_inputs, activation=sigmoid):
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(num_inputs)]
        self.bias = 0
        self.activation = activation

    def forward(self, inputs) -> float:
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        output += self.bias
        activated_output = self.activation(output)
        print(f"Forward pass - Output: {output}, Activated Output: {activated_output}")
        return activated_output

    def backward(self, error, inputs) -> None:
        output_error = error * self.activation_derivative(self.forward(inputs))
        output_error = max(min(output_error, optimizer), -optimizer)
        for i in range(len(inputs)):
            self.weights[i] += learning_rate * output_error * inputs[i]
        #self.bias -= learning_rate * output_error

    def activation_derivative(self, x):
        if self.activation == sigmoid:
            return sigmoid_derivative(x)
        elif self.activation == tanh:
            return tanh_derivative(x)
        elif self.activation == relu:
            return relu_derivative(x)
        elif self.activation == linear:
            return 1

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs) -> None:
        self.hidden_layer = [Neuron(num_inputs) for _ in range(num_hidden)]
        self.output_layer = [Neuron(num_hidden, activation=linear) for _ in range(num_outputs)]

    def forward(self, inputs) -> list[float]:
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_layer]
        outputs = [neuron.forward(hidden_outputs) for neuron in self.output_layer]
        return outputs

    def backward(self, targets, inputs) -> None:
        errors = [target - output for target, output in zip(targets, self.forward(inputs))]
        for neuron, error in zip(self.output_layer, errors):
            neuron.backward(error, inputs)
        for neuron in self.hidden_layer:
            neuron.backward(errors[0], inputs)

    def predict(self, inputs) -> list[float]:
        return self.forward(inputs)
