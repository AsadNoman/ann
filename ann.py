import numpy as np
#np.random.seed(1)

class TransferFunction():
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


class ANN(object):
    def __init__(self, shape, training_rate = 0.3, epochs = 10000):
        self.epochs = epochs
        self.training_rate = training_rate
        self.weights = self.generate(shape)

    def generate(self, shape):
    	weights_array = []
    	for i in range(0, len(shape) - 1):
    	    current_index = i
    	    next_index = i + 1
    	    weight_array = 2*np.random.rand(shape[next_index], shape[current_index]) - 1
    	    weights_array.append(weight_array)

    	return weights_array


    def run(self, inputs):
        current_input = inputs
        local_outputs = []
        for network_weight in self.weights:
            current_output_temp = np.dot(network_weight, current_input)
            current_output = TransferFunction.sigmoid(current_output_temp)
            local_outputs.append(current_output)
            current_input = current_output

        return current_output.T


    def train(self, inputs, outputs):
    	weights_array = self.weights
    	for i in range(self.epochs):
            current_input = inputs
            local_outputs = []
            for network_weight in weights_array:
                current_output_temp = np.dot(network_weight, current_input)
                current_output = TransferFunction.sigmoid(current_output_temp)
                local_outputs.append(current_output)
                current_input = current_output

            deltas = []

            final_error = outputs - local_outputs[len(local_outputs)-1]
            final_delta = final_error * TransferFunction.sigmoid_derivative(local_outputs[len(local_outputs)-1])
            deltas.append(final_delta)

            current_delta = final_delta
            back_index = len(local_outputs) - 2

            for network_weight in weights_array[::-1][:-1]:
                next_error = np.dot(network_weight.T, current_delta)
                next_delta = next_error * TransferFunction.sigmoid_derivative(local_outputs[back_index])
                deltas.append(next_delta)
                current_delta = next_delta
                back_index -= 1

            current_weight_index = len(weights_array) - 1

            for delta in deltas:
                input_used = None
                if current_weight_index - 1 < 0:
                    input_used = inputs
                else:
                    input_used = local_outputs[current_weight_index - 1]

                weights_array[current_weight_index] += self.training_rate*np.dot(delta, input_used.T)
                current_weight_index -= 1

    	self.weights = weights_array[:]

