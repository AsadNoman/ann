from ann import ANN
import numpy as np

#XOR gate example
inputs = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]]).T


outputs = np.array([[0],
                    [1],
                    [1],
                    [0]]).T

# first value in shape is number of inputs and last one is the number of outputs
#between them is the hidden layers and the number of neurons they contain

shape = [2,5, 10, 5, 1]
ann = ANN(shape)
ann.train(inputs, outputs)

test_input = np.array([[0, 1]]).T
test_output = ann.run(test_input)

print(f'{test_output}')