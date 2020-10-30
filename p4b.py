import numpy as np

np.random.seed(0)

X = np.array([
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
    ])

# weights tend to be initialised as random values between -1 to 1
# with neural networks you tend to want small values so things end up in the range of -1 to 1
# as data passes through the network, you're making it bigger with weights and biases = explosion
# you'll typically want to scale your inputs
# for biases, you'll typically initialise them as 0
# there are times though when you might not do that
# for example, if the inputs * weights isn't big enough to produce an output
# with a 0 bias added, your neural will produce a 0
# this will then be multiplied by next weight, and then if that bias is 0, it
# will also produce a 0 - you now have propagated all 0s and the network is dead
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # want to create weights of the number of inputs multiplied
        # by the number of neurons
        # use 0.1 * to make sure its small enough
        # shaping by inputs so we don't have to transpose anymore
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # first argument is shape passed as tuple
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# 4 features, 5 neurons
layer1 = Layer_Dense(4, 5)
# 2 neurons output by hidden layer
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)