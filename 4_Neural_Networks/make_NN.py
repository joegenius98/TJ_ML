'''
Steps to take:
1. Validate forward-prop works properly with simple example (take from a sample TJML Worksheet)
2. Validate backprop works properly (take an example from a sample TJML worksheet)
3. You know what to do from here. 
'''

import numpy as np

class Layer:
    def __init__(self, size):
        # size: (m, n) where m = prev "x" layer size and n = next "x" layer size
        # generates random layer (weights, biases)
        self.size = size
        self.weights = np.rand.randn(size[1], size[0])
        self.biases = np.random.randn(size[1], 1)

class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
    
    def forwardprop(self, input_vector):
        # input vector is size (m, 1) (or an m x 1 vector), np.array.shape = (m , 1)
        # to_ret: list of numpy arrays, where each numpy array reprents either an input layer, hidden layer, or output layer
        # of course, each element in to_ret has shape (n, 1), where n is an intger > 0

        sigmoid = lambda val: 1.0/(1.0 + np.exp(-val))
        #forward propogate thru layers
        to_ret = [input_vector]

        i = 1
        for layer in self.layers:
            curr_weights, curr_biases = layer.weights, layer.biases
            to_ret += [curr_weights.dot(to_ret[i - 1]) + curr_biases]
            i += 1
        
        return to_ret

    def backprop(self, y):
        sigmoid = lambda val: 1.0/(1.0 + np.exp(-val)) 
        sig_deriv = lambda val: sigmoid(val) * (1 - sigmoid(val))
        #Calculate gradients
        #Update weights 


def train(neural_net, iterations, training_data):
    for j in range(iterations): #number of iterations, to be tuned
        #train (forward, backprop)

def test(neural_net, testing_data):
	for data_point in testing_data:
	    #get output for each data_point
        #compare output to ground truth


layer_cts = [2, 3, 1]
layers = [Layer(layer_cts[i], layer_cts[i+1]) for i in range(len(layer_cts) - 1)]
my_NN = NeuralNet(layers)
#data i/o
#train network
#test network

