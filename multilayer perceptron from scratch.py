# 29/06/25 multilayer perceptron from scratch

import numpy as np

class Layer:
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.weights = np.random.randn(n, m)
        self.biases = np.zeros((n, m))
        
    def forward_pass(self, input):
        self.x = input
        self.z = np.dot(self.weights, input) + self.biases 
        self.a = np.maximum(0, self.z)  # ReLU
        return self.a

    def get_dim(self):
        return [self.n, self.m] 

    def set_input_dim(self, input_dim):  
        self.n = input_dim
        self.weights = np.random.randn(input_dim, self.m)  
        self.biases = np.zeros((input_dim, self.m)) 
    
    def backwards_pass(self, dA):  
        delta_a = (self.z > 0).astype(float)  # differentiate ReLU
        delta = dA * delta_a  # Hadamard product
        dW = np.dot(delta, np.transpose(self.x))
        db = delta
        dx = np.dot(np.transpose(self.weights), delta) 
        return dx, dW, db

    def gradient_descent(self, weights_gradient, biases_gradient, learning_rate): 
        self.weights -= weights_gradient * learning_rate
        self.biases -= biases_gradient * learning_rate   

class MLP:
    def __init__(self, input_dim, output_dim):
        self.layers = []
        layer = Layer(input_dim, input_dim)
        self.layers.append(layer)  # first layer
        layer = Layer(input_dim, output_dim)
        self.layers.append(layer)  # final layer
    
    def insert_layer(self, order, dim):
        # order is 0 indexed
        if order == 0: 
            layer = Layer(dim, dim)
        else:
            layer = Layer(self.layers[order-1].get_dim()[1], dim) # input dim is previous layers output dim
        if order != len(self.layers) - 1: # check if last item
            self.layers[order].set_input_dim(dim) # if it is not update next layers input dimensions 
        self.layers.insert(order, layer)  
    
    def forward_pass(self, input):
        for layer in self.layers:
            input = layer.forward_pass(input)
        return input
    
    def backwards_pass(self, loss_function, input_data, actual_output):
        output = self.forward_pass(input_data)
        if loss_function == "MSE":
            error = np.square(0.5 * (output - actual_output))
            dA = output - actual_output
        weights_gradient = []  
        biases_gradient = []   
        for layer in reversed(self.layers):
            dA, dW, db = layer.backwards_pass(dA)
            weights_gradient.insert(0, dW)
            biases_gradient.insert(0, db)
        return weights_gradient, biases_gradient

    def train(self, loss_function, input_data, actual_output, learning_rate):
        for i in range(len(input_data)):
            weights_gradient, biases_gradient = self.backwards_pass(loss_function, input_data[i], actual_output[i]) 
            for j in range(len(self.layers)):  
                self.layers[j].gradient_descent(weights_gradient[j], biases_gradient[j], learning_rate)  

# design neural network

# input training data thru text file

# check output
