# 24/06/25 multilayer perceptron from scratch work in progress

import numpy as np

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward_pass(input):
        output=np.dot(weights,inputs)
        output+=bias
        output=max(0,output) # ReLU
        return(output)
    
    def set_weights():
        self.weights=weights
    
class Layer:
    def __init__(self, dim, input_dim):
        self.neurons = neurons
        self.layer = layer
        self.neurons = []
        self.input_dim=input_dim
        for i in range(dim):
            neuron = Perceptron(np.random.randn(input_dim) * np.sqrt(2.0 / input_dim), 0)
            self.neurons.append(neuron)
        
    def forward_pass(input):
        for neuron in self.neurons:
            self.neuron.forward_pass(input)

    def get_dim():
        return(dim)

    def set_input_dim(input_dim):
        self.input_dim=input_dim
        for neuron in self.neurons:
            neuron.set_weights(np.random.randn(input_dim) * np.sqrt(2.0 / input_dim))
    


class MLP:
    def __init__(self, input_dim, output_dim):
        self.layers=[]
        layer=Layer(input_dim,input_dim)
        self.layers.append(layer) # first layer
        layer=Layer(input_dim,output_dim)
        self.layers.append(layer) # final layer
    
    def insert_layer(order,dim):
        # order is 0 indexed
        layer=Layer(dim, layers[order-1].get_dim())
        self.layers[order].set_input_dim(dim) # update new next layers input dimensions
        self.layers.append(layer)
    
    def forward_pass(input):
        for layer in self.layers:
            input=layer.forward_pass(input)
        return(input)
    
    def backwards_pass():



def train():
 # compute cost, back propogation,



    
