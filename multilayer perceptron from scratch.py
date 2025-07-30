# 29/06/25 multilayer perceptron from scratch

import numpy as np
import csv
import os

class Layer:
    def __init__(self, n, m, activation):
        self.n, self.m = n, m # n = input, m = output arcs
        self.weights = np.random.randn(m, n)
        self.biases = np.zeros((m, 1))
        self.activation = activation.lower()
        
    def forward_pass(self, input):
        self.x = input
        self.z = np.dot(self.weights, input) + self.biases # pre activation
        if self.activation == "relu":
            self.a = np.maximum(0, self.z) 
        elif self.activation == "sigmoid":
            self.a = 1 / (1 + np.exp(0 - self.z))
        return self.a

    def get_dim(self):
        return [self.n, self.m] 

    def set_input_dim(self, input_dim):  
        self.n = input_dim
        self.weights = np.random.randn(self.m, self.n)  
    
    def backwards_pass(self, dA):  
        if self.activation == "relu":
            delta_a = (self.z > 0).astype(float)  # differentiate activation
        if self.activation == "sigmoid":
            delta_a = self.a * (1 - self.a)
        delta = dA * delta_a  # reverse chain rule activation to get pre activation gradient
        dW = np.dot(delta, np.transpose(self.x)) # weight gradient
        db = np.sum(delta, axis=1, keepdims=True) # bias gradient
        dx = np.dot(np.transpose(self.weights), delta) # reverse chain rule to find gradient of previous layer
        return dx, dW, db

    def gradient_descent(self, weights_gradient, biases_gradient, learning_rate): 
        self.weights -= weights_gradient * learning_rate
        self.biases -= biases_gradient * learning_rate   

class MLP:
    def __init__(self):
        self.layers = []
    
    def insert_layer(self, order, dim, activation):
        # order is 0 indexed
        if order == 0:
            layer = Layer(dim, dim, activation)
        else:
            layer = Layer(self.layers[order-1].get_dim()[1], dim, activation) # input dim is previous layers output dim if not first layer
        if order != len(self.layers): # make sure not last layer
            self.layers[order].set_input_dim(dim) # if it is not update next layers input dimensions 
        self.layers.insert(order,layer)  
    
    def forward_pass(self, input):
        if isinstance(input, list):
            input = np.array(input).reshape(-1, 1)
        elif len(input.shape) == 1:
            input = input.reshape(-1, 1)
        if len(input) != self.layers[0].get_dim()[0]:
            return("invalid dimensions inputted")
        for layer in self.layers:
            input = layer.forward_pass(input)
        return input
    
    def backwards_pass(self, loss_function, input_data, actual_output):
        output = self.forward_pass(input_data)
        if output == "invalid dimensions inputted":
            return("invalid dimensions inputted")
        loss_function = loss_function.upper()
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
            if len(input_data[i]) != self.layers[0].get_dim()[0]:
                return("invalid dimensions inputted")
            weights_gradient, biases_gradient = self.backwards_pass(loss_function, input_data[i], actual_output[i]) 
            for j in range(len(self.layers)):  
                self.layers[j].gradient_descent(weights_gradient[j], biases_gradient[j], learning_rate)  


# design neural network here: 
# below is an XOR gate example

neural_network = MLP()
neural_network.insert_layer(0,2,"sigmoid") # 2 nodes input
neural_network.insert_layer(1,3,"sigmoid") # 3 nodes on hidden layer
neural_network.insert_layer(2,1,"sigmoid") # 1 node output

# input training data thru text file

file_name = input("enter filename (note must be in the same directory): ")
script_dir = os.path.dirname(__file__)  # Folder of the script
file_path = os.path.join(script_dir, file_name)
input_data, output_data = [], []
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Convert to numpy arrays and reshape as column vectors
        inputs = np.array([float(x) for x in row[:-1]]).reshape(-1, 1)
        output = np.array([float(row[-1])]).reshape(-1, 1)
        input_data.append(inputs)
        output_data.append(output)


# train 

epochs = int(input("enter number of epochs: "))
for i in range(epochs):
    neural_network.train("MSE",input_data,output_data,0.5)

# check output

print("Now you can test the network. type STOP to stop")
while True:

    user_input = input("Enter input data, if there are multiple values type in eg: 3 7 2 4 (or STOP to stop): ")

    if user_input == "STOP":
        break

    input_data = [float(x) for x in user_input.split()]
    input_array = np.array(input_data).reshape(-1, 1)
    result = neural_network.forward_pass(input_array)
    
    if isinstance(result, str):
        print(result)
    else:
        print(f"Output: {result.flatten()}")
