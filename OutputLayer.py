
import numpy as np
from functions import sigmoid, sigmoid_prime


#I denne klasse er der 2 mulige implementeringer af lagest aktiveringsfunktion.
class OutputLayer: 
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = 10
        #En matrix med vægtene mellem inputlaget og dette lag. I begyndelsen sættes deres værdi tilfældigt, dog sikres det at værdierne ikke er for høje.
        self.weights = np.random.randn(input_size, self.output_size) / np.sqrt(input_size + self.output_size)
        #En vektor med alle neuronernes bias. I begyndelsen sættes deres værdi tilfældigt, dog sikres det at værdierne ikke er for høje.
        self.bias = np.random.randn(1,self.output_size) / np.sqrt(input_size + self.output_size)
        
    def forward(self,input):
        self.input = input
        self.values =  np.dot(input, self.weights) + self.bias
        exp = np.exp(self.values)
        self.output = exp/np.sum(exp)
        #self.output = sigmoid(self.values) 
        return self.output
        
    def backward(self,output_error, learning_rate):

        out = np.tile(self.output.T, self.output_size)
        output_error = self.output * np.dot(output_error, np.identity(self.output_size) - out)
        #output_error = output_error * sigmoid_prime(self.values)

        input_error = np.dot(output_error, self.weights.T) 
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error 
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error
    