
import numpy as np

#hjælpefunktionerne importeres fra filen functions.py
from functions import sigmoid, sigmoid_prime

class HiddenLayer: 
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        #En matrix med vægtene mellem inputlaget og dette lag. I begyndelsen sættes deres værdi tilfældigt, dog sikres det at værdierne ikke er for høje.
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        #En vektor med alle neuronernes bias. I begyndelsen sættes deres værdi tilfældigt, dog sikres det at værdierne ikke er for høje.
        self.bias = np.random.randn(1,output_size) / np.sqrt(input_size + output_size)
    
    #Denne funktion beregner lagets aktiveringsvektor
    def forward(self,input):
        self.input = input

        self.values = np.dot(input, self.weights) + self.bias
        return sigmoid(self.values)

    #Denne funktion beregner gradienten og opdaterer lagets parametre
    def backward(self,output_error, learning_rate):
        output_error = output_error * sigmoid_prime(self.values)
        
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error 
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error
    