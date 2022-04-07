import numpy as np

#Denne mappe implementerer alle hjælpefunktioner

# Dette er den anvendte omkostningsfunktion og dens afledte funktion som beskrevet i afsnit 3.2
def mse(y_true,y_pred): 
    return np.mean((y_true - y_pred)**2)

def mse_prime(y_true,y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

# Dette er den anvendte aktiveringsfunktion og dens afledte funktion som beskrevet i afsnit 3
#I disse funktioner anvendes np.exp som eulers tal e
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def sigmoid_prime(value): 
    return np.exp(-value) / (1 + np.exp(-value))**2

