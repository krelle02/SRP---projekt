import numpy as np
from OutputLayer import OutputLayer
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer

#hjælpefunktioner importeres
from functions import mse, mse_prime


class Network: 
    def __init__(self, input_shape = (28,28), hidden_neurons = 128):
        self.input_layer = InputLayer(input_shape)
        self.hidden_layer = HiddenLayer(input_shape[0]*input_shape[1], hidden_neurons)
        self.output_layer = OutputLayer( hidden_neurons)
    #en funktion der sætter træninsprocessen for modellen i gang
    def train(self, learning_rate, epochs, x_train, y_train):
        #her angives en variabler der gemmer alle værdierne for omkosningsfunktionen
        history = []
        #her starter et for loop der itererer datasættet epochs antal gange
        for epoch in range(epochs):
            error = 0 # Dette er den summeret omkostning for en epoch
            #her starter et for loop der går gennem hele træningssættet og gemmer datapunkterne i nogle variabler
            for i, (x, y) in enumerate(zip(x_train, y_train)):
                #Herunder beregnes alle netværkets aktiveringsværdier og moddellen laver en fordusigelse der gemmes i variablen output
                output = x
                output = self.input_layer.forward(output)
                output = self.hidden_layer.forward(output)
                output = self.output_layer.forward(output) 
                #omkosningen beregnes med en hjælpefunktion og tilføjes til summen
                error += mse(y, output)
                print(i,"/",int(len(x_train)), end="\r")
                #backpropagation algorithmen udføres og netværkets parametre opdateres
                output_error = mse_prime(y, output)
                output_error = self.output_layer.backward(output_error, learning_rate) 
                output_error = self.hidden_layer.backward(output_error, learning_rate)
                output_error = self.input_layer.backward(output_error, learning_rate)
            error /= len(x_train)
            history.append(error)
            print(epoch + 1,"/", epochs,"error=", error)
        return history

    #Denne funktion gemmer en models parametre i nogle csv filer
    def save_params(self, filename = "params"):

        np.savetxt(filename + "_hidden_weights.csv", np.array(self.hidden_layer.weights), delimiter=',',fmt='%s')
        np.savetxt(filename + "_output_weights.csv", np.array(self.output_layer.weights), delimiter=',',fmt='%s')
        np.savetxt(filename + "_hidden_bias.csv", np.array(self.hidden_layer.bias), delimiter=',',fmt='%s')
        np.savetxt(filename + "_output_bias.csv", np.array(self.output_layer.bias), delimiter=',',fmt='%s')

    #denne funktion læser alle parametrene fra csv filerne og gemmer dem i nogle arrays
    def load_parameters(self):
        files = ["params_hidden_weights.csv", "params_hidden_bias.csv","params_output_weights.csv","params_output_bias.csv"]
        parameters = []
        for file in files:
            array = np.loadtxt(file, delimiter=",")
            parameters.append(array)
        #print(parameters)
        return parameters

    #denne funktion sætter netværkets parametre alle det gemte parametre fra load_pramaters 
    def setup_network(self,params):

        self.hidden_layer.weights = params[0]
        self.hidden_layer.bias = params[1]
        self.output_layer.weights = params[2]
        self.output_layer.bias = params[3]
        self.hidden_layer.output_size = 128
    #denne funktion evaluerer en model
    def evaluate(self, x_test, y_true):
        #her udføres testen og fordusigelserne gemmes i en array
        y_pred, y_prob = self.test(x_test)
        #formatet ændres
        y_pred = np.asarray(y_pred)
        #omkostningen for alle forudsigelserne beregnes
        mean_squared_error = mse(y_pred, y_true)

        #beregner netværkets præcision ved at sammenligne forudsigelsen med det korrespondernede label.
        accuracy = np.sum((y_true == y_pred).all(1)) / float(len(y_true))
        return accuracy, mean_squared_error

    #denne funktion udfører en test på modellen
    def test(self, x_test):
        y_pred = []
        y_prob = []
        #her anvendes itereres testdatasættet og modellens forudsigelser gemmes i en array
        for x in x_test:
            preditcion, probblity = self.predict(x)
            y_pred.append(preditcion)
            y_prob.append(probblity)

        return y_pred, y_prob
    

    #denne funktion udfører en enkel forudsigelse ved at beregne alle aktiveringsværdierne
    def predict(self, x):

        output = x
        output = self.input_layer.forward(output)
        output = self.hidden_layer.forward(output)
        output = self.output_layer.forward(output) 

        #konverterer forudsigelse til det samme format som labels
        new_output = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        pos = np.argmax(output)
        probability = output[0][pos] 
        new_output[pos] = 1.0

        return new_output, probability


        
    
    

