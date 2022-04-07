
import numpy as np

#dette er klassen for det første lag i netværket. Her får den en attribut input_shape der angiver antallet af neuroner
class InputLayer: 
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    #denne funktion anvendes til at beregne aktiveringsværdien for alle neuroner og omformer det 2-dimensionelle billede til en array
    def forward(self,image):
        return np.reshape(image, (1,-1))
        
    def backward(self,output_error, learning_rate):
        return output_error
    
