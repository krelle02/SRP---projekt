
#1.	Opsamling af data
#Her importeres de eksterne biblioteker
from random import randint
from time import sleep
import numpy as np
#fashion_mnist er det anvendte datasæt
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
print("-- trin 1 udføres og dataene opsamles --")
#importer datasættet i 2 dele. x er input data og y er dataenes labels 
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data() 

#2. Forberedelse af data
print("\n-- trin 2 udføres og dataene forberedes --")
#datsættet er allerede delvist forbedredt og af god kvalitet

#datatypen bliver omdannet til float
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

#datene bliver standardizeret ved at dividere med den maksimale værdi. 
x_train /= 255
x_test /= 255

#hjælpefunktion til at formatere et label til en array hvor indekset for 1'tallet indikerer typen af tøj. 
def label_correction(labels):
    new_labels = []
    for label in labels:
        new_label = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        new_label[label] = 1.0
        new_labels.append(new_label)
    return new_labels

#De 2 datasæt formateres med hjælpefunktionen
y_test = label_correction(y_test)
y_train = label_correction(y_train)

#En ordbog for de forskellige typer tøj
labels = {  0: "T-shirt/top,",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

#funktion der læser arrayformatet og angiver det korresponderende stykke tøj
def get_label(y):
    return labels[np.argmax(y)]

#3. Udvælge en model
print("\n-- trin 3 udføres og en model udvælges i form af et kunstigt neuralt netværk --")
#Dette er variablerne der kan ændres for at eksperimentere med modellen

epochs = 50             #Antallet af itterationer som netværket trænes med
learning_rate = 0.03    #Læringsraten der skalerer gradienten. Den standarde værdi er ofte 0.01 for lange træningsprocesser.
samples = 30000          #Antallet af datapunkter i træningsættet som modellen trænes med. Maksimum er 59999.
hidden_neurons = 128    #Antallet af neuroner i det middterste lag i netværket.
predictions = 8         #Antallet af gange som modellen laver forudsigelser. Dette kan ses i bunden.

#Et objekt for Nework klassen skabes.
from Network import Network
network = Network(input_shape = (28,28), hidden_neurons = 128)
'''
test_network  = Network(input_shape = (28,28), hidden_neurons = 128)
his = test_network.train(0.01,50, x_train[0:30000], y_train[0:30000])
plt.plot(his)
plt.show()
acc, error = test_network.evaluate(x_test, y_test)
print(f"Nøjagtigheden var {round(acc*100,2)}% og omkostningen er {error}")
'''
#4. Træning af modellen
print("\n-- trin 4 udføres og modellen gennemgår træninsprocessen --")
print("Dette er en kort træningsprocess med meget få datapunkter. Et plot af omkosningsfunktionens værdi vil også blevet genereret. Det skal bemærkes at funktionen ikke når at konvertere mod et lokalt minimum under denne korte træningsprocess")
history = network.train(learning_rate, epochs, x_train[0:samples], y_train[0:samples])
#5.	Evaluer modellen
print("\n-- trin 5 udføres og modellen evalueres --")
print("Modellen evalueres og nogle målinger bliver beregnet")
print("luk fanen når du er klar")
#omkosningfunktionens værdier plottes ved brug af et eksternt bibliotek 
plt.plot(history)
plt.show()
#nøjagtigheden og omkostningen fra evalueringen gemmes i nogle variabler
accuracy, mse = network.evaluate(x_test,y_test)
print(f"Nøjagtigheden var {round(accuracy*100,2)}% og omkostningen er {mse}")


#6. Optimering af modellen 
print("\n-- trin 6 ved at implementere en model der på forhånd er blevet omtimeret gennem en itterativ process --")
print("Nu implementeres et netværk der allerede er blevet trænet med 100 epochs/iterationer af hele datasættet")
#parametrene fra csv-filerne gemmes og netværkets parametre opdateres
parameters = network.load_parameters()
network.setup_network(parameters)

#5.	Evaluer modellen
print("\n-- trin 5 gentages igen for denne model --")
print("Modellen evalueres og nogle målinger bliver beregnet")
#nøjagtigheden og omkostningen fra evalueringen gemmes i nogle variabler
accuracy, mse = network.evaluate(x_test,y_test)
print(f"Nøjagtigheden var {round(accuracy*100,2)}% og omkostningen er {mse}")

sleep(3)

#7.	Afsluttende test og implementering
print("\n-- trin 7 udføres og modellen kan nu implementeres i et relevant it-system --")
print("Ud fra denne model laves nogle forudsigelse")
i = 0
#Et while loop sættes i gang og netværket udfører nogle forudsigelser
while i < predictions:
    #Genererer et tilfældigt indeks fra datasættet
    sample = randint(0,9999)
    #forudsigelsen og sandsynligheden gemmes i nogle variabler
    prediction, probabliity = network.predict(x_test[sample])
    predicted_item = get_label(prediction)
    print(f"Forudsigelsen var {predicted_item} med en sandsynlighed på {round(probabliity*100,2)}%. Det rigtige svar er {get_label(y_test[sample])}")
    #Et billede af tøjet laves
    image = np.reshape(x_test[sample], (28, 28))
    plt.imshow(image, cmap='binary')
    plt.show()
    i += 1
