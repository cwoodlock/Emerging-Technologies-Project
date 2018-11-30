#Adapted from: https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb

#Imports
import numpy as np 
import gzip #import gzip for unpacking images and the labels 
import sklearn.preprocessing as pre #For encoding categorical variables
import keras as kr #this will be used for the neural network

#Start a neural network, building it by layers
model = kr.models.Sequential()

#add a hidden layer with 1000 neurons and an input layer with 784
#Decided to use relu for activation function as it is tends to avoid dead neurons during backpropagation
#https://datascience.stackexchange.com/questions/26475/why-is-relu-used-as-an-activation-function
#https://ai.stackexchange.com/questions/6468/why-to-prefer-relu-over-linear-activation-functions
model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784))

# Add neurons ro the output layer
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
#optimiser times admam=771 sgd=515 adadelta=851
model.compile(loss='categorical_crossentropy', optimizer='sdg', metrics=['accuracy'])

# Open the gzipped files and read as bytes.
#Adapted from : https://docs.python.org/2/library/gzip.html
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

# Read all the images and labels into memory
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

#Flatten the array
inputs = train_img.reshape(60000, 784)

#Encoding categorical variables
#Encode the labels into binary format
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
encoder = pre.LabelBinarizer()
#the sze of the array for each category
encoder.fit(train_lbl)
#encode each label as binary outputs
#https://github.com/scikit-learn/scikit-learn/issues/5536
outputs = encoder.transform(train_lbl)

# print out each array
#In the console you will now see each number and its corressponding binary number
for i in range(10):
    print(i, encoder.transform([i]))

model.fit(inputs, outputs, epochs=50, batch_size=100)
