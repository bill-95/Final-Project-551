# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:43:47 2016

@author: lqi7
"""
import import_data
import random

def shuffle(X, Y):
    for i in range(len(Y)):
        swapWith = random.randint(i, len(Y) - 1)
        swap(X, i, swapWith)
        swap(Y, i, swapWith)
        
def cleanX(X):
    for row in X:
        for i in range(len(row)):
            if row[i] == "NA":
                row[i] = 0

            else:
                row[i] = float(row[i])
    return X

def swap(a, i, j):
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp 

def convertY(Y):
    classDict = {'Colony':[1,0,0,0],'Swimming':[0,1,0,0],'Diving':[0,0,1,0],'Flying':[0,0,0,1]}
    newY = []
    for label in Y:
        newY.append(classDict[label])
        
    return newY

def load_data(fields_to_extract):
    

    # adding depth helps
    #fields_to_extract.append("depth")
    
    X, Y = import_data.getData("GPS_Behaviours/Blue18_Accel_2s_Window_Sampled.csv", fields_to_extract)
    X2, Y2 = import_data.getData("GPS_Behaviours/Blue17_Accel_2s_Window_Sampled.csv", fields_to_extract)
    X3, Y3 = import_data.getData("GPS_Behaviours/Blue20_Accel_2s_Window_Sampled.csv", fields_to_extract)
    X4, Y4 = import_data.getData("GPS_Behaviours/Blue10_Accel_2s_Window_Sampled.csv", fields_to_extract)
    X5, Y5 = import_data.getData("GPS_Behaviours/Blue14_Accel_2s_Window_Sampled.csv", fields_to_extract)
    X6, Y6 = import_data.getData("GPS_Behaviours/Blue9_Accel_2s_Window_Sampled.csv", fields_to_extract)
    
    X = X + X2 + X3 + X4 + X5 + X6
    Y = Y + Y2 + Y3 + Y4 + Y5 + Y6
    #print X
    
    shuffle(X, Y)
    
    X = cleanX(X)
    Y = convertY(Y)
    print(Y[:10])
    return X, Y


def load_test(fields_to_extract):
    
    X_new_bird, Y_new_bird = import_data.getData("GPS_Behaviours/Blue13_Accel_2s_Window_Sampled.csv", fields_to_extract)
    X_new_bird = cleanX(X_new_bird)
    Y_new_bird = convertY(Y_new_bird)
    return X_new_bird, Y_new_bird
    
    
    
    
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, CSVLogger


model = Sequential()

#model.add(Dense(output_dim=64, input_dim=10))
#model.add(Activation("relu"))
#model.add(Dense(output_dim=4))
#model.add(Activation("softmax"))

model.add(Dense(128, input_dim=10, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(128, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(4, init='uniform'))
model.add(Activation('softmax'))

from keras.optimizers import SGD, RMSprop
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

fields_to_extract = ["X", "Y", "Z", "staticX", "staticY", "staticZ", "pitch", "dynamicX", "dynamicY", "dynamicZ"]  
X_train, Y_train = load_data(fields_to_extract)
X_test, Y_test = load_test(fields_to_extract)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
csv_logger = CSVLogger('NN.log')

model.fit(X_train, Y_train, validation_split=0.2, nb_epoch=5, batch_size=32, callbacks=[early_stopping, csv_logger])

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics)

from keras.utils.visualize_util import plot
plot(model, to_file='model.png')