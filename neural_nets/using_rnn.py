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

def load_data(file, fields_to_extract):
    

    # adding depth helps
    #fields_to_extract.append("depth")
    
    X, Y = import_data.getData(file, fields_to_extract)
    #X2, Y2 = import_data.getData("GPS_Behaviours/Blue17_Accel_2s_Window_Sampled.csv", fields_to_extract)
    #X3, Y3 = import_data.getData("GPS_Behaviours/Blue20_Accel_2s_Window_Sampled.csv", fields_to_extract)
    #X4, Y4 = import_data.getData("GPS_Behaviours/Blue10_Accel_2s_Window_Sampled.csv", fields_to_extract)
    #X5, Y5 = import_data.getData("GPS_Behaviours/Blue14_Accel_2s_Window_Sampled.csv", fields_to_extract)
    #X6, Y6 = import_data.getData("GPS_Behaviours/Blue9_Accel_2s_Window_Sampled.csv", fields_to_extract)
    
    #X = X + X2 + X3 + X4 + X5 + X6
    #Y = Y + Y2 + Y3 + Y4 + Y5 + Y6
    #print X
    
    #shuffle(X, Y)
    
    X = cleanX(X)
    Y = convertY(Y)
    #print(Y[:10])
    return X, Y


def load_test(file, fields_to_extract):
    
    X_new_bird, Y_new_bird = import_data.getData(file, fields_to_extract)
    X_new_bird = cleanX(X_new_bird)
    Y_new_bird = convertY(Y_new_bird)
    return X_new_bird, Y_new_bird
    
    
    
import numpy as np    
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, SimpleRNN
from keras.callbacks import EarlyStopping, CSVLogger


fields_to_extract = ["X", "Y", "Z", "staticX", "staticY", "staticZ", "pitch", "dynamicX", "dynamicY", "dynamicZ"]  
train_file = "GPS_Behaviours/Blue18_Accel_2s_Window_Sampled.csv"
X_train, Y_train = load_data(train_file, fields_to_extract)

test_file = "GPS_Behaviours/Blue13_Accel_2s_Window_Sampled.csv"
X_test, Y_test = load_test(test_file, fields_to_extract)

val_file = "GPS_Behaviours/Blue20_Accel_2s_Window_Sampled.csv"
X_val, Y_val = load_test(val_file, fields_to_extract)

seq_length = 1

X_train = np.reshape(X_train, (len(X_train), seq_length, 10))
X_test = np.reshape(X_test, (len(X_test), seq_length, 10))
X_val = np.reshape(X_val, (len(X_val), seq_length, 10))


model = Sequential()

model.add(SimpleRNN(128, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

#model.add(Dense(64, input_dim=10, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(LSTM(64))
#odel.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(Dense(4, init='uniform'))
#model.add(Activation('softmax'))

from keras.optimizers import SGD, RMSprop
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
csv_logger = CSVLogger('NN.log')

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=1, batch_size=32, callbacks=[early_stopping, csv_logger])

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics)

#prediction = model.predict(X_test)
"""
print(prediction[:10])
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cnf_matrix = confusion_matrix(Y_test, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=None,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=None, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
"""
#from keras.utils.visualize_util import plot
#plot(model, to_file='RNNmodel.png')
