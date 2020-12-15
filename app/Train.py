import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Open data
filename = 'fer2013.csv'
names = ['emotion', 'pixels', 'usage']

# Convert Data to a DF
df = pd.read_csv(filename, names=names, na_filter=False)

# DF of all pixels in image from CSV
im = df['pixels']


def get_data(filename):
    # Sinces images are 48x48, retrieve all the data from the images
    Y = []
    X = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y


# Get the image data from the csv
X, Y = get_data(filename)
num_class = len(set(Y))

# Keras with tensorflow backend
N, D = X.shape
X = X.reshape(N, 48, 48, 1)

# Assign train_test_split variables
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)


# Creates the model using Kera's Sequential Module
def create_model():
    model = Sequential()
    input_shape = (48, 48, 1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model


"""

This code trains the model as well as exports the model into model_filter.h5

We can use model_filter.h5 to further predict the emotions of a person in an image

"""
path_model = 'model_filter.h5'  # save model at this location after each epoch
K.clear_session()  # destroys the current graph and builds a new one
model = create_model()  # create the model
K.set_value(model.optimizer.lr, 1e-3)  # set the learning rate
h = model.fit(x=X_train,
              y=y_train,
              batch_size=64,
              epochs=20,
              verbose=1,
              validation_data=(X_test, y_test),
              shuffle=True,
              callbacks=[
                  ModelCheckpoint(filepath=path_model),
              ]
              )
