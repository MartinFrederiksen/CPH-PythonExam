import os

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.optimizers import adam
from keras.callbacks import Callback, ModelCheckpoint

from keras.utils import np_utils # Transfrom labels to categorical
from keras.datasets import cifar10 # To load the dataset

import numpy as np
import matplotlib.pyplot as plt

import keras.backend.common as K
K.set_image_dim_ordering('tf') # Tell TensorFlow the right order of dims

# Just to set some standard plot format
import matplotlib as mpl
mpl.style.use('classic')


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train[:500]
y_test = y_test[:100]

x_train = x_train[:500] / 255.0
x_test = x_test[:100] / 255.0

nClasses = 10
y_train = np_utils.to_categorical(y_train, nClasses)
y_test = np_utils.to_categorical(y_test, nClasses)

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))


    # Optimizer loss - klassifikation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def fit_model(model):
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Train the model with the new callback
    history = model.fit(x_train, y_train, batch_size=200, epochs=10, #verbose=0,
            validation_data=(x_test, y_test),
            callbacks=[cp_callback])  # Pass callback to training
    
    return history


def eval_test():
    # Create a basic model instance
    model = create_model()

    # Evaluate the model
    loss, acc = model.evaluate(x_test,  y_test, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss,acc = model.evaluate(x_test,  y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Create a basic model instance
model = create_model()

## Display the model's architecture
# model.summary()

# Fit the model
history = fit_model(model)

# # Load the weights and evaluate the model
# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(x_test,  y_test, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Loss curve
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'black', linewidth=3.0)
plt.plot(history.history['val_loss'], 'black', ls = '--', linewidth=3.0)
plt.legend(['Training loss', 'Validation loss'], fontsize = 18)
plt.xlabel('Epochs', fontsize=16)
plt.xlabel('Loss', fontsize=16)
plt.title('Loss curve', fontsize=16)
plt.savefig('loss.png', bbox_inches='tight', dpi=300)


# Accuracy curve
plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'black', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'black', ls = '--', linewidth=3.0)
plt.legend(['Training accuracy', 'Validation accuracy'], fontsize = 18)
plt.xlabel('Epochs', fontsize=16)
plt.xlabel('Accuracy', fontsize=16)
plt.title('Accuracy curve', fontsize=16)
plt.savefig('accuracy.png', bbox_inches='tight', dpi=300)

