import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.constraints import maxnorm
from keras.utils import np_utils # Transfrom labels to categorical.
from keras.datasets import cifar10 # To load the dataset.
import numpy as np
import matplotlib.pyplot as plt
import keras.backend.common as K
K.set_image_dim_ordering('tf') # Tell TensorFlow the right order of dims.
import matplotlib as mpl # Just to set some standard plot format.
mpl.style.use('classic')
import numpy as np
import argparse
from keras.preprocessing import image


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize the x_train and x_test to be a float between 0 and 1.
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding based on number of classes.
class_count = 10
y_train = np_utils.to_categorical(y_train, class_count)
y_test = np_utils.to_categorical(y_test, class_count)

def create_model():
    # Sequential model adds layer by layer.
    model = Sequential()
    # First perfrom convolution then add relu activation function.
    # Padding - Same/valid - same adds n-pixel border of 0-pixels to remove data loss.
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # Halves the image size because stride = pool_size.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Prevent overfitting.
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Converts matrix to single array.
    model.add(Flatten())
    # The nn with 512 neurons in the first hidden layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))

    # Optimizer loss - klassifikation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_model_v4():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
        
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(class_count , activation='softmax'))  

    # Optimizer loss - klassifikation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_model(model, path):
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=path, save_weights_only=True, verbose=1)

    # Train the model with the new callback
    history = model.fit(x_train, y_train, batch_size=64, epochs=25, #verbose=0,
            validation_data=(x_test, y_test),
            callbacks=[cp_callback])  # Pass callback to training
    
    # Saves the model
    model.save(path)
    
    return history


def evaluate(model):
    acc = model.evaluate(x_test,  y_test, verbose=2)
    print("Model accuracy: {:5.2f}%".format(100*acc))


def make_plots(history, path):
    # Loss curve
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'black', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'black', ls = '--', linewidth=3.0)
    plt.legend(['Training loss', 'Validation loss'], fontsize = 18)
    plt.xlabel('Epochs', fontsize=16)
    plt.xlabel('Loss', fontsize=16)
    plt.title('Loss curve', fontsize=16)
    plt.savefig('{}/loss.png'.format(path.split('/')[0]), bbox_inches='tight', dpi=300)

    # Accuracy curve
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'black', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'black', ls = '--', linewidth=3.0)
    plt.legend(['Training accuracy', 'Validation accuracy'], fontsize = 18, loc= 'lower right')
    plt.xlabel('Epochs', fontsize=16)
    plt.xlabel('Accuracy', fontsize=16)
    plt.title('Accuracy curve', fontsize=16)
    plt.savefig('{}/accuracy.png'.format(path.split('/')[0]), bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program that creates a cnn model.')
    parser.add_argument('output', help='path to output file')
    parser.add_argument('-mp', '--makeplots', default=False, help='make accuracy and loss graphs')    
    parser.add_argument('-sh', '--showsummary', default=False, help='show the model summary')
    args = parser.parse_args()
    model = create_model()
    if(args.showsummary == True):
        print(model.summary) 
    history = fit_model(model, args.output)
    if(args.makeplots == True):
        make_plots(history, path)
    evaluate(model)
    