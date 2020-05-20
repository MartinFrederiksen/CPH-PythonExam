import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization

from keras.optimizers import adam
from keras.callbacks import Callback, ModelCheckpoint

from keras.constraints import maxnorm

from keras.utils import np_utils # Transfrom labels to categorical
from keras.datasets import cifar10 # To load the dataset

import numpy as np
import matplotlib.pyplot as plt

import keras.backend.common as K
K.set_image_dim_ordering('tf') # Tell TensorFlow the right order of dims

# Just to set some standard plot format
import matplotlib as mpl
mpl.style.use('classic')

import numpy as np


from keras.preprocessing import image


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize the x_train and x_test to be a float between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding based on number of classes
class_count = 10
# y_train = np_utils.to_categorical(y_train, class_count)
# y_test = np_utils.to_categorical(y_test, class_count)
# checkpoint_path = "training/cp_v1.ckpt"

nClasses = 10
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

checkpoint_path = "training/cp_v5.ckpt"
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

    model.add(Dense(class_num , activation='softmax'))  

    # Optimizer loss - klassifikation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_model(model):
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    np.random.seed(21)

    # Train the model with the new callback
    history = model.fit(x_train, y_train, batch_size=64, epochs=25, #verbose=0,
            validation_data=(x_test, y_test),
            callbacks=[cp_callback])  # Pass callback to training
    
    # Saves the model
    model.save('training/cifar10_model_{}.h5'.format(checkpoint_path.split('_')[1].split('.')[0]))
    
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


def make_plots(history):
    # Loss curve
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'black', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'black', ls = '--', linewidth=3.0)
    plt.legend(['Training loss', 'Validation loss'], fontsize = 18)
    plt.xlabel('Epochs', fontsize=16)
    plt.xlabel('Loss', fontsize=16)
    plt.title('Loss curve', fontsize=16)
    plt.savefig('figures/loss_v5.png', bbox_inches='tight', dpi=300)

    # Accuracy curve
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'black', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'black', ls = '--', linewidth=3.0)
    plt.legend(['Training accuracy', 'Validation accuracy'], fontsize = 18, loc= 'lower right')
    plt.xlabel('Epochs', fontsize=16)
    plt.xlabel('Accuracy', fontsize=16)
    plt.title('Accuracy curve', fontsize=16)
    plt.savefig('figures/accuracy_v5.png', bbox_inches='tight', dpi=300)


def sort_unknown():
    loaded_model = load_model('training/cifar10_model_v1.h5')
    files = os.listdir('images/model_test')
    for file in files:
        img = image.load_img(os.path.join('images/model_test', file), target_size=(32, 32))
        img = np.expand_dims(img, axis=0)
        result=loaded_model.predict_classes(img)
        print('File: {} -- Prediction: {}'.format(file, class_names[result[0]]))


## Create a basic model instance
#model = create_model()
# Create a basic model instance
model = create_model_v4()
# print(model.summary())

## Display the model's architecture
# model.summary()

# Fit the model
history = fit_model(model)
make_plots(history)

# Load the weights and evaluate the model
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# print(model.input_names)

## Load the weights from a saved model
#model.load_weights(checkpoint_path)
model.save('training/cifar10_model_v5.h5')

loaded_model = load_model('training/cifar10_model_v5.h5')
#loaded_model.layers[0].input_shape #(None, 32, 32, 3)
image_path='images/model_test/Doog.png'
IMG_SIZE = 32
img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
img = np.expand_dims(img, axis=0)
result=loaded_model.predict_classes(img)
print(class_names[result[0]])
