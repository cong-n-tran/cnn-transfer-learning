# Khanh Nguyen Cong Tran
# 1002046419


from tensorflow import keras
from keras import layers
import numpy as np


def train_model(model, cifar_tr_inputs, cifar_tr_labels, 
            batch_size, epochs): 

    # Scale images to the [0, 1] range
    cifar_tr_inputs = cifar_tr_inputs.astype("float32") / 255

    # compile with SparseCategoricalCrossentropy and optimizer as adam
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="adam", metrics=['accuracy'])

    # train the model with given epoches and batch_size on the input data
    model.fit(cifar_tr_inputs, cifar_tr_labels, epochs=epochs, batch_size=batch_size)




def load_and_refine(filename, training_inputs, training_labels, 
                                batch_size, epochs): 
    
    # resize the input and labels 
    training_inputs, training_labels = resize(training_inputs, training_labels)


    #load the model
    model = keras.models.load_model(filename)

    num_layers = len(model.layers) # number of layers of the current model
    num_classes = training_labels.max()+1 # number of classes
    input_shape = training_inputs[0].shape #input shape

    # take the num_layers - 1 (minus the output layer) and add new layers to be trained
    refined_model = keras.Sequential([keras.Input(shape=input_shape)]+
                                 model.layers[:num_layers-1] + 
                                 [layers.Flatten(),
                                  layers.Dropout(0.5),
                                  layers.Dense(512, activation="tanh"),
                                  layers.Dropout(0.5),
                                  layers.Dense(num_classes, activation="softmax")])

    # freezes all the hidden layers from the old model
    for i in range(num_layers-1): 
        refined_model.layers[i].trainable = False

    refined_model.summary()

    # compile with SparseCategoricalCrossentropy and optimizer as adam
    refined_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="adam", metrics=['accuracy'])

    # train the refined model with given epoches and batch_size on the input data
    refined_model.fit(training_inputs, training_labels, epochs=epochs, batch_size=batch_size)

    return refined_model

def evaluate_my_model(model, test_inputs, test_labels): 

    # resize the input 
    test_inputs, test_labels = resize(test_inputs, test_labels)

    # compile the model and print summary 
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="adam", metrics=['accuracy'])
    model.summary()

    # evaulate
    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)

    # return test accurary
    return test_acc 


def resize(input, label): 


    # resizing the input data to be (_, 32, 32, 3)
    temp = input
    length = temp.shape[0] # get the number of inputs
    temp = temp.reshape(length, 28, 28, 1) # resize to get the last dimension
    temp = np.repeat(temp, 3, axis=3) #repeat the last dimension to be (_, 32, 32, 3)
    resized_input = np.zeros((length,32,32,3)) # add zeroes to the extra padding we included
    resized_input[:,2:30,2:30,:] = temp

    resized_shape = label.reshape(-1, 1)

    return (resized_input, resized_shape)



def print_shape(input, label, text = 'none bruh'): 
    print(text)
    print(f'this is the shape of the input: {input.shape}')
    print(f'this is the shpe of the label: {label.shape}')
    print()