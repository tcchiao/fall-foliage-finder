from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import SGD

def build_base_sequential_NN(nb_filters=16, nb_conv=3, map_dimensions=None, nb_pool=2):
    """
    Builds a sequential neural network using the Keras framework that predicts 2 categories. 
    The Theano backend was used for testing. Have not tested for Tensorflow. The model 
    contains 2 convolutional layers (2D) by default among others. 

    Parameters
    ----------
    nb_filters: int, number of convolution kernels to use (dimensionality of the output).
    nb_conv: int, the extension (spatial or temporal) of each filter.
             equivalent to "filter_length" in Keras documentation. 
    map_dimensions: tuple, shape of the input data for each sample/observation. 
    nb_pool: tuple of 2 integers, factors by which to downscale (vertical, horizontal). 
             (2, 2) will halve the image in each dimension.

    Returns
    --------
    model: an instance of the model built. 
    """

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=map_dimensions))
    
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD()
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def build_regressor_NN(ndim_conv=2, n_conv_layers=3, nb_filters=64, nb_conv=3, map_dimensions=None, 
                       nb_pool=2, optimizer='rmsprop'):
    """
    Builds a sequential neural network using the Keras framework for a regressor system. 
    The Theano backend was used for testing. Have not tested for Tensorflow. The model 
    contains 1 convolutional layers (2D) by default among others. 

    Parameters
    ----------
    nb_filters: int, number of convolution kernels to use (dimensionality of the output).
    nb_conv: int, the extension (spatial or temporal) of each filter.
             equivalent to "filter_length" in Keras documentation. 
    map_dimensions: tuple, shape of the input data for each sample/observation. 
    nb_pool: tuple of 2 integers, factors by which to downscale (vertical, horizontal). 
             (2, 2) will halve the image in each dimension.

    Returns
    --------
    model: an instance of the model built. 
    """

    model = Sequential()   
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,input_shape=map_dimensions))
    model.add(Activation('sigmoid'))
    model.add(AveragePooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))    
    
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(Dense(128, activation='linear'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='linear'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mae', optimizer=optimizer)
    return model