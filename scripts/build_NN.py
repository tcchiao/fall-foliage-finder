from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import SGD

def build_base_sequential_NN(n_conv_layers=2, nb_filters=16, nb_conv=3, map_dimensions=None, nb_pool=2):

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