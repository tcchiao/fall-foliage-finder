# import utility libraries
from netCDF4 import Dataset
import pandas as pd
import numpy as np
from collections import OrderedDict

# import machine learning tools
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K

# import utilities and classes I wrote
from clustering import Location_Clusterer

class NN_Input(object):
    """
    Stores the input data ready for feeding into a keras neural network. 

    To-Do:
    - add function to take the clustering data in some ways
    - add function to return the actual lat, lon, and time based on indices
    
    """
    def __init__(self, predict=2, history=3):
        """
        Initialize a class for storing neural network input data. 
        
        Parameters
        ----------
        predict: int, number of time points ahead that the model will predict. 
                 For example, if predict=2, the model will predict 2 time points away from the given time. 
        history: int, number of time points for which data would be included as input.
                 For example, if data_length=3, the model will receive 3 time points worth of data (current time
                 point, the previous time point, and the timep point before that).
        """
        self.lons = None
        self.lats = None
        self.times = None
        
        self.labels = None
        self.features = OrderedDict()
        self.feature_types = OrderedDict()
        
        self.predict = predict
        self.history = history
        
    def load_labels(self, f_path, var):
        """
        Load labels from netCDF file. 
        
        Parameters
        ----------
        f_path: string
        var: string
        """
        nc = Dataset(f_path, 'r')
        self.lons = nc.variables['lon'][:]
        self.lats = nc.variables['lat'][:]
        
        self.times = nc.variables['time'][self.history-1:-self.predict]
        n = self.predict + self.history - 1
        self.labels = nc.variables[var][n:,:,:]
        
    def load_features(self, f_path, var, name, feature_type):
        """
        Load feature values from netCDF files. Stores feature type information. 
        
        Parameters
        ----------
        f_path: string, path to input netCDF file.
        var: string, variable name as appeared in the netCDF file. 
        name: string, name of the variable to be stored. 
        feature_type: string, must be one of the following: 'history_time_series', 'forecast_time_series', 
        'multi_layers', 'single_layer'
        """
        nc = Dataset(f_path, 'r')
        temp_data = nc.variables[var][:]
        
        # Storing information on whether the input features 
        self.feature_types[name] = feature_type
        
        if self.feature_types[name] == 'history_time_series':
            self.features[name] = temp_data[:-self.predict, :, :]
        elif self.feature_types[name] == 'forecast_time_series':
            self.features[name] = temp_data[self.history-1:, :, :]
        else:
            self.features[name] = temp_data
        
    
    def get_features(self, i, j, k):
        """
        Given indices for latitude, longitude, and time point, returns the associated data from self.data. 
        
        Parameters
        ----------
        lat: int, index for the latitude desired. Must be within the range available in self.data. 
        lon: int, index for the longitude desired. Must be within the range available in self.data. 
        time: int, index for the time point desired. Must be within the range available in self.data. 
        """
        output = []
        for feat in self.variables:
            if self.feature_types[feat] == 'history_time_series':
                pass
            elif self.feature_types[feat] == 'forecast_time_series':
                pass
            elif self.feature_types[feat] == 'multi_layers':
                pass
            else: 
                pass
        
    def select(self, n, cutoff=None):
        if cutoff is None:
            cutoff = len(self.times)/2
            
        output = []
        while len(output) < n:
            i = np.random.randint(cutoff)
            j = np.random.randint(self.lats)
            k = np.random.randint(self.lons)
            features, masks = self.get_features(i, j, k)
            if not np.any(masks):
                indices = (i, j, k)
                label = self.labels[i, j, k]
                output.append([indices, label, features])
        return output
    
    def variables(self):
        return self.features.keys()