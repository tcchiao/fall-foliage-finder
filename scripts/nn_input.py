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
    Stores the input data ready for feeding into a Keras neural network. 

    To-Do:
    - add function to take the clustering data in some ways
    - add function to return the actual lat, lon, and time based on indices
    
    """
    def __init__(self, predict=2, history=2, box=5):
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
        self.features = {}
        self.feature_types = {}
        self.variables = []
        
        self.predict = predict
        self.history = history
        self.box = box
        
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
        
        self.times = nc.variables['time'][self.history:-self.predict]
        n = self.predict + self.history
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
        self.variables.append(name)
        
        if self.feature_types[name] == 'history_time_series':
            self.features[name] = temp_data[:-self.predict, :, :]
        elif self.feature_types[name] == 'forecast_time_series':
            self.features[name] = temp_data[self.history:, :, :]
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
        maps = None
        lst = None
        for ix, feat in enumerate(self.variables):
            if self.feature_types[feat] == 'history_time_series':
                temp_data = self.features[feat][i:i+self.history+1, j-self.box:j+self.box+1, k-self.box:k+self.box+1]
            elif self.feature_types[feat] == 'forecast_time_series':
                temp_data = self.features[feat][i:i+self.predict+1, j-self.box:j+self.box+1, k-self.box:k+self.box+1]
            elif self.feature_types[feat] == 'multi_layers':
                temp_data = self.features[feat][:, j, k].flatten()
            else: 
                temp_data = self.features[feat][j, k]
            
            if len(temp_data.shape) == 3:                
                if np.sum(temp_data.mask) > len(temp_data.flatten())/2:
                    return None
                elif np.any(temp_data.mask):
                    temp_data = temp_data.filled(-999)
                    
                if maps is None:
                    maps = temp_data
                else:
                    maps = np.ma.concatenate((maps, temp_data), axis=0)
            else:
                if lst is None:
                    lst = temp_data
                else:
                    lst = np.append(lst, temp_data)
        return [maps, lst]
        
    def select(self, n=None, cutoff=None, subset=None):
        """
        Selecting n data points randomly from the database before specified time cutoff. 
        
        Parameters
        ----------
        n: int, number of data points wanted. 
        cutoff: int, time cutoff for the training dataset. Default is half of the data available. 
        """
        if cutoff is None:
            cutoff = len(self.times)/2
        
        indices, labels, output_maps, output_lst = [], [], [], []
        
        mi = 0
        for ix, feat in enumerate(self.variables):
            if self.feature_types[feat] == 'history_time_series':
                mi += (self.history+1)
            elif self.feature_types[feat] == 'forecast_time_series':
                mi += (self.predict+1)
                
        map_dimensions = (mi, (2*self.box)+1, (2*self.box)+1)

        if n is None:
            for (k, j) in subset:
                for i in xrange(cutoff):
                    l = self.labels[i, j, k]
                    features = self.get_features(i, j, k)
                    if features is not None and l != np.nan and features[0].shape==map_dimensions:
                        indices.append([i, j, k])
                        labels.append(l)
                        output_maps.append(features[0])
                        output_lst.append(features[1])
        
        else:
            while len(labels) < n:
                if subset is not None:
                    (k, j) = subset[np.random.choice(len(subset))]
                else:
                    j = np.random.randint(self.box, len(self.lats)-self.box)
                    k = np.random.randint(self.box, len(self.lons)-self.box)
                i = np.random.randint(cutoff)
                l = self.labels[i, j, k]
                features = self.get_features(i, j, k)
                if features is not None and l != np.nan and features[0].shape==map_dimensions:
                    indices.append([i, j, k])
                    labels.append(l)
                    output_maps.append(features[0])
                    output_lst.append(features[1])
                    
        return indices, labels, output_maps, output_lst
