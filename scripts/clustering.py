
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from netCDF4 import Dataset
import numpy as np
import pandas as pd

class Location_Clusterer(KMeans):
    def __init__(self):
        KMeans.__init__(self)
        self.source_files = []
        self.coords = None
        self.mask = None
        self.raw_data = None
        self.clean_data = None

    def read_data(self, file_name, var_name):
        '''
        Read data from netCDF input file (1 file 1 variable at a time)
        
        INPUT:
            file_name -> string, path to input file
            var_name -> string, name of variable as appeared in the netCDF file
        '''
        
        # Initialize the coordinate system and dimension if this is the first file read
        if len(self.source_files) == 0:
            nc = Dataset(file_name, 'r')
            self.source_files.append((file_name, var_name))
            
            lats = nc.variables['lat'][:]
            lons = nc.variables['lon'][:]
            self.coords = np.meshgrid(lons, lats)          
        
        # Reading the actual data
        var = nc.variables[var_name][:]
        # Check dimensions of the new input data against existing data
        if var.shape[-2:] != self.coords[0].shape:
            raise InputError('Dimensions of input data do not match existing data.')
        # Initialize self.raw_data if it is previously empty
        elif self.raw_data is None:
            self.raw_data = np.copy(var)
        # Adding data if there's already some data stored
        else:
            self.raw_data = np.append(self.raw_data, var, axis0)
        
        # Check the mask from this file and initialize or update current mask 
        if len(var.mask.shape) == 3:
            mask = var.mask[0]
        else:
            mask = var.mask
            
        if self.mask is None:
            self.mask = np.copy(mask)
        else:
            self.mask = np.any(np.append(self.mask, var.mask, axis=0), axis=0)
            
        
        def clean_data(self):
            pass
        
        
            
        