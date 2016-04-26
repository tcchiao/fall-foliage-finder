from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from netCDF4 import Dataset
import numpy as np

class Location_Clusterer(KMeans):
    """
    Inherits from Scikit-Learn's KMeans class and adds functionality to read in
    location-based data from netCDF files. Transform 3D numpy arrays from netCDF
    files into 2D (# observations x # features) arrays that KMeans would accept. 
    
    Usage
    -----
    lc = Location_Clusterer(n_clusters=8)
    lc.read_data(filname, varname)
    data_2D = lc.transform_data()
    clusters = lc.fit_predict(data_2D)
    """
    def __init__(self, n_clusters=8, **kwargs):
        """
        Initialize and inherits from Scikit-Learn's KMeans class. 
        Currently only takes in 
        
        Parameters
        ----------
        n_clusters: int, number of clusters to use. Equivalent to n_clusters for
                    KMeans. 
        **kwargs: other acceptable arguments to KMeans.
        """
        KMeans.__init__(self, n_clusters=n_clusters, **kwargs)
        # A list of input data files from which data was read and stored
        self.source_files = []
        # Initialize indices and coordinates (lon, lat) in 3D and 2D formats
        self.ind = None
        self.ind2d = None
        self.coords = None
        self.coords2d = None
        # Initialize the mask that indicates null values
        # Note that True in the mask indicates null values/values out of domain
        self.mask = None
        # Initialize data in 3D and 2D formats
        self.raw_data = None
        self.data2d = None

    def read_data(self, file_name, var_name):
        """
        Read data from netCDF input file (1 file 1 variable at a time)
        
        Parameters
        -----------
        file_name: string, path to input file
        var_name: string, name of variable as appeared in the netCDF file
        """
        # Initialize the coordinate system and dimension if this is the first file read
        if len(self.source_files) == 0:
            nc = Dataset(file_name, 'r')
            self.source_files.append((file_name, var_name))
            
            lats = nc.variables['lat'][:]
            lons = nc.variables['lon'][:]
            self.coords = np.meshgrid(lons, lats)
            y = len(lats)
            x = len(lons)
            self.ind = np.meshgrid(np.arange(x), np.arange(y))
        
        # Reading the actual data
        nc = Dataset(file_name, 'r')
        var = nc.variables[var_name][:]
        if len(var.shape) < 3:
            var = var[np.newaxis]
        # Check dimensions of the new input data against existing data
        if var.shape[-2:] != self.coords[0].shape:
            raise InputError('Dimensions of input data do not match existing data.')
        # Initialize self.raw_data if it is previously empty
        elif self.raw_data is None:
            self.raw_data = np.copy(var)
        # Adding data if there's already some data stored
        else:
            self.raw_data = np.append(self.raw_data, var, axis=0)
        
        # Check the mask from this file and initialize or update current mask 
        # Note that True in the mask indicates null values/values out of domain
        if len(var.mask.shape) == 3:
            mask = var.mask[0][np.newaxis]
        else:
            mask = var.mask[np.newaxis]
            
        if self.mask is None:
            self.mask = np.copy(mask)
        else:
            self.mask = np.append(self.mask, mask, axis=0)
            self.mask = np.any(self.mask, axis=0)[np.newaxis]
            
    def transform_data(self):
        """
        Transform data and the corresponding indices/coordinates from 3D to 2D.
        """
        # Calculate the number of non-null values
        inmask = self.mask.flatten().shape[0] - sum(self.mask.flatten())
        # Get mask in 2D format (with True = values to use)
        # Note that the mask in 3D format has True = null values
        flatten_mask = (1-self.mask.flatten()).astype(bool)
        
        # Transform indices and coordinates to 2D
        self.ind2d = np.zeros((inmask, len(self.coords)))
        self.coords2d = np.zeros((inmask, len(self.coords)))
        for i in xrange(len(self.coords)):
            self.coords2d[:, i] = self.coords[i].flatten()[flatten_mask]
            self.ind2d[:, i] = self.ind[i].flatten()[flatten_mask]
        
        # Transform data to 2D
        self.data2d = np.zeros((inmask, self.raw_data.shape[0]))
        for i in xrange(self.raw_data.shape[0]):
            temp = self.raw_data[i].flatten().data
            self.data2d[:, i] = temp[flatten_mask]
        
        # Scale data by mean and standard deviation
        scaler = StandardScaler()
        self.data2d = scaler.fit_transform(self.data2d)
        return self.data2d