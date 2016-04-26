"""
A module for plotting outputs of a regressor neural network where the outputs are 
in a map format (i.e. the outputs include the longitutude, latitude, and the 
predicted value). Note: it is asssumed that the neural networks have been trained
and the model structures and weights were saved. Depending on the size of the 
input/output and the running time of the neural network, the plotter may take a while
to run.  
~~~~~~~~~~~~~~~~~~~~~~~~~
Author: Cindy Chiao
Last Edit Date: 04/26/2016
"""
# import utility libraries
import numpy as np
import pandas as pd
from geojson import FeatureCollection, Polygon, Feature, dump
from util import plot_diff_map, scale_data

class Regressor_Plotter(object):
    """
    A class that takes in the neural network input (as a NN_Input class), 
    clustering information, scaling information, and makes bias (predict - true) plots. 
    
    Usage:
    plotter = Regressor_Plotter(nn, n_var, n_clusters, df_cluster, models, 
				df_scaling, resolution=res, outdir=outdir)
    plotter.plot(timepoint)
    """
    
    def __init__(self, nn, nvar, n_clusters, df_cluster, models, df_scaling, resolution, outdir):
        """
	Initialize an instance. 
        
	Parameters
	-----------
	nn: NN_Input instance, see the nn_input.py file in the src folder. 
	nvar: int, number of variables stored in the NN_Input instance. Must be the same
              value used when training the neural networks. 
  	n_clusters: int, number of models to use (assumed to have 1 model per cluster).
	df_cluster: Pandas DataFrame, with the longitudes, latitudes, and cluster numbers as 			 the columns (the column names must be "xs", "ys", and "cluster_num". 
	models: dict, a dictionary with key values being the cluster number and the values
                being each model corresponding to that cluster. 
	df_scaling: Pandas DataFrame, with the cluster numbers, the means, and standard 
		    deviations for each variable for each cluster. The column names must 
		    be "mean", "std". And the values must be in numeric orders (i.e. 
		    cluster 0 must be in the first row). 
	resolution: float, indicates the resolution of the grid file DIVIDED BY 2.
	outdir: str, path to the directory where outputs are saved. 
	"""
	self.nn = nn
        self.nvar = nvar
        self.n_clusters = n_clusters
        self.cluster_list = df_cluster
        self.models = models
        self.scaler = df_scaling
        self.res = resolution
        self.outdir = outdir
        self.timeseries_output = {}
        
	# Calculates the number of map layers per variable
        _, _, test = nn.select(n=1)
        self.layers = (test[0].shape[0])/self.nvar
    
    def plot(self, timepoint, geojson=False, timeseries=False, plot=True):
        """
	Plot, outputs geojson, or save in a time series for a time point. 

	Paramters
	---------
	timepoint: int, the index for the time point to be analyzed. Must be within
		   the amount of data available in the NN_Input object.  
	geojson: boolean, indicates whether or not the output will be saved as a 
                 geojson file. 
	timeseries: boolean, indicates whether or not the output will be saved in the
		    time series dictionary (self.timeseries_output). 
	plot: boolean, indicates whether a bias plot will be made. 
	"""
	ids, y_true, y_predict = self._get_all_data(timepoint)
	timestamp = int(self.nn.times[timepoint])
        print 'for', str(timestamp)
	xs = self.nn.lons[(ids[:,2]).astype(int)]
        ys = self.nn.lats[(ids[:,1]).astype(int)]
        
        if geojson:
            self._save_geojson(xs, ys, y_true, y_predict, timestamp)
        
        if timeseries:
            index = pd.MultiIndex.from_tuples(list(zip(xs, ys)), names=['lon', 'lat'])
            s = pd.Series((y_predict-y_true).flatten(), index=index)
            self.timeseries_output[timestamp] = s
        
	if plot:
            plot_diff_map(xs, ys, y_true, y_predict, timestamp=timestamp, s=1, outdir=self.outdir)
    
    def _get_cluster_list(self, cluster_num):
	"""
	Getting a list of the lons and lats in the cluster. 

	Parameter
	---------
	cluster_num: int, the cluster to be analyzed. 
	"""
        index = self.cluster_list.cluster_num == cluster_num
        return self.cluster_list[index][['xs', 'ys']].values
    
    def _get_all_data(self, timepoint):
	"""
	Get all data from the NN_Input instance from all clusters
	for a specified timepoint

	Parameter
	---------
	timepoint: int, the index for the timepoint chosen. 
	"""
        ids, y_true, y_predict = [], [], []

        for cluster, model in self.models.iteritems():
            subset = self._get_cluster_list(cluster)
            
            id_test, y_test, X_map_test = self.nn.select(t=timepoint, subset=subset)
            y_test = y_test.reshape(-1,1)

            X_test = np.zeros(X_map_test.shape)

            for i in xrange(self.nvar):
                start = i*self.layers
                end = (i+1)*self.layers
                my_mean = self.scaler.iloc[cluster]['mean'+str(i+1)]
                my_std = self.scaler.iloc[cluster]['std'+str(i+1)]
                
                X_slice = X_map_test[:,start:end, :, :]
                X_test[:,start:end, :, :] = scale_data(X_slice, my_mean, my_std)

            test_predict = model.predict(X_test)
            if ids == []:
                ids = id_test
                y_true = y_test
                y_predict = test_predict
            else:
                ids = np.vstack((ids, id_test))
                y_true = np.vstack((y_true, y_test))
                y_predict = np.vstack((y_predict, test_predict))

        return ids, y_true, y_predict
    
    def _save_geojson(self, xs, ys, y_true, y_predict, timestamp):
        """
	Saves the output to a geojson file. 
	"""
	features=[]
        for x, y, val_true, val_predict in zip(xs, ys, y_true, y_predict):
            poly = Polygon([[(x-self.res, y-self.res), (x-self.res, y+self.res), 
                             (x+self.res, y+self.res), (x+self.res, y-self.res), 
                             (x-self.res, y-self.res)]])
            true_class = int(val_true*10)
            predict_class = int(val_predict*10)
            features.append(Feature(geometry=poly, properties={"true": float(true_class), 
                                                               "predict": float(predict_class)}))        
        
        fc = FeatureCollection(features)

        with open(self.outdir+str(int(timestamp))+'.geojson', 'w') as outfile:
            dump(fc, outfile)
