# import utility libraries
import numpy as np
import pandas as pd
from geojson import FeatureCollection, Polygon, Feature, dump
from util import plot_diff_map, scale_data

class Regressor_Plotter(object):
    """
    A class that takes in the neural network input (NN_Input class), clustering information, 
    scaling information, and makes bias (predict - true) plots. 
    """
    
    def __init__(self, nn, nvar, n_clusters, df_cluster, models, df_scaling, resolution, outdir):
        self.nn = nn
        self.nvar = nvar
        self.n_clusters = n_clusters
        self.cluster_list = df_cluster
        self.models = models
        self.scaler = df_scaling
        self.res = resolution
        self.outdir = outdir
        self.timeseries_output = {}
        
        _, _, test = nn.select(n=1)
        self.layers = (test[0].shape[0])/self.nvar
    
    def plot(self, timepoint, geojson=False, timeseries=False, plot=True):
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
        index = self.cluster_list.cluster_num == cluster_num
        return self.cluster_list[index][['xs', 'ys']].values
    
    def _get_all_data(self, timepoint):
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
