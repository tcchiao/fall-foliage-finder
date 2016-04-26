"""
A neural network training script using pre-defined network architecture. 
~~~~~~~~~~~~~~~~~~~
Author: Cindy Chiao
Last Edit Date: 04/26/2016

This script takes in data from netCDF files, performs K Means clustering 
on locations within the mask based on specified variables, then trains
1 neural network for each cluster to predict the target variable two time
points in the future. Note: this script takes approximately 4 hours to run
using GPU.
"""
# import utility libraries
from netCDF4 import Dataset
import numpy as np
from collections import defaultdict

# import machine learning tools
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D

from clustering import Location_Clusterer
from util import scale_data, eval_regressor_model
from nn_input import NN_Input
from build_NN import build_regressor_NN

# Clustering locations based on vegetation types, elevation, 
# and historic precip and mean temperature for each month
n_clusters = 8
folder = '/home/ubuntu/dataset/'
outdir = '/home/ubuntu/fall-foliage-finder/models/regressor/'
files = ['veg.nc', 'ppt.monthly.mask.nc', 'tmean.monthly.mask.nc', 'elev.nc']
var_names = ['Cv', 'Band1', 'Band1', 'elev']

lc = Location_Clusterer(n_clusters=n_clusters)
for f, var in zip(files, var_names):
    lc.read_data(folder+f, var)

lc.transform_data()
clusters = lc.fit_predict(lc.data2d)

# Loading weather data for training neural network 
folder = '/home/ubuntu/dataset/weather/'
nn = NN_Input(predict=2, history=2, box=5, random_seed=42)
nn.load_labels(folder+'all.ndvi.normed.nc', 'Band1')

# Recording the number of variables loaded. Later on each variable would get their own scaling factors
n_var = 4

f_paths = ['all.mean.of.Tmin.nc', 'all.mean.of.Tmin.nc' , 'all.min.of.Tmin.nc', 'all.min.of.Tmin.nc', 
           'all.sum.of.Prec.nc', 'all.sum.of.Prec.nc', 'all.max.of.Wind.nc', 'all.max.of.Wind.nc']
variables = ['Tmin', 'Tmin', 'Tmin', 'Tmin', 'Prec', 'Prec', 'Wind', 'Wind']
names = ['mean_tmin_history', 'mean_tmin_forecast', 'min_tmin_history', 'min_tmin_forecast',
         'sum_prec_history', 'sum_prec_forecast', 'max_wind_history', 'max_wind_forecast']
feature_types = ['history_time_series', 'forecast_time_series', 'history_time_series', 'forecast_time_series',
                'history_time_series', 'forecast_time_series', 'history_time_series', 'forecast_time_series']

for f_path, v, n, feature_type in zip(f_paths, variables, names, feature_types):
    nn.load_features(folder+f_path, v, n, feature_type)


# Train one neural network per cluster using the same structure
models = {}
mean_train = defaultdict(list)
std_train = defaultdict(list)

for cluster in xrange(n_clusters):
    print 'Modeling for cluster', cluster, '....'
    subset = lc.ind2d[clusters==cluster]

    np.savetxt(outdir+'cluster_list_'+str(cluster)+'.txt', subset)

    print '-- Getting training dataset'
    id_train, y_train, X_map_train = nn.select(n=100000, subset=subset, ndim_out=4)

    print '-- Getting testing dataset'
    id_test, y_test, X_map_test = nn.select(n=100000, cutoff=-200, subset=subset, ndim_out=4)
    
    # Scaling and reformatting data
    print '-- Scaling and reformatting data'
    map_dimensions = X_map_train[0].shape

    n_layers = map_dimensions[0]/n_var
    X_train = np.zeros(X_map_train.shape)
    X_test = np.zeros(X_map_test.shape)

    for i in xrange(n_var):
        X_slice = X_map_train[:,i*n_layers:(i+1)*n_layers,:,:]
        valid_vals = X_slice[X_slice != -999].flatten()
        
        mean_train[cluster].append(np.mean(valid_vals))
        std_train[cluster].append(np.std(valid_vals))

        X_train[:,i*n_layers:(i+1)*n_layers,:,:] = scale_data(X_map_train[:,i*n_layers:(i+1)*n_layers,:,:], mean_train[cluster][i], std_train[cluster][i])
        X_test[:,i*n_layers:(i+1)*n_layers,:,:] = scale_data(X_map_test[:,i*n_layers:(i+1)*n_layers,:,:], mean_train[cluster][i], std_train[cluster][i])


    # Building neural network
    print '-- Building neural network'
    model = build_regressor_NN(ndim_conv=2, nb_filters=64, map_dimensions=map_dimensions, optimizer='adagrad')
    model.fit(X_train, y_train, batch_size=20, nb_epoch=1, verbose=1, validation_data=(X_test, y_test))
    
    # Evaluating model
    print '-- Evaluating model'
    eval_regressor_model(model, cluster, X_train, y_train, X_test, y_test, 
                        outdir='/home/ubuntu/dataset/output/regressor/')
    
    # Adding model to the dictionary
    print '-- Packaging model'
    json_string = model.to_json()
    with open(outdir+'model_architecture_'+str(cluster)+'.json', 'w') as f:
        f.write(json_string)

    model.save_weights(outdir+'model_weights_'+str(cluster)+'.h5')

    print 

with open(outdir+'scaling_weights.txt', 'w') as f:
    f.write(','.join(['cluster', 'mean', 'std']))
    for cluster in xrange(n_clusters):
        means = [str(x) for x in mean_train[cluster]]
        stds = [str(x) for x in std_train[cluster]]
        f.write(','.join([str(cluster), ' '.join(means), ' '.join(stds)]))

