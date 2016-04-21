# import utility libraries
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import cv2

# import machine learning tools
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Graph
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import roc_auc_score, accuracy_score

from clustering import Location_Clusterer
from util import plot_list_in_2D, reformat_y, plot_compare_map, scale_data, evaluate_model
from nn_input import NN_Input
from build_NN import build_base_sequential_NN


n_clusters = 8
folder = '/home/ubuntu/dataset/'
out_dir = '/home/ubuntu/fall-foliage-finder/models/tmin_prec/'
files = ['veg.nc', 'ppt.monthly.mask.nc', 'tmean.monthly.mask.nc', 'elev.nc']
var_names = ['Cv', 'Band1', 'Band1', 'elev']

lc = Location_Clusterer(n_clusters=n_clusters)
for f, var in zip(files, var_names):
    lc.read_data(folder+f, var)

lc.transform_data()
clusters = lc.fit_predict(lc.data2d)


nn = NN_Input(predict=2, history=5, box=5, random_seed=42)
nn.load_labels(folder+'sign.label.nc', 'Band1')

# f_paths = ['all.ndvi.nc','all.max.of.Wind.nc', 'all.min.of.Tmin.nc', 'all.mean.of.Tmin.nc', 'all.sum.of.Prec.nc',
#            'all.max.of.Tmax.nc', 'all.mean.of.Tmax.nc','elev.nc', 'veg.nc']
# variables = ['Band1', 'Wind', 'Tmin', 'Tmin', 'Prec', 'Tmax', 'Tmax', 'elev', 'Cv']
# names = ['ndvi', 'max_wind', 'min_tmin', 'mean_tmin', 'total_prec', 'max_tmax', 'mean_tmax', 'elev', 'veg']
# feature_types = ['history_time_series', 'forecast_time_series', 'forecast_time_series', 'forecast_time_series',
#                  'forecast_time_series', 'forecast_time_series', 'forecast_time_series',
#                 'single_layer', 'multi_layers']

f_paths = ['all.mean.of.Tmin.nc', 'all.mean.of.Tmin.nc' , 'all.min.of.Tmin.nc', 'all.min.of.Tmin.nc', 
           'all.sum.of.Prec.nc', 'all.sum.of.Prec.nc']
variables = ['Tmin', 'Tmin', 'Tmin', 'Tmin', 'Prec', 'Prec']
names = ['mean_tmin_history', 'mean_tmin_forecast', 'min_tmin_history', 'min_tmin_forecast',
         'sum_prec_history', 'sum_prec_forecast']
feature_types = ['history_time_series', 'forecast_time_series', 'history_time_series', 'forecast_time_series',
                'history_time_series', 'forecast_time_series']

for f_path, v, n, feature_type in zip(f_paths, variables, names, feature_types):
    nn.load_features(folder+f_path, v, n, feature_type)


models = {}
mean_train = {}
std_train = {}

for cluster in xrange(n_clusters):
    print 'Modeling for cluster', cluster, '....'
    subset = lc.ind2d[clusters==cluster]

    np.savetxt(out_dir+'cluster_list_'+str(cluster)+'.txt', subset)

    print '-- Getting training dataset'
    id_train, y_train, X_map_train = nn.select(n=100000, subset=subset)

    print '-- Getting testing dataset'
    id_test, y_test, X_map_test = nn.select(n=100000, cutoff=-200, subset=subset)
    
    # Scaling and reformatting data
    print '-- Scaling and reformatting data'
    map_dimensions = X_map_train[0].shape
    mean_train[cluster] = np.mean(X_map_train.flatten())
    std_train[cluster] = np.std(X_map_train.flatten())
    
    X_train = scale_data(X_map_train, mean_train[cluster], std_train[cluster])
    X_test = scale_data(X_map_test, mean_train[cluster], std_train[cluster])

    y_train = reformat_y(y_train)
    y_test = reformat_y(y_test)

    # Building neural network
    print '-- Building neural network'
    model = build_base_sequential_NN(nb_filters=64, map_dimensions=map_dimensions)
    model.fit(X_train, y_train, batch_size=50, nb_epoch=2, verbose=True, validation_data=(X_test, y_test))
    
    train_predict = model.predict(X_train, verbose=True)
    test_predict = model.predict(X_test, verbose=True)
    
    # Evaluating model
    print '-- Evaluating model'
    evaluate_model(y_train, train_predict, y_test, test_predict, threshold=0.5)
    
    # Adding model to the dictionary
    print '-- Packaging model'
    #models[str(cluster)] = model
    json_string = model.to_json()
    with open(out_dir+'model_architecture_'+str(cluster)+'.json', 'w') as f:
        f.write(json_string)

    model.save_weights(out_dir+'model_weights_'+str(cluster)+'.h5')

    print 

with open(out_dir+'scaling_weights.txt', 'w') as f:
    f.write(' '.join(['cluster', 'mean', 'std']))
    for cluster in xrange(n_clusters):
        f.write(' '.join([str(cluster), str(mean_train[cluster]), str(std_train[cluster])]))

