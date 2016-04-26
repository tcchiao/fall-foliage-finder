# import utility libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from netCDF4 import Dataset

# import machine learning tools
from keras.models import model_from_json

# import functions and classese for use
from util import load_models
from nn_input import NN_Input
from regressor_plot import Regressor_Plotter

## Setting variables 

# dir path to saved models
modeldir = '/home/ubuntu/fall-foliage-finder/models/regressor/' 
# dir path to output
outdir = '/home/ubuntu/dataset/output/regressor/'   
# dir path to input weather data
folder = '/home/ubuntu/dataset/int_weather/'    
# number of clusters
n_clusters = 8
# number of variables loaded in NN_Input
n_var = 4  
# Spatial resolution of maps and data
res = 0.0625/2
# Time point start and finish 
time_start = 350
time_end = 400


# Loading saved models
models, df_cluster = load_models(n_clusters, modeldir, loss='mae', optimizer='adagrad')

# Loading data for training neural network 
nn = NN_Input(predict=2, history=2, box=5, random_seed=42)
nn.load_labels(folder+'all.ndvi.normed.nc', 'Band1')

f_paths = ['all.mean.of.Tmin.nc', 'all.mean.of.Tmin.nc' , 'all.min.of.Tmin.nc', 'all.min.of.Tmin.nc', 
           'all.sum.of.Prec.nc', 'all.sum.of.Prec.nc', 'all.max.of.Wind.nc', 'all.max.of.Wind.nc']
variables = ['Tmin', 'Tmin', 'Tmin', 'Tmin', 'Prec', 'Prec', 'Wind', 'Wind']
names = ['mean_tmin_history', 'mean_tmin_forecast', 'min_tmin_history', 'min_tmin_forecast',
         'sum_prec_history', 'sum_prec_forecast', 'max_wind_history', 'max_wind_forecast']
feature_types = ['history_time_series', 'forecast_time_series', 'history_time_series', 'forecast_time_series',
                'history_time_series', 'forecast_time_series', 'history_time_series', 'forecast_time_series']

for f_path, v, n, feature_type in zip(f_paths, variables, names, feature_types):
    nn.load_features(folder+f_path, v, n, feature_type)
    
# Loading scaling weights
df_scaling = pd.read_csv(modeldir+'scaling_weights.txt')

for i in xrange(n_var):
    df_scaling['mean'+str(i+1)] = df_scaling['mean'].apply(lambda x: float(x.split()[i]))
    df_scaling['std'+str(i+1)] = df_scaling['std'].apply(lambda x: float(x.split()[i]))

df_scaling = df_scaling.drop(['mean', 'std'], axis=1)

plotter = Regressor_Plotter(nn, n_var, n_clusters, df_cluster, models, df_scaling, 
                            resolution=res, outdir=outdir)

for timepoint in xrange(time_start, time_end):
    plotter.plot(timepoint, timeseries=True, plot=False)
    
# Aggregate time series data saved in a Regressor_Plotter instance by season
seasons = {'03':'Spring', '04':'Spring', '05':'Spring',
           '06':'Summer', '07':'Sumeer', '08':'Summer',
           '09':'Fall', '10':'Fall', '11':'Fall',
           '12':'Winter', '01':'Winter', '02':'Winter'}

df_diff = pd.DataFrame(plotter.timeseries_output).T
df_diff['season'] = [seasons[str(x)[4:6]] for x in df_diff.index]
df_diff_season = df_diff.groupby('season').mean().T

# Plot seasonal average bias 

cmap=mpl.cm.get_cmap('seismic')
norm = mpl.colors.Normalize(vmin=-1,vmax=1)
title = 'Seasonal mean bias across continental U.S.'

area_thresh=2500
land_color='grey'
ocean_color='lightblue'

fig = plt.figure(figsize=(18,11))
plt.axis('off')
for i, seas in enumerate(['Spring', 'Summer', 'Fall', 'Winter']):
    plt.subplot(2,2,i+1)
    y_diff = np.mean(df_diff_season[seas].dropna().values, axis=1)
    lons, lats = zip(*df_diff_season[seas].dropna().index.values)
    lons = np.array(lons)
    lats = np.array(lats)

    m = Basemap(projection='merc', llcrnrlat=lats.min(), llcrnrlon=lons.min(),
            urcrnrlat=lats.max(), urcrnrlon=lons.max(), resolution='f',
            area_thresh=area_thresh)
    xi, yi = m(lons, lats)
    m.scatter(xi, yi, c=y_diff, edgecolor='none', cmap=cmap, s=s, norm=norm)
    m.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=True)
    ax = plt.gca()
    ax.text(0.05, 0.05, seas, weight='semibold', horizontalalignment='left',
            verticalalignment='bottom', transform=ax.transAxes, fontsize=24)
plt.tight_layout()
img_path = outdir+'avg_anomaly_per_season.png'
plt.savefig(img_path, bbox_inches='tight', dpi=300)