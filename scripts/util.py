from netCDF4 import Dataset
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

def reformat_y(y):
    y[y == -1] = 0
    y = np.hstack((y.reshape(-1,1), 1-y.reshape(-1,1)))
    return y

def plot_time_series(i, j, time_series):
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(len(time_series)), time_series)
    note = 'lat = {0}, lon = {1}'.format(lats[i], lons[j])
    ax.text(0.9, 2, note, fontsize=20)
    plt.show()
    
def plot_list_in_2D(x, y, val):
    plt.scatter(x, y, c=val, edgecolors='none', s=3)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.show() 

def plot_compare_map(lons, lats, y_true, y_predict, timestamp, cmap=None, folder='/home/ubuntu/dataset/output/'):
    timestamp = str(int(timestamp))
    yr = timestamp[:4]
    mn = timestamp[4:6]
    dt = timestamp[6:]
    
    if cmap is None:
        cmap=mpl.cm.get_cmap('RdYlGn')
        
    norm = mpl.colors.Normalize(y_true.min, y_true.max)
    area_thresh=25000
    land_color='grey'
    ocean_color='lightblue'

    fig = plt.figure(figsize=(10,7))
    plt.title('/'.join([yr, mn, dt]), loc='right', fontsize=14)
    plt.axis('off')
    ax = fig.add_subplot(2,1,1)
    m = Basemap(projection='cyl', llcrnrlat=lats.min(), llcrnrlon=lons.min(),
            urcrnrlat=lats.max(), urcrnrlon=lons.max(), resolution='f',
            area_thresh=area_thresh)
    m.scatter(lons, lats, c=y_true, edgecolor='none', cmap=cmap)
    m.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=True)
    m.drawcountries()
    m.drawcoastlines()
    
    ax = fig.add_subplot(2,1,2)
    m = Basemap(projection='cyl', llcrnrlat=lats.min(), llcrnrlon=lons.min(),
            urcrnrlat=lats.max(), urcrnrlon=lons.max(), resolution='f',
            area_thresh=area_thresh)
    m.scatter(lons, lats, c=y_predict,edgecolor='none',cmap=cmap)
    m.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=True)
    m.drawcountries()
    m.drawcoastlines()
    img_path = folder+str(int(timestamp))+'.png'
    plt.savefig(img_path)
    return img_path