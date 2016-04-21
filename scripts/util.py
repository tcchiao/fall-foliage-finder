from netCDF4 import Dataset
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.animation as animation
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score, roc_auc_score

def reformat_y(y):
    y[y == -1] = 0
    y = np.hstack((y.reshape(-1,1), 1-y.reshape(-1,1)))
    return y

def scale_data(data, mean, std):
    return (data-mean)/std

def evaluate_model(y_train, train_predict, y_test, test_predict, threshold=0.5):
    bm = np.sum(y_train[:,1])/float(len(y_train))
    acc = accuracy_score(y_train[:,0], (train_predict[:,0]>threshold))
    try:
        auc = roc_auc_score(y_train[:,0], train_predict[:,0])
    except ValueError:
        auc = 'N/A'
        
    print 'Training set:'
    print 'Bench mark:', bm
    print 'Accuracy:', acc
    print 'ROC AUC:', auc
    
    bm = np.sum(y_test[:,1])/float(len(y_test))
    acc = accuracy_score(y_test[:,0], (test_predict[:,0]>threshold))
    try:
        auc = roc_auc_score(y_test[:,0], test_predict[:,0])
    except ValueError:
        auc = 'N/A'
        
    print 'Testing set:'
    print 'Bench mark:', bm
    print 'Accuracy:', acc
    print 'ROC AUC:', auc

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

#     rows, row_pos = np.unique(lats, return_inverse=True)
#     cols, col_pos = np.unique(lons, return_inverse=True)

#     y_true_array = np.zeros((len(rows), len(cols)))
#     y_true_array[row_pos, col_pos] = y_true
#     y_predict_array = np.zeros((len(rows), len(cols)))
#     y_predict_array[row_pos, col_pos] = y_true    
     
    timestamp = str(int(timestamp))
    yr = timestamp[:4]
    mn = timestamp[4:6]
    dt = timestamp[6:]
    
    if cmap is None:
        cmap=mpl.cm.get_cmap('RdYlGn')
        
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
#     xi, yi = m(lons, lats)
#     xi, yi = np.meshgrid(xi, yi)
#     m.pcolormesh(xi, yi, y_true, cmap=cmap, norm=norm)
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

def animate_maps(img_paths, ani_path):
    ims = []
    fig = plt.figure()
    plt.axis('off')

    for p in img_paths:
        img=mpimg.imread(p)    
        ims.append((plt.imshow(img),))

    im_ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=3000, blit=False)
    mywriter = animation.FFMpegWriter(fps=10)
    im_ani.save(ani_path, writer=mywriter)