import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, cm
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

def reformat_y(y):
    y[y == -1] = 0
    y = np.hstack((y.reshape(-1,1), 1-y.reshape(-1,1)))
    return y

def scale_data(data, mean, std):
    return (data-mean)/std

def eval_regressor_model(model, cluster, X_train, y_train, 
                         X_test, y_test, outdir='/home/ubuntu/dataset/output/'):
    
    train_predict = model.predict_proba(X_train, verbose=0)
    test_predict = model.predict_proba(X_test, verbose=0)

    train_err = train_predict - y_train.reshape(-1,1)
    test_err = test_predict - y_test.reshape(-1,1)

    plt.figure(figsize=(7,4))
    plt.title('Error Histogram for Cluster '+str(cluster))
    plt.hist([train_err, test_err], label=['Train', 'Test'], bins=20)
    plt.xlabel('Error')
    plt.ylabel('# Observations')
    plt.legend()
    plt.savefig(outdir+'hist_error_'+str(cluster)+'.png', bbox_inches='tight', dpi=300)
    
    slope, intercept, r_train, p_value, std_err = stats.linregress(y_train.flatten(), train_predict.flatten())
    print "Train r-squared:", r_train**2
    slope, intercept, r_test, p_value, std_err = stats.linregress(y_test.flatten(), test_predict.flatten())
    print "Test r-squared:", r_test**2

    plt.figure(figsize=(7,7))
    plt.title("Cluster {0}: Train r-squared = {1:.4f}, Test r-sqaured = {2:.4f}".format(cluster, r_train**2, r_test**2))
    plt.scatter(y_train, train_predict, label='Train', color='r', edgecolor='none', alpha=0.5, s=1)
    plt.scatter(y_test, test_predict, label='Test', color='b', edgecolor='none', alpha=0.5, s=1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(outdir+'actual_predict_'+str(cluster)+'.png', bbox_inches='tight', dpi=300)

def eval_classifier_model(y_train, train_predict, y_test, test_predict, threshold=0.5):
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

def plot_compare_map(lons, lats, y_true, y_predict, timestamp, s=None, 
                     cmap=None, outdir='/home/ubuntu/dataset/output/'):
     
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
    m.scatter(lons, lats, c=y_true, edgecolor='none', cmap=cmap, s=s)
    m.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=True)
    m.drawcountries()
    m.drawcoastlines()
    
    ax = fig.add_subplot(2,1,2)
    m = Basemap(projection='cyl', llcrnrlat=lats.min(), llcrnrlon=lons.min(),
            urcrnrlat=lats.max(), urcrnrlon=lons.max(), resolution='f',
            area_thresh=area_thresh)
    m.scatter(lons, lats, c=y_predict,edgecolor='none',cmap=cmap, s=s)
    m.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=True)
    m.drawcountries()
    m.drawcoastlines()
    img_path = outdir+str(int(timestamp))+'.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    return img_path

def plot_diff_map(lons, lats, y_true=None, y_predict=None, y_diff=None, timestamp=None, 
                  s=None, cmap=None, outdir='/home/ubuntu/dataset/output/'):
     
    timestamp = str(int(timestamp))
    yr = timestamp[:4]
    mn = int(timestamp[4:6])
    dt = timestamp[6:]
    months_lst = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    title = months_lst[mn]+' '+dt+', '+yr
    
    if cmap is None:
        cmap=mpl.cm.get_cmap('seismic')
    norm = mpl.colors.Normalize(vmin=-1.5,vmax=1.5)
        
    area_thresh=2500
    land_color='grey'
    ocean_color='lightblue'

    fig = plt.figure(figsize=(9,7))
    plt.title(title, fontsize=18, fontweight='bold', loc='right')
    plt.axis('off')
    m = Basemap(projection='merc', llcrnrlat=lats.min(), llcrnrlon=lons.min(),
            urcrnrlat=lats.max(), urcrnrlon=lons.max(), resolution='f',
            area_thresh=area_thresh)
    xi, yi = m(lons, lats)
    if y_diff is None:
        y_diff = y_predict-y_true
    m.scatter(xi, yi, c=y_diff, edgecolor='none', cmap=cmap, s=s, norm=norm)
    m.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=True)
    plt.tight_layout()

    img_path = outdir+str(int(timestamp))+'.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
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