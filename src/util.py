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
from keras.models import model_from_json


def scale_data(data, mean, std):
    """
    Scales data by substracting mean and dividing by standard deviation. 
    """
    return (data-mean)/std

def load_models(n_clusters, modeldir, loss='mae', optimizer='adagrad'):
    """
    Loads Keras neural network models from file. Model structure is stored in 
    json format and model weights are stored in h5 (hdf5) files. The models 
    are compiled with the provided loss and optimizer, then put into a 
    dictionary to be used. This function was designed to be used with clustered 
    data, with 1 model per cluster. Thus, the keys in the dictionary is the 
    cluster number. In addition to the models, a DataFrame of cluster list is 
    also returned containing the x, y index of each entry in the cluster. 
    """

    models = {}
    df_cluster = None

    for cluster in xrange(n_clusters):
        json_file = modeldir+'model_architecture_'+str(cluster)+'.json'
        weights_file = modeldir+'model_weights_'+str(cluster)+'.h5'

        models[cluster] = model_from_json(open(json_file).read())
        models[cluster].load_weights(weights_file)
        models[cluster].compile(loss=loss, optimizer=optimizer)

        df_subset = pd.read_table(modeldir+'cluster_list_'+str(cluster)+'.txt',
                                 delimiter=' ', names = ['xs', 'ys'])
        df_subset['cluster_num'] = np.ones(len(df_subset), dtype=int)*cluster
        df_cluster = pd.concat((df_cluster, df_subset))
    
    return models, df_cluster

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
    plt.title(title, fontsize=18, fontweight='bold', loc='center')
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