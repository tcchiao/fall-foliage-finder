# Obtained historic climate normals from PRISM project
# http://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf
# http://www.prism.oregonstate.edu/normals/

import subprocess
import numpy as np

variables = ['ppt', 'tmean']
folder = '/Users/Chiao/google-drive/projects/Galvanize/fall-foliage-finder/data/normals/'
base_fname = 'PRISM_{var}_30yr_normal_4kmM2_all_asc/PRISM_{var}_30yr_normal_4kmM2_{m:02d}_asc.asc'
base_nc_name = '{var}.{m:02d}.nc'
gdal = '/opt/local/bin/gdal_translate'

months = np.arange(1, 13)

for var in variables:
    for month in months:
        subprocess.call([gdal, '-of', 'netCDF', 
                         folder+base_fname.format(var=var, m=month), 
                         folder+base_nc_name.format(var=var, m=month)])