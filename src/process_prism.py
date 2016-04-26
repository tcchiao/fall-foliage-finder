"""
This script is used to process downloaded historic normal data from PRISM. 
~~~~~~~~~~~~~~~~~~~~
Author: Cindy Chiao
Last Edit Date: 04/19/2016

Obtained historic climate normals from PRISM project
http://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf
http://www.prism.oregonstate.edu/normals/
"""
import subprocess
import numpy as np

# Setting local variables
variables = ['ppt', 'tmean']
folder = '../data/normals/'
grid = '../data/mask.nc'

base_fname = 'PRISM_{var}_30yr_normal_4kmM2_all_asc/PRISM_{var}_30yr_normal_4kmM2_{m:02d}_asc.asc'
base_nc_name = '{var}.{m:02d}.nc'
# the version of gdal to be used
gdal = '/opt/local/bin/gdal_translate'

months = np.arange(1, 13)

# Use gdal to translate the ascii data into netCDF format
# Use CDO to merge monthly data into 1 file
for var in variables:
    files = []
    for month in months:
        subprocess.call([gdal, '-of', 'netCDF', 
                         folder+base_fname.format(var=var, m=month), 
                         folder+base_nc_name.format(var=var, m=month)])
        files.append(folder+base_nc_name.format(var=var, m=month))
    subprocess.call(['cdo', 'merge', ' '.join(files), folder+'{var}.monthly.nc'.format(var=var)])
    for f in files:
        subprocess.call(['rm', f])

# Use CDO to remap PRISM data into the grid file used for this project
subprocess.call(['cdo', 'remapcon,{grid}'.format(grid=grid), 
                    folder+'ppt.monthly.nc', folder+'ppt.monthly.mask.nc'])

subprocess.call(['cdo', 'remapbil,{grid}'.format(grid=grid), 
                    folder+'tmean.monthly.nc', folder+ 'tmean.monthly.mask.nc'])