# Obtained historic climate normals from PRISM project
# http://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf
# http://www.prism.oregonstate.edu/normals/

import subprocess
import numpy as np

variables = ['ppt', 'tmean']
folder = '/Users/Chiao/google-drive/projects/Galvanize/fall-foliage-finder/data/normals/'
base_fname = 'PRISM_{var}_30yr_normal_4kmM2_all_asc/PRISM_{var}_30yr_normal_4kmM2_{m:02d}_asc.asc'
grid = '/Users/Chiao/google-drive/projects/Galvanize/fall-foliage-finder/data/mask.nc'
base_nc_name = '{var}.{m:02d}.nc'
gdal = '/opt/local/bin/gdal_translate'

months = np.arange(1, 13)

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

subprocess.call(['cdo', 'remapcon,{grid}'.format(grid=grid), 
                    folder+'ppt.monthly.nc', folder+'ppt.monthly.mask.nc'])

subprocess.call(['cdo', 'remapbil,{grid}'.format(grid=grid), 
                    folder+'tmean.monthly.nc', folder+ 'tmean.monthly.mask.nc'])