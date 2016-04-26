"""
This script is used to download gridded metereological (weather) data from UW ftp website. 
~~~~~~~~~~~~~~~~~~~~
Author: Cindy Chiao
Last Edit Date: 04/09/2016

Reference: http://www.hydro.washington.edu/SurfaceWaterGroup/Data/livneh/livneh.et.al.2013.page.html
Citation: Livneh, B., E. A. Rosenberg, C. Lin, B. Nijssen, V. Mishra, K. M. Andreadis, E. P. Maurer, and 
          D. P. Lettenmaier, 2013: A Long-Term Hydrologically Based Dataset of Land Surface Fluxes and 
          States for the Conterminous United States: Update and Extensions. J . Climate, 26.
Download site: ftp://ftp.hydro.washington.edu/pub/blivneh/CONUS/Meteorology.nc.v.1.2.1915.2011.bz2/VERSION_ID
Version: Version 1.2
Date: March, 2014
Contact: ben.livneh@colorado.edu
"""
import numpy as np
import subprocess

# Set local variables 
f_path = '../data/weather/' # path to save downloaded file
mask = '../data/mask.nc'# mask to use for remapping

# Website path variables
base_url = 'ftp://ftp.hydro.washington.edu/pub/blivneh/CONUS/Meteorology.nc.v.1.2.1915.2011.bz2/'
base_fname = 'Meteorology_Livneh_CONUSExt_v.1.2_2013.{t}.nc.bz2'

# Data is available by month at the time of writing
# Specify the years and months to download
yrs = np.arange(2004, 2012)
months = np.arange(1, 13)
# Specify the variables to unpack
variables = ['Prec', 'Tmax', 'Tmin', 'Wind']

# Downloading and processing files
for yr in yrs:
    for month in months:
        time = '{yr}{month:02d}'.format(yr=yr, month=month)
        fname = f_path + time + '{mod}{ext}'
        print 'Downloading for', time
        # Obtaining compressed file from UW ftp server
        subprocess.call(['wget', '-O', fname.format(mod='', ext='.nc.bz2'), base_url + base_fname.format(t=time)])
        # Unzip the file
        subprocess.call(['bunzip2', fname.format(mod='', ext='.nc.bz2')])
        
        print 'Processing for', time
        # Convert float data into integer, change data type into short to reduce file size (by half!)
        subprocess.call(['cdo', 'int', fname.format(mod='', ext='.nc'), fname.format(mod='.int', ext='.nc')])
        subprocess.call(['ncap2', '-s', 'Prec=short(Prec);Tmax=short(Tmax);Tmin=short(Tmin);Wind=short(Wind)',\
                        fname.format(mod='.int', ext='.nc'), fname.format(mod='.short', ext='.nc')])
        
        # Subsetting out each variable since different remap functions are being used
        for var in variables:
            subprocess.call(['cdo', 'selname,{var}'.format(var=var), \
                            fname.format(mod='.short', ext='.nc'), fname.format(mod='.short.'+str(var), ext='.nc')])
        subprocess.call(['cdo', 'remapcon,{grid}'.format(grid=mask), \
                        fname.format(mod='.short.Prec', ext='.nc'), fname.format(mod='.Prec', ext='.nc')])
        subprocess.call(['cdo', 'remapbil,{grid}'.format(grid=mask), \
                        fname.format(mod='.short.Tmax', ext='.nc'), fname.format(mod='.Tmax', ext='.nc')])
        subprocess.call(['cdo', 'remapbil,{grid}'.format(grid=mask), \
                        fname.format(mod='.short.Tmin', ext='.nc'), fname.format(mod='.Tmin', ext='.nc')])
        subprocess.call(['cdo', 'remapbil,{grid}'.format(grid=mask), \
                        fname.format(mod='.short.Wind', ext='.nc'), fname.format(mod='.Wind', ext='.nc')])
        
        # Clean up the directories
        subprocess.call(['rm', fname.format(mod='', ext='.nc'), fname.format(mod='.int', ext='.nc'), \
                        fname.format(mod='.short', ext='.nc')])
        for var in variables:
            subprocess.call(['rm', fname.format(mod='.short.'+str(var), ext='.nc')])
    
    # Merging files download by time, produces 1 file per variable
    print 'Merging year', yr
    for var in variables:
        file_list = [f_path+'{yr}{month:02d}.{var}.nc'.format(yr=yr, month=month, var=var) for month in months]
        subprocess.call(['cdo', 'mergetime', ' '.join(file_list), f_path+'{yr}.{var}.nc'.format(yr=yr, var=var)])
        for f in file_list:
            subprocess.call(['rm', f])