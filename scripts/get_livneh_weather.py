import numpy as np
import subprocess

base_url = 'ftp://ftp.hydro.washington.edu/pub/blivneh/CONUS/Meteorology.nc.v.1.2.1915.2011.bz2/'
base_fname = 'Meteorology_Livneh_CONUSExt_v.1.2_2013.{t}.nc.bz2'

f_path = '/Users/Chiao/google-drive/projects/Galvanize/fall-foliage-finder/data/weather/'
mask = '/Users/Chiao/google-drive/projects/Galvanize/fall-foliage-finder/data/mask.nc'

yrs = np.arange(2002, 2012)
months = np.arange(1, 13)
variables = ['Prec', 'Tmax', 'Tmin', 'Wind']

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