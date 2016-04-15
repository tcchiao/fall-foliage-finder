'''
A MODIS Data Scraper module
~~~~~~~~~~~~~~~~~~~~
Author: Cindy Chiao
Last Edit Date: 04/09/2016

The module included is intended to scrape MODIS satellite data from USGS Data Pool 
website. This module is a byproduct of my Galvanize Data Science Capstone project.

This is only a first iteration, please feel free to contact me if you see 
something wrong.  Error handling has not been included yet.
'''

import requests
import os
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import subprocess


class MODIS_Scraper(object):
    '''
    Usage: 

    modis = modis_scraper.MODISScraper()
    modis.download_latlon(product='MOLT/MOD13A2.006', latlon=[-90, -60, 35, 50])

    '''
    def __init__(self, tiled=False, latlon=None, convert=True, mask=None):
        '''
        Initiates the MODISScraper class
        '''
        self.url = 'http://e4ftl01.cr.usgs.gov/{0}/'
        self.is_date = re.compile(r"[0-9]{4}\.[0-9]{2}\.[0-9]{2}")
        self.is_tile = re.compile(r"h[0-9]{2}v[0-9]{2}")
        self.convert = convert
        self.mask = mask
        self.product = None
        self.pages = []
        if tiled:
            self.tiles = self._parse_latlon(latlon)
        else:
            self.tiles = None

    def get_dates(self, product):
        '''
        INPUT:
            product -> string, name of MODIS product desired including the version
        '''
        self.product = product
        pages = self._get_links(self.url.format(self.product))
        for p in pages:
            if self.is_date.search(p):
                self.pages.append(p)
        return self.pages
        #print 'There are {0} available dates for {1}'.format(len(self.pages), product)

    def get_data(self, f_path, pages=None, format='hdf'):
        ''' 
        INPUT:
            tiled -> boolean, True if the data is tiled 
            latlon -> list, bounding latitude and longitude coordinates 
                      in the format of [lon_min    lon_max   lat_min   lat_max]
            f_path -> string; path to save scraped data
        '''
        if pages == None:
            pages = self.pages
        for page in pages:   
            print 'Downloading for', page         
            file_names = self._download(page, format, f_path)
            if self.convert:
                nc_files = self._convert_hdf_to_nc(file_names)
            if self.mask is not None:
                self._extract_area(nc_files)

    def _download(self, page, format, f_path):
        links = self._get_links(self.url.format(self.product)+page)
        file_names = []
        for l in links:
            if l[-len(format):] == format:
                url = self.url.format(self.product) + page + l
                path = f_path + page.replace('/', '.') + format
                self._save_to_file(url, path)
                file_names.append(path)
        return file_names

    def _convert_hdf_to_nc(self, file_names):
        
        nc_files = []
        for file_name in file_names:
            os.system('/opt/local/bin/gdal_translate -of netCDF "HDF4_EOS:EOS_GRID:{0}:MODIS_Grid_16Day_VI_CMG:CMG 0.05 Deg 16 days NDVI" {1}'.format(file_name, file_name.replace('hdf', 'nc')))
        
            nc_files.append(file_name.replace('hdf', 'nc'))
            subprocess.call(['rm', file_name])
        return nc_files

    def _extract_area(self, nc_files):
        for nc_file in nc_files:
            subprocess.call(['cdo', 'remapdis,{0}'.format(self.mask), nc_file, nc_file.replace('.nc', '.mask.nc')])
            subprocess.call(['rm', nc_file])

    def _get_links(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.findAll('a', href=True)

        return [link['href'] for link in links]

    def _save_to_file(self, url, path):
        r = requests.get(url)
        with open(path, 'w') as f:
            f.write(r.content)

    def _parse_latlon(self, latlon, grid_src='http://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt'):
        '''
        Takes bounding latlon list and returns the corresponding tile number for MODLAND Tiles
        Reference: http://modis-land.gsfc.nasa.gov/MODLAND_grid.html

        INPUT:
            latlon -> list, bounding latitude and longitude coordinates 
                      in the format of [lon_min    lon_max   lat_min   lat_max]
        OUTPUT:
            list; strings of MODLAND tile number formatted as 'h##v##'
        '''
        lon_min, lon_max, lat_min, lat_max = latlon

        r = requests.get(grid_src)
        table = [x.split() for x in r.content.split('\r') if len(x.split()) == 6]
        table = pd.DataFrame(table[1:], columns=table[0], dtype=float)

        conditions = np.vstack((table.lon_max >= lon_min, table.lon_min <= lon_max,
                                table.lat_max >= lat_min, table.lat_min <= lat_max))

        table['in_bounds'] = conditions.all(axis=0)
        v_list = table[table.in_bounds].iv.values
        h_list = table[table.in_bounds].ih.values

        return ('h{0:02d}v{1:02d}'.format(int(h), int(v))
                for (v, h) in zip(v_list, h_list))
