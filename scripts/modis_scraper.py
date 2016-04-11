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

CWD = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_PATH = CWD + '/../data/nvdi/hdf/'

class MODISScraper(object):
    '''
    Usage: 

    modis = modis_scraper.MODISScraper()
    modis.download_latlon(product='MOLT/MOD13A2.006', latlon=[-90, -60, 35, 50])

    '''
    def __init__(self):
        '''
        Initiates the MODISScraper class
        '''
        self.url = 'http://e4ftl01.cr.usgs.gov/{0}/'
        self.is_date = re.compile(r"[0-9]{4}\.[0-9]{2}\.[0-9]{2}")
        self.is_tile = re.compile(r"h[0-9]{2}v[0-9]{2}")
        self.f_path = None
        self.format = None
        self.data_list = []


    def download_latlon(self, product, tiled=False, latlon=None, format='.hdf', f_path=DOWNLOAD_PATH):
        '''
        INPUT:
            product -> string, name of MODIS product desired including the version
            #latlon -> list, bounding latitude and longitude coordinates 
            #          in the format of [lon_min    lon_max   lat_min   lat_max]
            f_path -> string; path to save scraped data
        '''
        self.f_path = f_path

        if tiled:
            tiles = set(self._parse_latlon(latlon))

        product_page = self.url.format(product)
        self.format = format
        print 'Downloading product:', product

        links = self._get_links(product_page)
        for link in links:
            if self.is_date.search(link):
                if tiled:
                    folder = f_path+link
                    if not os.path.exists(folder):
                        os.mkdir(folder)

                print 'Downloading date', link
                files = self._get_links(product_page+link)
                for f in files:
                    if tiled:
                        tile = self.is_tile.search(f).group()
                        if format in f and tile in tiles:
                            url = product_page+link+f
                            path = folder+f
                            self._save_to_file(url, path)
                    else:
                        if format in f:
                            url = product_page+link+f
                            path = f_path+link.replace('/', '.')+f.split('.')[-1]
                            self._save_to_file(url, path)


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
        
        return ('h{0:02d}v{1:02d}'.format(int(h), int(v)) for (v, h) in zip(v_list, h_list))

