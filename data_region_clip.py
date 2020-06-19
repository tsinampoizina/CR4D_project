#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:24:44 2020

@author: sr0046
"""
import pdb
import geopandas
import rioxarray
import xarray
from shapely.geometry import mapping
from pathlib import Path
from tqdm import tqdm

import models
def shapefile(region):
    if region == 'region-cwest':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar3region-centrewest.shp'
    elif region == 'region-east':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar3region-east.shp'
    elif region == 'region-south':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar3region-ssw.shp'
    elif region == 'madagascar':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar.shp'
    
def input_file(folder, model, year):
    '''Get data from filename, slice using Madagascar coordinates'''
    return folder + model.name + '/' + model.version + '-region-' + str(year) + model.ext

def output_folder(folder, model, region):
    return folder + region + '/' + model.name + '/'

def output_filename(model, year, region):
    return model.version + '-' + region + '-' + str(year) + '.nc'


def clip_for_region(output_folder, input_file, output_file, region, year):
    ds_disk = xarray.open_dataset(input_file)
    ds_disk.rio.set_spatial_dims(x_dim=model.lon, y_dim=model.lat, inplace=True)
    ds_disk.rio.write_crs("epsg:4326", inplace=True)
    region_shape = geopandas.read_file(shapefile(region), crs="epsg:4326")    
    clipped = ds_disk.rio.clip(region_shape.geometry.apply(mapping), region_shape.crs, drop=False)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    clipped.to_netcdf(output_file)

model = models.chirps

def main():
    FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'
    for year in tqdm(model.year_range):
        #pdb.set_trace()
        region = 'madagascar'
        input_f = input_file(FOLDER,model,year)
        output_dir = output_folder(FOLDER,model,region)
        output_f = output_dir + output_filename(model,year,region)
        clip_for_region(output_dir, input_f, output_f, region, year)
    
main()