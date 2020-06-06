#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:07:39 2020

@author: sr0046
"""

import xarray as xr
import numpy as np
import time

from collections import namedtuple

from matplotlib import pyplot as plt

from cartopy.feature import ShapelyFeature, OCEAN, LAKES
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import Reader as ShapeReader, natural_earth

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil


def get_by_file(filex):
    '''Get data from filename, slice using Madagascar coordinates'''
    ds = xr.open_dataset(filex)
    # ds = ds.sel(longitude=slice(42.125,54.125),latitude=slice(-26.125,-11.125))
    return ds
def countour_plot(dat, contour_levels, title):
    
    # Download the Natural Earth shapefile for country boundaries at 10m resolution
    shapefile = natural_earth(category='cultural',
                              resolution='10m',
                              name='admin_0_countries')
    
    # Sort the geometries in the shapefile into Madagascar or other
    country_geos = []
    other_land_geos = []
    for record in ShapeReader(shapefile).records():
        if record.attributes['ADMIN'] in ['Madagascar']:
            country_geos.append(record.geometry)
        else:
            other_land_geos.append(record.geometry)
    
    # Define map projection to allow Cartopy to transform ``lat`` and ``lon`` values accurately into points on the
    # matplotlib plot canvas.
    projection = PlateCarree()
    
    # Define a Cartopy Feature for the country borders and the land mask (i.e.,
    # all other land) from the shapefile geometries, so they can be easily plotted
    countries = ShapelyFeature(country_geos,
                               crs=projection,
                               facecolor='none',
                               edgecolor='black',
                               lw=1.5)
    land_mask = ShapelyFeature(other_land_geos,
                               crs=projection,
                               facecolor='white',
                               edgecolor='none')
    
    # Download the Natural Earth shapefile for the states/provinces at 10m resolution
    shapefile = natural_earth(category='cultural',
                              resolution='10m',
                              name='admin_1_states_provinces')
    
    # Extract the Madagascar region borders
    province_geos = [record.geometry for record in ShapeReader(shapefile).records()
                     if record.attributes['admin'] == 'Madagascar']
    
    # Define a Cartopy Feature for the province borders, so they can be easily plotted
    provinces = ShapelyFeature(province_geos,
                               crs=projection,
                               facecolor='none',
                               edgecolor='black',
                               lw=0.25)
    
    
    # Generate figure (set its size (width, height) in inches) and axes using Cartopy
    fig = plt.figure(figsize=(12,15))
    ax = plt.axes(projection=projection)
    
    ax.set_extent([42, 52, -26, -11], crs=projection)
    
    # Define the contour levels
    clevs = contour_levels
    
    
    # Import an NCL colormap, truncating it by using geocat.viz.util convenience function
    newcmp = gvutil.truncate_colormap(gvcmaps.precip_11lev, minval=0, maxval=.8, n=len(clevs))
    
    # Draw the temperature contour plot with the subselected colormap
    # (Place the zorder of the contour plot at the lowest level)
    cf = ax.contourf(lon, lat, dat, levels=clevs, cmap=newcmp, zorder=1)
    
    # Draw horizontal color bar
    cax = plt.axes((0.14, 0.08, 0.74, 0.02))
    cbar = plt.colorbar(cf, ax=ax, cax=cax, ticks=clevs[1:-1], drawedges=True, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    
    # Add the land mask feature on top of the contour plot (higher zorder)
    ax.add_feature(land_mask, zorder=2)
    
    # Add the OCEAN and LAKES features on top of the contour plot
    ax.add_feature(OCEAN.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
    ax.add_feature(LAKES.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
    
    # Add the country and province features (which are transparent) on top
    ax.add_feature(countries, zorder=3)
    ax.add_feature(provinces, zorder=3)
    
    # Use geocat.viz.util convenience function to set axes tick values
    gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
    
    # Use geocat.viz.util convenience function to make plots look like NCL plots by using latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)
    
    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, x_minor_per_major=4, y_minor_per_major=5, labelsize=18)
    
    # Use geocat.viz.util convenience function to add main title as well as titles to left and right of the plot axes.
    gvutil.set_titles_and_labels(ax, lefttitle=title, lefttitlefontsize=20)
    

def month_start_end(year):
    month_start_end_list = [
    (0,31), (31,59), (59,90), (90,120), (120,151), (151,181), (181,212),
    (212,243), (243,273), (273,304), (304,334), (334,365) ]
    if year in [1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020]:
        month_start_end_list = [
            (0,31), (31,60), (60,91), (91,121), (121,152), (152,182),
            (182,213), (213,244), (244,274), (274,305), (305,335), (335,366)]
    return month_start_end_list
def compute_freq_year_season(year, season):
    file = file_name(year)
    ds_disk = get_by_file(file)
    precip = ds_disk[model.precip]
    freqs_month = []
    for start, end in month_start_end(year):
        prec = precip.sel(time=slice(start,end))
        # Make 1 if >=threshold, otherwise make 0
        freqs = prec.where(prec<THRESHOLD,1)
        freqs = freqs.where(prec>=THRESHOLD-0.000001,0) # you never know with python
        freq = freqs.sum('time')
        freqs_month.append(freq)
    freq = sum([freqs_month[i] for i in season]) # may-oct
    return freq
def compute_climatology_season(years, season):
    climatology = xr.zeros_like(example_precip)
    if season == 'DJFM':
        for year in years:
            climatology += compute_freq_year_season(year-1, [11])
            climatology += compute_freq_year_season(year, [0,1,2])
    else:
        for year in years:
            climatology += compute_freq_year_season(year, season)
    return climatology

def file_name(year):
    return INPUT_FOLDER + model.name + '/' + model.version + '-region-' + str(year) + '.nc4'
 
# Take one day to get the corect data shape, to initialise
# example_precip is used in compute_climatology
# lat, lon, are used in contour_plot
INPUT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'
EXAMPLE_YEAR = '2000'
EXAMPLE_DATE = 360

Model = namedtuple('Model', 'name version lat lon precip')
trmm = Model('trmm','TRMM_3B42','lat','lon','precipitation')
chirps = Model('chirps','chrips-v2','latitude','longitude','precip') 
model = trmm

file = file_name(EXAMPLE_YEAR)
ds_disk = get_by_file(file)
example_precip = ds_disk[model.precip]
example_precip = example_precip.sel(time=EXAMPLE_DATE)
lat = ds_disk[model.lat]
lon = ds_disk[model.lon]
precip = ds_disk[model.precip]

# CONSTANTS
THRESHOLD = 30
YEARS = range(1999,2009)
#YEARS = [2018]
SEASON = [0]         # Use 0,1,..., 11 for a month. Use [0,1,2] for jan-mar
SEASON_NAME = 'jan'
#SEASON = range(10,11)                        # BUT USE 'djfm' for djfm over one season
MAX_VALUE = 30*len(SEASON)
MAX_VALUE = 10*len(SEASON)

CONTOUR_LEVELS = np.arange(0, MAX_VALUE, MAX_VALUE/20, dtype=float)
title = model.name 
plot_filename = 'climatology-wet-days-frequency-'+SEASON_NAME+'-'+str(YEARS[0])+'-'+str(YEARS[-1])+'-'+title
PROJECT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada'
PLOT_FOLDER = PROJECT_FOLDER+'/plot-images/climatology-'+str(THRESHOLD)+'mm-wdf-jan-1999-2008/'

t0 = time.time()
  
climatology_freq = compute_climatology_season(YEARS, SEASON)
climatology_freq = climatology_freq / len(YEARS)
if model.name == 'trmm':
    climatology_freq = climatology_freq.transpose()
countour_plot(climatology_freq, CONTOUR_LEVELS, title)
plt.savefig(PLOT_FOLDER+plot_filename+'.png')
t1 = time.time()
print(t1-t0)
