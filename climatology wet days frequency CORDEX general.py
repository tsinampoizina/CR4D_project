#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:07:39 2020

@author: sr0046
"""

import xarray as xr
import numpy as np
import time

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
def countour_plot(model, dat, contour_levels, title):
    example_year = '2000' 
    file = file_name(model, example_year)
    ds_disk = get_by_file(file)
    lat = ds_disk[LAT]
    lon = ds_disk[LON]  
    
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
    (str(year)+'-01-01T00:00:00.000000000',str(year)+'-01-31T00:00:00.000000000'),
    (str(year)+'-02-01T00:00:00.000000000',str(year)+'-02-28T00:00:00.000000000'),
    (str(year)+'-03-01T00:00:00.000000000',str(year)+'-03-31T00:00:00.000000000'),
    (str(year)+'-04-01T00:00:00.000000000',str(year)+'-04-30T00:00:00.000000000'),
    (str(year)+'-05-01T00:00:00.000000000',str(year)+'-05-31T00:00:00.000000000'),
    (str(year)+'-06-01T00:00:00.000000000',str(year)+'-06-30T00:00:00.000000000'),
    (str(year)+'-07-01T00:00:00.000000000',str(year)+'-07-31T00:00:00.000000000'),
    (str(year)+'-08-01T00:00:00.000000000',str(year)+'-08-31T00:00:00.000000000'),
    (str(year)+'-09-01T00:00:00.000000000',str(year)+'-09-30T00:00:00.000000000'),
    (str(year)+'-10-01T00:00:00.000000000',str(year)+'-10-31T00:00:00.000000000'),
    (str(year)+'-11-01T00:00:00.000000000',str(year)+'-11-30T00:00:00.000000000'),
    (str(year)+'-12-01T00:00:00.000000000',str(year)+'-12-31T00:00:00.000000000')]
    if year in [1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020]:
        month_start_end_list[1] = (str(year)+'-02-01T00:00:00.000000000',str(year)+'-02-29T00:00:00.000000000')
    return month_start_end_list
def compute_freq_year_season(model, year, season):
    file = file_name(model, year)
    ds_disk = get_by_file(file)
    precip = ds_disk["pr"]
    #dates_pan = dates.to_dataframe()
    #example_precip_pan = example_precip.to_dataframe()
    precip = precip*86400
    freqs_month = []
    for start, end in month_start_end(year):
        prec = precip.sel(time=slice(start,end))
        # Make 1 if >=threshold, otherwise make 0
        freqs = prec.where(prec<THRESHOLD,1)
        freqs = freqs.where(prec>=THRESHOLD-0.001,0)
        freq = freqs.sum('time')
        #freq = freq*(100/year_length)
        freqs_month.append(freq)
    #freq = sum([freqs_month[i] for i in range(0,12)]) # all year
    #freq = freqs_month[11] + freqs_month[0] + freqs_month[1] + freqs_month[2] # djfm 
    #freq = sum([freqs_month[i] for i in range(3,11)]) # apr-nov
    #freq = sum([freqs_month[i] for i in [0,1,2,3,10,11]]) # NDJFMA
    freq = sum([freqs_month[i] for i in season]) # may-oct
    return freq
def compute_climatology_season(model,years, season):
    example_year = '2000'
    example_date = '2000-01-04 12:00:00'    
    file = file_name(model, example_year)
    ds_disk = get_by_file(file)
    example_precip = ds_disk["pr"]
    example_precip = example_precip.sel(time=example_date)
    climatology = xr.zeros_like(example_precip)
    if season == 'DJFM':
        for year in years:
            climatology += compute_freq_year_season(model,year-1, [11])
            climatology += compute_freq_year_season(model,year, [0,1,2])
    else:
        for year in years:
            climatology += compute_freq_year_season(year, season)
    return climatology

def file_name(model, year):
    file = 'pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_'+ model +'_day_'
    folder = '/media/sr0046/WD-exFAT-50/CORDEX/evaluation/'+ model +'/ECMWF-ERAINT_r1i1p1/day/native/'
    return folder+file+str(year)+'.nc'

def generate_plot(model):
        title = model
        plot_filename = 'climatology-wet-days-frequency-'+SEASON+'-'+str(YEARS[0])+'-'+str(YEARS[-1])+'-'+title
         
        t0 = time.time()
          
        climatology_freq = compute_climatology_season(model, YEARS, SEASON)
        climatology_freq = climatology_freq / len(YEARS)
        countour_plot(model,climatology_freq, CONTOUR_LEVELS, title)
        plt.savefig(PLOT_FOLDER+plot_filename+'.png')
        plt.show()
        
        t1 = time.time()
        print(model,'time', t1-t0)
        
# CONSTANTS
THRESHOLD = 20
YEARS = range(1999,2009)
#YEARS = [2018]
SEASON = 'DJFM'         # Use 0,1,..., 11 for a month. Use [0,1,2] for jan-mar
#SEASON = range(10,11)                        # BUT USE 'djfm' for djfm over one season
MAX_VALUE = 30*len(SEASON)
MAX_VALUE = 10*len(SEASON)
CONTOUR_LEVELS = np.arange(0, MAX_VALUE, MAX_VALUE/20, dtype=float)
PROJECT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada'
PLOT_FOLDER = PROJECT_FOLDER+'/plot-images/climatology-'+str(THRESHOLD)+'mm-wdf-DJFM-1999-2008/'
MODELS = [
'CLMcom-CCLM4-8-17_v1',
'DMI-HIRHAM5_v2',
'KNMI-RACMO22T_v1',
'MOHC-HadGEM3-RA_v1', 
'MOHC-HadRM3P_v1',
'MPI-CSC-REMO2009_v1', 
'SMHI-RCA4_v1',
'UQAM-CRCM5_v1'
]


for model in MODELS:
    if model in ["MOHC-HadGEM3-RA_v1", 'MOHC-HadRM3P_v1']:
        LAT ='lat' # lat for MOHC-Had..., rlat for others
        LON = 'lon' # lon for MOHC-Had..., rlon for others
    else:
        LAT = 'rlat'
        LON = 'rlon'
    generate_plot(model)
