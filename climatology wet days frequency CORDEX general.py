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
import pdb

from matplotlib import pyplot as plt
from cartopy.feature import ShapelyFeature, OCEAN, LAKES
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import Reader as ShapeReader, natural_earth

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

def add_lon_ticklabels(ax, zero_direction_label=False, dateline_direction_label=False):
    """
    Utility function to make plots look like NCL plots by using latitude, longitude tick labels
    Args:
        ax (:class:`matplotlib.axes._subplots.AxesSubplot` or :class:`cartopy.mpl.geoaxes.GeoAxesSubplot`):
            Current axes to the current figure
        zero_direction_label (:class:`bool`):
            Set True to get 0 E / O W or False to get 0 only.
        dateline_direction_label (:class:`bool`):
            Set True to get 180 E / 180 W or False to get 180 only.
    """
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    lon_formatter = LongitudeFormatter(zero_direction_label=zero_direction_label,
                                       dateline_direction_label=dateline_direction_label)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

def month_start_end(year):
    month_lengths = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    if year in [1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000,
                    2004, 2008, 2012, 2016, 2020]:
        month_lengths[2] = 29
    month_start_end_list = [(0,1)]
    for mo in range(1,13):
        s,e = month_start_end_list[-1]
        month_start_end_list.append((e, e + month_lengths[mo]-1))
    return month_start_end_list

def length(year, lmonths):
    start_ends = [month_start_end(year)[mo] for mo in lmonths]
    lengths = [y-x+1 for x,y in start_ends]
    return sum(lengths)
def get_by_file(filex):
    '''Get data from filename, slice using Madagascar coordinates'''
    ds = xr.open_dataset(filex)
    # ds = ds.sel(longitude=slice(42.125,54.125),latitude=slice(-26.125,-11.125))
    return ds
def countour_plot(model, dat, contour_levels, title, xyz, abc):
    if model in ['MOHC-HadGEM3-RA_v1','MOHC-HadRM3P_v1']:
        lat = dat.lat
        lon = dat.lon
    else:
        lat = dat.rlat
        lon = dat.rlon
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



    ax = plt.subplot(xyz, projection=projection)
    #ax = plt.axes(projection=projection)

    ax.set_extent([42, 52, -26, -11], crs=projection)

    # Define the contour levels
    clevs = contour_levels


    # Import an NCL colormap, truncating it by using geocat.viz.util convenience function
    newcmp = gvutil.truncate_colormap(gvcmaps.precip3_16lev, minval=charac.min_val, maxval=charac.max_val, n=len(clevs))

    # Draw the temperature contour plot with the subselected colormap
    # (Place the zorder of the contour plot at the lowest level)
    cf = ax.contourf(lon, lat, dat, levels=clevs, cmap=newcmp, zorder=1)

    # Draw horizontal color bar
    cax = plt.axes((0.14, 0.08, 0.74, 0.02))
    cbar = plt.colorbar(cf, ax=ax, cax=cax, ticks=clevs[1:-1:2], drawedges=True, orientation='horizontal', pad=3)
    cbar.ax.tick_params(labelsize=ticklabelsize)

    # Add the land mask feature on top of the contour plot (higher zorder)
    ax.add_feature(land_mask, zorder=2)

    # Add the OCEAN and LAKES features on top of the contour plot
    ax.add_feature(OCEAN.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
    ax.add_feature(LAKES.with_scale('50m'), edgecolor='black', lw=1, zorder=2)

    # Add the country and province features (which are transparent) on top
    ax.add_feature(countries, zorder=3)
    ax.add_feature(provinces, zorder=3)

    # Use geocat.viz.util convenience function to set axes tick values
    z = xyz%10
    if z not in [1,6]:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        ax.set_yticklabels(['','',''])
    else:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        # Use geocat.viz.util convenience function to make plots look like NCL plots by using latitude, longitude tick labels
        add_lon_ticklabels(ax)

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, x_minor_per_major=4, y_minor_per_major=5, labelsize=ticklabelsize)

    # Use geocat.viz.util convenience function to add main title as well as titles to left and right of the plot axes.
    gvutil.set_titles_and_labels(ax, lefttitle=title, lefttitlefontsize=titlesize)


def compute_freq_year_season(model, year, months_list):
    example_date = str(year)+'-01-04 12:00:00'
    file = file_name(model, year)
    ds_disk = get_by_file(file)
    precip = ds_disk["pr"]
    #dates_pan = dates.to_dataframe()
    #example_precip_pan = example_precip.to_dataframe()
    precip = precip*86400  # UNIT Conversion
    freq_sum = xr.zeros_like(precip.sel(time=example_date))
    for i in months_list:
        mo = str(year) + '-' + str(i)
        prec = precip.sel(time=mo)
        if charac == 'wdfreq':
            freqs = prec.where(prec < THRESHOLD, 1)
            freqs = freqs.where(prec >= THRESHOLD-0.000001, 0)  # to be sure
        else:
            freqs = prec
        freq = freqs.sum('time')
        #freq = freq*(100/year_length)
        freq_sum += freq
    #freq = sum([freqs_month[i] for i in range(0,12)]) # all year
    #freq = freqs_month[11] + freqs_month[0] + freqs_month[1] + freqs_month[2] # djfm
    #freq = sum([freqs_month[i] for i in range(3,11)]) # apr-nov
    #freq = sum([freqs_month[i] for i in [0,1,2,3,10,11]]) # NDJFMA
    return freq_sum
def compute_climatology_season(model,years, season):
    example_year = '2000'
    example_date = '2000-01-04 12:00:00'
    file = file_name(model, example_year)
    ds_disk = get_by_file(file)
    example_precip = ds_disk["pr"]
    example_precip = example_precip.sel(time=example_date)
    climatology = xr.zeros_like(example_precip)
    if season.name == 'DJFM':
        for year in years:
            climatology += compute_freq_year_season(model,year-1, [12])
            climatology += compute_freq_year_season(model,year, [1,2,3])
        if charac == average_daily:
            climatology = climatology/length(year, [1,2,3,12])
    else:
        for year in years:
            climatology += compute_freq_year_season(model,year, season.months)
        if charac == average_daily:
            climatology = climatology/length(year, season.months)
    return climatology

def file_name(model, year):
    file = 'pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_'+ model +'_day_'
    folder = '/media/sr0046/WD-exFAT-50/CORDEX/evaluation/'+ model +'/ECMWF-ERAINT_r1i1p1/day/native/'
    return folder+file+str(year)+'.nc'

def generate_plot(model, dat, xyz, abc):
    title = abc+') '+model
    t0 = time.time()
    countour_plot(model, dat, charac.contour, title, xyz, abc)

    t1 = time.time()
    print(model,'time', t1-t0)

# CONSTANTS
THRESHOLD = 30
YEARS = range(1999,2009)
#YEARS = [2018]
Season = namedtuple('Season', 'name months')
djfm = Season('DJFM', [1,2,3,12])
all_year = Season('all-year', range(1, 13))
amjjaso = Season('Apr-Oct', [4,5,6,7,8,9,10])
seasons = [djfm, all_year, amjjaso]
#seasons = [all_year]

for season in seasons:
    Characteristic = namedtuple('Characteristic',
                              'name contour unit threshold min_val max_val')
    if season == all_year:
      wd_freq1 = Characteristic('WDF', np.arange(0, 280, 20, dtype=float), 'days', 1, 0,1)
      wd_freq30 = Characteristic('WDF', np.arange(0, 40, 2, dtype=float), 'days', 30,0,1)
      total_precip = Characteristic('TOTAL-RAINFALL',
                                    np.arange(200,4800, 200, dtype=float), 'mm',
                                    '',0,1.5)
      average_daily = Characteristic('AVERAGE-DAILY-RAINFALL',
                                    np.arange(0, 20, 1, dtype=float), 'mm',
                                    '',0,1)
    if season == djfm:
      wd_freq1 = Characteristic('WDF', np.arange(0, 120, 10, dtype=float), 'days', 1,0,1)
      wd_freq30 = Characteristic('WDF', np.arange(0, 40, 2, dtype=float), 'days', 30,0,1)
      total_precip = Characteristic('TOTAL-RAINFALL',
                                    np.arange(200,4800, 200, dtype=float), 'mm',
                                    '',0,1.5)
      average_daily = Characteristic('AVERAGE-DAILY-RAINFALL',
                                    np.arange(0, 20, 1, dtype=float), 'mm',
                                    '',0,1)
    if season == amjjaso:
      wd_freq1 = Characteristic('WDF', np.arange(0, 180, 10, dtype=float), 'days', 1,0,1)
      wd_freq30 = Characteristic('WDF', np.arange(0, 40, 2, dtype=float), 'days', 30,0,1)
      total_precip = Characteristic('TOTAL-RAINFALL',
                                    np.arange(200,4800, 200, dtype=float), 'mm',
                                    '',0,1.5)
      average_daily = Characteristic('AVERAGE-DAILY-RAINFALL',
                                    np.arange(0, 20, 1, dtype=float), 'mm',
                                    '',0,1)
    charac = average_daily
    charac = total_precip
    #charac = wd_freq1
    #charac = wd_freq30

    PROJECT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada'
    PLOT_FOLDER = PROJECT_FOLDER + '/plot-images/climatology-'  + charac.name + str(charac.threshold) + '/'


    MODELS = [
        'CLMcom-CCLM4-8-17_v1',
        'DMI-HIRHAM5_v2',
        'KNMI-RACMO22T_v1',
        'MOHC-HadGEM3-RA_v1',
        'MOHC-HadRM3P_v1',
        'MPI-CSC-REMO2009_v1',
        'SMHI-RCA4_v1',
        'UQAM-CRCM5_v1',
        'Ensemble Mean'
        ]
# Generate figure (set its size (width, height) in inches) and axes using Cartopy
    fig = plt.figure(figsize=(18,12))
    ticklabelsize = 18
    titlesize = 22

    XYZ = [251,252,253,254,259,256,257,258,255]
    abc = ['a','b','c','d','i','f','g','h','e']
    for model, xyz, voy in zip(MODELS,XYZ,abc):
        if model == 'MOHC-HadGEM3-RA_v1':
            YEARS = range(1999,2008)
        else:
            YEARS = range(1999,2009)
        LAT ='rlat' # lat for MOHC-Had..., rlat for others
        LON = 'rlon' # lon for MOHC-Had..., rlon for others
        if model in ['MOHC-HadGEM3-RA_v1','MOHC-HadRM3P_v1']:
            LAT = 'lat'
            LON = 'lon'
        if model != MODELS[-1]:
            climatology_freq = compute_climatology_season(model, YEARS, season)
            climatology_freq = climatology_freq / len(YEARS)
            generate_plot(model, climatology_freq, xyz, voy)
        if model == MODELS[0]:
            initial_climato = climatology_freq
            ensemble_climato = climatology_freq.assign_coords(rlon=((initial_climato.rlon*100)//1000)/100)
            ensemble_climato = climatology_freq.assign_coords(rlat=((initial_climato.rlat*100)//1000)/100)
        else:
            if model in ['MOHC-HadGEM3-RA_v1','MOHC-HadRM3P_v1']:
                climatology_freq = climatology_freq.rename({'lat':'rlat'})
                climatology_freq = climatology_freq.rename({'lon':'rlon'})
            ensemble_climato = ensemble_climato.assign_coords(rlon=(initial_climato.rlon))
            climatology_freq = climatology_freq.assign_coords(rlon=(initial_climato.rlon))
            ensemble_climato = ensemble_climato.assign_coords(rlat=(initial_climato.rlat))
            climatology_freq = climatology_freq.assign_coords(rlat=(initial_climato.rlat))
            ensemble_climato += climatology_freq
        if model == MODELS[-1]:
            generate_plot(model, ensemble_climato/len(MODELS)-1, xyz, voy)
    plot_filename = 'climatology-CORDEX'+ charac.name + str(charac.threshold) + \
      '-' + season.name + '-' + str(YEARS[0]) + '-' + str(YEARS[-1])
    plt.savefig(PLOT_FOLDER + plot_filename + '.png')
    plt.show()
