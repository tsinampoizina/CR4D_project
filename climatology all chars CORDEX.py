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


def add_lon_ticklabels(ax, zero_direction_label=False,
                       dateline_direction_label=False):
    """
    Utility function to make plots look like NCL plots by using latitude,
    longitude tick labels
    Args:
        ax (:class:`matplotlib.axes._subplots.AxesSubplot` or
        :class:`cartopy.mpl.geoaxes.GeoAxesSubplot`):
            Current axes to the current figure
        zero_direction_label (:class:`bool`):
            Set True to get 0 E / O W or False to get 0 only.
        dateline_direction_label (:class:`bool`):
            Set True to get 180 E / 180 W or False to get 180 only.
    """
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    lon_formatter = LongitudeFormatter(zero_direction_label=zero_direction_label,
                                       dateline_direction_label=dateline_direction_label)
    ax.xaxis.set_major_formatter(lon_formatter)


def month_start_end(year):
    month_lengths = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year in range(1964, 2021, 4):
        month_lengths[2] = 29
    month_start_end_list = [(0, 1)]
    for mo in range(1, 13):
        s, e = month_start_end_list[-1]
        month_start_end_list.append((e, e + month_lengths[mo]-1))
    return month_start_end_list


def length(year, lmonths):
    start_ends = [month_start_end(year)[mo] for mo in lmonths]
    lengths = [y-x+1 for x, y in start_ends]
    return sum(lengths)


def get_by_file(filex):
    '''Get data from filename, slice using Madagascar coordinates'''
    ds = xr.open_dataset(filex)
    # ds = ds.sel(longitude=slice(42.125,54.125),latitude=slice(-26.125,-11.125))
    return ds


def countour_plot(model, dat, contour_levels, title, xyz, abc):
    if model in ['MOHC-HadGEM3-RA_v1', 'MOHC-HadRM3P_v1']:
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

    # Define map projection to allow Cartopy to transform ``lat`` and `
    # lon`` values accurately into points on the
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
    # ax = plt.axes(projection=projection)

    ax.set_extent([42, 52, -26, -11], crs=projection)

    # Define the contour levels
    clevs = contour_levels

    # Import an NCL colormap, truncating it by using geocat.viz.util
    newcmp = gvutil.truncate_colormap(gvcmaps.precip3_16lev,
                                      minval=charac.min_val,
                                      maxval=charac.max_val,
                                      n=len(clevs))

    # Draw the temperature contour plot with the subselected colormap
    # (Place the zorder of the contour plot at the lowest level)
    cf = ax.contourf(lon, lat, dat, levels=clevs, cmap=newcmp, zorder=1)

    # Draw horizontal color bar
    cax = plt.axes((0.14, 0.08, 0.74, 0.02))
    cbar = plt.colorbar(cf, ax=ax, cax=cax, ticks=clevs[1:-1:2],
                        drawedges=True, orientation='horizontal', pad=3)
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
    z = xyz % 10
    if z not in [1,6]:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50],
                                         yticks=[-25, -20, -15])
        ax.set_yticklabels(['', '', ''])
        add_lon_ticklabels(ax)
    else:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50],
                                         yticks=[-25, -20, -15])
        # Use geocat.viz.util convenience function to make plots look
        # like NCL plots by using latitude, longitude tick labels
        gvutil.add_lat_lon_ticklabels(ax)

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, x_minor_per_major=4, y_minor_per_major=5,
                                 labelsize=ticklabelsize)

    # Use geocat.viz.util convenience function to add main title
    # as well as titles to left and right of the plot axes.
    gvutil.set_titles_and_labels(ax, lefttitle=title,
                                 lefttitlefontsize=titlesize)


def compute_freq_year_season(model, year, months_list):
    example_date = str(year)+'-01-04 12:00:00'
    file = file_name(model, year)
    ds_disk = get_by_file(file)
    precip = ds_disk["pr"]
    precip = precip*86400  # UNIT Conversion
    freq_sum = xr.zeros_like(precip.sel(time=example_date))
    for i in months_list:
        mo = str(year) + '-' + str(i)
        prec = precip.sel(time=mo)
        if charac.name == 'WDF':
            freqs = prec.where(prec < charac.threshold, 1)
            freqs = freqs.where(prec >= charac.threshold-0.01, 0)  # to be sure
        else:
            freqs = prec
        freq = freqs.sum('time')
        freq_sum += freq
    return freq_sum


def compute_climato_season(model,years, season):
    example_year = '2000'
    example_date = '2000-01-04 12:00:00'
    file = file_name(model, example_year)
    ds_disk = get_by_file(file)
    example_precip = ds_disk["pr"]
    example_precip = example_precip.sel(time=example_date)
    climato = xr.zeros_like(example_precip)
    if season.name == 'DJFM':
        for year in years:
            climato += compute_freq_year_season(model, year-1, [12])
            climato += compute_freq_year_season(model, year, [1, 2, 3])
        if charac == ave_daily:
            climato = climato/length(year, [1, 2, 3, 12])
    else:
        for year in years:
            climato += compute_freq_year_season(model, year, season.months)
        if charac == ave_daily:
            climato = climato/length(year, season.months)
    return climato


def file_name(model, year):
    file = 'pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_' + model + '_day_'
    folder = '/media/sr0046/WD-exFAT-50/CORDEX/evaluation/' + model +\
        '/ECMWF-ERAINT_r1i1p1/day/native/'
    return folder + file + str(year) + '.nc'


def generate_plot(model, dat, xyz, abc):
    title = abc + ') ' + model
    t0 = time.time()
    countour_plot(model, dat, charac.contour, title, xyz, abc)

    t1 = time.time()
    print(model, 'time', t1-t0)

# CONSTANTS
# YEARS = range(1999, 2009)
# YEARS = [2018]
Season = namedtuple('Season', 'name months')
djfm = Season('DJFM', [1, 2, 3, 12])
all_year = Season('all-year', range(1, 13))
amjjaso = Season('Apr-Oct', [4, 5, 6, 7, 8, 9, 10])
seasons = [djfm, amjjaso, all_year]
#seasons = [all_year]

for season in seasons:
    Charact = namedtuple('Charact',
                         'name contour unit threshold min_val max_val')
    wd_freq1 = Charact('WDF', np.arange(10, 330, 10, dtype=float), 'days',
                       1, 0, 1.2)
    wd_freq30 = Charact('WDF', np.arange(0, 48, 2, dtype=float), 'days', 30,
                        0, 1)
    tot_prec = Charact('TOTAL-RAINFALL', np.arange(200, 4800, 200, dtype=float),
                       'mm', '', 0, 1.5)
    ave_daily = Charact('AVERAGE-DAILY-RAINFALL', np.arange(0, 22, 1, dtype=float),
                        'mm', '', 0, 1)
    charac = ave_daily
    charac = tot_prec
    #charac = wd_freq1
    charac = wd_freq30

    PROJECT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada'
    PLOT_FOLDER = PROJECT_FOLDER + '/plot-images/climatology-'  +\
        charac.name + str(charac.threshold) + '/'


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
# Generate figure (size (width, height) in inches) and axes using Cartopy
    fig = plt.figure(figsize=(18, 12))
    ticklabelsize = 18
    titlesize = 15

    XYZ = [251, 252, 253, 254, 259, 256, 257, 258, 255]
    abc = ['a', 'b', 'c', 'd', 'i', 'f', 'g', 'h', 'e']
    for model, xyz, voy in zip(MODELS, XYZ, abc):
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
            climato_freq = compute_climato_season(model, YEARS, season)
            climato_freq = climato_freq / len(YEARS)
            generate_plot(model, climato_freq, xyz, voy)
        if model == MODELS[0]:
            ini_climato = xr.zeros_like(climato_freq)
            # very slight differences in lat lon causes error when adding datta.array
            ens_climato = ini_climato.assign_coords(rlon = ((ini_climato.rlon*100)//1000)/100)
            ens_climato = ini_climato.assign_coords(rlat = ((ini_climato.rlat*100)//1000)/100)
        if model!= MODELS[-1]:
            if model in ['MOHC-HadGEM3-RA_v1','MOHC-HadRM3P_v1']:
                climato_freq = climato_freq.rename({'lat':'rlat'})
                climato_freq = climato_freq.rename({'lon':'rlon'})
            ens_climato = ens_climato.assign_coords(rlon = (ini_climato.rlon))
            climato_freq = climato_freq.assign_coords(rlon = (ini_climato.rlon))
            ens_climato = ens_climato.assign_coords(rlat = (ini_climato.rlat))
            climato_freq = climato_freq.assign_coords(rlat = (ini_climato.rlat))
            ens_climato += climato_freq
        if model == MODELS[-1]:
            generate_plot(model, ens_climato/(len(MODELS)-1), xyz, voy)
    plot_filename = 'climatology-CORDEX' + charac.name + str(charac.threshold) + \
      '-' + season.name + '-' + str(YEARS[0]) + '-' + str(YEARS[-1])
    plt.savefig(PLOT_FOLDER + plot_filename + '.png')
    plt.show()
