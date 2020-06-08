#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: S F Rajaona & Tsinampoizina Marie Sophie
TODOS : use date functionality of xarray for chirps & co.
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
    return ds


def countour_plot(dat, contour_levels, title):
    # Download the Natural Earth shapefile for country boundaries at
    # 10m resolution
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
    projection = PlateCarree()
    countries = ShapelyFeature(country_geos,
                               crs=projection,
                               facecolor='none',
                               edgecolor='black',
                               lw=1.5)
    land_mask = ShapelyFeature(other_land_geos,
                               crs=projection,
                               facecolor='white',
                               edgecolor='none')
    shapefile = natural_earth(category='cultural',
                              resolution='10m',
                              name='admin_1_states_provinces')
    # Extract the Madagascar region borders
    province_geos = [record.geometry for record in
                     ShapeReader(shapefile).records()
                     if record.attributes['admin'] == 'Madagascar']
    # Define a Cartopy Feature for the province borders,
    # so they can be easily plotted
    provinces = ShapelyFeature(province_geos,
                               crs=projection,
                               facecolor='none',
                               edgecolor='black',
                               lw=0.25)
    # Generate figure (set its size (width, height) in inches)
    # and axes using Cartopy
    ax = plt.subplot(model.plot_pos, projection=projection)
    ax.set_extent([42, 52, -26, -11], crs=projection)
    # Define the contour levels
    clevs = contour_levels
    # Import an NCL colormap, truncating it by using geocat.viz.util\
    # convenience function
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
                        drawedges=True, orientation='horizontal')
    cbar.ax.tick_params(labelsize=ticklabelsize)
    # Add the land mask feature on top of the contour plot (higher zorder)
    ax.add_feature(land_mask, zorder=2)
    # Add the OCEAN and LAKES features on top of the contour plot
    ax.add_feature(OCEAN.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
    ax.add_feature(LAKES.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
    # Add the country and province features (which are transparent) on top
    ax.add_feature(countries, zorder=3)
    ax.add_feature(provinces, zorder=3)
    if model.plot_pos != 151: # the most left image
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        ax.set_yticklabels(['','',''])
        add_lon_ticklabels(ax)
    else:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        # Use geocat.viz.util convenience function to make plots look like NCL plots by using latitude, longitude tick labels
        gvutil.add_lat_lon_ticklabels(ax)

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, x_minor_per_major=4, y_minor_per_major=5,
                                 labelsize=ticklabelsize)
    # Use geocat.viz.util convenience function to add main title as well as \
    # itles to left and right of the plot axes.
    gvutil.set_titles_and_labels(ax, lefttitle=title, lefttitlefontsize=titlesize)

def compute_freq_year_season(year, months):
    file = file_name(year)
    ds_disk = get_by_file(file)
    precip = ds_disk[model.precip]

    if model.name == 'trmm':
        freqs_month = []
        for start, end in month_start_end(year):
            prec = precip.sel(time=slice(start, end))
            # Make 1 if >= threshold, otherwise make 0
            if charac.name == 'WDF':
                freqs = prec.where(prec < charac.threshold, 1)
                freqs = freqs.where(prec > charac.threshold-0.01, 0)  # to be sure
            else:
                freqs = prec
            freq = freqs.sum('time')
            freqs_month.append(freq)
        freqs_season = sum([freqs_month[i-1] for i in months])
    else:
        example_date = str(year)+'-11-30'
        if model.name == 'arc2':
            freqs_season = xr.zeros_like(precip.sel(T=example_date))
        else:
            freqs_season = xr.zeros_like(precip.sel(time=example_date))
        for i in months:
            mo = str(year) + '-' + str(i).zfill(2)
            if model.name == 'arc2':
                prec = precip.sel(T=mo)
            else:
                prec = precip.sel(time=mo)
            # Make 1 if >= threshold, otherwise make 0
            if charac.name == 'WDF':
                freqs = prec.where(prec < charac.threshold, 1)
                freqs = freqs.where(prec > charac.threshold-0.01, 0)  # to be sure
                freq = freqs.sum(model.time)
            else:
                freqs = prec
                freq = freqs.sum(model.time)
            print(freq.max(model.lat))
            freqs_season += freq
    if model.name == 'arc2':
        freqs_season = freqs_season.squeeze('T')
    return freqs_season


def compute_climatology_season(years, season):
    climatology = xr.zeros_like(example_precip)
    if season == djfm:
        for year in years:
            climatology += compute_freq_year_season(year-1, [12])
            climatology += compute_freq_year_season(year, [1, 2, 3])
        if charac == average_daily:
            climatology = climatology/length(year, [1,2,3,12])
    else:
        for year in years:
            climatology += compute_freq_year_season(year, season.months)
        if charac == average_daily:
            climatology = climatology/length(year, season.months)
    if model.name == 'arc2':
        climatology = climatology.squeeze('T')
    return climatology


def file_name(year):
    fname = INPUT_FOLDER + model.name + '/' + model.version + '-region-' +\
            str(year) + model.ext
    return fname


# Take one day to get the corect data shape, to initialise
# example_precip is used in compute_climatology
# lat, lon, are used in contour_plot
INPUT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'
EXAMPLE_YEAR = '2001'
Season = namedtuple('Season', 'name months')
djfm = Season('DJFM', [1,2,3,12])
all_year = Season('all-year', range(1, 13))
amjjaso = Season('Apr-Oct', [4,5,6,7,8,9,10])
seasons = [djfm, all_year, amjjaso]

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

   Model = namedtuple('Model', 'name version ext lat lon precip plot_pos date_example time')
   XYZ = [151, 152, 153, 154, 155]
   EXAMPLE_DATES = [360,  '2001-11-30']
   trmm = Model('trmm', 'TRMM_3B42', '.nc4', 'lat', 'lon', 'precipitation',
               XYZ[0], EXAMPLE_DATES[0], 'time')
   chirps = Model('chirps', 'chirps-v2', '.nc', 'latitude', 'longitude', 'precip',
                  XYZ[1], EXAMPLE_DATES[1], 'time')
   gpcc = Model('gpcc', 'gpcc_v2018', '.nc', 'lat', 'lon', 'precip',
               XYZ[4], EXAMPLE_DATES[1], 'time')
   tamsat = Model('tamsat', 'rfe_v3', '.nc', 'lat', 'lon', 'rfe',
               XYZ[2], EXAMPLE_DATES[1], 'time')
   arc2 = Model('arc2', 'arc2', '.nc', 'Y', 'X', 'est_prcp',
               XYZ[3], EXAMPLE_DATES[1], 'T')
   MODELS = [chirps]
   #MODELS = [arc2]
   MODELS = [tamsat]
   MODELS = [chirps, trmm, tamsat, arc2, gpcc]

   fig = plt.figure(figsize=(18, 6))
   ticklabelsize = 18
   titlesize = 18

   for model in MODELS:
      if model.name == 'arc2':
         YEARS = range(2001, 2010)
      else:
         YEARS = range(1999, 2010)
      plot_filename = 'climatology-'+ charac.name + str(charac.threshold) + \
      '-' + season.name + '-' + str(YEARS[0]) + '-' + str(YEARS[-1])
      file = file_name(EXAMPLE_YEAR)
      ds_disk = get_by_file(file)
      if model.name == 'arc2':
         example_precip = ds_disk[model.precip]
         example_precip = example_precip.sel(T=model.date_example)
      else:
         example_precip = ds_disk[model.precip]
         example_precip = example_precip.sel(time=model.date_example)
      lat = ds_disk[model.lat]
      lon = ds_disk[model.lon]
      precip = ds_disk[model.precip]

      title = model.name

      t0 = time.time()

      climatology_freq = compute_climatology_season(YEARS, season)
      climatology_freq = climatology_freq / len(YEARS)
      if model.name == 'trmm':
         climatology_freq = climatology_freq.transpose()
      countour_plot(climatology_freq, charac.contour, title)
      t1 = time.time()
      print(model.name, t1-t0)

   plt.savefig(PLOT_FOLDER + plot_filename + '.png')
   plt.show()
