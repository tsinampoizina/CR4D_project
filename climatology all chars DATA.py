#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Tsinampoizina Marie Sophie R. & SFR
"""

import xarray as xr
import numpy as np

import time
from tqdm import tqdm
import pdb

from pathlib import Path
from collections import namedtuple

from matplotlib import pyplot as plt
from cartopy.feature import ShapelyFeature, OCEAN, LAKES
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import Reader as ShapeReader, natural_earth

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil



def secs_to_dhms(seconds):
    from datetime import datetime, timedelta
    d = datetime(1,1,1) + timedelta(seconds=int(seconds))
    if seconds > 3600:
        output = f"{d.hour} hours, {d.minute} minutes {d.second} seconds"
    elif seconds > 60:
        output = f"{d.minute} minutes, {d.second} seconds"
    else:
        output = f"{d.second} seconds"
    return output


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
    cbar = plt.colorbar(cf, ax=ax, cax=cax, ticks=charac.ticks_step,
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


def get_precip(year):
    file = file_name(year)
    ds_disk = xr.open_dataset(file)
    precip = ds_disk[model.precip]
    precip = precip.rename({model.time:'time'})
    return precip

def precip_of_spell(stacked_precip, t1, t2, xy):
    spell = stacked_precip.loc[dict(time=slice(t1, t2), z=xy)]
    return spell.sum().values

def compute_climatology_dry_spell(years, season, len_or_count):
    #pdb.set_trace()
    climatology = xr.zeros_like(example_precip)
    for year in years:
        path_data_file = PLOT_DATA_FOLDER + season.name + str(year) + '.nc'
        if not Path(path_data_file).is_file() or save_data_file:
            this_year = compute_dry_spell_freq_year_season(year, season)
            Path(PLOT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
            this_year.to_netcdf(path_data_file)
            print(year, '...', 'data saved to', path_data_file)
            this_year = this_year[len_or_count]
            time.sleep(120)
        else:
            print(year, '...')
            ds_disk = xr.open_dataset(path_data_file)
            this_year = ds_disk[len_or_count]
        climatology += this_year
        
    if model.name == 'arc2':
        climatology = climatology.squeeze('time')
    return climatology

def compute_dry_spell_freq_year_season(year, season):
    precip = get_precip(year)

    if season == djfm:
        precip_prev = get_precip(year-1)
        decb = str(year-1) + '-' + '12-01'
        dece = str(year-1) + '-' + '12-31'
        jan = str(year) + '-01-01'
        mar = str(year) + '-03-31'
        sliced = xr.concat([precip_prev.sel(time=slice(decb,dece)), precip.sel(time=slice(jan,mar))], dim='time')
    elif season == all_year:
        start = str(year) + '-01-01'
        end = str(year) + '-12-31'
        sliced = precip.sel(time=slice(start,end))
    else:
        start = str(year) + '-' + str(season.months[0]) + '-01'
        end = str(year) + '-' + str(season.months[-1]) + '-' + str(length(year, [season.months[-1]]))
        sliced = precip.sel(time=slice(start,end))
    # transform sliced time dimension into number 1 2 3 4 5 ...
    # first get sliced time length
    days = range(1, sliced.time.size+1)
    days_da = np.array(days)
    sliced_days = sliced.assign_coords(time=days_da)

    stacked = sliced_days.stack(z=(model.lon, model.lat))
    stacked_len = xr.zeros_like(stacked)

    tmax = len(days)
    print(year, '...')
    for xy in tqdm(stacked.z):
        time_lead = 1
        spell_len_thresh = 5
        precip_thresh = 5
        while time_lead <= tmax:
            wet = False
            dry_spell = 4
            while time_lead + dry_spell <= tmax and not wet:
                extra_dry = dry_spell - (spell_len_thresh - 1)
                if  precip_of_spell(stacked, time_lead + extra_dry, time_lead + extra_dry + spell_len_thresh - 1, xy) < precip_thresh:
                    dry_spell += 1
                else: 
                    wet = True
            if dry_spell < spell_len_thresh:
                stacked_len.loc[dict(time=time_lead, z=xy)] = 0
                time_lead += 1
            if dry_spell >= spell_len_thresh:
                stacked_len.loc[dict(time=time_lead, z=xy)] = dry_spell
                time_lead += dry_spell

    stacked_count = xr.where(stacked_len > 4, 1, 0)
    dry_spell_count = stacked_count.unstack().sum(dim='time')
    dry_spell_len_sum = stacked_len.unstack().sum(dim='time')
    dry_spell_len_ave = xr.where(dry_spell_count>0.5, dry_spell_len_sum/dry_spell_count, 0)
    dry_spell_count = dry_spell_count.rename('spell_count')
    dry_spell_len = dry_spell_len_ave.rename('ave_spell_len')
    dry_spells = xr.merge([dry_spell_count, dry_spell_len])
    return dry_spells


def compute_freq_year_season(year, months):
    precip = get_precip(year)
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
        freqs_season = xr.zeros_like(precip.sel(time=example_date))
        for i in months:
            mo = str(year) + '-' + str(i).zfill(2)
            prec = precip.sel(time=mo)
            # Make 1 if >= threshold, otherwise make 0
            if charac.name == 'WDF':
                freqs = prec.where(prec < charac.threshold, 1)
                freqs = freqs.where(prec > charac.threshold-0.01, 0) # to be sure
                freq = freqs.sum('time')
            else:
                freqs = prec
                freq = freqs.sum('time')
            freqs_season += freq
    if model.name == 'arc2':
        freqs_season = freqs_season.squeeze('time')
    return freqs_season


def compute_climatology_season(years, season):
    if charac == dry_spell_freq:
        return compute_climatology_dry_spell(years, season, 'spell_count')
    if charac == dry_spell_ave_len:
        return compute_climatology_dry_spell(years, season, 'ave_spell_len')
    climatology = xr.zeros_like(example_precip)
    if season == djfm:
        for year in years:
            path_data_file = PLOT_DATA_FOLDER + 'DJFM' + str(year) + '.nc'
            path_data_file_prev = PLOT_DATA_FOLDER + 'DJFM' + str(year-1) + '.nc'
            if not Path(path_data_file).is_file() or save_data_file:
                the_dec = compute_freq_year_season(year-1, [12])
                this_year = the_dec
                this_year += compute_freq_year_season(year, [1, 2, 3])
                Path(PLOT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
                this_year.to_netcdf(path_data_file)
                print(year, '...', 'data saved to', path_data_file)
            else:
                ds_disk = xr.open_dataset(path_data_file)
                this_year = ds_disk[model.precip]
                print(year, '...', sep=' ', end='', flush=True)
            climatology += this_year
        if charac == average_daily:
            climatology = climatology/length(year, [1,2,3,12])
    elif season == all_year:
        for year in years:
            path_data_file = PLOT_DATA_FOLDER + str(year) + '.nc'
            if not Path(path_data_file).is_file() or save_data_file:
                this_year = compute_freq_year_season(year, season.months)
                save_this_year = this_year.expand_dims('time')
                Path(PLOT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
                save_this_year.to_netcdf(path_data_file)
                print(year, '...', 'data saved to', path_data_file)
            else:
                ds_disk = xr.open_dataset(path_data_file)
                this_year = ds_disk[model.precip]
                print(year, '...', sep=' ', end='', flush=True)
            climatology += this_year
        if charac == average_daily:
            climatology = climatology/length(year, season.months)
    if model.name == 'arc2':
        climatology = climatology.squeeze('time')
    # call ../data-scripts/nco-merger.sh to combine into one climatology data
    return climatology


def file_name(year):
    fname = INPUT_FOLDER + model.name + '/' + model.version + '-' +region + '-' +\
            str(year) + model.ext
    return fname


# Take one day to get the corect data shape, to initialise
# example_precip is used in compute_climatology
# lat, lon, are used in contour_plot
region = 'madagascar'
INPUT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'+region+'/'
EXAMPLE_YEAR = '2001'
Season = namedtuple('Season', 'name months')
djfm = Season('DJFM', [1,2,3,12])
all_year = Season('all-year', range(1, 13))
amjjaso = Season('Apr-Oct', [4,5,6,7,8,9,10])
seasons = [djfm, amjjaso]
seasons = [all_year]
seasons = [djfm]
for season in seasons:
   Characteristic = namedtuple('Characteristic',
                              'name contour unit threshold min_val max_val ticks_step')
   if season == djfm:
       dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', np.arange(0, 10.5, 0.5, dtype=float),
                               'spells', 5, 0, 1, np.arange(0, 11, 1, dtype=float))
       dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', np.arange(0, 17, 1, dtype=float),
                               'spells', 5, 0, 1, np.arange(0, 16, 1, dtype=float))
   if season == all_year:
       dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', np.arange(0, 30, 1, dtype=float),
                               'spells', 5, 0, 1, np.arange(0, 30, 1, dtype=float))
       dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', np.arange(0, 50, 1, dtype=float),
                               'spells', 5, 0, 1, np.arange(5, 50, 5, dtype=float))


   wd_freq1 = Characteristic('WDF', np.arange(10, 330, 10, dtype=float), 'days', 1, 0,1.2, np.arange(10, 330, 10, dtype=float))
   wd_freq30 = Characteristic('WDF', np.arange(0, 48, 2, dtype=float), 'days', 30,0,1, np.arange(0, 48, 2, dtype=float))
   total_precip = Characteristic('TOTAL-RAINFALL',
                                  np.arange(200,4800, 200, dtype=float), 'mm',
                                  '',0,1.5,np.arange(200,4800, 200, dtype=float))
   average_daily = Characteristic('AVERAGE-DAILY-RAINFALL',
                                  np.arange(0, 22, 1, dtype=float), 'mm',
                                  '',0,1,np.arange(0, 22, 1, dtype=float))

   charac = average_daily
   charac = total_precip
   #charac = wd_freq1
   #charac = wd_freq30
   charac = dry_spell_freq
   #charac = dry_spell_ave_len
   save_plot_file = False
   save_data_file = True

   print("Processing ", charac.name, 'for', season.name)
   print("Save plot file:", save_plot_file)
   print("Force save data for plot:", save_data_file)
   PROJECT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada'
   PLOT_FOLDER = PROJECT_FOLDER + '/plot-images/climatology-'  + charac.name + str(charac.threshold) + '/'
   
   if charac == dry_spell_ave_len:
       PLOT_DATA_FOLDER_ROOT = PROJECT_FOLDER + '/plot-data/' + 'DRY-SPELL-FREQUENCY' + str(charac.threshold) + '/'
   else:
       PLOT_DATA_FOLDER_ROOT = PROJECT_FOLDER + '/plot-data/' + charac.name + str(charac.threshold) + '/'

   from models import *
   MODELS = [arc2]
   #MODELS = [tamsat]
   MODELS = [chirps, trmm, tamsat, arc2, gpcc]
   MODELS = [gpcc]
   MODELS = [ichirps]


   fig = plt.figure(figsize=(18, 7))
   ticklabelsize = 18
   titlesize = 18

   for model in MODELS:
      print("Working on model:", model.name)
      if model.name == 'arc2':
         YEARS = range(2001, 2010)
      else:
         YEARS = range(1999, 2009)
      plot_filename = 'climatology-'+ charac.name + str(charac.threshold) + \
      '-' + season.name + '-' + str(YEARS[0]) + '-' + str(YEARS[-1])
      PLOT_DATA_FOLDER =  PLOT_DATA_FOLDER_ROOT + model.name + '/'
      file = file_name(EXAMPLE_YEAR)
      ds_disk = xr.open_dataset(file)

      example_precip = ds_disk[model.precip]
      example_precip = example_precip.rename({model.time:'time'})
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
      print('\n','Execution time', model.name, ':', secs_to_dhms(t1-t0))

   if save_plot_file:
       print('file saved as ', PLOT_FOLDER + plot_filename + '.png')
       plt.savefig(PLOT_FOLDER + plot_filename + '.png')
   plt.show()
tfinal = time.time()
print('\n','Total execution time:', secs_to_dhms(tfinal - t0))
