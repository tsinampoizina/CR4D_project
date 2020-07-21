#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Tsinampoizina Marie Sophie R. & SFR
"""

import xarray as xr
import numpy as np
import pandas as pd

import time
from tqdm import tqdm
import pdb

from pathlib import Path
from collections import namedtuple

from matplotlib import pyplot as plt
from cartopy.feature import ShapelyFeature, OCEAN, LAKES
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import Reader as ShapeReader, natural_earth

from dry_spells_counter import dry_spells, pool_dry_spells, asafa_dry_spells, wet_spells, asafa_wet_spells
from characteristics import wd_freq1, wd_freq30, total_precip, average_daily, wet_spell_freq, wet_spell_ave_len, dry_spell_freq, dry_spell_ave_len, djfm, all_year, amjjaso
from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil
def shapefile(region):
    if region == 'region-cwest':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar3region-centrewest.shp'
    elif region == 'region-east':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar3region-east.shp'
    elif region == 'region-south':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar3region-ssw.shp'
    elif region == 'madagascar':
        return '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-qgis/madagascar.shp'


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

def add_lat_ticklabels(ax, zero_direction_label=False,
                       dateline_direction_label=True):
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

    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

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

def colorbar(dat, contour_levels):
    if 'rlat' in dat.dims:
        lat = dat.rlat
        lon = dat.rlon
    else:
        lat = dat.lat
        lon = dat.lon
    clevs = contour_levels
    projection = PlateCarree()
    # Import an NCL colormap, truncating it by using geocat.viz.util
    newcmp = gvutil.truncate_colormap(gvcmaps.precip3_16lev,
                                      minval=charac.min_val,
                                      maxval=charac.max_val,
                                      n=len(clevs)+2)
    newcmp.set_under("white")
    newcmp.set_over("darkred")
    # Draw the temperature contour plot with the subselected colormap
    # (Place the zorder of the contour plot at the lowest level)
    ax = plt.subplot(model.plot_pos[0],model.plot_pos[1], model.plot_pos[2],projection=projection)
    cf = ax.contourf(lon, lat, dat, levels=clevs, cmap=newcmp, zorder=1, extend='both')
    # Draw horizontal color bar
    cax = plt.axes((0.14, 0.08, 0.74, 0.02))
    cbar = plt.colorbar(cf, ax=ax, cax=cax, ticks=charac.ticks_step, extend='both',
                        drawedges=True, orientation='horizontal', pad=6)
    cbar.ax.tick_params(labelsize=ticklabelsize)

def countour_plot(model, dat, contour_levels, title):
    if 'rlat' in dat.dims:
        lat = dat.rlat
        lon = dat.rlon
    else:
        lat = dat.lat
        lon = dat.lon
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
    ax = plt.subplot(model.plot_pos[0],model.plot_pos[1], model.plot_pos[2],projection=projection)
    ax.set_extent([42, 52, -26, -11], crs=projection)

    
    # Define the contour levels
    clevs = contour_levels
    
    # Import an NCL colormap, truncating it by using geocat.viz.util\
    # convenience function
    if model.name == 'Ensemble STD':
        cax = plt.axes((0.14, 0.08, 0.74, 0.02))
        #print('can I add', type(clevs))
        #print('and', type(np.array([1,2,3])))
        #clevs = np.concatenate([np.array([1,2,3]),clevs])
    newcmp = gvutil.truncate_colormap(gvcmaps.precip3_16lev,
                                      minval=charac.min_val,
                                      maxval=charac.max_val,
                                      n=len(clevs)+2)
    if model.name == 'Ensemble STD' and 'LENGTH' in charac.name:
        newcmp = gvutil.truncate_colormap(gvcmaps.precip3_16lev,
                                      minval=0,
                                      maxval=1,
                                      n=len(clevs)+2)
    newcmp.set_under("white")
    newcmp.set_over("darkred")
    # Draw the temperature contour plot with the subselected colormap
    # (Place the zorder of the contour plot at the lowest level)
    cf = ax.contourf(lon, lat, dat, levels=clevs, cmap=newcmp, zorder=1, extend='both')
    # Draw horizontal color bar
    if model.name == 'Ensemble STD':
        cbar = plt.colorbar(cf, ax=ax, cax=cax, ticks=charac.ticks_step, extend='both',
                        drawedges=True, orientation='horizontal', pad=6)
        cbar.ax.tick_params(labelsize=ticklabelsize)
        
        
    # Add the land mask feature on top of the contour plot (higher zorder)
    ax.add_feature(land_mask, zorder=2)
    # Add the OCEAN and LAKES features on top of the contour plot
    ax.add_feature(OCEAN.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
    ax.add_feature(LAKES.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
    # Add the country and province features (which are transparent) on top
    ax.add_feature(countries, zorder=3)
    ax.add_feature(provinces, zorder=3)
    nplots = model.plot_pos[0]*model.plot_pos[1]

    left_plots = [1+i*model.plot_pos[1] for i in range(0,model.plot_pos[0])]
    bottom_plots = range(nplots-model.plot_pos[1]+1, nplots+1)
    if model.plot_pos[2] in left_plots and model.plot_pos[2] not in bottom_plots:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        # Use geocat.viz.util convenience function to make plots look like NCL plots by using latitude, longitude tick labels
        ax.set_xticklabels(['','',''])
        add_lat_ticklabels(ax)

    elif model.plot_pos[2] in bottom_plots and  model.plot_pos[2] not in left_plots:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        ax.set_yticklabels(['','',''])
        add_lon_ticklabels(ax)
    elif model.plot_pos[2] not in left_plots and model.plot_pos[2] not in bottom_plots:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        ax.set_yticklabels(['','',''])
        ax.set_xticklabels(['','',''])
    elif model.plot_pos[2] in bottom_plots and model.plot_pos[2] in left_plots:
        gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])
        gvutil.add_lat_lon_ticklabels(ax)
    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, x_minor_per_major=4, y_minor_per_major=5,
                                 labelsize=ticklabelsize)
    # Use geocat.viz.util convenience function to add main title as well as \
    # itles to left and right of the plot axes.
    gvutil.set_titles_and_labels(ax, lefttitle=title, lefttitlefontsize=titlesize)


def get_precip(modx, year):
    file = file_name(modx, year)
    # if Path(file).is_file():
    #     print("input file exists", file)
    # else:
    #     print("input does not exist")
    dsx = xr.open_dataset(file)
    prec = dsx[modx.precip]
    if modx == tamsat:
        dsx = xr.open_dataset(file)
        prec = dsx['rfe']

    if modx.precip != 'pr' and modx != tamsat:
        prec = prec.rename({modx.time:'time'})
        prec = prec.rename({modx.lat:'lat'})
        prec = prec.rename({modx.lon:'lon'})
    if modx == trmm:
        dates = pd.date_range(str(year)+'-01-01', periods=prec.time.size)
        prec = xr.DataArray(prec.values, coords=[dates, prec.lon, prec.lat], dims=['time', 'lon', 'lat'])
        prec = prec.transpose()
    if modx.name == 'arc2':
        dates = pd.date_range(str(year)+'-01-01', periods=prec.time.size)
        prec = xr.DataArray(prec.values, coords=[dates, prec.lat, prec.lon], dims=['time', 'lat', 'lon'])
        prec = prec.transpose('lat', 'lon', 'time',transpose_coords=False)
    if modx.precip == 'pr':
        prec= prec*86400  # UNIT Conversion
    return prec

def compute_climatology_dry_spell(mody, years, season, len_or_count):
    for year in years:
        path_data_file = PLOT_DATA_FOLDER + season.name + str(year) + '.nc'
        if not Path(path_data_file).is_file() or save_data_file:
            this_year = compute_dry_spell_freq_year_season(mody, year, season)
            Path(PLOT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
            this_year.to_netcdf(path_data_file)
            print(year, '...', 'data saved to', path_data_file)
            print('this year after compute_dry_spell...', this_year)
            this_year = this_year[len_or_count]
            t=200
            print('sleep for', t, 'seconds')
            #time.sleep(t)
        else:
            print(year, '...')
            ds_disk = xr.open_dataset(path_data_file)
            this_year = ds_disk[len_or_count]
        if year == YEARS[0]:
            climatology = this_year
            climatology = xr.zeros_like(this_year)
        if 'rlat' in this_year.dims:
            #print('this year rlons', this_year.rlon)
            this_year = this_year.assign_coords(rlat=climatology.rlat)
            this_year = this_year.assign_coords(rlon=climatology.rlon)
        else:
            this_year = this_year.assign_coords(lat=climatology.lat)
            this_year = this_year.assign_coords(lon=climatology.lon)
        climatology += this_year
        
    if time in climatology.dims:
        climatology = climatology.squeeze('time')
    path_data_file_all = PLOT_DATA_FOLDER + season.name + str(YEARS[0]) + '-' + str(YEARS[-1]) + '.nc'
    if not Path(path_data_file_all).is_file() or True:
        t=120
        print('sleep for', t, 'seconds')
        #time.sleep(t)
        climatology.to_netcdf(path_data_file_all)
    return climatology

def compute_dry_spell_freq_year_season(mod, year, season):
    precip = get_precip(mod, year)
    precip_prev = get_precip(mod, year-1)
    if season == djfm:
        dec = str(year-1) + '-12'
        jan = str(year) + '-01-01'
        mar = str(year) + '-03-31'
        sliced = xr.concat([precip_prev.sel(time=dec), precip.sel(time=slice(jan,mar))], dim='time')
    elif season == all_year:
        start = str(year) + '-01-01'
        end = str(year) + '-12-31'
        sliced = precip.sel(time=slice(start,end))
    elif season == amjjaso:
        start = str(year) + '-04-01'
        end = str(year) + '-10-31'
        sliced = precip.sel(time=slice(start,end))
    if 'DRY-SPELL' in charac.name:
        if asafast:
            return asafa_dry_spells(sliced, 5, mod)
        return dry_spells(sliced, 5)
    else:
        if asafast:
            return asafa_wet_spells(sliced, 20, mod)
        else:
            return wet_spells(sliced,20)
    # print(dask_dry_spells(sliced, 5).values)
    # print('matory amizay')




def compute_freq_year_season(mod, year, months):
    precip = get_precip(mod, year)
    example_date = str(year)+'-11-30'
    freqs_season = xr.zeros_like(precip.sel(time=example_date))
    if time in freqs_season.dims:
        freqs_season = freqs_season.squeeze('time')
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
    if 'time' in freqs_season.dims:
        freqs_season = freqs_season.squeeze('time')
    return freqs_season


def compute_climatology_season(mod, years, season):

    if charac.name == 'DRY-SPELL-LENGTH':
        return compute_climatology_dry_spell(mod, years, season, 'dry_spell_ave_len')
    if charac.name == 'DRY-SPELL-FREQUENCY':
        return compute_climatology_dry_spell(mod, years, season, 'dry_spell_freq')
    if charac.name == 'WET-SPELL-LENGTH':
        return compute_climatology_dry_spell(mod, years, season, 'wet_spell_ave_len')
    if charac.name == 'WET-SPELL-FREQUENCY':
        return compute_climatology_dry_spell(mod, years, season, 'wet_spell_freq')


    for year in years:
        print(year, '...', sep=' ', end='', flush=True)
        path_data_file = PLOT_DATA_FOLDER + season.name + str(year) + '.nc'
        path_data_file_prev = PLOT_DATA_FOLDER + season.name + str(year-1) + '.nc'
        if Path(path_data_file).is_file() and not save_data_file: # if plot data is present
           ds_disk = xr.open_dataset(path_data_file)
           this_year = ds_disk[charac.variable_name]
        else: # compute climatology and save data for future replots
           if season == djfm:
               the_dec = compute_freq_year_season(mod, year-1, [12])
               this_year = the_dec
               this_year += compute_freq_year_season(mod, year, [1, 2, 3])
           else:
               this_year = compute_freq_year_season(mod, year, season.months)
           this_year = this_year.rename(charac.variable_name)
           Path(PLOT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
           this_year.to_netcdf(path_data_file)
           print(year, '...', 'data saved to', path_data_file)
        if year == YEARS[0]:
            climatology = xr.zeros_like(this_year)
        this_year = this_year.assign_coords(lat=climatology.lat)
        this_year = this_year.assign_coords(lon=climatology.lon)
        if 'time' in this_year.dims:
            this_year = this_year.squeeze('time')
        climatology += this_year
    if charac == average_daily:
        climatology = climatology/length(year, season.months)
    if 'time' in climatology.dims:
        climatology = climatology.squeeze('time')
    path_data_file_all = PLOT_DATA_FOLDER + season.name + str(YEARS[0]) + '-' + str(YEARS[-1]) + '.nc'
    if not Path(path_data_file_all).is_file() or True:
        climatology = climatology.rename(charac.variable_name)
        climatology.to_netcdf(path_data_file_all)
    return climatology

def compute_climatology_cordex_ens(years, season):
    CORDEX_MODELS_ENS = [x for x in CORDEX_MODELS if x in MODELS]
    cordex_climatologies = []
    maxes = []
    for cordex_model, i in zip(CORDEX_MODELS_ENS, range(0,len(CORDEX_MODELS))):
        print('cordex model', cordex_model)
        PLOT_DATA_FOLDER_ROOT_x = PROJECT_FOLDER + '/plot-data/' + charac.name + str(charac.threshold) + '/'
        PLOT_DATA_FOLDER_x = PLOT_DATA_FOLDER_ROOT_x + cordex_model.name + '/'
        begin = max(cordex_model.year_range[0], YEARS[0])
        end = min(cordex_model.year_range[-1], YEARS[-1])
        climato_file_full = PLOT_DATA_FOLDER_x + season.name + str(YEARS[0]) + '-' + str(YEARS[-1]) + '.nc'
        climato_file = PLOT_DATA_FOLDER_x + season.name + str(begin) + '-' + str(end) + '.nc'
        if Path(climato_file_full).is_file():
            dsx = xr.open_dataset(climato_file_full)
            print('path of dsx',climato_file_full)
            print('dsx',dsx)
            climatology = dsx[charac.variable_name]
        elif Path(climato_file).is_file():
            dsx = xr.open_dataset(climato_file)
            climatology = dsx[charac.variable_name]
        else:
            print('print some climato for', cordex_model.name)
        if cordex_model == CORDEX_MODELS_ENS[0]:
            ens_climatology = xr.zeros_like(climatology)
        # Always use rlon and rlat
        if 'lat' in climatology.dims:
            climatology = climatology.expand_dims('model')
            climatology = climatology.transpose('lon','lat','model',transpose_coords=False)
            climatology = xr.DataArray(climatology.values,
                                       coords={'rlat':ens_climatology.rlat,
                                               'rlon':ens_climatology.rlon,
                                               'model':np.array([i])},
                                       dims=['rlon','rlat','model'])
        else:
            climatology = climatology.assign_coords(rlat = ens_climatology.rlat)
            climatology = climatology.assign_coords(rlon = ens_climatology.rlon)
            if not 'model' in climatology.dims:
                climatology = climatology.expand_dims('model')
            climatology = climatology.transpose('rlon','rlat','model',transpose_coords=False)
            climatology = xr.DataArray(climatology.values,
                                       coords={'rlat':ens_climatology.rlat,
                                               'rlon':ens_climatology.rlon,
                                               'model':np.array([i])},
                                       dims=['rlon','rlat','model'])

        cordex_climatologies.append(climatology)
        maxes.append(climatology.max().values)
    climatologies = xr.concat(cordex_climatologies, dim='model')
    climatologies = climatologies.rename(charac.variable_name)
    mean =  climatologies.mean("model")
    std =  climatologies.std("model")
    mean = mean.transpose('rlat', 'rlon',transpose_coords=False)
    std = std.transpose('rlat','rlon',transpose_coords=False)
    std.to_netcdf('/home/sr0046/Desktop/stdev.nc')
    return {'mean':mean, 'stdev':std}


def file_name(mod, year):
    if mod.precip == 'pr':    # Cordex people
        file = 'pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_' + mod.version + '_day_'
        folder = '/media/sr0046/WD-exFAT-50/CORDEX/evaluation/' + mod.version +\
            '/ECMWF-ERAINT_r1i1p1/day/native/'
        return folder + file + str(year) + '.nc'
    else:
        fname = INPUT_FOLDER + mod.name + '/' + mod.version + '-' +region + '-' +\
                str(year) + mod.ext
        return fname


# Take one day to get the corect data shape, to initialise
# example_precip is used in compute_climatology
# lat, lon, are used in contour_plot
region = 'madagascar'
INPUT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'+region+'/'

#EXAMPLE_YEAR = '2001'
seasons = [djfm]
seasons = [all_year]
seasons = [djfm]

#seasons = [amjjaso]

dsx = xr.open_dataset('/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/madagascar/trmm/TRMM_3B42-madagascar-1999.nc')
for season in seasons:
   # choice charac
   charac = average_daily
   charac = total_precip
   charac = wd_freq1(season)
   #charac = wd_freq30(season)
   charac = dry_spell_freq(season)
   charac = dry_spell_ave_len(season)
   charac = wet_spell_freq(season)
   #charac = wet_spell_ave_len(season)
   save_plot_file = True
   save_data_file = False
   asafast = True

   PROJECT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada'
   PLOT_FOLDER = PROJECT_FOLDER + '/plot-images/climatology-'  + charac.name + str(charac.threshold) + '/'

   if charac.name == 'DRY-SPELL-LENGTH' and False:
       PLOT_DATA_FOLDER_ROOT = PROJECT_FOLDER + '/plot-data/' + 'DRY-SPELL-FREQUENCY' + str(charac.threshold) + '/'
   else:
       PLOT_DATA_FOLDER_ROOT = PROJECT_FOLDER + '/plot-data/' + charac.name + str(charac.threshold) + '/'

   from models import *
   #MODELS = [trmm]
   MODELS = [itamsat] #
   #MODELS += [chirps, trmm, arc2, gpcc]
   #MODELS = [gpcc]
   # MODELS = [ichirps, gpcc]
   #MODELS = [ichirps]
   #MODELS = [gpcc]
   CORDEX_MODELS = [remo,  hirham, crcm5, racmo, rca4, clm, rm3p, gem3]
   MODELS = [remo,  hirham,  crcm5, racmo, rca4, clm, rm3p, gem3, ens, stdev]
   #MODELS.remove(gem3)
   #MODELS = [itamsat]
   #MODELS = [remo]
   MODELS += [ichirps, trmm, arc2, gpcc]
   #MODELS = [remo, ens, stdev]
   

   fig = plt.figure(figsize=(14, 14))
   ticklabelsize = 18
   titlesize = 18
   for model in MODELS:

      if model.name == 'arc2':
          YEARS = range(2002, 2009) # 2002 to 2009
      elif model.name == 'tamsat' and season == djfm:
          YEARS = range(1999, 2009)
      elif model == gem3 and season != all_year:
          YEARS = range(1999,2008)
      elif model == gem3 and season == all_year:
          YEARS = range(1999, 2007)
      else:
          YEARS = range(1999, 2009)  # 2005 izao

      print("Processing ", charac.name, 'for', season.name, 'of', YEARS)
      print("Save plot file:", save_plot_file)
      print("Force save data for plot:", save_data_file)
      print("Working on model:", model.long_name)
      plot_filename = 'climatology-'+ charac.name + str(charac.threshold) + \
      '-' + season.name + '-' + str(YEARS[0]) + '-' + str(YEARS[-1])
      PLOT_DATA_FOLDER =  PLOT_DATA_FOLDER_ROOT + model.name + '/'
      # file = file_name(EXAMPLE_YEAR)
      # ds_disk = xr.open_dataset(file)

      indiv_title = model.long_name
      title = charac.long_name + ' ' + season.name + ' ' + str(YEARS[0]) + '-' + str(YEARS[-1])

      t0 = time.time()
      if model == ens:
          climatology_freq = compute_climatology_cordex_ens(YEARS, season)['mean']
      elif model == stdev:
          climatology_freq = compute_climatology_cordex_ens(YEARS, season)['stdev']
      else:
          climatology_freq = compute_climatology_season(model, YEARS, season)
      climatology_freq = climatology_freq / len(YEARS)
      countour_plot(model, climatology_freq, charac.contour, indiv_title)
      t1 = time.time()
      print('\n','Execution time', model.name, ':', secs_to_dhms(t1-t0))
   
   if save_plot_file:
       Path(PLOT_FOLDER).mkdir(parents=True, exist_ok=True)
       if len(MODELS) == 9:
           data_grid_or_cordex = 'cordex'
       elif len(MODELS) == 5 and chirps in MODELS:
           data_grid_or_cordex = 'data'
       elif len(MODELS) <= 2:
           data_grid_or_cordex = "".join([model.name for model in MODELS])
       else:
           data_grid_or_cordex = ''
       print('file saved as ', PLOT_FOLDER + data_grid_or_cordex + plot_filename + '.png')
       plt.suptitle(title, fontsize=16)
       plt.savefig(PLOT_FOLDER + data_grid_or_cordex + plot_filename + '.png')
   plt.show()
tfinal = time.time()
print('\n','Total execution time:', secs_to_dhms(tfinal - t0))
