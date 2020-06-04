#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:47:09 2020

@author: sr0046
"""
import xarray as xr
import numpy as np
import time
import pandas as pd

from matplotlib import pyplot as plt

from cartopy.feature import ShapelyFeature, OCEAN, LAKES
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import Reader as ShapeReader, natural_earth

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

def month_range(year, mo):
    month_start_end_list = [
        (0,31),
        (31,59),
        (59,90),
        (90,120),
        (120,151),
        (151,181),
        (181,212),
        (212,243),
        (243,273),
        (273,304),
        (304,334),
        (334,365)
        ]
    if year in [1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020]:
        month_start_end_list = [
                (0,31),
                (31,60),
                (60,91),
                (91,121),
                (121,152),
                (152,182),
                (182,213),
                (213,244),
                (244,274),
                (274,305),
                (305,335),
                (335,366)        
                ] 
    start,end = month_start_end_list[mo]
    return start, end

def input_file(year, region):
    if MODEL == 'chirps':
        PRECIPITATION = "precip"
        LATITUDE = "latitude"
        LONGITUDE = "longitude"
        YEAR_RANGE = range(1981,2017+1)
    if MODEL == 'trmm':
        PRECIPITATION = "precipitation"
        LATITUDE = "lat"
        LONGITUDE = "lon"
        YEAR_RANGE = range(1998,2017+1)
    '''Get data from filename, slice using Madagascar coordinates'''
    return FOLDER + region+'/'+MODEL+'/'+MODEL+'-'+region+'-'+str(year)+'.nc'

def compute_climatology_monthly_mean(region):
    INPUT_F = input_file(EXAMPLE_YEAR, 'region-cwest')
    if MODEL == 'chirps':
        PRECIPITATION = "precip"
        LATITUDE = "latitude"
        LONGITUDE = "longitude"
        YEAR_RANGE = range(1981,2017+1)
    if MODEL == 'trmm':
        PRECIPITATION = "precipitation"
        LATITUDE = "lat"
        LONGITUDE = "lon"
        YEAR_RANGE = range(1998,2017+1)
    if MODEL == 'chirps':
        ds_disk = xr.open_dataset(INPUT_F)
        monthly = ds_disk.groupby('time.month')
        monthly_mean = monthly.apply(np.mean)
        monthly_mean_precip_df = monthly_mean.to_dataframe().precip
        zeros_df = np.zeros_like(monthly_mean_precip_df)
        climatology = zeros_df
        for year in YEAR_RANGE:
            input_f = input_file(year, region)    
            ds_disk = xr.open_dataset(input_f)
            monthly = ds_disk.groupby('time.month')               
            monthly_mean = monthly.apply(np.mean)
            #print(monthly_mean)
            #print(monthly_mean.precip)
            monthly_mean_precip_df = monthly_mean.to_dataframe().precip
            climatology += monthly_mean_precip_df
        print(climatology/len(YEAR_RANGE))
    if MODEL == 'trmm': 
        climatology = np.zeros(12)
        for year in YEAR_RANGE:
            this_year_climato = np.zeros(12)
            input_f = input_file(year, region)    
            daily = xr.open_dataset(input_f)
            daily = daily.groupby(daily.time)
            daily_mean = daily.apply(np.mean)
            daily_mean_precip_df = daily_mean.precipitation.to_dataframe()
            daily_mean_precip_df = daily_mean_precip_df.precipitation
            for mo in range(0,12):
                s,e = month_range(year, mo)
                this_month_daily_precip_df = daily_mean_precip_df[s:e]
                this_year_climato[mo] = np.mean(this_month_daily_precip_df)
            climatology += this_year_climato
        climatology = climatology/len(YEAR_RANGE)
        print(climatology/len(YEAR_RANGE))
    return climatology
def plot_add():
    if MODEL == 'chirps':
        PRECIPITATION = "precip"
        LATITUDE = "latitude"
        LONGITUDE = "longitude"
        YEAR_RANGE = range(1981,2017+1)
    if MODEL == 'trmm':
        PRECIPITATION = "precipitation"
        LATITUDE = "lat"
        LONGITUDE = "lon"
        YEAR_RANGE = range(1998,2017+1)
    for region in ['region-east','region-cwest','region-south']:
        climatology_monthly_mean_precip_df = compute_climatology_monthly_mean(region)
        if MODEL == 'trmm':
            double = np.concatenate((climatology_monthly_mean_precip_df,climatology_monthly_mean_precip_df),axis=None)
        if MODEL == 'chirps':
            double = np.concatenate((climatology_monthly_mean_precip_df/len(YEAR_RANGE),climatology_monthly_mean_precip_df/len(YEAR_RANGE)),axis=None)
        xs = range(0,24)
        crs,cre = (6,-6) # cycle start and end for display
        month_labels = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        double_months= month_labels + month_labels
        if  MODEL == 'trmm':
            ax.plot(xs[crs:cre],double[crs:cre], marker='o', label=region+' '+MODEL, color=col_dic[region])
        if MODEL == 'chirps':
            ax.plot(xs[crs:cre],double[crs:cre], marker='<', linestyle=':', label=region+' '+MODEL, color=col_dic[region])

FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'
OUTPUT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/plot-images/'
MODEL = 'trmm'
EXAMPLE_YEAR = 2000
EXAMPLE_DATE = '2000-11-30T00:00:00.000000000'
if MODEL == 'chirps':
    PRECIPITATION = "precip"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    YEAR_RANGE = range(1981,2017+1)
if MODEL == 'trmm':
    PRECIPITATION = "precipitation"
    LATITUDE = "lat"
    LONGITUDE = "lon"
    YEAR_RANGE = range(1998,2017+1)



    
fig=plt.figure(figsize=(7,5))
ax=fig.add_subplot(111)
col_dic={'region-east':'tab:green','region-cwest':'tab:purple','region-south':'tab:orange'}

MODEL = 'chirps'
plot_add()
MODEL = 'trmm'
plot_add()

xs = range(0,24)
crs,cre = (6,-6) # cycle start and end for display
month_labels = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
double_months= month_labels + month_labels
ax.set_xticks(xs[crs:cre])
ax.set_xticklabels(double_months[crs:cre])  
ax.set_ylabel("average precipitation (mm per day)")  
plt.legend(loc=2)
ax.grid(linestyle='--', linewidth=.25)
output_f = OUTPUT_FOLDER + 'climatology_seasonal-cycle_daily-rainfall/' + 'seasonal-cycle-all-models' + '.eps'
plt.savefig(output_f, format='eps', dpi=900)
plt.show()