#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:12:43 2020

@author: sr0046
"""

import xarray as xr
import numpy as np
from tqdm import tqdm
#thresh = 5

def pentad(precip_days):
    pentad_precip = xr.zeros_like(precip_days)
    for t0 in range(1, precip_days.day.size+1-4):  # assume
        pentad_precip.loc[dict(day=t0)] = precip_days.loc[dict(day=slice(t0,t0+4))].sum(dim='day')
    return pentad_precip

def find_spell_at_interval(series, day_start, begin, end):
    '''look for the end of dry spell at interval, divide and conquer'''
    #pdb.set_trace()
    if end >= series.size:
        return find_spell_at_interval(series, day_start, begin, end-1)
    if end - begin <= 1:
        if series.loc[dict(day=end)] == 1:
            return (end - day_start + 1)
        else:
            return (begin - day_start + 1)
    else:
        mid = ((begin+end)//2)
        if series.loc[dict(day=slice(day_start, mid))].sum() > mid-day_start+0.99: # i.e all ones from day_start to mid
            return find_spell_at_interval(series, day_start, mid, end)
        else:
            return find_spell_at_interval(series, day_start, begin, mid)


def dry_spells(precip, thresh):
    if 'rlat' in precip.dims:
        precip = precip.sel(rlon=slice(42.625,50.625),rlat=slice(-26.125,-11.125))
        precip = precip.transpose('rlat', 'rlon', 'time')
        precip.to_netcdf('/home/sr0046/Desktop/check-this.nc')
        precip_days = xr.DataArray(precip.values, coords={'rlat':precip.rlat, 'rlon':precip.rlon, 'day':np.arange(1,precip.time.size+1), 'lat':precip.lat, 'lon':precip.lon}, dims=['rlat', 'rlon', 'day'])
        #print('cordex data get days', precip_days)
    elif precip.name != 'rfe':
        precip = precip.sel(lon=slice(42.625,50.625),lat=slice(-26.125,-11.125))
        precip = precip.transpose('lat', 'lon', 'time')
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    else:
        precip = precip.transpose('lat', 'lon', 'time')
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    #precip_days = precip.swap_dims({'time': 'day'})
    precip_pentad = pentad(precip_days)
    dry_pentad = xr.where(precip_pentad<thresh, 1, 0)
    if 'rlat' in precip.dims:
        stacked_precip = precip_days.stack(z=('rlat', 'rlon'))
        stacked_dry_pentad = dry_pentad.stack(z=('rlat', 'rlon'))
    else:   
        stacked_precip = precip_days.stack(z=('lat', 'lon'))
        stacked_dry_pentad = dry_pentad.stack(z=('lat', 'lon'))
    dry_spell = xr.zeros_like(stacked_dry_pentad)
    #print('eo ary dask',dry_spell.z)
    for xy in tqdm(dry_spell.z):
        series = stacked_dry_pentad.loc[dict(z=xy)]
        print('type of series', type(series))
        day_start = 1
        while day_start <= dry_spell.day.size-thresh+1:
            if stacked_precip.loc[dict(z=xy)].sum(dim='day') < thresh:   # for xy outside region, all zero, eliminate them
                break
            wet = False
            if stacked_dry_pentad.loc[dict(day=day_start,z=xy)] == 0:
                day_start += 1
                wet = True
            elif  stacked_dry_pentad.loc[dict(z=xy, day=slice(day_start, day_start+19))].sum(dim='day') >= 19.5:
                spell_len = find_spell_at_interval(series, day_start, day_start+19, dry_spell.day.size+1)
            else:
                spell_len = find_spell_at_interval(series, day_start, day_start+4, day_start+19)
            if not wet:
                dry_spell.loc[dict(day=day_start,z=xy)] = spell_len
                day_start += spell_len
    dry_spell = dry_spell.unstack().swap_dims({'day': 'time'})
    #print('time is',dry_spell.time)
    # if 'late' in dry_spell.dims:
    #     dry_spell = dry_spell.unstack().swap_dims({'late': 'rlat'})
    #     dry_spell = dry_spell.unstack().swap_dims({'lone': 'rlon'})
    dry_spell_len_sum = dry_spell.sum(dim='time')
    dry_spell_zeros_ones = xr.where(dry_spell > 4, 1, 0)
    dry_spell_count = dry_spell_zeros_ones.sum(dim='time')
    dry_spell_len_ave = xr.where(dry_spell_count>0.5, dry_spell_len_sum/dry_spell_count, 0)
    dry_spell_count.name = 'dry_spell_freq'
    dry_spell_len_ave.name = 'dry_spell_ave_len'
    dry_spell = xr.merge([dry_spell_count, dry_spell_len_ave])
    #print(dry_spell)
    return dry_spell
