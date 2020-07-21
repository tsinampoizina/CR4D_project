#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:12:43 2020

@author: sr0046
"""
from tqdm.contrib.concurrent import process_map
import xarray as xr
import numpy as np
from tqdm import tqdm
import time
import pdb
from functools import partial
#thresh = 5

def pentad(precip_days):
    pentad_precip_x = xr.zeros_like(precip_days)
    pentad_precip = pentad_precip_x.load()
    for t0 in range(1, precip_days.day.size+1-4):  # assume
        pentad_precip.loc[dict(day=t0)] = precip_days.loc[dict(day=slice(t0,t0+4))].sum(dim='day')
    return pentad_precip

def pentad_dask(precip_days):
    pentad_precip = np.zeros_like(precip_days)
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

def dask_dry_spells(precip, thresh):
    # avadika days aloha ilay big precip
    if 'rlat' in precip.dims:
        precip = precip.sel(rlon=slice(42.625,50.625),rlat=slice(-26.125,-11.125))
        precip = precip.transpose('rlat', 'rlon', 'time')
        precip.to_netcdf('/home/sr0046/Desktop/check-this.nc')
        precip_days = xr.DataArray(precip.values, coords={'rlat':precip.rlat, 'rlon':precip.rlon, 'day':np.arange(1,precip.time.size+1), 'lat':precip.lat, 'lon':precip.lon}, dims=['rlat', 'rlon', 'day'])
        chunked = precip_days.chunk({'lat': precip.lat.size//4, 'lon': precip.lon.size//4})
        #print('cordex data get days', precip_days)
    elif precip.name != 'rfe':
        precip = precip.sel(lon=slice(42.625,50.625),lat=slice(-26.125,-11.125))
        precip = precip.transpose('lat', 'lon', 'time')
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
        chunked = precip_days.chunk({'lat': precip.lat.size//4, 'lon': precip.lon.size//4})
    else:
        precip = precip.transpose('lat', 'lon', 'time')
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
        chunked = precip_days.chunk({'lat': precip.lat.size//4, 'lon': precip.lon.size//4})

    print("the dims", chunked.dims)
    # dask_dry_spell = xr.apply_ufunc(dask_dry_spell_chunk, chunked, input_core_dims=[['day']],
    ddsc = partial(dask_dry_spell_chunk, precip=precip, thresh=thresh)
    dask_dry_spell = ddsc(chunked)
    print('type of output', type(dask_dry_spell))
    print('output', (dask_dry_spell))
    return dask_dry_spell

def dask_dry_spell_chunk(precip_days, precip, thresh):
    print('type of input chunked', type(precip_days))
    print('chunked dims', (precip_days))
    precip_pentad = pentad(precip_days)
    dry_pentad = xr.where(precip_pentad<thresh, 1, 0)
    stacked_precip = precip_days.stack(z=('lat', 'lon'))
    stacked_dry_pentad = dry_pentad.stack(z=('lat', 'lon'))
    dry_spell = xr.zeros_like(stacked_dry_pentad)
    #print('eo ary dask',dry_spell.z)
    for xy in tqdm(dry_spell.z):
        series = stacked_dry_pentad.loc[dict(z=xy)]
        #print('type of series', type(series))
        day_start = 1
        while day_start <= dry_spell.day.size-5+1:
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

from multiprocessing import Pool


def pool_dry_spells(precip, thresh):
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
    spells_at_xy = []
    # for xy in tqdm(dry_spell.z):
    #     spells_at_xy.append((xy, compute_dry_spells_at_xy(precip, xy, thresh)))
    pool = Pool(processes=4)              # start 4 worker processes
    part=partial(compute_dry_spells_at_xy, precip=precip, thresh=thresh) # prod_x has only one argument x (y is fixed to 10)
    #print('map against',list(dry_spell.z.values))
    #result = process_map(part, list(dry_spell.z.values), max_workers=2)
    result = list(tqdm(pool.imap(part, list(dry_spell.z.values)), total=dry_spell.z.size))
    # result = pool.map(part, list(dry_spell.z.values))

    # for xy_day_spell_list in result:
    #     xiy = xy_day_spell_list[0]
    #     day_spell_list = xy_day_spell_list[1:]
    #     for day0, spell in day_spell_list[1:]:
    #         dry_spell.loc[dict(day=day0,z=xiy)] = spell
    for xy, day_spell_list in zip(dry_spell.z, result):
        for day0, spell in day_spell_list:
            dry_spell.loc[dict(day=day0,z=xy)] = spell

    dry_spell = dry_spell.unstack().swap_dims({'day': 'time'})
    dry_spell_len_sum = dry_spell.sum(dim='time')
    dry_spell_zeros_ones = xr.where(dry_spell > 4, 1, 0)
    dry_spell_count = dry_spell_zeros_ones.sum(dim='time')
    dry_spell_len_ave = xr.where(dry_spell_count>0.5, dry_spell_len_sum/dry_spell_count, 0)
    dry_spell_count.name = 'dry_spell_freq'
    dry_spell_len_ave.name = 'dry_spell_ave_len'
    dry_spell = xr.merge([dry_spell_count, dry_spell_len_ave])
    # print('type necessary',type(dry_spell))
    return dry_spell

def compute_dry_spells_at_xy(xy, precip, thresh):
    if 'rlat' in precip.dims:
        precip = precip.sel(rlon=slice(42.625,50.625),rlat=slice(-26.125,-11.125))
        precip = precip.transpose('rlat', 'rlon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'rlat':precip.rlat, 'rlon':precip.rlon, 'day':np.arange(1,precip.time.size+1), 'lat':precip.lat, 'lon':precip.lon}, dims=['rlat', 'rlon', 'day'])
        #print('cordex data get days', precip_days)
    elif precip.name != 'rfe':
        precip = precip.sel(lon=slice(42.625,50.625),lat=slice(-26.125,-11.125))
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    else:
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
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
    spell_at_day = []
    series = stacked_dry_pentad.loc[dict(z=xy)]
    #print('type of series', type(series))
    day_start = 1
    while day_start <= precip_days.day.size-thresh+1:
        if stacked_precip.loc[dict(z=xy)].sum(dim='day') < thresh:   # for xy outside region, all zero, eliminate them
            break
        wet = False
        if stacked_dry_pentad.loc[dict(day=day_start,z=xy)] == 0:
            day_start += 1
            wet = True
        elif  stacked_dry_pentad.loc[dict(z=xy, day=slice(day_start, day_start+19))].sum(dim='day') >= 19.5:
            spell_len = find_spell_at_interval(series, day_start, day_start+19, precip_days.day.size+1)
        else:
            spell_len = find_spell_at_interval(series, day_start, day_start+4, day_start+19)
        if not wet:
            ## dry_spell.loc[dict(day=day_start,z=xy)] = spell_len
            spell_at_day.append((day_start, spell_len))
            day_start += spell_len
    return spell_at_day



def asafa_dry_spells(precip,thresh,model):
    #if model.name == 'tamsat':
        #xlat = np.array([-11.024999999999999,-11.0625,-11.100000000000001,-11.137500000000003,-11.174999999999997,-11.212499999999999,-11.25,-11.287500000000001,-11.325000000000003,-11.362499999999997,-11.399999999999999,-11.4375,-11.475000000000001,-11.512500000000003,-11.549999999999997,-11.587499999999999,-11.625,-11.662500000000001,-11.700000000000003,-11.737499999999997,-11.774999999999999,-11.8125,-11.850000000000001,-11.887500000000003,-11.924999999999997,-11.962499999999999,-12.0,-12.037500000000001,-12.075000000000003,-12.112499999999997,-12.149999999999999,-12.1875,-12.225000000000001,-12.262500000000003,-12.299999999999997,-12.337499999999999,-12.375,-12.412500000000001,-12.450000000000003,-12.487499999999997,-12.524999999999999,-12.5625,-12.600000000000001,-12.637500000000003,-12.674999999999997,-12.712499999999999,-12.75,-12.787500000000001,-12.825000000000003,-12.862499999999997,-12.899999999999999,-12.9375,-12.975000000000001,-13.012500000000003,-13.049999999999997,-13.087499999999999,-13.125,-13.162500000000001,-13.200000000000003,-13.237499999999997,-13.274999999999999,-13.3125,-13.350000000000001,-13.387500000000003,-13.424999999999997,-13.462499999999999,-13.5,-13.537500000000001,-13.575000000000003,-13.612499999999997,-13.649999999999999,-13.6875,-13.725000000000001,-13.762500000000003,-13.799999999999997,-13.837499999999999,-13.875,-13.912500000000001,-13.950000000000003,-13.987499999999997,-14.024999999999999,-14.0625,-14.100000000000001,-14.137500000000003,-14.174999999999997,-14.212499999999999,-14.25,-14.287500000000001,-14.325000000000003,-14.362499999999997,-14.399999999999999,-14.4375,-14.475000000000001,-14.512500000000003,-14.549999999999997,-14.587499999999999,-14.625,-14.662500000000001,-14.700000000000003,-14.737499999999997,-14.774999999999999,-14.8125,-14.850000000000001,-14.887500000000003,-14.924999999999997,-14.962499999999999,-15.0,-15.037500000000001,-15.075000000000003,-15.112499999999997,-15.149999999999999,-15.1875,-15.225000000000001,-15.262500000000003,-15.299999999999997,-15.337499999999999,-15.375,-15.412500000000001,-15.450000000000003,-15.487499999999997,-15.524999999999999,-15.5625,-15.600000000000001,-15.637500000000003,-15.674999999999997,-15.712499999999999,-15.75,-15.787500000000001,-15.825000000000003,-15.862499999999997,-15.899999999999999,-15.9375,-15.975000000000001,-16.012500000000003,-16.049999999999997,-16.0875,-16.125,-16.1625,-16.200000000000003,-16.237499999999997,-16.275,-16.3125,-16.35,-16.387500000000003,-16.424999999999997,-16.4625,-16.5,-16.5375,-16.575000000000003,-16.612499999999997,-16.65,-16.6875,-16.725,-16.762500000000003,-16.799999999999997,-16.8375,-16.875,-16.9125,-16.950000000000003,-16.987499999999997,-17.025,-17.0625,-17.1,-17.137500000000003,-17.174999999999997,-17.2125,-17.25,-17.2875,-17.325000000000003,-17.362499999999997,-17.4,-17.4375,-17.475,-17.512500000000003,-17.549999999999997,-17.5875,-17.625,-17.6625,-17.700000000000003,-17.737499999999997,-17.775,-17.8125,-17.85,-17.887500000000003,-17.924999999999997,-17.9625,-18.0,-18.0375,-18.075000000000003,-18.112499999999997,-18.15,-18.1875,-18.225,-18.262500000000003,-18.299999999999997,-18.3375,-18.375,-18.4125,-18.450000000000003,-18.487499999999997,-18.525,-18.5625,-18.6,-18.637500000000003,-18.674999999999997,-18.7125,-18.75,-18.7875,-18.825000000000003,-18.862499999999997,-18.9,-18.9375,-18.975,-19.012500000000003,-19.049999999999997,-19.0875,-19.125,-19.1625,-19.200000000000003,-19.237499999999997,-19.275,-19.3125,-19.35,-19.387500000000003,-19.424999999999997,-19.4625,-19.5,-19.5375,-19.574999999999996,-19.612499999999997,-19.65,-19.6875,-19.725,-19.762499999999996,-19.799999999999997,-19.8375,-19.875,-19.9125,-19.949999999999996,-19.987499999999997,-20.025,-20.0625,-20.1,-20.137499999999996,-20.174999999999997,-20.2125,-20.25,-20.2875,-20.324999999999996,-20.362499999999997,-20.4,-20.4375,-20.475,-20.512499999999996,-20.549999999999997,-20.5875,-20.625,-20.6625,-20.699999999999996,-20.737499999999997,-20.775,-20.8125,-20.85,-20.887499999999996,-20.924999999999997,-20.9625,-21.0,-21.0375,-21.074999999999996,-21.112499999999997,-21.15,-21.1875,-21.225,-21.262499999999996,-21.299999999999997,-21.3375,-21.375,-21.4125,-21.449999999999996,-21.487499999999997,-21.525,-21.5625,-21.6,-21.637499999999996,-21.674999999999997,-21.7125,-21.75,-21.7875,-21.824999999999996,-21.862499999999997,-21.9,-21.9375,-21.975,-22.012499999999996,-22.049999999999997,-22.0875,-22.125,-22.1625,-22.199999999999996,-22.237499999999997,-22.275,-22.3125,-22.35,-22.387499999999996,-22.424999999999997,-22.4625,-22.5,-22.5375,-22.574999999999996,-22.612499999999997,-22.65,-22.6875,-22.725,-22.762499999999996,-22.799999999999997,-22.8375,-22.875,-22.9125,-22.949999999999996,-22.987499999999997,-23.025,-23.0625,-23.1,-23.137499999999996,-23.174999999999997,-23.2125,-23.25,-23.2875,-23.324999999999996,-23.362499999999997,-23.4,-23.4375,-23.475,-23.512499999999996,-23.549999999999997,-23.5875,-23.625,-23.6625,-23.699999999999996,-23.737499999999997,-23.775,-23.8125,-23.85,-23.887499999999996,-23.924999999999997,-23.9625,-24.0,-24.0375,-24.074999999999996,-24.112499999999997,-24.15,-24.1875,-24.225,-24.262499999999996,-24.299999999999997,-24.3375,-24.375,-24.4125,-24.449999999999996,-24.487499999999997,-24.525,-24.5625,-24.6,-24.637499999999996,-24.674999999999997,-24.7125,-24.75,-24.7875,-24.824999999999996,-24.862499999999997,-24.9,-24.9375,-24.975,-25.012499999999996,-25.049999999999997,-25.0875,-25.125,-25.1625,-25.199999999999996,-25.237499999999997,-25.275,-25.3125,-25.35,-25.387499999999996,-25.424999999999997,-25.4625,-25.5,-25.5375,-25.574999999999996,-25.612499999999997,-25.65,-25.6875,-25.725,-25.762499999999996,-25.799999999999997,-25.8375,-25.875,-25.9125,-25.949999999999996,-25.987500000000004]
#, dtype=float)
    #    precip = xr.DataArray(precip.values, coords={'time':precip.time,'lon':precip.lon, 'lat':xlat[::-1]}, dims=['time', 'lat','lon'])
    if 'rlat' in precip.dims:
        precip = precip.sel(rlon=slice(42,52),rlat=slice(-26.125,-11.125))
        precip = precip.transpose('rlat', 'rlon', 'time',transpose_coords=False)
        precip.to_netcdf('/home/sr0046/Desktop/check-this.nc')
        #print('cordex data get days', precip_days)
    else:
        precip = precip.sel(lon=slice(42,52),lat=slice(-26.125,-11.125))
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)

    if 'rlat' in precip.dims:
        r1, r2, r3, r4 = np.array_split(precip.rlat, 4)
        precip1 = precip.sel(rlat=r1)
        precip2 = precip.sel(rlat=r2)
        precip3 = precip.sel(rlat=r3)
        precip4 = precip.sel(rlat=r4)
    else:
        r1, r2, r3, r4 = np.array_split(precip.lat, 4)
        precip1 = precip.sel(lat=r1)
        precip2 = precip.sel(lat=r2)
        precip3 = precip.sel(lat=r3)
        precip4 = precip.sel(lat=r4)
    print('precip1',precip1)
    pool = Pool(processes=4)              # start 4 worker processes
    part=partial(asafa_chunk, thresh=thresh) # prod_x has only one argument x (y is fixed to 10)
    #print('map against',list(dry_spell.z.values))
    #result = process_map(part, list(dry_spell.z.values), max_workers=2)
    result = list(tqdm(pool.imap(part, [precip1,precip2,precip3,precip4]), total=106800))
    if 'rlat' in precip.dims:
        return xr.concat(result, dim="rlat")
    else:
        return xr.concat(result, dim="lat")

def asafa_wet_spells(precip,thresh,model):
    #pdb.set_trace()
    #if model.name == 'tamsat':
        #xlat = np.array([-11.024999999999999,-11.0625,-11.100000000000001,-11.137500000000003,-11.174999999999997,-11.212499999999999,-11.25,-11.287500000000001,-11.325000000000003,-11.362499999999997,-11.399999999999999,-11.4375,-11.475000000000001,-11.512500000000003,-11.549999999999997,-11.587499999999999,-11.625,-11.662500000000001,-11.700000000000003,-11.737499999999997,-11.774999999999999,-11.8125,-11.850000000000001,-11.887500000000003,-11.924999999999997,-11.962499999999999,-12.0,-12.037500000000001,-12.075000000000003,-12.112499999999997,-12.149999999999999,-12.1875,-12.225000000000001,-12.262500000000003,-12.299999999999997,-12.337499999999999,-12.375,-12.412500000000001,-12.450000000000003,-12.487499999999997,-12.524999999999999,-12.5625,-12.600000000000001,-12.637500000000003,-12.674999999999997,-12.712499999999999,-12.75,-12.787500000000001,-12.825000000000003,-12.862499999999997,-12.899999999999999,-12.9375,-12.975000000000001,-13.012500000000003,-13.049999999999997,-13.087499999999999,-13.125,-13.162500000000001,-13.200000000000003,-13.237499999999997,-13.274999999999999,-13.3125,-13.350000000000001,-13.387500000000003,-13.424999999999997,-13.462499999999999,-13.5,-13.537500000000001,-13.575000000000003,-13.612499999999997,-13.649999999999999,-13.6875,-13.725000000000001,-13.762500000000003,-13.799999999999997,-13.837499999999999,-13.875,-13.912500000000001,-13.950000000000003,-13.987499999999997,-14.024999999999999,-14.0625,-14.100000000000001,-14.137500000000003,-14.174999999999997,-14.212499999999999,-14.25,-14.287500000000001,-14.325000000000003,-14.362499999999997,-14.399999999999999,-14.4375,-14.475000000000001,-14.512500000000003,-14.549999999999997,-14.587499999999999,-14.625,-14.662500000000001,-14.700000000000003,-14.737499999999997,-14.774999999999999,-14.8125,-14.850000000000001,-14.887500000000003,-14.924999999999997,-14.962499999999999,-15.0,-15.037500000000001,-15.075000000000003,-15.112499999999997,-15.149999999999999,-15.1875,-15.225000000000001,-15.262500000000003,-15.299999999999997,-15.337499999999999,-15.375,-15.412500000000001,-15.450000000000003,-15.487499999999997,-15.524999999999999,-15.5625,-15.600000000000001,-15.637500000000003,-15.674999999999997,-15.712499999999999,-15.75,-15.787500000000001,-15.825000000000003,-15.862499999999997,-15.899999999999999,-15.9375,-15.975000000000001,-16.012500000000003,-16.049999999999997,-16.0875,-16.125,-16.1625,-16.200000000000003,-16.237499999999997,-16.275,-16.3125,-16.35,-16.387500000000003,-16.424999999999997,-16.4625,-16.5,-16.5375,-16.575000000000003,-16.612499999999997,-16.65,-16.6875,-16.725,-16.762500000000003,-16.799999999999997,-16.8375,-16.875,-16.9125,-16.950000000000003,-16.987499999999997,-17.025,-17.0625,-17.1,-17.137500000000003,-17.174999999999997,-17.2125,-17.25,-17.2875,-17.325000000000003,-17.362499999999997,-17.4,-17.4375,-17.475,-17.512500000000003,-17.549999999999997,-17.5875,-17.625,-17.6625,-17.700000000000003,-17.737499999999997,-17.775,-17.8125,-17.85,-17.887500000000003,-17.924999999999997,-17.9625,-18.0,-18.0375,-18.075000000000003,-18.112499999999997,-18.15,-18.1875,-18.225,-18.262500000000003,-18.299999999999997,-18.3375,-18.375,-18.4125,-18.450000000000003,-18.487499999999997,-18.525,-18.5625,-18.6,-18.637500000000003,-18.674999999999997,-18.7125,-18.75,-18.7875,-18.825000000000003,-18.862499999999997,-18.9,-18.9375,-18.975,-19.012500000000003,-19.049999999999997,-19.0875,-19.125,-19.1625,-19.200000000000003,-19.237499999999997,-19.275,-19.3125,-19.35,-19.387500000000003,-19.424999999999997,-19.4625,-19.5,-19.5375,-19.574999999999996,-19.612499999999997,-19.65,-19.6875,-19.725,-19.762499999999996,-19.799999999999997,-19.8375,-19.875,-19.9125,-19.949999999999996,-19.987499999999997,-20.025,-20.0625,-20.1,-20.137499999999996,-20.174999999999997,-20.2125,-20.25,-20.2875,-20.324999999999996,-20.362499999999997,-20.4,-20.4375,-20.475,-20.512499999999996,-20.549999999999997,-20.5875,-20.625,-20.6625,-20.699999999999996,-20.737499999999997,-20.775,-20.8125,-20.85,-20.887499999999996,-20.924999999999997,-20.9625,-21.0,-21.0375,-21.074999999999996,-21.112499999999997,-21.15,-21.1875,-21.225,-21.262499999999996,-21.299999999999997,-21.3375,-21.375,-21.4125,-21.449999999999996,-21.487499999999997,-21.525,-21.5625,-21.6,-21.637499999999996,-21.674999999999997,-21.7125,-21.75,-21.7875,-21.824999999999996,-21.862499999999997,-21.9,-21.9375,-21.975,-22.012499999999996,-22.049999999999997,-22.0875,-22.125,-22.1625,-22.199999999999996,-22.237499999999997,-22.275,-22.3125,-22.35,-22.387499999999996,-22.424999999999997,-22.4625,-22.5,-22.5375,-22.574999999999996,-22.612499999999997,-22.65,-22.6875,-22.725,-22.762499999999996,-22.799999999999997,-22.8375,-22.875,-22.9125,-22.949999999999996,-22.987499999999997,-23.025,-23.0625,-23.1,-23.137499999999996,-23.174999999999997,-23.2125,-23.25,-23.2875,-23.324999999999996,-23.362499999999997,-23.4,-23.4375,-23.475,-23.512499999999996,-23.549999999999997,-23.5875,-23.625,-23.6625,-23.699999999999996,-23.737499999999997,-23.775,-23.8125,-23.85,-23.887499999999996,-23.924999999999997,-23.9625,-24.0,-24.0375,-24.074999999999996,-24.112499999999997,-24.15,-24.1875,-24.225,-24.262499999999996,-24.299999999999997,-24.3375,-24.375,-24.4125,-24.449999999999996,-24.487499999999997,-24.525,-24.5625,-24.6,-24.637499999999996,-24.674999999999997,-24.7125,-24.75,-24.7875,-24.824999999999996,-24.862499999999997,-24.9,-24.9375,-24.975,-25.012499999999996,-25.049999999999997,-25.0875,-25.125,-25.1625,-25.199999999999996,-25.237499999999997,-25.275,-25.3125,-25.35,-25.387499999999996,-25.424999999999997,-25.4625,-25.5,-25.5375,-25.574999999999996,-25.612499999999997,-25.65,-25.6875,-25.725,-25.762499999999996,-25.799999999999997,-25.8375,-25.875,-25.9125,-25.949999999999996,-25.987500000000004]
#, dtype=float)
    #    precip = xr.DataArray(precip.values, coords={'time':precip.time,'lon':precip.lon, 'lat':xlat[::-1]}, dims=['time', 'lat','lon'])
    if 'rlat' in precip.dims:
        precip = precip.sel(rlon=slice(42,52),rlat=slice(-26.125,-11.125))
        precip = precip.transpose('rlat', 'rlon', 'time',transpose_coords=False)
        precip.to_netcdf('/home/sr0046/Desktop/check-this.nc')
        #print('cordex data get days', precip_days)
    elif model.name != 'tamsat':
        precip = precip.sel(lon=slice(42,52),lat=slice(-26.125,-11.125))
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)

    if 'rlat' in precip.dims:
        r1, r2, r3, r4 = np.array_split(precip.rlat, 4)
        precip1 = precip.sel(rlat=r1)
        precip2 = precip.sel(rlat=r2)
        precip3 = precip.sel(rlat=r3)
        precip4 = precip.sel(rlat=r4)
    else:
        r1, r2, r3, r4 = np.array_split(precip.lat, 4)
        precip1 = precip.sel(lat=r1)
        precip2 = precip.sel(lat=r2)
        precip3 = precip.sel(lat=r3)
        precip4 = precip.sel(lat=r4)
    #print('precip1',precip1)
    pool = Pool(processes=4)              # start 4 worker processes
    part=partial(asafa_chunk_wet, thresh=thresh) # prod_x has only one argument x (y is fixed to 10)
    #print('map against',list(wet_spell.z.values))
    #result = process_map(part, list(wet_spell.z.values), max_workers=2)
    result = list(tqdm(pool.imap(part, [precip1,precip2,precip3,precip4]), total=106800))
    if 'rlat' in precip.dims:
        return xr.concat(result, dim="rlat")
    else:
        return xr.concat(result, dim="lat")

def asafa_chunk(precip, thresh):
    print("going to asafa chunk")
    if 'rlat' in precip.dims:
        precip = precip.transpose('rlat', 'rlon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'rlat':precip.rlat, 'rlon':precip.rlon, 'day':np.arange(1,precip.time.size+1), 'lat':precip.lat, 'lon':precip.lon}, dims=['rlat', 'rlon', 'day'])
        #print('cordex data get days', precip_days)
    elif precip.name != 'rfe':
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    else:
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
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
        #print('type of series', type(series))
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
    # print('type necessary',type(dry_spell))
    return dry_spell

def asafa_chunk_wet(precip, thresh):
    print("going to asafa chunk")
    if 'rlat' in precip.dims:
        precip = precip.transpose('rlat', 'rlon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'rlat':precip.rlat, 'rlon':precip.rlon, 'day':np.arange(1,precip.time.size+1), 'lat':precip.lat, 'lon':precip.lon}, dims=['rlat', 'rlon', 'day'])
        #print('cordex data get days', precip_days)
    elif precip.name != 'rfe':
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    else:
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    #precip_days = precip.swap_dims({'time': 'day'})
    precip_pentad = pentad(precip_days)
    wet_pentad = xr.where(precip_pentad>thresh, 1, 0)
    if 'rlat' in precip.dims:
        stacked_precip = precip_days.stack(z=('rlat', 'rlon'))
        stacked_wet_pentad = wet_pentad.stack(z=('rlat', 'rlon'))
    else:
        stacked_precip = precip_days.stack(z=('lat', 'lon'))
        stacked_wet_pentad = wet_pentad.stack(z=('lat', 'lon'))
    wet_spell = xr.zeros_like(stacked_wet_pentad)
    #print('eo ary dask',wet_spell.z)
    for xy in tqdm(wet_spell.z):
        series = stacked_wet_pentad.loc[dict(z=xy)]
        #print('type of series', type(series))
        day_start = 1
        while day_start <= wet_spell.day.size-5+1:
            if stacked_precip.loc[dict(z=xy)].sum(dim='day') < 1:   # for xy outside region, all zero, eliminate them
                break
            dry = False
            if stacked_wet_pentad.loc[dict(day=day_start,z=xy)] == 0:
                day_start += 1
                dry = True
            elif  stacked_wet_pentad.loc[dict(z=xy, day=slice(day_start, day_start+19))].sum(dim='day') >= 19.5:
                spell_len = find_spell_at_interval(series, day_start, day_start+19, wet_spell.day.size+1)
            else:
                spell_len = find_spell_at_interval(series, day_start, day_start+4, day_start+19)
            if not dry:
                wet_spell.loc[dict(day=day_start,z=xy)] = spell_len
                day_start += spell_len
    wet_spell = wet_spell.unstack().swap_dims({'day': 'time'})
    #print('time is',wet_spell.time)
    # if 'late' in wet_spell.dims:
    #     wet_spell = wet_spell.unstack().swap_dims({'late': 'rlat'})
    #     wet_spell = wet_spell.unstack().swap_dims({'lone': 'rlon'})
    wet_spell_len_sum = wet_spell.sum(dim='time')
    wet_spell_zeros_ones = xr.where(wet_spell > 4, 1, 0)
    wet_spell_count = wet_spell_zeros_ones.sum(dim='time')
    wet_spell_len_ave = xr.where(wet_spell_count>0.5, wet_spell_len_sum/wet_spell_count, 0)
    wet_spell_count.name = 'wet_spell_freq'
    wet_spell_len_ave.name = 'wet_spell_ave_len'
    wet_spell = xr.merge([wet_spell_count, wet_spell_len_ave])
    # print('type necessary',type(wet_spell))
    return wet_spell



def dry_spells(precip, thresh):
    if 'rlat' in precip.dims:
        precip = precip.sel(rlon=slice(42.625,52),rlat=slice(-26.125,-11.125))
        precip = precip.transpose('rlat', 'rlon', 'time',transpose_coords=False)
        precip.to_netcdf('/home/sr0046/Desktop/check-this.nc')
        precip_days = xr.DataArray(precip.values, coords={'rlat':precip.rlat, 'rlon':precip.rlon, 'day':np.arange(1,precip.time.size+1), 'lat':precip.lat, 'lon':precip.lon}, dims=['rlat', 'rlon', 'day'])
        #print('cordex data get days', precip_days)
    elif precip.name != 'rfe':
        precip = precip.sel(lon=slice(42.625,52),lat=slice(-26.125,-11.125))
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    else:
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
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
        #print('type of series', type(series))
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
                day_start += spell_len + 1 # start looking not the day after but the day after the day after
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
    # print('type necessary',type(dry_spell))
    return dry_spell

def wet_spells(precip, thresh):
    if 'rlat' in precip.dims:
        precip = precip.sel(rlon=slice(42.625,52),rlat=slice(-26.125,-11.125))
        precip = precip.transpose('rlat', 'rlon', 'time',transpose_coords=False)
        precip.to_netcdf('/home/sr0046/Desktop/check-this.nc')
        precip_days = xr.DataArray(precip.values, coords={'rlat':precip.rlat, 'rlon':precip.rlon, 'day':np.arange(1,precip.time.size+1), 'lat':precip.lat, 'lon':precip.lon}, dims=['rlat', 'rlon', 'day'])
        #print('cordex data get days', precip_days)
    elif precip.name != 'rfe':
        precip = precip.sel(lon=slice(42.625,52),lat=slice(-26.125,-11.125))
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    else:
        precip = precip.transpose('lat', 'lon', 'time',transpose_coords=False)
        precip_days = xr.DataArray(precip.values, coords={'lat':precip.lat, 'lon':precip.lon, 'day':np.arange(1,precip.time.size+1)}, dims=['lat', 'lon', 'day'])
    #precip_days = precip.swap_dims({'time': 'day'})
    precip_pentad = pentad(precip_days)
    wet_pentad = xr.where(precip_pentad>thresh, 1, 0)
    if 'rlat' in precip.dims:
        stacked_precip = precip_days.stack(z=('rlat', 'rlon'))
        stacked_wet_pentad = wet_pentad.stack(z=('rlat', 'rlon'))
    else:
        stacked_precip = precip_days.stack(z=('lat', 'lon'))
        stacked_wet_pentad = wet_pentad.stack(z=('lat', 'lon'))
    wet_spell = xr.zeros_like(stacked_wet_pentad)
    #print('eo ary dask',wet_spell.z)
    for xy in tqdm(wet_spell.z):
        series = stacked_wet_pentad.loc[dict(z=xy)]
        #print('type of series', type(series))
        day_start = 1
        while day_start <= wet_spell.day.size-5+1: # 5 for pentad
            if stacked_precip.loc[dict(z=xy)].sum(dim='day') < 1:   # for xy outside region, all zero, eliminate them
                break
            dry = False
            if stacked_wet_pentad.loc[dict(day=day_start,z=xy)] == 0:
                day_start += 1
                dry = True
            elif  stacked_wet_pentad.loc[dict(z=xy, day=slice(day_start, day_start+19))].sum(dim='day') >= 19.5:
                spell_len = find_spell_at_interval(series, day_start, day_start+19, wet_spell.day.size+1)
            else:
                spell_len = find_spell_at_interval(series, day_start, day_start+4, day_start+19)
            if not dry:
                wet_spell.loc[dict(day=day_start,z=xy)] = spell_len
                day_start += spell_len  # start looking not the day after but the day after the day after
    wet_spell = wet_spell.unstack().swap_dims({'day': 'time'})
    #print('time is',wet_spell.time)
    # if 'late' in wet_spell.dims:
    #     wet_spell = wet_spell.unstack().swap_dims({'late': 'rlat'})
    #     wet_spell = wet_spell.unstack().swap_dims({'lone': 'rlon'})
    wet_spell_len_sum = wet_spell.sum(dim='time')
    wet_spell_zeros_ones = xr.where(wet_spell > 4, 1, 0)
    wet_spell_count = wet_spell_zeros_ones.sum(dim='time')
    wet_spell_len_ave = xr.where(wet_spell_count>0.5, wet_spell_len_sum/wet_spell_count, 0)
    wet_spell_count.name = 'wet_spell_freq'
    wet_spell_len_ave.name = 'wet_spell_ave_len'
    wet_spell = xr.merge([wet_spell_count, wet_spell_len_ave])
    # print('type necessary',type(wet_spell))
    return wet_spell
