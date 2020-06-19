#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:06:47 2020

@author: sr0046
"""
from collections import namedtuple
import numpy as np

Model = namedtuple('Model', 'name version ext lat lon precip plot_pos date_example time year_range grid_lat_out grid_lon_out')
XYZ = [151, 152, 153, 154, 155]
EXAMPLE_DATES = [360,  '2001-11-30']
trmm = Model(name='trmm',
             version='TRMM_3B42',
             ext='.nc4',
             lat='lat',
             lon='lon',
             precip='precipitation',
             plot_pos=XYZ[0],
             date_example=EXAMPLE_DATES[0], # need to go out
             time='time',
             year_range=range(1998,2017+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
chirps = Model(name='chirps',
               version='chirps-v2',
               ext='.nc',
               lat='latitude',
               lon='longitude',
               precip='precip',
               plot_pos=XYZ[1],
               date_example=EXAMPLE_DATES[1],
               time='time',
               year_range=range(1981,2017+1),
               grid_lat_out = np.arange(-26.125, -11, 0.25),
               grid_lon_out = np.arange(42.125, 54.25, 0.25)
               )
ichirps = Model(name='chirps',
               version='interpolated-chirps-v2',
               ext='.nc',
               lat='lat',
               lon='lon',
               precip='precip',
               plot_pos=XYZ[1],
               date_example=EXAMPLE_DATES[1],
               time='time',
               year_range=range(1981,2017+1),
               grid_lat_out = np.arange(-26.125, -11, 0.25),
               grid_lon_out = np.arange(42.125, 54.25, 0.25)
               )
gpcc = Model(name='gpcc',
             version='gpcc_v2018',
             ext='.nc',
             lat='lat',
             lon='lon',
             precip='precip',
             plot_pos=XYZ[4],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(1982,2016+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
tamsat = Model(name='tamsat',
               version='rfe_v3',
               ext='.nc',
               lat='lat',
               lon='lon',
               precip='rfe',
               plot_pos=XYZ[2],
               date_example=EXAMPLE_DATES[1],
               time='time',
               year_range=range(1983,2019+1),
               grid_lat_out = np.arange(-26.125, -11, 0.25),
               grid_lon_out = np.arange(42.125, 54.25, 0.25)
               )
arc2 = Model(name='arc2',
             version='arc2',
             ext='.nc',
             lat='Y',
             lon='X',
             precip='est_prcp',
             plot_pos=XYZ[3],
             date_example=EXAMPLE_DATES[1],
             time='T',
             year_range=range(2001,2019+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
