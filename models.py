#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:06:47 2020

@author: sr0046
"""
from collections import namedtuple
import numpy as np

Model = namedtuple('Model', 'name version ext lat lon precip plot_pos date_example time year_range grid_lat_out grid_lon_out')
ABC = [(3,5,1), (3,5,2), (3,5,3), (3,5,4), (3,5,5)]
EXAMPLE_DATES = [360,  '2001-11-30']

chirps = Model(name='chirps',
               version='chirps-v2',
               ext='.nc',
               lat='latitude',
               lon='longitude',
               precip='precip',
               plot_pos=ABC[1],
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
               plot_pos=ABC[1],
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
             plot_pos=ABC[0],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(1982,2016+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
trmm = Model(name='trmm',
             version='TRMM_3B42',
             ext='.nc',
             lat='lat',
             lon='lon',
             precip='precipitation',
             plot_pos=ABC[2],
             date_example=EXAMPLE_DATES[0], # need to go out
             time='time',
             year_range=range(1998,2017+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
tamsat = Model(name='tamsat',
               version='rfe_v3',
               ext='.nc',
               lat='lat',
               lon='lon',
               precip='rfe',
               plot_pos=ABC[3],
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
             plot_pos=ABC[4],
             date_example=EXAMPLE_DATES[1],
             time='T',
             year_range=range(2001,2019+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )

XYZ = [(3,5,6), (3,5,7), (3,5,8), (3,5,9), (3,5,10), (3,5,11), (3,5,12), (3,5,13), (3,5,14)]
clm = Model(name='cordex-clm',
             version='CLMcom-CCLM4-8-17_v1',
             ext='.nc',
             lat='rlat',
             lon='rlon',
             precip='pr',
             plot_pos=XYZ[0],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
rm3p = Model(name='rm3p',
             version='MOHC-HadRM3P_v1',
             ext='.nc',
             lat='lat',
             lon='lon',
             precip='pr',
             plot_pos=XYZ[3],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )

gem3 = Model(name='gem3',
             version='MOHC-HadGEM3-RA_v1',
             ext='.nc',
             lat='lat',
             lon='lon',
             precip='pr',
             plot_pos=XYZ[8],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
hihram = Model(name='hirham5',
             version='DMI-HIRHAM5_v2',
             ext='.nc',
             lat='rlat',
             lon='rlon',
             precip='pr',
             plot_pos=XYZ[1],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
remo = Model(name='remo',
             version='MPI-CSC-REMO2009_v1',
             ext='.nc',
             lat='rlat',
             lon='rlon',
             precip='pr',
             plot_pos=XYZ[5],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
crcm5 = Model(name='crcm5',
             version='UQAM-CRCM5_v1',
             ext='.nc',
             lat='rlat',
             lon='rlon',
             precip='pr',
             plot_pos=XYZ[7],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
racmo = Model(name='racmo',
             version='KNMI-RACMO22T_v1',
             ext='.nc',
             lat='rlat',
             lon='rlon',
             precip='pr',
             plot_pos=XYZ[2],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )
rca4 = Model(name='rca4',
             version='SMHI-RCA4_v1',
             ext='.nc',
             lat='rlat',
             lon='rlon',
             precip='pr',
             plot_pos=XYZ[6],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(2001,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )

ens = Model(name='ens',
             version='Ensemble Mean',
             ext='.nc',
             lat='rlat',
             lon='rlon',
             precip='pr',
             plot_pos=XYZ[4],
             date_example=EXAMPLE_DATES[1],
             time='time',
             year_range=range(1999,2008+1),
             grid_lat_out = np.arange(-26.125, -11, 0.25),
             grid_lon_out = np.arange(42.125, 54.25, 0.25)
             )

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

