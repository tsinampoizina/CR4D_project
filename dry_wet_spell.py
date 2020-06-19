# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# The average precipitation over wet days for each period  [mm/d or kg/m²/s]
#  -------------------------------------------------------------------------
#
#  Let $\mathbf{p}=𝑝_0,𝑝_1,\ldots,𝑝_𝑛$ be the daily precipitation and
# $\mathit{threshold}$ be the precipitation threshold to define a wet day.
# Then the average rainfall intensity over wet days is defined as
#
#  $$
#  \frac{\sum_{i\le n} p_i {\mathbf{1}}_{p_i\ge \mathit{threshold}}}
# {\sum_{i\le n} {\mathbf{1}}_{p_i\ge \mathit{threshold}}}
#  $$
#
# where ${\mathbf{1}}_{p_i\ge \mathit{threshold}} =
# 1$ if $p_i\ge \mathit{threshold}$ and $0$ otherwise.
#

# Computing dry spells over a season (DJFM, or whole year, or April-October)
# --------------------------------------------------------------------------
# 1. First slice each year into the season

# +

import xarray as xr
import numpy as np
from collections import namedtuple

INPUT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'

def file_name(year):
    fname = INPUT_FOLDER + model.name + '/' + model.version + '-region-' +\
            str(year) + model.ext
    return fname

def get_by_file(filex):
    '''Get data from filename, slice using Madagascar coordinates'''
    ds = xr.open_dataset(filex)
    return ds


# -

Charact = namedtuple('Charact', 'name contour unit threshold min_val max_val')
wd_freq1 = Charact('WDF', np.arange(10, 330, 10, dtype=float), 'days', 1, 0, 1.2)
wd_freq30 = Charact('WDF', np.arange(0, 48, 2, dtype=float), 'days', 30, 0, 1)
tot_prec = Charact('TOTAL-RAINFALL', np.arange(200, 4800, 200, dtype=float), 'mm', '', 0, 1.5)
ave_daily = Charact('AVERAGE-DAILY-RAINFALL', np.arange(0, 22, 1, dtype=float), 'mm', '', 0, 1)
dry_spell = Charact('DRY-SPELLS', np.arange(0, 22, 1, dtype=float), 'mm', '', 0, 1)
charac = dry_spell
Model = namedtuple('Model', 'name version ext lat lon precip plot_pos date_example time')

# +
XYZ = [151, 152, 153, 154, 155]
EXAMPLE_DATES = [360, '2000-10-30']
trmm = Model('trmm', 'TRMM_3B42', '.nc4', 'lat', 'lon', 'precipitation', XYZ[0], EXAMPLE_DATES[0],
             'time')
chirps = Model('chirps', 'chirps-v2', '.nc', 'latitude', 'longitude', 'precip', XYZ[1],
               EXAMPLE_DATES[1], 'time')
gpcc = Model('gpcc', 'gpcc_v2018', '.nc', 'lat', 'lon', 'precip', XYZ[4], EXAMPLE_DATES[1], 'time')

PROJECT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada'
PLOT_FOLDER = PROJECT_FOLDER + '/plot-images/climatology-' + charac.name + str(
    charac.threshold) + '/'
model = gpcc
# -
if model.name == 'arc2':
    YEARS = range(2001, 2010)
else:
    YEARS = range(1999, 2010)
Season = namedtuple('Season', 'name months')
djfm = Season('DJFM', [1, 2, 3, 12])
all_year = Season('all-year', range(1, 13))
amjjaso = Season('Apr-Oct', [4, 5, 6, 7, 8, 9, 10])
seasons = [djfm, amjjaso]
seasons = [all_year]
season = all_year



plot_filename = 'climatology-'+ charac.name + str(charac.threshold) + \
'-' + season.name + '-' + str(YEARS[0]) + '-' + str(YEARS[-1])
file = file_name(2000)
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

# Split your data into multiple independents grid $(x,y)\in \text{lat}\times \text{lon}$ 

stacked = precip.stack(z=('lat', 'lon'))
print(stacked)

#
