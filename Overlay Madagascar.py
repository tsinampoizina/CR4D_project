#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:08:26 2020

@author: sr0046
"""

import xarray as xr
import numpy as np
import time



from matplotlib import pyplot as plt

from cartopy.feature import ShapelyFeature, OCEAN, LAKES
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import Reader as ShapeReader, natural_earth

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

LON = 'lon'
LAT = 'lat'
#ds_disk = xr.open_dataset('/media/sr0046/WD-exFAT-50/DATA/chirps/chirps-v2.0.2000.days_p05.nc')
#dir = '/media/sr0046/WD-exFAT-50/CORDEX/historical/DMI-HIRHAM5_v1/'
dir = '/media/sr0046/WD-exFAT-50/CORDEX/evaluation/MOHC-HadRM3P_v1/ECMWF-ERAINT_r1i1p1/day/native/'
file = 'pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_day_19910101-19951231.nc'
ds_disk = xr.open_dataset(dir+file)
#ds_disk = ds_disk.sel(lon=slice(42.125,54.125),lat=slice(-26.399999618530273,-11.125))
lat = ds_disk["lat"]
lon = ds_disk["lon"]
date = ds_disk["time"]
time_pan = date.to_dataframe()

precip = ds_disk["pr"]
precip = precip.sel(time='1995-01-01 12:00:00')
precip_pan = precip.to_dataframe()
lon_pan = lon.to_dataframe()
lat_pan = lat.to_dataframe()
ds = ds_disk

t0 = time.time()

# Download the Natural Earth shapefile for country boundaries at 10m resolution
shapefile = natural_earth(category='cultural',
                          resolution='10m',
                          name='admin_0_countries')

# Sort the geometries in the shapefile into Chinese/Taiwanese or other
country_geos = []
other_land_geos = []
for record in ShapeReader(shapefile).records():
    if record.attributes['ADMIN'] in ['Madagascar']:
        country_geos.append(record.geometry)
    else:
        other_land_geos.append(record.geometry)

# Define map projection to allow Cartopy to transform ``lat`` and ``lon`` values accurately into points on the
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

# Extract the Chinese province borders
province_geos = [record.geometry for record in ShapeReader(shapefile).records()
                 if record.attributes['admin'] == 'Madagascar']

# Define a Cartopy Feature for the province borders, so they can be easily plotted
provinces = ShapelyFeature(province_geos,
                           crs=projection,
                           facecolor='none',
                           edgecolor='black',
                           lw=0.25)


# Generate figure (set its size (width, height) in inches) and axes using Cartopy
fig = plt.figure(figsize=(12,15))
ax = plt.axes(projection=projection)

ax.set_extent([42, 52, -26, -11], crs=projection)

# Define the contour levels
clevs = np.arange(0, 50, 3, dtype=float)

# Import an NCL colormap, truncating it by using geocat.viz.util convenience function
newcmp = gvutil.truncate_colormap(gvcmaps.precip_diff_12lev, minval=0, maxval=1, n=len(clevs))

# Draw the temperature contour plot with the subselected colormap
# (Place the zorder of the contour plot at the lowest level)
cf = ax.contourf(lon, lat, precip*86400, levels=clevs, cmap=newcmp, zorder=1)

# Draw horizontal color bar
cax = plt.axes((0.14, 0.08, 0.74, 0.02))
cbar = plt.colorbar(cf, ax=ax, cax=cax, ticks=clevs[1:-1], drawedges=True, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)

# Add the land mask feature on top of the contour plot (higher zorder)
ax.add_feature(land_mask, zorder=2)

# Add the OCEAN and LAKES features on top of the contour plot
ax.add_feature(OCEAN.with_scale('50m'), edgecolor='black', lw=1, zorder=2)
ax.add_feature(LAKES.with_scale('50m'), edgecolor='black', lw=1, zorder=2)

# Add the country and province features (which are transparent) on top
ax.add_feature(countries, zorder=3)
ax.add_feature(provinces, zorder=3)

# Use geocat.viz.util convenience function to set axes tick values
gvutil.set_axes_limits_and_ticks(ax, xticks=[45, 50], yticks=[-25, -20, -15])

# Use geocat.viz.util convenience function to make plots look like NCL plots by using latitude, longitude tick labels
gvutil.add_lat_lon_ticklabels(ax)

# Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax, x_minor_per_major=4, y_minor_per_major=5, labelsize=18)

# Use geocat.viz.util convenience function to add main title as well as titles to left and right of the plot axes.
gvutil.set_titles_and_labels(ax, lefttitle="Precipitation", lefttitlefontsize=20)

# End timing
t1 = time.time()

time_total = t1-t0

# Show the plot
print(time_total)
plt.show()