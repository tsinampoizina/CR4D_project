---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

 The average precipitation over wet days for each period  [mm/d or kg/m²/s]
 -------------------------------------------------------------------------
 
 Let $\mathbf{p}=𝑝_0,𝑝_1,\ldots,𝑝_𝑛$ be the daily precipitation and $\mathit{threshold}$ be the precipitation threshold to define a wet day. Then the average rainfall intensity over wet days is defined as
 
 $$
 \frac{\sum_{i\le n} p_i {\mathbf{1}}_{p_i\ge \mathit{threshold}}}{\sum_{i\le n} {\mathbf{1}}_{p_i\ge \mathit{threshold}}}
 $$
 
where ${\mathbf{1}}_{p_i\ge \mathit{threshold}} = 1$ if $p_i\ge \mathit{threshold}$ and $0$ otherwise.


Computing dry spells over a season (DJFM, or whole year, or April-October)
--------------------------------------------------------------------------
1. First slice each year into the season


```python
INPUT_FOLDER = '/home/sr0046/Documents/asa_sophie/Cordex-Mada/data-region/'
def file_name(year):
    fname = INPUT_FOLDER + model.name + '/' + model.version + '-region-' +\
            str(year) + model.ext
    return fname
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
```
