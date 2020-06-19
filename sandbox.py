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

# Counting dry spells in precipitation data
# =========================================
#
# ## What is a dry spell?
# There exists different definitions in climate research literature. We name a few here.
#
# #### **Definition 1.** A pentad (5 days period) that receives a precipitation < 5mm.
#
#
# #### **Definition 2.** A pentad (5 days period) or more that receives an average precipitation < 1mm per day
#
#
# #### **Definition 3.** A pentad (5 days period) that receives a precipitation amount < 5mm. Consecutive (overlapping) pentads are combined into one dry spell
#
#
# #### **Definition 4.** A dry spell is a period of 5 or more days, in which each day receives <1mm precipitation.
#
# Some definitions use a minimum time period of 3 instead of 5.
#
# ## Example
# Consider the precipitation pattern
#
# ```
# 0 0 0 2 2 0 2 0 1 0
# ```
#
# * With **Definition 1** this pattern counts two dry spells.
# This definition does not allow the study of dry spells duration.
# * With **Definition 2** this pattern counts one dry spell of length 10.
# Both the patterns `0 0 0 0 2 0 2 2 2` and `0 0 0 0 0 0 0 0 5 4` also count one dry spell of length 10 with this definition altought they contain a wet pentad.
# * With **Definition 3** this pattern counts one dry spell of length 6.
#
# ```
#          0 0 0 2 2 0 2 0 1 0
#          v v v v v                      (dry spell)
#            v v v v v                    (dry spell)
#              x x x x x                  (wet spell)
#
# ```
#
# * With **Definition 4** this pattern counts no dry spell.
#

# ## Implementation
#
# ### We will use xarray dataset to store the precipitation data from an netcdf file.

import xarray as xr
import numpy as np

array = xr.DataArray(np.random.normal(20,15,[2, 3, 30]), coords=[('x', ['a', 'b']), ('y', [0, 1, 2]), ('t',range(1,31))])
array.name = 'precip'
stacked = array.stack(z=('x', 'y'))
stacked_old = stacked.copy()


# +
def precip_of_spell(stacked_precip, t1, t2, xy):
    spell = stacked_precip.loc[dict(time=slice(t1, t2), z=xy)]
    return spell.sum().values

### DRY SPELL COMBINED CONSECUTIVES
days = range(1, sliced.time.size+1)
    days_da = np.array(days)
    sliced_days = sliced.assign_coords(time=days_da)

    stacked = sliced_days.stack(z=(model.lon, model.lat))
    stacked_len = xr.zeros_like(stacked)

    tmax = len(days)

    for xy in stacked.z:
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
    dry_spell_len_ave = xr.where(dry_spell_count>0, dry_spell_len_sum/dry_spell_count, 0)
    print(dry_spell_count)
    dry_spell_count = dry_spell_count.rename('spell_count')
    dry_spell_len = dry_spell_len_ave.rename('ave_spell_len')
    dry_spells = xr.merge([dry_spell_count, dry_spell_len])
    return dry_spells

# +
import xarray as xr
import numpy as np
import pdb
array = xr.DataArray(np.random.normal(20,15,[2, 3, 30]), coords=[('x', ['a', 'b']), ('y', [0, 1, 2]), ('t',range(1,31))])
array.name = 'precip'
stacked = array.stack(z=('x', 'y'))
stacked_old = stacked.copy()

def precip_of_spell(t1, t2, xy):
    spell = stacked.loc[dict(t=slice(t1, t2), z=xy)]
    return spell.sum().values

### DRY SPELL UNCOMBINED CONSECUTIVES
for xy in stacked.z:
    time_lead = 1
    spell_thresh_len = 2
    prec_thresh = 15
    while time_lead < 31:
        wet = False
        dry_spell_len = 4
        extra_dry = (spell_thresh_len - 1) - dry_spell_len
        while time_lead + dry_spell_len < 30 and not wet:
            if precip_of_spell(time_lead + extra_dry, time_lead + extra_dry + spell_thresh_len, xy) < prec_thresh:
                dry_spell_len += 1
            else:
                wet = True
        if dry_spell_len < spell_thresh_len:
            stacked.loc[dict(t=time_lead, z=xy)] = 0
            time_lead+=1
        if dry_spell_len >= spell_thresh_len:
            stacked.loc[dict(t=time_lead, z=xy)] = dry_spell_len
            time_lead += dry_spell_len
print(stacked.loc[dict(z=('a',0))].values)
print(stacked_old.loc[dict(z=('a',0))].values)
print(stacked.loc[dict(t=slice(1, 10), z=('a',0))])
print(stacked)
# -

stacked.sel(t=time_lead, z=xy)


print(stacked.z)



array.groupby('t')
threshold = 1
array.where(array < 1, 1)
array = array.where(array > 1, 0)
print(array)

# +
import xarray as xr
import numpy as np
import pdb
array = xr.DataArray(np.random.normal(20,15,[2, 3, 10]), coords=[('x', ['a', 'b']), ('y', [0, 1, 2]), ('t',range(1,11))])
array.name = 'precip'
stacked = array.stack(z=('x', 'y'))
stacked_old = stacked.copy()

def precip_of_spell(t1, t2, xy):
    spell = stacked.loc[dict(t=slice(t1, t2), z=xy)]
    return spell.sum().values

### DRY SPELL COMBINED CONSECUTIVES
for xy in stacked.z:
    time_lead = 1
    spell_thresh_len = 3
    prec_thresh = 30
    while time_lead < 11:
        long_dry_spell = 0
        end_of_spell = False
        new_time_lead = time_lead
        while not end_of_spell:
            wet = False
            dry_spell = 0

            while new_time_lead+dry_spell < 10 and not wet:
                if precip_of_spell(new_time_lead, new_time_lead+dry_spell, xy) < prec_thresh:
                    dry_spell += 1
                else:
                    wet = True
            long_dry_spell += dry_spell
            new_time_lead = time_lead + long_dry_spell
            if not wet:
                end_of_spell = True
            elif precip_of_spell(new_time_lead, new_time_lead+spell_thresh_len-1, xy) >= prec_thresh:
                end_of_spell = True

        if long_dry_spell < spell_thresh_len:
            stacked.loc[dict(t=time_lead, z=xy)] = 0
            time_lead += 1
        else:
            for tt in range(time_lead, time_lead+long_dry_spell+1):
                stacked.loc[dict(t=tt, z=xy)] = long_dry_spell
            time_lead += long_dry_spell



print(stacked.loc[dict(z=('a',0))].values)
print(stacked_old.loc[dict(z=('a',0))].values)
print(stacked.loc[dict(t=slice(1, 10), z=('a',0))])
print(stacked)


# -

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
secs_to_dhms(26)




