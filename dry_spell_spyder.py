import xarray as xr
import numpy as np
import string
import pdb
tmax = 10
array = xr.DataArray(np.random.normal(20,15,[26, 40, tmax]),
                     coords=[('x', list(string.ascii_lowercase)), ('y', range(1,41)), ('t',range(1,tmax+1))])
array.name = 'precip'
stacked = array.stack(z=('x', 'y'))
stacked_old = stacked.copy()

def precip_of_spell(t1, t2, xy):
    spell = stacked.loc[dict(t=slice(t1, t2), z=xy)]
    return spell.sum().values

### DRY SPELL COMBINED CONSECUTIVES
for xy in stacked.z:
    time_lead = 1
    spell_thresh = 2
    prec_thresh = 40
    while time_lead <= tmax:
        long_dry_spell = 0
        end_of_spell = False
        new_time_lead = time_lead
        while not end_of_spell:
            wet = False
            dry_spell = 0

            while new_time_lead+dry_spell < tmax and not wet:
                if precip_of_spell(new_time_lead, new_time_lead+dry_spell, xy) < prec_thresh:
                    dry_spell += 1
                else:
                    wet = True
            long_dry_spell += dry_spell
            new_time_lead = time_lead + long_dry_spell
            if not wet:
                end_of_spell = True
            elif precip_of_spell(new_time_lead, new_time_lead+spell_thresh-1, xy) >= prec_thresh:
                end_of_spell = True

        if long_dry_spell < spell_thresh:
            stacked.loc[dict(t=time_lead, z=xy)] = 0
            time_lead += 1
        else:
            for tt in range(time_lead, time_lead+long_dry_spell+1):
                stacked.loc[dict(t=tt, z=xy)] = long_dry_spell
            time_lead += long_dry_spell
    print(stacked.loc[dict(z=xy)].values)
    print(stacked_old.loc[dict(z=xy)].values)



