#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:11:40 2020

@author: sr0046
"""

import xarray as xr
import numpy as np
import pdb
import string
import time

t0 = time.time()
tmax = 10
precip = xr.DataArray(np.random.normal(20, 15, [26, 10, tmax]),
                      coords=[('x', list(string.ascii_lowercase)), ('y', range(1, 11)),
                              ('t', range(1, tmax + 1))])
precip.name = 'precip'
stacked = precip.stack(z=('x', 'y'))
stacked_len = xr.zeros_like(stacked)


def precip_of_spell(t1, t2, xy):
    spell = stacked.loc[dict(t=slice(t1, t2), z=xy)]
    return spell.sum().values


# DRY SPELL UNCOMBINED CONSECUTIVES
for xy in stacked.z:
    time_lead = 1
    spell_len_thresh = 3
    precip_thresh = 30
    while time_lead <= tmax:
        wet = False
        dry_spell = 0
        while time_lead + dry_spell < tmax and not wet:
            if precip_of_spell(time_lead, time_lead + dry_spell, xy) < precip_thresh:
                dry_spell += 1
            else:
                wet = True
        if dry_spell < spell_len_thresh:
            stacked_len.loc[dict(t=time_lead, z=xy)] = 0
            time_lead += 1
        if dry_spell >= spell_len_thresh:
            stacked_len.loc[dict(t=time_lead, z=xy)] = dry_spell
            time_lead += dry_spell
    print(stacked.loc[dict(z=xy)].values)
    print(stacked_len.loc[dict(z=xy)].values)
t1 = time.time()
stacked_count = xr.where(stacked_len > .99, 1, 0)
dry_spell_count = stacked_count.unstack().sum(dim='t')
dry_spell_ave_len = stacked_len.unstack().mean(dim='t')
print('time for', tmax, ':', t1 - t0)
