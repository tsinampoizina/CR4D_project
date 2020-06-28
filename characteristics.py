import numpy as np
from collections import namedtuple

Season = namedtuple('Season', 'name months')
djfm = Season('DJFM', [1,2,3,12])
all_year = Season('all-year', range(1, 13))
amjjaso = Season('Apr-Oct', [4,5,6,7,8,9,10])

Characteristic = namedtuple('Characteristic', 'name long_name contour unit threshold min_val max_val ticks_step variable_name')

def wd_freq1(season):
    if season == djfm:
        wd_freq1 = Characteristic('WDF', 'Wet days ($\geq 1mm$)',np.arange(0, 330, 20, dtype=float), 'days', 1, 0,1, np.arange(10, 330, 20, dtype=float), 'precip')
    elif season == all_year:
        wd_freq1 = Characteristic('WDF', 'Wet days ($\geq 1mm$)',np.arange(0, 330, 20, dtype=float), 'days', 1, 0,1, np.arange(10, 330, 20, dtype=float), 'precip')
    elif season == amjjaso:
        wd_freq1 = Characteristic('WDF', 'Wet days ($\geq 1mm$)',np.arange(0, 330, 20, dtype=float), 'days', 1, 0,1, np.arange(10, 330, 20, dtype=float), 'precip')
    return wd_freq1

def wd_freq30(season):
    if season == djfm:
        wd_freq30 = Characteristic('WDF', 'Heavy rain days ($\geq 30mm$)',np.arange(0, 42, 2, dtype=float), 'days', 30,0,1, np.arange(0, 42, 2, dtype=float), 'precip')
    elif season == all_year:
        wd_freq30 = Characteristic('WDF', 'Heavy rain days ($\geq 30mm$)',np.arange(0, 42, 2, dtype=float), 'days', 30,0,1, np.arange(0, 42, 2, dtype=float), 'precip')
    elif season == amjjaso:
        wd_freq30 = Characteristic('WDF', 'Heavy rain days ($\geq 30mm$)',np.arange(0, 42, 2, dtype=float), 'days', 30,0,1, np.arange(0, 42, 2, dtype=float), 'precip')
    return wd_freq30



total_precip = Characteristic('TOTAL-RAINFALL','TOTAL-RAINFALL',
                                np.arange(200,4800, 200, dtype=float), 'mm',
                                '',0,1,np.arange(200,4800, 200, dtype=float), 'precip')
average_daily = Characteristic('AVERAGE-DAILY-RAINFALL','AVERAGE-DAILY-RAINFALL',
                                np.arange(0, 22, 1, dtype=float), 'mm',
                                '',0,1,np.arange(0, 22, 1, dtype=float), 'precip')
def dry_spell_freq(season):
    if season == djfm:
        dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', 'Frequency of dry spells',np.arange(0, 9, 0.5, dtype=float),
                                'spells', 5, 0, 1, np.array([1,2,3,4,5,6,7,8]), 'dry_spell_freq')
    if season == all_year:
        dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', 'Frequency of dry spells',np.arange(0, 29, 1, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 29, 2, dtype=float), 'dry_spell_freq')
    if season == amjjaso:
        dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', 'Frequency of dry spells',np.arange(0, 17, 1, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 17, 1, dtype=float), 'dry_spell_freq')
    return dry_spell_freq

def wet_spell_freq(season):
    if season == djfm:
        wet_spell_freq = Characteristic('WET-SPELL-FREQUENCY', 'Frequency of wet spells',np.arange(0, 9, 0.5, dtype=float),
                                'spells', 5, 0, 1, np.array([1,2,3,4,5,6,7,8]), 'wet_spell_freq')
    if season == all_year:
        wet_spell_freq = Characteristic('WET-SPELL-FREQUENCY', 'Frequency of wet spells',np.arange(0, 29, 1, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 29, 2, dtype=float), 'wet_spell_freq')
    if season == amjjaso:
        wet_spell_freq = Characteristic('WET-SPELL-FREQUENCY', 'Frequency of wet spells',np.arange(0, 17, 1, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 17, 1, dtype=float), 'wet_spell_freq')
    return wet_spell_freq

def dry_spell_ave_len(season):
    if season == djfm:
        dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', 'Average length of a dry spell',np.arange(0, 20, 1, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 20, 2, dtype=float), 'dry_spell_ave_len')
    if season == all_year:
        dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', 'Average length of a dry spell',np.arange(0, 30, 1.5, dtype=float),
                                'spells', 5, 0, 1.5, np.arange(0, 30, 5, dtype=float), 'dry_spell_ave_len')
    if season == amjjaso:
        dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', 'Average length of a dry spell',np.arange(0, 151, 10, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 151, 10, dtype=float), 'dry_spell_ave_len')
    return dry_spell_ave_len
