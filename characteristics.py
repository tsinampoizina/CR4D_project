import numpy as np
from collections import namedtuple

Season = namedtuple('Season', 'name months')
djfm = Season('DJFM', [1,2,3,12])
all_year = Season('all-year', range(1, 13))
amjjaso = Season('Apr-Oct', [4,5,6,7,8,9,10])

Characteristic = namedtuple('Characteristic', 'name contour unit threshold min_val max_val ticks_step variable_name')
wd_freq1 = Characteristic('WDF', np.arange(10, 330, 10, dtype=float), 'days', 1, 0,1.2, np.arange(10, 330, 10, dtype=float), 'precip')
wd_freq30 = Characteristic('WDF', np.arange(0, 48, 2, dtype=float), 'days', 30,0,1, np.arange(0, 48, 2, dtype=float), 'precip')
total_precip = Characteristic('TOTAL-RAINFALL',
                                np.arange(200,4800, 200, dtype=float), 'mm',
                                '',0,1.5,np.arange(200,4800, 200, dtype=float), 'precip')
average_daily = Characteristic('AVERAGE-DAILY-RAINFALL',
                                np.arange(0, 22, 1, dtype=float), 'mm',
                                '',0,1,np.arange(0, 22, 1, dtype=float), 'precip')
def dry_spell_freq(season):
    if season == djfm:
        dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', np.arange(0, 10.5, 0.5, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 15, 1, dtype=float), 'dry_spell_freq')
    if season == all_year:
        dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', np.arange(0, 29, 1, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 29, 2, dtype=float), 'dry_spell_freq')
    if season == amjjaso:
        dry_spell_freq = Characteristic('DRY-SPELL-FREQUENCY', np.arange(0, 19, 1, dtype=float),
                                'spells', 5, 0, 1, np.arange(0, 19, 1, dtype=float), 'dry_spell_freq')
    return dry_spell_freq

def dry_spell_ave_len(season):
    if season == djfm:
        dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', np.arange(5, 18, 0.25, dtype=float),
                                'spells', 5, 0, 1.4, np.arange(5, 18, 1, dtype=float), 'dry_spell_ave_len')
    if season == all_year:
        dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', np.arange(5, 30, 1, dtype=float),
                                'spells', 5, 0, 1.5, np.arange(5, 30, 5, dtype=float), 'dry_spell_ave_len')
    if season == amjjaso:
        dry_spell_ave_len = Characteristic('DRY-SPELL-LENGTH', np.arange(5, 151, 7.5, dtype=float),
                                'spells', 5, 0, 1, np.arange(5, 151, 15, dtype=float), 'dry_spell_ave_len')
    return dry_spell_ave_len
