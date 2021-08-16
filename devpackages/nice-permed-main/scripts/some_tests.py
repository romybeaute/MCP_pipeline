#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:08:51 2021

@author: dragana.manasova
"""
import numpy as np
import mne

fname = '/home/dragana.manasova/Documents/data/permed/germany/subjects/p03_lg/' \
        'D-09pTZB_LG_1_20210420_113254_lg_patient_permed-lg-egi400.mff'

raw = mne.io.read_raw_egi(fname, preload=True, verbose=True)

din_ch_names = ['DIN2', 'DIN3', 'DIN4', 'DIN5', 'DIN6', 'DIN7', 'DIN8']
stims_data = raw.copy().pick(din_ch_names).get_data().astype(np.int)


# Check this logic:
stims_data = (stims_data != 0).astype(np.int)
trig_mult = np.array([0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])
raw_trig_data = np.multiply(stims_data.T, trig_mult).T
sumi = np.sum(raw_trig_data, axis=0)  # sum of all the values in one sample

# in the case of 184, the value was encoded in two consecutive samples,
# so I sum the two samples in the ith place and add a 0 in the i+1th place
# for i in range(sumi.shape[0] - 1):
#     if (sumi[i] != 0) and (sumi[i + 1] != 0) and (sumi[i + 2] == 0): # 2 consecutive values
#         suma = sumi[i] + sumi[i + 1]
#         sumi[i] = suma
#         sumi[i + 1] = 0
#     if (sumi[i] != 0) and (sumi[i + 1] != 0) and (sumi[i + 2] != 0): # 3 consecutive values
#         suma = sumi[i] + sumi[i + 1] + sumi[i + 2]
#         sumi[i] = suma
#         sumi[i + 1] = 0
#         sumi[i + 2] = 0


# for i in range(sumi.shape[0] - 1):
#     if sumi[i] !=0:
#         for j in range(5): # check the next 5 samples
            

# 
k=10 # size of the sliding window
for i in range(len(sumi) - k + 1):
    if sumi[i] != 0:
        sliding_list=[]
        for j in range(k):
            sliding_list.append(sumi[i + j])
        # Check if there are non-zero elements
        non_zero_el_idx = np.where(np.array(sliding_list) != 0)[0]
        non_zero_el_num = len(non_zero_el_idx) # number of elements that are non-0 in the sliding window
        non_zero_el_vals = np.array(sliding_list)[non_zero_el_idx]
        if non_zero_el_num > 1:
            flag = len(set(non_zero_el_vals)) == len(non_zero_el_vals) # checks if all the values are original, True is original
            if flag == True: 
                sum_sliding = np.sum(sliding_list) # sum the vals in the sliding window
                sumi[i] = sum_sliding
                for l in range(1,k):
                    sumi[i+l] = 0 # set all the rest of the values to 0
            


# Should I remove the first staircase of events? Yes!
find_seq = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
seq_len = len(find_seq)
non_zero_idx = np.where(sumi != 0)[0]
non_zero_vals = sumi[non_zero_idx]
max_samples = int(raw.info['sfreq'])


am=sumi[650*500:]
#%%
# I need the non_zero_vals and a column by it with the event times

triggers_times = np.column_stack((non_zero_vals, non_zero_idx))

#%%
found_seq = []
# look for the ocurrences of the sequence
for i in range(len(non_zero_vals) - seq_len + 1):
    if all(non_zero_vals[i:i + seq_len] == find_seq):
        # If the sequence match, get the IDX of the beggining and the end
        idx_end = non_zero_idx[i + seq_len - 1]

        # The start is the last time that value appears, get the first time
        idx_st = non_zero_idx[i]

        st_val = sumi[idx_st]
        while (sumi[idx_st] == st_val):
            idx_st -= 1

        if idx_end - idx_st < max_samples:
            found_seq.append((idx_st + 1, idx_end))

    if len(found_seq) == 0:
        logger.warn('Warning: check sequence not found.')
    else:
        logger.info('Found {} check sequences.'.format(len(found_seq)))

    for seq_st, seq_end in found_seq:
        sumi[seq_st:seq_end + 1] = 0

    sumi = sumi & 0xF8  # remove ignored bits
    # map the values from arduino to the other values used in the LG paradigm
    trig_data = np.zeros_like(sumi)
    for k, v in _arduino_trigger_map.items():
        mask = sumi == k
        t_v = _icm_lg_event_id[v]
        trig_data[mask] = t_v

    # sumi_changed_numbers = [_egi400_trigger_map_values[i] for i in sumi]

    if 'STI 014' not in raw.ch_names:
        stim_name = [
            x['ch_name'] for x in raw.info['chs']
            if x['kind'] == mne.io.constants.FIFF.FIFFV_STIM_CH][0]
        mne.rename_channels(raw.info, {stim_name: 'STI 014'})
        din_ch_names.remove(stim_name)
    stim = mne.pick_channels(raw.info['ch_names'], include=['STI 014'])
    raw._data[stim] = trig_data
    raw.event_id = _icm_lg_event_id
    raw.drop_channels(din_ch_names)
    
#%%
#%%    
#%% Germany rs
import mne
from mne.utils import logger
from nice_ext.api.io import register_suffix, _check_io_suffix, _check_path
from nice_ext.equipments import get_montage



path = '/home/dragana.manasova/Documents/data/permed/germany/subjects/p002_01_rs/'
config = 'permed/rs/mff/egi400'
files = _check_io_suffix(path, config, multiple=True)


# def _read_rs_egi400_generic(files, config_params):
logger.info('Reading {} files'.format(len(files)))

raws = []
for fname in files:
    # fname = path
    fname = _check_path(fname)
    raw = mne.io.read_raw_egi(fname.as_posix(), preload=True, verbose=True)
    raws.append(raw)

raws = sorted(raws, key=lambda x: x.info['meas_date'])
raw = mne.io.concatenate_raws(raws)

# Cleanup raw
n_eeg = 0
for idx in range(len(raw.ch_names)):
    n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')
replacement = {k: k.replace('EG', '')
                   .replace(' 00', '')
                   .replace(' 0', '')
                   .replace(' ', '')
               for k in raw.ch_names}  # removes spaces etc

if n_eeg == 257:
    if 'EEG 257' in raw.ch_names:
        replacement['EEG 257'] = 'Cz'
    elif 'E257' in raw.ch_names:
        replacement['E257'] = 'Cz'
if 'STI 014' in replacement.keys():
    del replacement['STI 014']
mne.rename_channels(raw.info, replacement)
if 'Cz' in raw.ch_names:
    n_eeg -= 1
    raw.drop_channels(['Cz'])

eq_config = 'egi/{}'.format(n_eeg)
montage = get_montage(eq_config)

to_drop = [x for x in raw.ch_names if x not in montage.ch_names
           and x != 'STI 014']  # noqa

raw.drop_channels(to_drop)
raw.set_montage(montage)
if n_eeg in (257, 129, 65):
    ch_pos_is_not_zero = \
        not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
    assert ch_pos_is_not_zero

raw.info['description'] = eq_config
logger.info('Reading done')
# return raw




#%% Israel rs

import numpy as np

from pathlib import Path
import tempfile

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix
from nice_ext.equipments import get_montage

# from .constants import _permed_biosemi64_chan_names
from next_permed.equipments import (_permed_biosemi64_chan_names, 
                                    _biosemi64_AB_names, _permed_ch_names)


path = '/home/dragana.manasova/Documents/data/permed/israel/subjects/p006_01_rs/'


config = 'permed/rs/bdf/biosemi'
files = _check_io_suffix(path, config, multiple=False)

raw = mne.io.read_raw_bdf(files[0], preload=True, verbose=True)

# There are some extra channels which are EEG type, but we don't need them.
# Here I take all the channels that are different than the 64 we need,
# and then I remove them.

if raw.ch_names[0][0] == 'A' or raw.ch_names[0][0] == 'B':
    ch_to_drop = [x for x in raw.ch_names if x not in set(_biosemi64_AB_names)]
    raw.drop_channels(ch_to_drop)
    print('Channels named with Ax Bx notation.')
    # Change the channel names from Ax and Bx to scalp locations
    raw.rename_channels(_permed_biosemi64_chan_names)
else:
    ch_to_drop = [x for x in raw.ch_names if x not in set(_permed_ch_names['bs64'])]
    raw.drop_channels(ch_to_drop)
    print('Channels named with positions.')

n_eeg = 0
for idx in range(len(raw.ch_names)):
    n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')

logger.info('Adding standard channel locations to info.')

ch_config = 'permed/bs{}'.format(n_eeg)
montage = get_montage(ch_config)
raw.set_montage(montage)

if n_eeg in (257, 129, 65):
    ch_pos_is_not_zero = \
        not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
    assert ch_pos_is_not_zero
    # The assert keyword lets you test if a condition in your code 
    # returns True, if not, the program will raise an AssertionError.

raw.info['description'] = ch_config
logger.info('Reading done')



