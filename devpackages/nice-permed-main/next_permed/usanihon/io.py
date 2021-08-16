"""
Template copied from /.../nice_ext/.../rs/io.py
"""

import numpy as np

from pathlib import Path
import tempfile

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix
from nice_ext.equipments import get_montage

from ..equipments import _permed_ch_names


def register():
    # RS
    register_module('io', 'permed/rs/eeg/nihonkohden', _read_rs_eeg_nihonkohden)
    register_suffix('permed/rs/eeg/nihonkohden', 'permed-rs-nihonkohden.EEG')
    # RS
    register_module('io', 'permed/rs/edf/nihonkohden', _read_rs_edf_nihonkohden)
    register_suffix('permed/rs/edf/nihonkohden', 'permed-rs-nihonkohden.edf')

def _read_rs_eeg_nihonkohden(path, config_params):
    config = 'permed/rs/eeg/nihonkohden'
    files = _check_io_suffix(path, config, multiple=False)
    return _read_rs_nihonkohden_generic(files, config_params)

def _read_rs_nihonkohden_generic(files, config_params):
    # print('This is the path: ')
    # print(files)
    # print('\nand this the [0]')
    # print(files[0])
    raw = mne.io.read_raw_nihon(files[0], preload=True, verbose=True)

    # There are some extra channels which are EEG type, but we don't need them.
    # Here I keep only a predefined set of channels - the EEG ones. 

    raw.pick_channels(ch_names = _permed_ch_names['nk23'], ordered=False)

    n_eeg = 0
    for idx in range(len(raw.ch_names)):
        n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')
    # n_eeg = 23

    logger.info('Adding standard channel locations to info.')

    ch_config = 'permed/nk{}'.format(n_eeg)
    montage = get_montage(ch_config)
    raw.set_montage(montage)

    # if n_eeg in (257, 129, 65):
    #     ch_pos_is_not_zero = \
    #         not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
    #     assert ch_pos_is_not_zero
    #     # The assert keyword lets you test if a condition in your code 
    #     # returns True, if not, the program will raise an AssertionError.

    raw.info['description'] = ch_config
    logger.info('Reading done')

    return raw


def _read_rs_edf_nihonkohden(path, config_params):
    config = 'permed/rs/edf/nihonkohden'
    files = _check_io_suffix(path, config, multiple=False)
    return _read_rs_nihonkohden_edf_generic(files, config_params)

def _read_rs_nihonkohden_edf_generic(files, config_params):
    # print('This is the path: ')
    # print(files)
    # print('\nand this the [0]')
    # print(files[0])
    raw = mne.io.read_raw_edf(files[0], preload=True, verbose=True)

    # There are some extra channels which are EEG type, but we don't need them.
    # Here I keep only a predefined set of channels - the EEG ones. 

    raw.pick_channels(ch_names = _permed_ch_names['nk23'], ordered=False)

    n_eeg = 0
    for idx in range(len(raw.ch_names)):
        n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')

    logger.info('Adding standard channel locations to info.')

    ch_config = 'permed/nk{}'.format(n_eeg)
    montage = get_montage(ch_config)
    raw.set_montage(montage)

    raw.info['description'] = ch_config
    logger.info('Reading done')

    return raw