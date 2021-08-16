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

from ..equipments import (_permed_biosemi64_chan_names, 
                          _biosemi64_AB_names, 
                          _permed_ch_names)


def register():
    # RS
    register_module('io', 'permed/rs/bdf/biosemi', _read_rs_bdf_biosemi)
    register_suffix('permed/rs/bdf/biosemi', 'permed-rs-biosemi.bdf')

    register_module('io', 'permed/rs/bdfzip/biosemi', _read_rs_bdfzip_biosemi)
    register_suffix('permed/rs/bdfzip/biosemi', 'permed-rs-biosemi.zip')

    # LG
    register_module('io', 'permed/lg/bdf/biosemi', _read_lg_bdf_biosemi)
    register_suffix('permed/lg/bdf/biosemi', 'permed-lg-biosemi.bdf')


def _read_rs_bdf_biosemi(path, config_params):
    config = 'permed/rs/bdf/biosemi'
    files = _check_io_suffix(path, config, multiple=False)
    return _read_rs_biosemi_generic(files, config_params)

def _read_rs_bdfzip_biosemi(files, config_params):
    import zipfile
    config = 'permed/rs/bdfzip/biosemi'
    files = _check_io_suffix(path, config, multiple=False)
    fname = files[0]
    logger.info('Extracting zip file')
    zip_ref = zipfile.ZipFile(fname, 'r')
    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        zip_ref.extractall(tdir.as_posix())

        logger.info('Checking content of zip file')
        fnames = tdir.glob('*')
        fnames = [x for x in fnames if x.name not in ['__MACOSX']]
        fnames = [x for x in fnames if not x.name.startswith('.')]
        if len(fnames) != 1:
            raise ValueError('Wrong ZIP file content (n files = {})'.format(
                len(fnames)))

        new_fname = Path(f'{fnames[0].as_posix()}-permed-rs-biosemi.bdf')
        fnames[0].rename(new_fname)

        raw = _read_rs_biosemi_generic([new_fname], config_params=config_params)
    return raw

def _read_rs_biosemi_generic(files, config_params):
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

    return raw


def _read_lg_bdf_biosemi(path, config_params):
    config = 'permed/lg/bdf/biosemi'
    files = _check_io_suffix(path, config, multiple=False)
    return _read_lg_biosemi_generic(files, config_params)


def _read_lg_bdf_generic(files, config_params):
    raw = mne.io.read_raw_bdf(files[0], preload=True, verbose=True)
    logger.info('Reading {} files'.format(len(files)))
    raws = []

    # TODO check if this is the function and the way to deal w/ events
    events, _ = mne.events_from_annotations(raw) 
    valid_events = np.array([x for x in _arduino_trigger_map.keys()])
    events[:, 2] -= 1
    events = events[np.in1d(events[:, 2], valid_events)]

    events[:, 2] = [_icm_lg_event_id[_arduino_trigger_map[x]]
                    for x in events[:, 2]]

    # Adding the events in the trigger channel
    if 'STI 014' not in raw.ch_names:
        # Creating a new stim channel and adding it to the raw
        stim_data = np.zeros((1, len(raw.times)))
        info = mne.create_info(
            ['STI 014'], raw.info['sfreq'], ch_types=['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        raw.add_channels([stim_raw], force_update_info=True)
    # add the events to the stim channel
    raw.add_events(events, stim_channel='STI 014')
    raw.event_id = _icm_lg_event_id

    # Cleanup raw
    raw.pick_types(eeg=True, stim=True)

    ch_config = 'permed/bs{}'.format(n_eeg)
    ch_names = get_ch_names(eq_config)
    if len(ch_names) + 1 != len(raw.ch_names):
        raise ValueError('Wrong number of channels')
    for i_ch, ch_name in enumerate(ch_names):
        if ch_name != raw.ch_names[i_ch]:
            raise ValueError('Wrong channel order')

    montage = get_montage(eq_config)
    raw.set_montage(montage)

    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw