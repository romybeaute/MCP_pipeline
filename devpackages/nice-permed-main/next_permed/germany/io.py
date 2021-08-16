import numpy as np

import mne
from mne.utils import logger
from pathlib import Path
import tempfile

from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix, _check_path
from nice_ext.equipments import get_montage

from next_icm.lg.constants import _arduino_trigger_map, _icm_lg_event_id


def register():
    # LG
    register_module('io', 'permed/lg/mff/egi400', _read_lg_mff_egi400)
    register_suffix('permed/lg/mff/egi400', 'permed-lg-egi400.mff')

    register_module('io', 'permed/lg/mffzip/egi400', _read_lg_mffzip_egi400)
    register_suffix('permed/lg/mffzip/egi400', 'permed-lg-egi400.mff.zip')

    # RS
    register_module('io', 'permed/rs/mff/egi400', _read_rs_mff_egi400)
    register_suffix('permed/rs/mff/egi400', 'permed-rs-egi400.mff')

    register_module('io', 'permed/rs/mffzip/egi400', _read_rs_mffzip_egi400)
    register_suffix('permed/rs/mffzip/egi400', 'permed-rs-egi400.mff.zip')

def _read_lg_mff_egi400(path, config_params):
    config = 'permed/lg/mff/egi400'
    raw = None

    files = _check_io_suffix(path, config, multiple=True)
    raw = _read_lg_egi400_generic(files, config_params)
    return raw


def _read_lg_mffzip_egi400(path, config_params):
    import zipfile
    config = 'permed/lg/mffzip/egi400'
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

        new_fname = Path(f'{fnames[0].as_posix()}-permed-lg-egi400.mff')
        fnames[0].rename(new_fname)

        raw = _read_lg_egi400_generic([new_fname], config_params=config_params)
    return raw

def _read_lg_egi400_generic(files, config_params):
    logger.info('Reading {} files'.format(len(files)))

    raws = []
    for fname in files:
        # fname = path
        fname = _check_path(fname)
        raw = mne.io.read_raw_egi(fname.as_posix(), preload=True, verbose=True)

        # Interpret triggers
        _change_triggers(raw)

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
    return raw

def _change_triggers(raw):
    """Assign triggers to conditions and clean channel values.

    Parameters
    ----------
    raw : instance of mne.io.egi.Raw
        The egi imported raw.
    """
    din_ch_names = ['DIN2', 'DIN3', 'DIN4', 'DIN5', 'DIN6', 'DIN7', 'DIN8']
    raw.drop_channels(['DIN1'])
    stims_data = raw.copy().pick(din_ch_names).get_data().astype(np.int)

    # Check this logic:
    stims_data = (stims_data != 0).astype(np.int)
    trig_mult = np.array([0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])
    raw_trig_data = np.multiply(stims_data.T, trig_mult).T
    sumi = np.sum(raw_trig_data, axis=0)  # sum of all the values in one sample

    # A sliding window of k going through the trigger values,
    # the sliding window starts only when it detects a non-0 value,
    # and checks if there are more than 1 non-zero elements,
    # and then checking if they are in original values,
    # and if yes, then it's summing them up, 
    # and setting the first value to the sum, and the rest to 0.
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

    # Removing the first staircase of triggers
    find_seq = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
    seq_len = len(find_seq)
    non_zero_idx = np.where(sumi != 0)[0]
    non_zero_vals = sumi[non_zero_idx]
    max_samples = int(raw.info['sfreq'])
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

def _read_rs_mff_egi400(path, config_params):
    config = 'permed/rs/mff/egi400'
    raw = None

    files = _check_io_suffix(path, config, multiple=True)
    raw = _read_rs_egi400_generic(files, config_params)
    return raw

def _read_rs_mffzip_egi400(path, config_params):
    import zipfile
    config = 'permed/rs/mffzip/egi400'
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

        new_fname = Path(f'{fnames[0].as_posix()}-permed-rs-egi400.mff')
        fnames[0].rename(new_fname)

        raw = _read_rs_egi400_generic([new_fname], config_params=config_params)
    return raw

def _read_rs_egi400_generic(files, config_params):
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
    raw.set_montage(montage)

    to_drop = [x for x in raw.ch_names if x not in montage.ch_names
               and x != 'STI 014']  # noqa
    raw.drop_channels(to_drop)
    
    if n_eeg in (257, 129, 65):
        ch_pos_is_not_zero = \
            not np.all(raw.info['chs'][n_eeg - 1]['loc'][:3] == 0.0)
        assert ch_pos_is_not_zero

    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw