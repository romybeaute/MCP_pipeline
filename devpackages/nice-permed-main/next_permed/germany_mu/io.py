import numpy as np
import mne
import tempfile
from pathlib import Path

from mne.utils import logger
from mne.utils import _TempDir
from nice_ext.api.modules import register_module
from nice_ext.api.io import register_suffix, _check_io_suffix, _check_path
from nice_ext.equipments import get_montage, get_ch_names

from next_icm.lg.constants import _arduino_trigger_map, _icm_lg_event_id


def register():
    # PACKAGE/PROTOCOL/FORMAT/SYSTEM
    # LG
    register_module('io', 'permed/lg/raw/bv62', _read_lg_raw_bv62)
    register_suffix('permed/lg/raw/bv62', 'permed-lg-bv62.vhdr')

    # register_module('io', 'permed/lg/rawzip/bv', _read_lg_rawzip_bv)
    # register_suffix('permed/lg/rawzip/bv', 'permed-lg-bv.zip')

    # RS
    register_module('io', 'permed/rs/raw/bv62', _read_rs_raw_bv62)
    register_suffix('permed/rs/raw/bv62', 'permed-rs-bv62.vhdr')

def _read_lg_raw_bv62(path, config_params):
    config = 'permed/lg/raw/bv62'
    files = _check_io_suffix(path, config, multiple=True)
    return _read_lg_bv62_generic(files, config_params)

# TODO ?
# Trial, not finished. Maybe it's not needed.
# def _read_lg_rawzip_bv(path, config_params):
#    import zipfile
#    config = 'permed/lg/rawzip/bv'
#    files = _check_io_suffix(path, config, multiple=False)
#    fname = files[0]
#    logger.info('Extracting zip file')
#    zip_ref = zipfile.ZipFile(fname, 'r')
#    with tempfile.TemporaryDirectory() as tdir:
#        tdir = Path(tdir)
#        zip_ref.extractall(tdir.as_posix())

#        logger.info('Checking content of zip file')
#        fnames = tdir.glob('*')
#        fnames = [x for x in fnames if x.name not in ['__MACOSX']]
#        fnames = [x for x in fnames if not x.name.startswith('.')]
#        if len(fnames) != 1:
#            raise ValueError('Wrong ZIP file content (n files = {})'.format(
#                len(fnames)))

#        new_fname = Path(f'{fnames[0].as_posix()}-permed-lg-bv.vhdr')
#        fnames[0].rename(new_fname)

#        raw = _read_lg_bv_generic([new_fname], config_params=config_params)
#    return raw

def _read_lg_bv62_generic(files, config_params):
    # config = 'permed/lg/raw/bv'
    # files = _check_io_suffix(path, config, multiple=True)
    logger.info('Reading {} files'.format(len(files)))
    raws = []
    for fname in files:
        raw = mne.io.read_raw_brainvision(fname, preload=True, verbose=True)
        raws.append(raw)
    raws = sorted(raws, key=lambda x: x.info['meas_date'])
    raw = mne.io.concatenate_raws(raws)

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
    raw.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
    raw.pick_types(eeg=True, stim=True)

    n_eeg = 0
    for idx in range(len(raw.ch_names)):
        n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')

    eq_config = 'permed/bv{}'.format(n_eeg)
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

def _read_rs_raw_bv62(path, config_params):
    config = 'permed/rs/raw/bv62'
    files = _check_io_suffix(path, config, multiple=True)
    return _read_rs_bv62_generic(files, config_params)

def _read_rs_bv62_generic(files, config_params):
    logger.info('Reading {} files'.format(len(files)))
    raws = []
    for fname in files:
        raw = mne.io.read_raw_brainvision(fname, preload=True, verbose=True)
        raws.append(raw)
    raws = sorted(raws, key=lambda x: x.info['meas_date'])
    raw = mne.io.concatenate_raws(raws)

    # Cleanup raw
    raw.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
    raw.pick_types(eeg=True, stim=True)

    n_eeg = 0
    for idx in range(len(raw.ch_names)):
        n_eeg += int(mne.io.pick.channel_type(raw.info, idx) == 'eeg')

    eq_config = 'permed/bv{}'.format(n_eeg)
    ch_names = get_ch_names(eq_config)
    if len(ch_names) != len(raw.ch_names): # without +1 because we dont have STI 014
        raise ValueError('Wrong number of channels')
    for i_ch, ch_name in enumerate(ch_names):
        if ch_name != raw.ch_names[i_ch]:
            raise ValueError('Wrong channel order')

    montage = get_montage(eq_config)
    raw.set_montage(montage)

    # Drop extra channels which are not in the montage,
    # except for STI014.
    to_drop = [x for x in raw.ch_names if x not in montage.ch_names]  # noqa
    raw.drop_channels(to_drop)

    raw.info['description'] = eq_config
    logger.info('Reading done')
    return raw
