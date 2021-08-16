#%% First option from https://github.com/fraimondo/nice-ld/tree/master/next_ld


from copy import deepcopy
import mne
from mne.utils import logger


import numpy as np

# import mne
# from mne.utils import logger

from nice_ext.api.modules import register_module

# from .equipments.filters import _ld_filter

def _ld_filter(raw, params=None, summary=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get('lpass', 45.)
    hpass = params.get('hpass', 0.5)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, ecg=False,
                           exclude=[])
    _filter_params = dict(method='iir',
                          l_trans_bandwidth=0.1,
                          iir_params=dict(ftype='butter', order=6))
    filter_params = [
        dict(l_freq=hpass, h_freq=None,
             iir_params=dict(ftype='butter', order=6)),
        dict(l_freq=None, h_freq=lpass,
             iir_params=dict(ftype='butter', order=8))
    ]

    for fp in filter_params:
        if fp['l_freq'] is None and fp['h_freq'] is None:
            continue
        _filter_params2 = deepcopy(_filter_params)
        if fp.get('method') == 'fft':
            _filter_params2.pop('iir_params')
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        if summary is not None:
            summary['steps'].append(
                {'step': 'filter', 'params':
                    {'hpass': '{} Hz'.format(_filter_params2['l_freq']),
                     'lpass': '{} Hz'.format(_filter_params2['h_freq'])}})
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = [50]
    if raw.info['sfreq'] > 200:
        notches.append(100)
    logger.info('Notch filters at {}'.format(notches))
    if summary is not None:
        params = {(k + 1): '{} Hz'.format(v) for k, v in enumerate(notches)}
        summary['steps'].append({'step': 'notches', 'params': params})
    if raw.info['sfreq'] != 250:
        logger.info('Resampling to 250Hz')
        raw = raw.resample(250)
        if summary is not None:
            summary['steps'].append({'step': 'resample',
                                     'params': {'sfreq': 250}})
    raw.notch_filter(notches, method='fft', n_jobs=n_jobs)




def register():
    # register_module('preprocess', 'generic/rs/raw/ld', _preprocess_rs_raw_ld)
    register_module('preprocess', 'permed/rs/raw/nk', _preprocess_rs_raw_nk)


def _preprocess_rs_raw_nk(raw, config_params):
    from nice_ext.api.preprocessing import _check_min_events

    # Cut 800 ms epochs (-200, 600 ms)
    t_cut = config_params.get('t_cut', 0.8)
    # between 550ms (550 + 800 = 1350 ms space between triggers)
    min_jitter = config_params.get('min_jitter', 0.55)
    # and 850ms (850 + 800 = 1650 ms space between triggers)
    max_jitter = config_params.get('max_jitter', 0.85)
    # reject = config_params.get('reject', None)  # was commented out
    tmin = config_params.get('tmin', -.2)
    baseline = config_params.get('baseline', None)
    n_jobs = config_params.get('n_jobs', 1)
    min_events = config_params.get('min_events', 50)

    # if reject is None:
    #     reject = {'eeg': 100e-6} # was commented out

    summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            summary = dict(steps=[], bad_channels=[])
    # Filter
    _ld_filter(raw, config_params, summary=summary, n_jobs=n_jobs)

    # Cut
    logger.info(f'Cutting events (sfreq = {raw.info["sfreq"]})')
    max_events = int(np.ceil(len(raw) / (raw.info['sfreq'] * t_cut))) + 1
    evt_times = []
    if isinstance(min_jitter, float):
        min_jitter = int(np.ceil(min_jitter * raw.info['sfreq']))
    if isinstance(max_jitter, float):
        max_jitter = int(np.ceil(max_jitter * raw.info['sfreq']))
    jitters = np.random.random_integers(
        min_jitter, max_jitter, max_events)
    epoch_len = int(np.ceil(t_cut * raw.info['sfreq']))
    this_sample = 0
    this_jitter = 0
    while this_sample < len(raw):
        evt_times.append(this_sample + raw.first_samp)
        this_sample += epoch_len + jitters[this_jitter]
        this_jitter += 1
    evt_times = np.array(evt_times)
    events = np.concatenate((evt_times[:, None],
                             np.zeros((len(evt_times), 1), dtype=np.int),
                             np.ones((len(evt_times), 1), dtype=np.int)),
                            axis=1)
    event_id = 1

    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=t_cut + tmin,
                        preload=True, reject=None, picks=None,
                        baseline=baseline, verbose=False)

    logger.info('Using autoreject')
    from autoreject import AutoReject
    ar = AutoReject(
        n_interpolate=np.array([1, 2, 4, 8])
    )
    epochs_clean = ar.fit_transform(epochs)
    reject_log = ar.get_reject_log(epochs)
    if summary is not None:
        summary['autoreject'] = reject_log
        summary['steps'].append(
            dict(step='Autoreject',
                 params={'n_interpolate': ar.n_interpolate_['eeg'],
                         'consensus_perc': ar.consensus_['eeg']},
                 bad_epochs=np.where(reject_log.bad_epochs)[0]))
    _check_min_events(epochs, min_events)
    logger.info('found bad epochs: {} {}'.format(
        np.sum(reject_log.bad_epochs),
        np.where(reject_log.bad_epochs)[0]))
    epochs = epochs_clean

    ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
    epochs.add_proj(ref_proj)
    epochs.apply_proj()

    out = epochs
    if summary is not None:
        out = out, summary
    return out
