"""
Created by Romy Beauté
romybeaute.univ@gmail.com
Basic configurations for the Motor Command Protocol
Stores the global variables to be used across modules
"""

import mne
from mne.utils import logger
from copy import deepcopy



#*********************************************************************************************************************
# ==> Multicenter configs
#*********************************************************************************************************************
# => Basic variables for PREPROCESSING
preproc = ['filt', 'csd','drop_ears','chan_to_drop','bios']
fmin, fmax = 1.0,30.0
nb_chans = 256
# => Variable for EVENTS & EPOCHS creation
n_epo_segments = 5
# => SVM FEATURES : Frequency bands of interest, used as features for the SVM
bands = ((1, 3), (4, 7), (8, 13), (14, 30))
SMR_bands = ((7, 12), (13, 24), (25, 35))
all_bands = [(0, 4, 'Delta'), (4, 7, 'Theta'), (8, 12, 'Mu/Alpha'),(13, 24, 'Low beta'),(12, 30, 'Beta'), (25, 35, 'High beta'),(30, 45, 'Gamma')]
# => Event basic DICT of events
event_id = {1: 'keep/right', 2: 'stop/right', 3: 'stop/left', 4: 'keep/left'}
event_dict = {'keep/right': 1, 'stop/right': 2, 'stop/left': 3, 'keep/left': 4}



#*********************************************************************************************************************
# ==> PERMED
# ==> Ressources : https://github.com/fraimondo/nice-permed/blob/main/next_permed/equipments.py
#*********************************************************************************************************************
_permed_montages = {
    'bv60': 'easycap-M1',
    'bv62': 'easycap-M1',
    'bs64': 'biosemi64',
    'nk23': 'standard_1020'
}


_permed_ch_names = {
    'bv60': [
        'Iz',  'O2',  'Oz',  'O1',  'PO8',  'PO4',  'POz',  'PO3',  'PO7',
        'P8',  'P6',  'P4',  'P2',  'Pz',  'P1',  'P3',  'P5',  'P7',
        'TP8',  'CP6',  'CP4',  'CP2',  'CPz',  'CP1',  'CP3',  'CP5',  'TP7',
        'T8',  'C6',  'C4',  'C2',  'Cz',  'C1',  'C3',  'C5',  'T7',
        'FT8',  'FC6',  'FC4',  'FC2',  'FCz',  'FC1',  'FC3',  'FC5',  'FT7',
        'F8',  'F6',  'F4',  'F2',  'Fz',  'F1',  'F3',  'F5',  'F7',  'AF4',
        'AFz',  'AF3',  'Fp2',  'Fpz',  'Fp1'],
    'bv62': [
        'Iz',  'O2',  'Oz',  'O1',  'PO8',  'PO4',  'POz',  'PO3',  'PO7',
        'P8',  'P6',  'P4',  'P2',  'Pz',  'P1',  'P3',  'P5',  'P7',
        'TP10', # diff
        'TP8',  'CP6',  'CP4',  'CP2',  'CPz',  'CP1',  'CP3',  'CP5',  'TP7',
        'TP9', # diff
        'T8',  'C6',  'C4',  'C2',  'Cz',  'C1',  'C3',  'C5',  'T7',
        'FT8',  'FC6',  'FC4',  'FC2',  'FCz',  'FC1',  'FC3',  'FC5',  'FT7',
        'F8',  'F6',  'F4',  'F2',  'Fz',  'F1',  'F3',  'F5',  'F7',  'AF4',
        'AFz',  'AF3',  'Fp2',  'Fpz',  'Fp1'],
    'bs64': [
        'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
        'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1',
        'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
        'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6',
        'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6',
        'T8', 'TP8', 'CP6', 'CP4', 'CP2',
        'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
    'nk23' : [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
        'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
        # 'T7', 'T8', 'P7', 'P8', #
        'Fz', 'Cz', 'Pz', 'A1', 'A2',
        # 'FT9', 'FT10' #
        ]
}




#*********************************************************************************************************************
# ==> Configs for EGI256
#*********************************************************************************************************************
fr_setup = 'egi/256'
# For classifc conditions
fr_event_key = {61: 1, 125: 2, 45: 3, 109: 4}
# For imaginary conditions :
event_id_im = {1: 'open/right', 2: 'close/right', 3: 'open/left', 4: 'close/left', 11: 'open/imagine/right', 22: 'close/imagine/right', 33: 'open/imagine/left', 44: 'close/imagine/left'}
event_key_im = {61: 1, 125: 2, 45: 3, 109: 4, 610: 11, 1250: 22, 450: 33, 1090: 44}
event_dict_im = {value: key for key, value in event_id_im.items()}
# Self defined bad chans :
ears_outlines = ['E190', 'E191', 'E201', 'E209', 'E218', 'E217', 'E216', 'E208', 'E200', 'E81', 'E72', 'E66', 'E67',
                     'E68', 'E73', 'E82', 'E92', 'E91']
outer_outlines = ['E9', 'E17', 'E24', 'E30', 'E31', 'E36', 'E45', 'E243', 'E240', 'E241', 'E242', 'E246', 'E250',
                  'E255', 'E90', 'E101', 'E110', 'E119', 'E132', 'E144', 'E164', 'E173', 'E186', 'E198', 'E207',
                  'E215', 'E228', 'E232', 'E236', 'E239', 'E238', 'E237', 'E233', 'E257']
chan_to_drop = ['E67', 'E73', 'E247', 'E251', 'E256', 'E243', 'E246', 'E250',
                'E255', 'E82', 'E91', 'E254', 'E249', 'E245', 'E242', 'E253',
                'E252', 'E248', 'E244', 'E241', 'E92', 'E102', 'E103', 'E111',
                'E112', 'E120', 'E121', 'E133', 'E134', 'E145', 'E146', 'E156',
                'E165', 'E166', 'E174', 'E175', 'E187', 'E188', 'E199', 'E200',
                'E208', 'E209', 'E216', 'E217', 'E228', 'E229', 'E232', 'E233',
                'E236', 'E237', 'E240', 'E218', 'E227', 'E231', 'E235', 'E239',
                'E219', 'E225', 'E226', 'E230', 'E234', 'E238']
bios = ['ECG', 'EMG-Leg','E257'] #biochans + ref



#*********************************************************************************************************************
# ==> Configs for ITALY
#*********************************************************************************************************************
it_setup = 'eeg/62'
it_preproc = ['filt', 'csd','bad_it_chans']
bad_it_chans = ['TP9','TP10','VEOG', 'HEOG']
_italy_mcp_event_id = {
    197:'move/right',
    133:'stop/right',
    213:'move/left',
    149:'stop/left',
}
_italy_mcp_event_id2 = {
    13: 'move/right',  # type 2
    5: 'stop/right',  # type 2
    29: 'move/left',  # type 2
    21: 'stop/left'  # type 2
}
# _italy_mcp_event_id = _italy_mcp_event_id2 #useful for some subject (SMN011)
it_event_key = {197: 1, 133: 2, 213: 3, 149: 4}
it_event_key2 = {13: 1, 5: 2, 29: 3, 21: 4}


#*********************************************************************************************************************
# ==> Configs for Standard 21
#*********************************************************************************************************************
std_setup = 'standard/1020'
std_preproc = ['filt', 'csd']
# EEG channels to use in the analysis (=19 after excluding A1, A2)
use_ch21 = ['C3', 'C4', 'O1', 'O2', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz',
          'Fp1', 'Fp2', 'P3', 'P4', 'Pz', 'T7', 'T8', 'P7', 'P8']



#*********************************************************************************************************************
# ==> Configs for GERMANY
#*********************************************************************************************************************
ge_setup = 'egi/256'
_germany_mcp_event_id = {
    'move/right': 196,
    'stop/right': 132,
    'move/left': 212,
    'stop/left': 148
}
ge_event_key = {196: 1, 132: 2, 212: 3, 148: 4}


#*********************************************************************************************************************
# ==> Configs for MUNICH
#*********************************************************************************************************************
mun_preproc = ['filt', 'csd','bad_mun_chans']
bad_mun_chans = ['VEOG', 'HEOG']
_munich_mcp_event_id = {
    196: 'move/right',
    132: 'stop/right',
    212: 'move/left',
    148: 'stop/left'
}


#*********************************************************************************************************************
# ==> SETUPS MAPPING
#*********************************************************************************************************************
_egi256_21_map = {
    'E37': 'Fp1', 'E18': 'Fp2', 'E47': 'F7', 'E36': 'F3',
    'E21': 'Fz', 'E224': 'F4', 'E2': 'F8', 'E69': 'T3',
    'E59': 'C3', 'E90': 'Cz', 'E183': 'C4', 'E202': 'T4',
    'E96': 'T5', 'E87': 'P3', 'E101': 'Pz', 'E153': 'P4',
    'E170': 'T6', 'E116': 'O1', 'E150': 'O2', 'E26': 'Fpz',
    'E126': 'Oz'}



#*********************************************************************************************************************
# ==> PREPROCESSING
# ==> Ressources : https://github.com/fraimondo/nice-permed/blob/main/next_permed/usanihon/preprocessing.py
#*********************************************************************************************************************
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

















