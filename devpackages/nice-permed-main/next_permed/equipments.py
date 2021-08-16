import numpy as np
import os.path as op

from nice_ext.equipments import define_equipment, define_rois
from nice_ext.equipments.montages import define_neighbor, define_map_montage

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

# I checked the neighbors of Fz, Cz and Pz, from the data/neighbor file. 
# Then I found which numbers they correspond to in the epochs.info["ch_names"].
_biosemi64_rois = {
    'p3a': np.array([10, 11, 18, 31, 45, 46, 47, 48, 55, 3, 36, 37, 38, 46]), # Cz & Fz
    'p3b': np.array([10, 11, 18, 31, 45, 46, 47, 48, 55, 19, 29, 30, 31, 56]), # Cz & Pz
    'mmn': np.array([10, 11, 18, 31, 45, 46, 47, 48, 55, 3, 36, 37, 38, 46]), # Cz & Fz
    'cnv': np.array([3, 36, 37, 38, 46]), # Fz
    'Fz': np.array([3, 36, 37, 38, 46]), 
    'Cz': np.array([10, 11, 18, 31, 45, 46, 47, 48, 55]), 
    'Pz': np.array([19, 29, 30, 31, 56]), 
    'scalp': np.arange(64),
    'nonscalp': None
}

# I checked the neighbors of Fz, Cz and Pz, from the data/neighbor file. 
# Then I found which numbers they correspond to in the epochs.info["ch_names"].
_bv_60_rois = {
    'p3a': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41, 39, 40, 41, 48, 49, 50, 55]), # Cz & Fz
    'p3b': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41, 6, 12, 13, 14, 21, 22, 23]), # Cz & Pz
    'mmn': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41, 39, 40, 41, 48, 49, 50, 55]), # Cz & Fz
    'cnv': np.array([39, 40, 41, 48, 49, 50, 55]), # same as Fz
    'Fz': np.array([39, 40, 41, 48, 49, 50, 55]), 
    'Cz': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41]),
    'Pz': np.array([6, 12, 13, 14, 21, 22, 23]), 
    'scalp': np.arange(60),
    'nonscalp': None
}

_bv_62_rois = {
    'p3a': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41, 39, 40, 41, 48, 49, 50, 55]), # Cz & Fz
    'p3b': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41, 6, 12, 13, 14, 21, 22, 23]), # Cz & Pz
    'mmn': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41, 39, 40, 41, 48, 49, 50, 55]), # Cz & Fz
    'cnv': np.array([39, 40, 41, 48, 49, 50, 55]), # same as Fz
    'Fz': np.array([39, 40, 41, 48, 49, 50, 55]), 
    'Cz': np.array([21, 22, 23, 30, 31, 32, 39, 40, 41]),
    'Pz': np.array([6, 12, 13, 14, 21, 22, 23]), 
    'scalp': np.arange(62),
    'nonscalp': None
}

_nk_23_rois = {
    'p3a' : np.array([2, 16, 3, 4, 5, 6, 18, 7, 0, 1, 2, 3, 4, 17, 5]), # Cz & Fz
    'p3b' : np.array([2, 16, 3, 4, 5, 6, 18, 7, 4, 17, 5, 6, 7, 8, 9]), # Cz & Pz
    'mmn' : np.array([2, 16, 3, 4, 5, 6, 18, 7, 0, 1, 2, 3, 4, 17, 5]), # Cz & Fz
    'cnv' : np.array([0, 1, 2, 3, 4, 17, 5]), # same as Fz
    'Fz' : np.array([0, 1, 2, 3, 4, 17, 5]), 
    'Cz' : np.array([2, 16, 3, 4, 5, 6, 18, 7]), 
    'Pz' : np.array([4, 17, 5, 6, 7, 8, 9]),
    'scalp' : np.arange(23),
    'nonscalp' : None
}


_biosemi64_AB_names = ['A{}'.format(x) for x in range(1, 33)] + ['B{}'.format(x) for x in range(1, 33)]

_permed_biosemi64_chan_names = dict(zip(_biosemi64_AB_names, _permed_ch_names['bs64']))

def register():
    define_equipment('permed', _permed_montages, _permed_ch_names)
    define_rois('permed/bv60', _bv_60_rois)
    define_rois('permed/bv62', _bv_62_rois) # TP9 and TP10 aren't around Fz, Cz, Pz. The ROIs are mostly the same, except for scalp. 
    define_rois('permed/bs64', _biosemi64_rois)
    # define_map_montage('egi/256', 'egi/128', _egi256_egi128_map)
    define_rois('permed/nk23', _nk_23_rois)

    # for the neighbours:
    c_path = op.abspath(op.dirname(op.realpath(__file__)))
    fname = op.join(c_path, 'data', 'biosemi_64_neighbours.mat')
    define_neighbor('permed/bs64', fname)
    #
    c_path = op.abspath(op.dirname(op.realpath(__file__)))
    fname = op.join(c_path, 'data', 'bv_60_neighbours.mat')
    define_neighbor('permed/bv60', fname)
    #
    c_path = op.abspath(op.dirname(op.realpath(__file__)))
    fname = op.join(c_path, 'data', 'bv_60_neighbours.mat')
    define_neighbor('permed/bv62', fname)
    #
    c_path = op.abspath(op.dirname(op.realpath(__file__)))
    fname = op.join(c_path, 'data', 'nk_23_neighbours.mat')
    define_neighbor('permed/nk23', fname)

