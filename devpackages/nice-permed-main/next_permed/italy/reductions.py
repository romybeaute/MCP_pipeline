import numpy as np

from nice_ext.api.reductions import (get_avaialable_functions,
                                     get_function_by_name)
from nice_ext.api.modules import register_module
from nice_ext.stats import entropy
from nice_ext.equipments import get_roi


def register():
    for f in get_avaialable_functions():
        register_module('reductions', 'permed/rs/bv60/{}'.format(f),
                        _get_rs_bv60)
        register_module('reductions', 'permed/rs/bv60gfp/{}'.format(f),
                        _get_rs_bv60gfp)

def _get_rs_bv60gfp(config, config_params):
    if len(config) == 0:
        config == 'mean'
    epochs_fun = get_function_by_name(config)
    channels_fun = np.std
    return _get_rs_bv60_generic(epochs_fun, channels_fun)


def _get_rs_bv60(config, config_params):
    if len(config) == 0:
        config == 'mean'
    epochs_fun = get_function_by_name(config)
    channels_fun = np.mean
    return _get_rs_bv60_generic(epochs_fun, channels_fun)

def _get_rs_bv60_generic(epochs_fun, channels_fun):
    return _get_rs_montage_generic(
        montage='permed/bv60', epochs_fun=epochs_fun, channels_fun=channels_fun)

def _get_rs_montage_generic(montage, epochs_fun, channels_fun,
                            config_params=None):
    reduction_params = {}
    scalp_roi = get_roi(config=montage, roi_name='scalp')
    if config_params is None:
        config_params = {}
    epochs_picks = config_params.get('epochs_picks', None)

    reduction_params['PowerSpectralDensity'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': np.sum},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensity/summary_se'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': entropy},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensitySummary'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['PermutationEntropy'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    reduction_params['SymbolicMutualInformation'] = {
        'reduction_func':
            [{'axis': 'channels_y', 'function': np.median},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels_y': scalp_roi,
            'channels': scalp_roi}}

    reduction_params['KolmogorovComplexity'] = {
        'reduction_func':
            [{'axis': 'channels', 'function': channels_fun},
             {'axis': 'epochs', 'function': epochs_fun}],
        'picks': {
            'epochs': epochs_picks,
            'channels': scalp_roi}}

    return reduction_params
