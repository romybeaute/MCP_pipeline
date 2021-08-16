"""
Created by Romy Beauté
romybeaute.univ@gmail.com
Define personalized functions to be used in the rest of the MCP pipeline (for the different centers)
"""


import numpy
import matplotlib
import seaborn
import sklearn
import pandas
import tqdm
import mne  # https://github.com/mne-tools/mne-python
import pycsd  # https://github.com/nice-tools/pycsd
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm_notebook

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, plot_roc_curve

from mne.time_frequency import psd_multitaper
from mne.decoding import LinearModel, get_coef

# Perso functions
import configs as cfg

"""
==> GLOBAL variables
"""
fmin, fmax = 1.0, 30.0  # frequency cut-off, in Hz
n_epo_segments = 5

cv = LeaveOneGroupOut()  # cross-validation method

# Basic parameters (defined in configs modules) used in the majority of cases
bands = cfg.bands


# event_dict = cfg.event_dict # ==> basic dictionnary to map the keep/rest right/left events

#==> Replaced by the MCP_vars Params (but kept it in case)
def permed_df(subject_type):  # ('patients', 'controls')
    """
    :param subject_type: 'patients' or 'controls'
    :return: df
    ==> Read PERMED CSV data from google sheet into Pandas """
    sheet_id = '1wvrkO8mHMx-fGP9HrwpBs57iQmEwBpcJ-DdtcfjV0JE'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={subject_type}'
    df = pd.read_csv(url)
    if not df.index.name == 'ID':
        df.set_index('ID', inplace=True)  # change the index if not already done
    return df

# ==> change here your datapath if needed
def load_setups(subject_id, df):
    """ Load the parameters from the CSV file """
    center = df.loc[subject_id]['Center']
    setup = df.loc[subject_id]['Setup']
    filename = df.loc[subject_id]['Filename']
    filename2 = df.loc[subject_id]['Filename2']
    is_control = bool(df.loc[subject_id]['Control'])
    is_imagine = df.loc[subject_id]['Condition'] == 'imagine'

    # ==> Define datapath according to used computer
    if 'romybeaute' in os.getcwd():
        # MAC
        datapath = '/Users/romybeaute/MOTOR'  # file with ALL the MCP datas
    elif 'romy.beaute' in os.getcwd():
        # LINUX
        datapath = '/home/romy.beaute/Bureau/CMP_romy/MOTOR'
    elif 'dragana.manasova' in os.getcwd():
        #datapath = '/media/dragana.manasova/Romy_SSDT5/MOTOR'
        datapath = '/network/lustre/dtlake01/cohen/globus/Romy_data/DATA' #LUSTRE
    elif 'melanie' in os.getcwd():
        datapath = '/Users/melanie/Desktop/MCP_Romy/MOTOR' 

    # ==> patients or controls
    if int(is_control) == 0:  # patients
        filepath = os.path.join(datapath, center, 'patients')
    else:  # controls
        filepath = os.path.join(datapath, center, 'controls')
    '''
    # ==> get file path
    if center not in ['Munich', 'Italy']:  # vhdr files, stored in a different way
        filepath = os.path.join(filepath, filename)
        if center == 'Paris':
            preproc = cfg.preproc  # ==> basic preprocessing parameters : ['filt', 'csd','drop_ears','chan_to_drop','bios']
        elif center == 'Germany':
            preproc = cfg.preproc
        elif center == 'Columbia':
            preproc = cfg.std_preproc
    # ==>  .vhdr files
    else:
        #datapath = os.path.join(filepath,subject_id)
        filepath = os.path.join(filepath, subject_id)
        if center == 'Munich':
            preproc = cfg.mun_preproc
        elif center == 'Italy':
            preproc = cfg.it_preproc
        #files = [f for f in os.listdir(filepath) if f.endswith('.vhdr')]
        if str(filename2) == 'nan': #only one file
            filepath = os.path.join(filepath, filename)
        else:
            #datapath = os.path.join(datapath, filename)
            datapath = filepath
    '''

    # ==> get file path
    filepath = os.path.join(filepath, filename)
    if center == 'Paris':
        preproc = cfg.preproc  # ==> basic preprocessing parameters : ['filt', 'csd','drop_ears','chan_to_drop','bios']
    elif center == 'Germany':
        preproc = cfg.preproc
    elif center == 'Columbia':
        preproc = cfg.std_preproc
    elif center == 'Munich':
        preproc = cfg.mun_preproc
    elif center == 'Italy':
        preproc = cfg.it_preproc

    return center, setup, filename, filename2, is_control, datapath, filepath, preproc, is_imagine


def get_badchans(preproc):
    """ Get the self defined bad chans lists, stored in the configs file """

    drop_bads = list()
    dropped_bads = list()
    print("Preprocessing parameters: ", preproc)
    if 'drop_outer' in preproc:
        drop_bads += cfg.outer_outlines
    if 'chan_to_drop' in preproc:
        drop_bads += cfg.chan_to_drop
    if 'drop_ears' in preproc:
        drop_bads += cfg.ears_outlines
    if 'bios' in preproc:  # bio chans (ECG,respi,EMG)
        drop_bads += cfg.bios
    if 'bad_it_chans' in preproc:  # chans to drop for Italy
        drop_bads += cfg.bad_it_chans
    if 'bad_mun_chans' in preproc:  # chans to drop for Munich
        drop_bads += cfg.bad_mun_chans
    return drop_bads


def read_raw(filepath, preproc):
    """ Read and filter raw data according to preprocessing parameters defined in 'configs' """

    print(preproc)

    try:
        if filepath.endswith('.fif'):  # Setup for Columbia
            raw = mne.io.read_raw_fif(filepath, preload=True)
            # Add an extension description to target the setup
            raw.info['ext'] = '.fif'

        elif filepath.endswith('.vhdr'):  # Setup for Italy
            raw = mne.io.read_raw_brainvision(filepath, preload=True)
            # Add an extension description to target the setup
            raw.info['ext'] = '.vhdr'

        elif filepath.endswith('.mff'):  # Setup for Paris & Germany
            raw = mne.io.read_raw_egi(filepath, preload=True)
            raw.info['ext'] = '.mff'

        else:
            print("Couldn't find this file extension : '{}'".format(filepath[-4:]))

        # Band-pass filter between 1 and 30 Hz over continuous recordings
        # By default, MNE uses a zero-phase FIR filter with a hamming window

        raw.filter(fmin, fmax)

    except:
        pass
        print("A problem occured while trying to open the file '{}'".format(filepath))

    return raw

def diy_drop_bads(raw,preproc):
    drop_bads = get_badchans(preproc)  # manually defined chans we want to drop for this setup
    dropped_bads = list()  # will store the chans to drop

    # Drop manually chosen bad chans (stored in preproc & configs)
    if len(drop_bads) > 0:
        for i in range(len(drop_bads)):
            if drop_bads[i] in raw.ch_names:
                dropped_bads.append(drop_bads[i])
                raw.drop_channels(drop_bads[i])
        print('Dropping %i unwanted chans' % (len(dropped_bads)), dropped_bads)
    
    return raw

def get_instructions(raw, df, subject_id, run_imaginev, run_activev):
    """ Load the reconstructed instructions and events from self defined functions, according to the EEG setup """

    #setup = df.loc[subject_id]['Setup']
    raw.info['center'] = df.loc[subject_id]['Center']
    is_imagine = df.loc[subject_id]['Condition'] == 'imagine'
    setup = df.loc[subject_id]['Setup']
    raw.info['setup'] = setup
    raw.info['montage'] = df.loc[subject_id]['Montage']
    raw.info['lang'] = df.loc[subject_id]['LANG']

    if setup == 'EGI256':
        events, events_info, instructions = egi256_instruction(raw)
    elif setup == 'easycap':
        events, events_info, instructions = easycap_instructions(raw)
    elif setup == 'EGI400':
        events, events_info, instructions = egi400_instructions(raw)
    else:  # 10/20standard montage
        events, events_info, instructions = std_instructions(raw)

    # ==> reconstruct dict of events from available events
    event_dict = dict()
    uniquev = np.unique(events[:, 2])
    if 1 in uniquev:  # right events
        event_dict['keep/right'] = 1
        event_dict['stop/right'] = 2
    if 3 in uniquev:  # left events
        event_dict['keep/left'] = 3
        event_dict['stop/left'] = 4

    # check for imaginary condition to reconstruct the events if present
    if is_imagine:
        print('Reconstructing imaginary condition')
        events, events_info, instructions, im_event_dict = retrieve_im_events(raw, subject_id, instructions, df)
        if run_imaginev == True:
            if run_activev == True:
                event_dict.update(im_event_dict)  # ALL events (active & imaginary)
            else:
                event_dict = im_event_dict  # only IM events
        else:
            event_dict = event_dict  # only ACT events

    print('Code of events : ', event_dict)
    return events, events_info, event_dict


def std_instructions(raw):
    instructions = mne.find_events(raw)
    n_samples = raw.info['sfreq'] * 10. / n_epo_segments

    # ==> /!\ mne.find_events takes the onset, so get the offset of the instruction
    # ==> /!\ Modifications for Columbia ! Might have to change the duration for other setups (according to the legnth of the instructions)
    for indx, instr in enumerate(instructions):
        if instructions[indx, 2] == 1:
            instructions[indx, 0] += 794  # the "keep op" instr lasts 794/raw.info['sfreq'] sec (env 3.10sec)
        else:
            instructions[indx, 0] += 830  # the "stop op" instr lasts 830/raw.info['sfreq'] sec (env 3.24sec)

    # ==> check up that the instructions match
    # HVinstr = np.array(mne.read_events('/Users/romybeaute/MOTOR/Columbia/controls/healthy_volunteer_events-eve.fif'))
    # HVinstr == instructions

    events = list()
    events_info = list()

    # For each instruction
    for instr_id, (onset, _, code) in enumerate(instructions):

        # Generate 5 epochs
        for repeat in range(n_epo_segments):
            event = [onset + repeat * n_samples, 0, code]

            # Store this into a new event array
            events.append(event)
            events_info.append(instr_id)
    events = np.array(events, int)

    return events, events_info, instructions


def easycap_instructions(raw):
    # ==> Reconstructing the events and segmenting trials into epochs

    n_samples = raw.info['sfreq'] * 10. / n_epo_segments  # freq : 1000.0

    event_id = {'move/right': 1, 'stop/right': 2, 'move/left': 3, 'stop/left': 4}
    

    if raw.info['center'] == 'Italy':
        mcp_event_id = cfg._italy_mcp_event_id
        print(mcp_event_id)
        #mcp_event_id = cfg._italy_mcp_event_id2 #for SMN011_2
    elif raw.info['center'] == 'Munich':
        mcp_event_id = cfg._munich_mcp_event_id

    # Instructions
    instructions = mne.events_from_annotations(raw)[0]


    # Re create events from instructions

    event = list()
    events = list()
    events_info = list()
    instr_id = -1

    for indx, (onset, _, code) in enumerate(instructions):
        # Generate 5 epochs
        if code in mcp_event_id.keys():
            instr_id += 1
            instr_type = mcp_event_id[code]  # récupère le type de commande
            new_code = event_id[instr_type]
            for repeat in range(n_epo_segments):  # creation of 5 epochs per instruction
                event = [onset + repeat * n_samples, 0, new_code]
                events.append(event)
                events_info.append(instr_id)

    events = np.array(events, int)

    return events, events_info, instructions


def egi256_instruction(raw):
    """ Get instructions and events for EGI256 setup """

    # Events parameters
    event_id = {1: 'keep/right', 2: 'stop/right', 3: 'keep/left', 4: 'stop/left'}  # correspondance of events

    if raw.info['lang'] == 'fr':
        event_key = {194: 1, 130: 2, 210: 3, 146: 4}  # keys of events on stim_data for French language
    elif raw.info['lang'] == 'en':
        event_key = {193: 1, 129: 2, 209: 3, 145: 4}  # keys of events on stim_data for English language

    ### Instructions
    # Creation of steam_data
    dnames = [x for x in raw.ch_names if x.startswith('D')]
    ttl_fix = True
    if 'DIN3' in raw.ch_names:
        din3 = mne.pick_channels(raw.ch_names, ['DIN3'])[0]
        if np.sum(raw._data[din3, :]) > 10:
            ttl_fix = False

    values = [int(x.replace('DIN', '').replace('DI', '').replace('D', ''))
              for x in dnames]
    if ttl_fix:
        values = 255 - np.array(values)
    stim_data = np.zeros_like(raw._data[raw.ch_names.index(dnames[0]), :])

    for dchan, dvalue in zip(dnames, values):
        if dvalue == 0:
            continue
        ddata = raw._data[raw.ch_names.index(dchan), :]
        idx = np.where(ddata != 0)[0]
        stim_data[idx] = dvalue

    # ==> Creation of INSTRUCTIONS object
    event_dict = {value: key for key, value in event_id.items()}

    instructions = []
    for onset, code in enumerate(stim_data):
        for key, value in event_key.items():
            if code == key:
                instructions.append([onset, 0, value])
    instructions = np.array(instructions)

    # ==>
    #if subject_id == 'AH303', suppress some events
    '''
    goodinstr=instructions[:64].tolist()
    goodinstr.extend(instructions[77:].tolist())
    new_instr=np.array(goodinstr)
    instructions = new_instr'''

    # ==> Creation of EVENTS objects
    n_samples = raw.info['sfreq'] * 10. / n_epo_segments  # lenght of 1 epoch

    events = list()
    events_info = list()

    # Iteration of events and trials : each 10 sec following a given instruction is split in 5 epochs (of 2 seconds)
    for instr_id, (onset, _, code) in enumerate(instructions):

        # generate 5 epochs
        for repeat in range(n_epo_segments):
            event = [onset + repeat * n_samples, 0, code]

            events.append(event)
            events_info.append(instr_id)
    events = np.array(events, int)

    return events, events_info, instructions


def egi400_instructions(raw):
    """ Get instructions and events for EGI400 setup """
    print('Reconstructing events from EGI400 setup')

    event_key = {196: 1, 132: 2, 212: 3, 148: 4}  # keys of events on stim_data for GERMAN language

    dnames = [x for x in raw.ch_names if x.startswith('D')]
    if 'DIN1' in dnames:
        raw.drop_channels('DIN1')
        dnames.remove('DIN1')

    stims_data = raw.copy().pick(dnames).get_data().astype(np.int)  # prend data pour chaque point temp et chaque stim chan (n=8 chans)
    stims_data = (stims_data != 0).astype(np.int)  # binary: 1 if event
    #trig_mult = np.array([0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])
    trig_array = list()
    trig_dict = {
    'DIN1' : 0x1,
    'DIN2' : 0x2,
    'DIN3' : 0x4,
    'DIN4' : 0x8,
    'DIN5' : 0x10,
    'DIN6' : 0x20,
    'DIN7' : 0x40,
    'DIN8' : 0x80
    }
    
    for i, dname in enumerate(dnames):
        trig_array.append(trig_dict[dname])
    trig_array = np.array(trig_array)
    print('Stim chans : {} ==> Trig array : {}'.format(dnames,trig_array))

    # Crée 8 arrays codant pour chaque trigger : len(raw_trig_data) = 8
    #raw_trig_data = np.multiply(stims_data.T,trig_mult).T  # associe num à chaque stim #==> multiplie chaque stim trig par sa valeur associée
    raw_trig_data = np.multiply(stims_data.T,trig_array).T
    
    sumi = np.sum(raw_trig_data, axis=0)  # sum of all the values in one sample

    _germany_mcp_event_id = {
        'move/right': 196,
        'stop/right': 132,
        'move/left': 212,
        'stop/left': 148
    }

    trig_data = np.zeros_like(sumi)

    non_zero_idx = np.where(sumi != 0)[0]  # times of stim events
    non_zero_vals = sumi[non_zero_idx]  # stim events' codes
    max_samples = int(raw.info['sfreq'])  # 500Hz in Germany

    triggers_times = np.column_stack((non_zero_vals, non_zero_idx))

    if 'STI 014' not in raw.ch_names:
        stim_name = [
            x['ch_name'] for x in raw.info['chs']
            if x['kind'] == mne.io.constants.FIFF.FIFFV_STIM_CH][0]
        mne.rename_channels(raw.info, {stim_name: 'STI 014'})
        dnames.remove(stim_name)

    # For the MCP : the values 196, 132, 202 and 148 are encoded in two or three consecutive samples,
    # so sum the two/three samples in the ith place and add a 0 in the i+1th and i+2nd place
    for i in range(sumi.shape[0] - 1):
        if (sumi[i] != 0) and (sumi[i + 1] != 0):
            # print(raw_sumi[i-2],raw_sumi[i-1], raw_sumi[i], raw_sumi[i + 1], raw_sumi[i+2])
            suma = sumi[i] + sumi[i + 1] + sumi[i + 2]
            # if suma in _icm_mcp_event_id.values(): #avoid summing if not in our list
            sumi[i] = suma
            sumi[i + 1] = 0
            sumi[i + 2] = 0

    stim = mne.pick_channels(raw.info['ch_names'], include=['STI 014'])
    raw._data[stim] = sumi  # trig_data
    raw.event_id = _germany_mcp_event_id
    # raw.drop_channels(dnames)

    instructions_sumi = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)

    n_samples = raw.info['sfreq'] * 2
    events = list()
    events_info = list()

    # iteration of events and trials

    a = -1
    for instr_id, (onset, _, code) in enumerate(instructions_sumi):
        if instructions_sumi[instr_id, 2] in _germany_mcp_event_id.values():
            a += 1
            # generate 5 epochs
            for repeat in range(n_epo_segments):
                code = change_code(code)
                event = [onset + repeat * n_samples, 0, code]
                events.append(event)
                events_info.append(a)

    events = np.array(events, int)

    return events, events_info, instructions_sumi


def retrieve_im_events(raw, subject_id, instructions, df):
    """ Enter the id of IM blocks, registered in df (list)"""
    # load parameters
    # event_dict_im = cfg.event_dict #basic dict to be updated with im event's codes
    event_dict_im = dict()
    nblocks = int(df.loc[subject_id]['nblocks'])  # number of blocks
    btypes = df.loc[subject_id]['Block1':'Block{}'.format(nblocks)].values.tolist()  # list of blocks' type
    idx_imblocks = [idx for idx, block in enumerate(btypes) if block.startswith('I')]  # indexes of im blocks

    # variables
    im_events = list()
    im_events_info = list()
    n_samples = raw.info['sfreq'] * 10. / n_epo_segments  # lenght of 1 epoch

    # ==> retrieve & update instructions with im condition
    for block in range(len(idx_imblocks)):  # for each im block
        im_id = idx_imblocks[block] * 16  # 8 x 2 instructions / block
        for instr_id, (onset, _, code) in enumerate(instructions[im_id:im_id + 16]):
            instructions[im_id + instr_id][2] = instructions[im_id + instr_id][2] * 10 + instructions[im_id + instr_id][
                2]

    # ==> retrieve & update im events & trials
    for instr_id, (onset, _, code) in enumerate(instructions):
        # generate 5 epochs
        for repeat in range(n_epo_segments):
            event = [onset + repeat * n_samples, 0, code]
            im_events.append(event)
            im_events_info.append(instr_id)

    if 11 in np.unique(instructions[:, 2]):  # right imagine events
        event_dict_im['keep/imagine/right'] = 11
        event_dict_im['stop/imagine/right'] = 22
    if 33 in np.unique(instructions[:, 2]):  # left imagine events
        event_dict_im['keep/imagine/left'] = 33
        event_dict_im['stop/imagine/left'] = 44

    im_events = np.array(im_events, int)
    return im_events, im_events_info, instructions, event_dict_im


def change_code(code):
    """ Change the code of the events for EGI 400 (shouldn't be necessary anymore)"""

    if code == 196:  # move right
        code = 1
    elif code == 132:  # stop right
        code = 2
    elif code == 212:  # move left
        code = 3
    elif code == 148:  # stop left
        code = 4
    return code


def create_epochs(raw, events, events_info, event_dict):
    """
    Creation of Epochs objects and Metadata from specific events and segmented trials from instructions
    :param raw:
    :param events:
    :param events_info:
    :param event_dict:
    :return: instructions, epochs, metadata
    """

    # ==> Specify USE_CH : channels used to create the epochs


    if raw.info['setup'] == 'easycap':  # Italy & Munich
        #use_ch = raw.info['ch_names']
        raw_setup = 'eeg/62'

    elif raw.info['setup'] == 'standard/1020':  # 21 chans
        #use_ch = cfg.use_ch21
        raw_setup = 'standard/1020'

    elif 'EGI' in raw.info['setup']:  # EGI
        #use_ch = [ch for ch in raw.info['ch_names'] if ch.startswith('E') and ch not in cfg.bios]
        raw_setup = 'egi/256'


    # ==> Creation of METADATA
    metadata = DataFrame(dict(
        time_sample=events[:, 0],  # time sample in the EEG file
        id=events[:, 2],  # the unique code of the epoch
        move=(events[:, 2] % 2) == 1,  # whether the code corresponds to a 'move' trial
        is_right=(events[:, 2] < 3),  # whether the code corresponds to a 'left' trial
        is_imagine=(events[:, 2] > 10),  # whether the code corresponds to an 'imaginary' condition
        instr=events_info,  # the instruction from which the epoch comes
        trial=np.array(events_info) // 2,  # trial number: there are two instructions per trial
    ))
    # There are 8 trials per block
    metadata['block'] = metadata['trial'] // 8

    # ==> Creation of EPOCHS (segment 10 s post-instructions into n epochs lasting 2 s)
    picks = mne.pick_types(raw.info, eeg=True, stim=False, bio=False)

    for indx, key in enumerate(event_dict):
        #update the metadata according to the dict to plot all the SP configurations
        metadata[key] = (events[:, 2] == event_dict[key])


    epochs = mne.Epochs(
        raw,
        tmin=0.0, tmax=10. / n_epo_segments,
        picks=picks,  # Selected channels
        events=events,
        metadata=metadata,
        event_id=event_dict,  # Event information
        preload=True,
        proj=False,  # No additional reference
        baseline=None
    )

    # ==> Compute Curent Source Densisty (CSD)
    if raw.info['description']:
        epochs.info['description'] = raw.info['description']
    else:
        epochs.info['description'] = raw_setup
    print('Computing CSD with {} setup for epochs : '.format(epochs.info['description']))
    epochs = pycsd.epochs_compute_csd(epochs)

    return epochs, metadata


def concatenate_rec(datapath, files, preproc, subject_id, center):
    """ concatenate recordings if in several files ("files" : list of rec files) """
    print('Concatenating {} recordings : {}'.format(len(files), files))
    raws = list()  # will store the partial raws
    
    if center in ['Munich','Italy']:
        print(datapath,1)
        datapath = os.path.join(datapath, subject_id)
        print(datapath,2)
    
    for i in range(len(files)):
        filepath = os.path.join(datapath, files[i])
        try:
            splitraw = read_raw(filepath, preproc)
            raws.append(splitraw)
            del splitraw
        except:
            pass
            print("A problem occured while trying to open the file '{}'".format(filepath))
    rawss = raws.copy()
    try:
        raws = mne.equalize_channels(raws)  # Equalize channel picks and ordering across multiple MNE-Python objects
        raw = mne.concatenate_raws(raws)
    except:
        mne.equalize_channels(rawss)
        raw = mne.concatenate_raws(rawss)
    del raws,rawss
    return raw


def compute_psd(epochs, bands=bands, fmin=fmin, fmax=fmax):
    """
    Compute power spectral densities for each frequency band
    :param epochs:
    :param bands:
    :param fmin:
    :param fmax:
    :return:
    """
    print('Computing PSD in {} bands ... '.format(bands))
    full_psd_data, frequencies = psd_multitaper(epochs, fmin=fmin, fmax=fmax)
    n_epochs, n_chans, n_freqs = full_psd_data.shape
    print('Full PSD shape : ', full_psd_data.shape)

    # Setup X array: average PSD within a given frequency band
    psd_data = np.zeros((n_epochs, n_chans, len(bands)))
    for ii, (fmin, fmax) in enumerate(bands):
        # Find frequencies
        freq_index = np.where(np.logical_and(frequencies >= fmin,
                                             frequencies <= fmax))[0]

        # Mean across frequencies
        psd_data[:, :, ii] = full_psd_data[:, :, freq_index].mean(2)

        # Vectorize PSD: i.e. matrix of n_trials x (channel x 4 frequency bands)
    psd_data = psd_data.reshape(n_epochs, n_chans * len(bands))
    print('Reshaped PSD data (shape of feature array) : {} (n_epochs, n_chans * len(bands))'.format(psd_data.shape))

    return psd_data


"""
In order to prevent training the classifier on potentially temporally-correlated 
neighboring epochs, we apply a leave one-trial-out cross-validation (CV).
Consequently, the training set consists of 2-second long epochs that are not temporally adjacent to those used during the test set.
"""


def compute_probas(psd_data, epochs, subject_id, plot):
    """
    # ==> Defining CV and CLASSIFIER (SVM), then calculate the accuracy of the classifier, when predicting 1) p('keep moving') and 2) class for each epoch
    Plot the probas if plot_plobs == True
    :param psd_data:
    :param epochs:
    :return: y_pred (class proba)
    """

    # ==> Defining cross validation
    cv = LeaveOneGroupOut()

    # ==> Defining classifier (Support Vector Machine)

    clf = make_pipeline(
        StandardScaler(),  # z-score centers data
        SVC(kernel='linear', probability=True))  # linear SVM to decode two classes

    # ==> Calculate p("keep moving") for each epoch using cross_val_predict associated with "predict_proba" method (decode performance over time)
    # compute class PROBABILITY for each epoch using the same previously described CV
    y_pred = cross_val_predict(
        clf,
        X=psd_data,
        y=epochs.metadata['move'],
        method="predict_proba",
        cv=cv,
        groups=epochs.metadata['trial']
    )
    # The predictions are probabilitic
    # Consequently, P(move|EEG) = 1 - P(not move|EEG) : we can keep only one category (/!\ change if multiclass /!\)
    epochs.metadata['y_pred'] = y_pred[:, 1]

    if plot == True:

        # Average the proba over the blocks to obtain the temporal pattern
        proba = np.mean([block['y_pred'] for _, block in epochs.metadata.groupby('block')],
                        axis=0)
        """"
        #(prob won't need this, I used it to calculate the mean of Temporal patterns)
        print('Saving probs ...')
        #np.save("numpy_probs/mean_probas_across_blocks_{}".format(subject_id),proba)
        print(os.getcwd())
        np.save("mean_probas_across_blocks_{}".format(subject_id), proba)
        print('Saved probs!')
        """

        nblocks = len(epochs.metadata.groupby('block'))

        # Plot the probability of 'keep moving'
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_ylabel('P ("keep moving ...")')
        ax.set_xlabel('Time (trial number)')
        ax.plot(proba, marker='o', linestyle='-',
                linewidth=3, markersize=6)
        ax.set_ylim([0.0, 1.0])
        plt.axhline(0.5, linestyle=':', color='k', label='Chance')

        for x in np.arange(-0.5, 79, 10):
            plt.axvline(x, color='g', label="'keep moving ...'" if x < 4 else None)
            plt.axvline(x + 5., color='r', label="'stop moving ...'" if x < 4 else None)
        plt.legend(loc='lower right', framealpha=1.)
        plt.title("Average predicted probability of 'keep moving ...' across all {} blocks for {} ".format(nblocks,
                                                                                                           subject_id))
        sns.despine()

        plt.close(fig)

    return y_pred, fig if 'fig' in locals() else None

    


def compute_prob_class(psd_data, epochs):
    """
    # ==> Defining CV and CLASSIFIER (SVM), then calculate the accuracy of the classifier, when predicting 1) p('keep moving') and 2) class for each epoch
    :param psd_data:
    :param epochs:
    :return: y_pred (class proba)
    """

    # ==> Defining cross validation
    cv = LeaveOneGroupOut()

    # ==> Defining classifier (Support Vector Machine)

    clf = make_pipeline(
        StandardScaler(),  # z-score centers data
        SVC(kernel='linear', probability=True))  # linear SVM to decode two classes

    # ==> Compute the PREDICTED CLASS class of each epoch using cross_val_predict associated with "predict" method (decode class prediction)
    y_pred_class = cross_val_predict(
        clf,
        X=psd_data,
        y=epochs.metadata['move'],
        cv=cv,
        groups=epochs.metadata['trial']
    )

    # ==> Check that we get the same accuracy with the method used to calculate the AUC score
    acc_scores = cross_val_score(
        estimator=clf,  # The SVM
        X=psd_data,  # The 4 bands of PSD for each channel
        y=epochs.metadata['move'],  # The epoch categories
        # scoring=‘roc_auc’,               # Summarize performance with the Area Under the Curve
        cv=cv,  # The cross-validation scheme
        groups=epochs.metadata['trials'])  # use for cv)
    # mean of scores’ array of the estimator for each run of the cross validation.
    print("{:.2f} accuracy with a standard deviation of {:.2f}".format(acc_scores.mean(), acc_scores.std()))

    prediction_accuracy = accuracy_score(epochs.metadata['move'], y_pred_class,
                                         normalize=False)  # if normalize=False, give the number of correct predicted epochs
    prediction_accuracy_frac = prediction_accuracy / len(epochs)  # same as if normalize=True
    print(' - Prediction_accuracy : {:.3f} \n - Number of correctly classified samples : {} / {}'.format(
        prediction_accuracy_frac, prediction_accuracy, len(epochs)))
    print(' - Prediction_accuracy with "predict_proba" method : ', )

    return y_pred_class, acc_scores

def metadata_for_sp(metadata,event_dict,events):
    #update the metadata according to the dict to plot all the SP configurations
    for indx, key in enumerate(event_dict):
        metadata[key] = (events[:, 2] == event_dict[key])
    return metadata



def plot_sp(psd_data, epochs, raw, show_names):
    """
    # ==> Define the classifier and stode spatial patterns
    To plot the SVM patterns, it is necessary to compute the data covariance (Haufe et al Neuroimage 2014).
    Spatial patterns are automatically stored by MNE LinearModel.
    """

    # Define classifier
    clf = make_pipeline(
        StandardScaler(),  # z-score to center data
        LinearModel(LinearSVC()))  # Linear SVM augmented with an automatic storing of spatial patterns

    # fit classifier
    clf.fit(X=psd_data,
            y=epochs.metadata['move'])

    # Unscale the spatial patterns before plotting
    patterns = get_coef(clf, 'patterns_', inverse_transform=True)

    # In our study, the SVM is trained on all frequencies simultanouesly
    # we thus pull the corresponding spatial topographies apart
    n_elect = int(patterns.shape[0] / len(bands))
    spatial_pattern = patterns.reshape(n_elect, len(bands))

    # Plot

    #raw.set_montage(montage=raw.info['montage'])

    posarray = np.array([raw.info['chs'][i]['loc'][:2].tolist() for i, p in enumerate(raw.info['chs']) if raw.info['chs'][i]['kind'] == 2])
    posarray_names = [raw.info['chs'][i]['ch_name'] for i, p in enumerate(raw.info['chs']) if raw.info['chs'][i]['kind'] == 2]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    fig.tight_layout()

    for idx, (band, sp, ax) in enumerate(zip(bands, spatial_pattern.T, axes)):
        scale = np.percentile(np.abs(sp), 99)
        print(' Spatial patterns over %i - %i Hz' % band)
        im, _ = mne.viz.plot_topomap(sp,
                                     pos=posarray,
                                     names = posarray_names,
                                     show_names=show_names,
                                     vmin=-scale, vmax=+scale,
                                     # extrapolate = 'head',
                                     cmap='RdBu_r',
                                     axes=ax,
                                     show=False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('%i - %i Hz' % band)

    plt.close(fig)
    return fig


def compute_auc(psd_data, epochs,permutations, report_folder):
    """
    Computing cross-validated AUC scores : measure of separability of the epoch types (move vs rest)
    => tells us how well the model is capable of distinguishing between different classes
    :param psd_data:
    :param epochs:
    :param permutations:
    :return: scores, mean_score, pvalue
    """

    cv = LeaveOneGroupOut()

    # Set the SVM classifyer with a linear kernel
    clf = make_pipeline(
        StandardScaler(),  # z-score to center data
        LinearSVC()  # Fast implementation of linear support vector machine
    )

    # Computes SVM decoding score with cross-validation
    scores = cross_val_score(
        estimator=clf,  # The SVM
        X=psd_data,  # The 4 bands of PSD for each channel
        y=epochs.metadata['move'],  # The epoch categories
        scoring='roc_auc',  # Summarize performance with the Area Under the Curve
        cv=cv,  # The cross-validation scheme
        groups=epochs.metadata['trial'],  # use for cv => the CV compare one trial with the (k-1) others
    )

    mean_score = scores.mean(0)
    print('Mean scores across split: AUC=%.3f' % mean_score)

    if mean_score > 0.505:
        """
        # == > Diagnosis of cognitive motor dissociation (CMD) : performing PERMUTATION TEST
        -> To evaluate whether the mean AUC score significantly differs from chance, we performed a permutation test (Good P. 2006; Noirhomme Q et al. Neuroimage Clin. 2014; Noirhomme et al. Neuroimage. 2017).
        -> This procedure consists of training and evaluating the same classifier several times (n=500) after randomly shuffling the target labels (i.e. “keep moving …” and “stop moving …”).
        -> A recording is considered to reveal command following (i.e., diagnosis of Cognitive Motor Dissociation in clinically unresponsive patients) if less than 5% of the AUCs obtained with scrambled labels are superior or equal to the mean AUC obtained using the real labels.
        """

        if permutations is not None:

            permutation_scores = []
            n_permutations = permutations
            order = np.arange(len(epochs))

            for _ in tqdm_notebook(range(n_permutations)):
                # Shuffle order
                np.random.shuffle(order)

                # Compute score with similar parameters
                permutation_score = cross_val_score(
                    estimator=clf,
                    X=psd_data,
                    y=epochs.metadata['move'].values[order],
                    scoring='roc_auc',
                    cv=cv,
                    groups=epochs.metadata['trial'].values[order],
                    n_jobs=-1,  # multiple core
                )

                # Store the results
                permutation_scores.append(permutation_score.mean(0))

            # The p-value is computed from the number of permutations which leads to a higher score than the one obtained without permutation
            # p = n_higher + 1 / (n_permutation + 1) ==> (Ojala M GG. Journal of Machine Learning Research. 2010).

            n_higher = sum([s >= scores.mean(0) for s in permutation_scores])
            pvalue = (n_higher + 1.) / (n_permutations + 1.)
            print("Empirical AUC = %.2f +/-%.2f" % (scores.mean(0), scores.std(0)))
            print("Shuffle AUC = %.2f" % np.mean(permutation_scores, 0))
            print("p-value = %.4f" % pvalue)

            # plot permutation and empirical distributions
            fig = plt.figure()

            sns.kdeplot(permutation_scores, label='permutation scores')
            sns.kdeplot(scores)

            plt.axvline(.5, linestyle='--', label='theoretical chance')
            plt.axvline(scores.mean(), color='orange', label='mean score')
            plt.scatter(scores, 6. + np.random.randn(len(scores)) / 10., color='orange',
                        s=5, label='split score')
            plt.xlim(-.1, 1.1)

            plt.legend()
            plt.xlabel('AUC Score')
            plt.ylabel('Probability')
            plt.yticks([])
            fig.savefig(report_folder+"/permutations.png")
            plt.close(fig)

    return scores, mean_score, pvalue if 'pvalue' in locals() else None


### PLOT FUNCTIONS ###

def plotcv(cv, psd_data, epochs):
    # Here, we plot 4 splits as examples: CV #1, #2, #3 as well as split #9.
    fig, axes = plt.subplots(4, 1, figsize=[14, 9])
    axes = iter(axes)

    for split, (train, test) in enumerate(cv.split(
            X=psd_data,  # The 4 bands of PSD for each channel
            y=epochs.metadata['move'],  # The epoch category
            groups=epochs.metadata['trial'],  # used to avoid neighboring epochs
    )):

        # Avoid plotting all splits
        if split >= 3 and split != 8:
            continue
        ax = next(axes)

        # Plot train and test epochs sample
        ax.scatter(epochs.metadata['time_sample'].values[train],
                   1 - epochs.metadata['move'].values[train],
                   color='k', label='train')
        ax.scatter(epochs.metadata['time_sample'].values[test],
                   1 - epochs.metadata['move'].values[test],
                   edgecolor='b', color='w', label='test')
        ax.set_title('CV Split #%i' % (split + 1))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['"Keep moving..."', '"Stop moving..."'])
        ax.set_ylabel('Instruction')
        ax.set_xlabel('Time')
        ax.set_xticks([])
        ax.set_xlim(0, 100000)  # zoom for clarity
        ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig

# ==> save the raw file in a '.fif' file
def write_processed_raw(filepath1,filepath2=None):
    raw = mne.io.read_raw_brainvision(filepath1, preload=True)
    if filepath2 is not None:
        raws = list()
        raws.append(raw)
        raw2 = mne.io.read_raw_brainvision(filepath2, preload=True)
        raws.append(raw2)
        raws = mne.equalize_channels(raws)
        del raw,raw2
        raw = mne.concatenate_raws(raws)
    name = os.path.normpath(filepath1).split(os.sep)[-1]
    raw.save('RAWS/{}_eeg.fif'.format(name))
    return raw



'''
If you want to save all your files in eeg-fif (useful when >1 rec files), you can run the following :
for f in range(len(ids)):
    try:
        subject_id = ids[f]
        filepath = os.path.join(datapath,subject_id)
        files = [file for file in os.listdir(filepath) if file.endswith('.vhdr')]
        filepath1 = os.path.join(filepath,files[0])
        filepath2 = os.path.join(filepath,files[1])
        raw = write_processed_raw(filepath1,filepath2)
        del raw,filepath,filepath1,filepath2
    except:
        pass
'''

