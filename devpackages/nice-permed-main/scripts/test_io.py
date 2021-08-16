import nice_ext
import mne
import numpy as np
# raw = nice_ext.api.read(
#     'C:/Users/Dragana/Documents/data/germany_lg/', 
#     'permed/lg/mff/egi400') #CK_LocGlob1_20210312_105209_permed-lg-egi400.mff

raw = nice_ext.api.read(
    '/Users/fraimondo/data/permed/', 'permed/lg/mff/egi400')

events = mne.find_events(raw, shortest_event=1)
mne.viz.plot_events(events, sfreq=raw.info['sfreq'])

# raw = mne.io.read_raw_egi('/Users/fraimondo/data/permed/CK_LocGlob1_20210312_105209-permed-lg-egi400.mff')


# idx_start = int(335 * raw.info['sfreq'])
# idx_end = int(340 * raw.info['sfreq'])
# din_ch_names = ['DIN1', 'DIN2', 'DIN3', 'DIN4', 'DIN5', 'DIN6', 'DIN7',
#                 'DIN8']
# stims_data = raw.copy().pick(din_ch_names).get_data()
# stims_data = (stims_data != 0).astype(np.int)
# stims_data[-1, :] = np.r_[stims_data[-1, 1:], [0]]
# idx = mne.pick_channels(raw.ch_names, include=din_ch_names)
# import matplotlib.pyplot as plt

# plt.figure()

# for i_idx, t_ch_name in enumerate(din_ch_names):
#     plt.plot((stims_data[i_idx, idx_start:idx_end] > 0).astype(np.int) + i_idx * 2)



# from munich
fname = "/home/dragana.manasova/Documents/data/permed/germany/subjects/EEG box test Munich 260721/TestMuc_260721_localglobal.vhdr"

fname = "/home/dragana.manasova/Documents/data/permed/germany/subjects/munich/rs/TestMuc_cogEEG_26jul2021_rs_healthyPhD/TestMuc_260721_restingstate_permed-rs-bv62.vhdr"
raw = mne.io.read_raw_brainvision(fname, preload=True, verbose=True)
events = mne.events_from_annotations(raw)[0]
mne.viz.plot_events(events, sfreq=raw.info['sfreq'])
info = raw.info

raw.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})

channel_type = mne.io.pick.channel_type(info, 63)
print('Channel #63 is of type:', channel_type)

raw.pick_types(eeg=True, stim=True)

# from italy
fname = "/home/dragana.manasova/Documents/data/permed/italy/subjects/lg/SMN020_cogEEG_01jun2021_lg_patient/SMN020_cogEEG_01jun2021_0003_permed-lg-bv.vhdr"
raw = mne.io.read_raw_brainvision(fname, preload=True, verbose=True)
events = mne.events_from_annotations(raw)[0]
mne.viz.plot_events(events, sfreq=raw.info['sfreq'])


from next_icm.lg.constants import _arduino_trigger_map, _icm_lg_event_id

events, _ = mne.events_from_annotations(raw)
valid_events = np.array([x for x in _arduino_trigger_map.keys()])
events[:, 2] -= 1
events = events[np.in1d(events[:, 2], valid_events)]

events[:, 2] = [_icm_lg_event_id[_arduino_trigger_map[x]]
                for x in events[:, 2]]