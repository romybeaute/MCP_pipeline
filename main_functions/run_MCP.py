"""
Created by Romy Beauté
romybeaute.univ@gmail.com

Main script to run the Motor Command Protocol
All the used functions are defined in the "main_functions" folder (especially use automatized_MCP.py"

# **************        METHOD SUMMARY        **************

1) Preprocessing raw data
2) Reading events and segmenting trials into epochs
3) Performing Power Spectral Density (PSD) analysis
4) Defining cross validation
5) Defining classifier (Support vector machine)
6) Computing cross-validated AUC scores
7) Diagnosis of cognitive motor dissociation (CMD)

**************        MAIN VARIABLES        **************

- raw : continuous EEG recordings, typically a n_channels x n_times matrix
- epochs : 2 second long EEG recording segments. One trial results in 10 epochs: 5 epochs during which the subject is supposed to move followed by 5 epochs during which the subject is supposed not to move, typically a n_epochs x n_channels x n_ntimes matrix
- psd_data: averaged power spectrum density (psd) within the 4 frequency bands of interest [(1, 3), (4, 7), (8, 13), (14, 30)] typically a n_epochs x n_channels x freq_band


********************************************************
# ==> Initialisation for a given subject type don't need to use it if only want to run a type of subject (default is set for "patients")
while True:
    try:
        subject_type = input('Enter subject type : type "patients" or "controls" : ')
        assert subject_type in ['patients', 'controls'], 'Error in subject type.'
    except AssertionError as msg:
        print(msg)
        continue  # return to the start of the loop to type it again
    else:
        break
"""


import time
import sys
import os

# ==> Self defined functions
# add the sys access to the 'main_functions' personnal folder with all the created functions



if 'romybeaute' in os.getcwd():
    sys.path.append('/Users/romybeaute/Desktop/CMP_pipeline/main_functions') #MAC
    name_for_savings = 'Romy'
elif 'romy.beaute' in os.getcwd():
    sys.path.append('/home/romy.beaute/Bureau/CMP_pipeline/main_functions') #LINUX
    name_for_savings = 'Romy'
elif 'dragana.manasova' in os.getcwd():
    sys.path.append('/home/dragana.manasova/Desktop/CMP_pipeline/main_functions') #LINUX
    name_for_savings = 'Dragana'
elif 'melanie' in os.getcwd():
    sys.path.append('/Users/melanie/Desktop/CMP_pipeline/main_functions')
    name_for_savings = 'Melanie'
else:
    sys.path.append('main_functions')
    name_for_savings = 'Users'



from mapping_functions import diy_change_montage
from automatized_MCP import *
from automatized_MCP import concatenate_rec
from MCP_vars import *
import configs as cfg

plot = True
permutations = None # if True, will perform 500 permutations if the AUC > 0.5
lat = ['right','left'] #change here if only wants to run 'right' or 'left' events
run_imaginev = False  # if True, calculate imaginary events (only for controls)
run_activev = True  # if True, calculate active events (only for controls)
change_montage = 19  # can be 4, 8, 19, 128, 256 or None (only None apply the personnalized and optimazed channel preprocessing)
save_parameters = True #if true, will store the plots (if plot == True) and the subject just runned

subject_type = 'patients'
p = Params(subject_type)
df = p.df
multifiles = "" #list of files if want to run several files at the same time





# ==> Main function to run subject_id

def run_MCP(subject_id, plot=plot, lat=lat):
    startrun = time.time()

    # ************************************************************************
    # PROCESSING
    # ************************************************************************

    # center, setup, filename, filename2, datapath, filepath, preproc, is_imagine = load_setups(subject_id,df)
    p.get_setups(subject_id)
    center = p.center
    setup = p.setup
    filename = p.filename
    filename2 = p.filename2
    datapath = os.path.join(p.datapath, center, subject_type)
    filepath = p.filepath
    preproc = p.preproc
    is_imagine = p.is_imagine

    # ==> Load specific SETUPS

    bands = cfg.bands
    print('------------------------------------------')
    print('Loading {}\n------------------------------------------'.format(filename))
    print('Preprocessing parameters : ', preproc)
    print('Setup : ', setup)

    # create folder for this subject
    if save_parameters == True:
        global report_folder
        #report_folder = os.path.join(p.datapath, "Analysis_reports/{}/{}".format(subject_type, subject_id))
        report_folder = "Analysis_from_main_functions/{}/{}/{}".format(name_for_savings,subject_type, subject_id)
        if change_montage is not None:
            report_folder = os.path.join(report_folder, "{}_montage".format(change_montage))
        if len(lat) != 2:
            report_folder = os.path.join(report_folder, "{}".format(lat[0]))
        if is_imagine:
            if run_imaginev:
                report_folder = os.path.join(report_folder, "Imaginary")
        if plot == True:
            os.makedirs(report_folder, exist_ok=True)
    else:
        report_folder = ''

    # ==> Load & preprocess RAW data
    if str(filename2) != 'nan':
        files = list([filename, filename2])
        raw = concatenate_rec(datapath, files, preproc, subject_id, center)
    else:
        if center in ['Munich','Italy']:
            filepath = os.path.join(datapath,subject_id,filename)
        raw = read_raw(filepath, preproc)
        print('Loading data from {}.'.format(filepath))

    # ==> Define INSTRUCTIONS & EVENTS (before an eventual remapping)
    events, events_info, event_dict = get_instructions(raw, df, subject_id, run_imaginev=run_imaginev,run_activev=run_activev)

    # ==> change montage if needed
    n_eeg_chans = len(mne.pick_types(raw.info, eeg=True, stim=False))

    if change_montage is not None:
        if 'E257' in raw.info['ch_names']:
            raw.drop_channels(['E257'])
        try:
            raw, use_ch = diy_change_montage(raw, nb_eletrodes=change_montage)
            print('Applied a {} --> {} electrodes mapping'.format(n_eeg_chans, change_montage))
            print('---------------------------------')
        except:
            print('Cannot apply this montage for {}'.format(subject_id))
            pass
        """
        if change_montage == 256:
            raw = diy_drop_bads(raw,preproc)
        """

    # ==> no re-mapping
    else:
        # ==> get rid of bad chans we defined
        raw.info['description'] = ""
        raw = diy_drop_bads(raw, preproc)

    used_eeg = mne.pick_types(raw.info, eeg=True, stim=False)  # kept eeg chans after preprocessing & remapping
    print('Used chans : ',
          [raw.info['chs'][i]['ch_name'] for i, p in enumerate(raw.info['chs']) if raw.info['chs'][i]['kind'] == 2])

    print('Keeping {} channels over {} after preprocessing steps ({:.1f} %)'.format(len(used_eeg), n_eeg_chans,len(used_eeg) / n_eeg_chans * 100))
    print('---------------------------------')

    # ==> Construct EPOCHS & define METADATA
    epochs, metadata = create_epochs(raw, events, events_info, event_dict)

    #if save_parameters == True:
    #epochs.save('{}_epo.fif'.format(subject_id))

    if lat != ['right', 'left']:  # only one lateralization
        epochs = epochs['keep/{}'.format(lat[0]), 'stop/{}'.format(lat[0])]
        print('------------------------------------------')
        print('Using {} events.\nUsed epochs : {} '.format(lat[0], epochs))
        print('------------------------------------------')
        metadata = epochs.metadata

    # ************************************************************************
    # FEATURES
    # ************************************************************************

    # ==> Calculate the POWER SPECTRAL DENSITY in the n (=4) previously defined frequency bands
    psd_data = compute_psd(epochs, bands=bands, fmin=fmin, fmax=fmax)

    # ************************************************************************
    # PERFORMANCE SCORES/MEASURES
    # ************************************************************************

    # ==> Compute cross-validated AUC scores
    scores, mean_score, pvalue = compute_auc(psd_data, epochs, permutations=permutations, report_folder=report_folder)

    # ************************************************************************
    # PLOT (if plot == True)
    # ************************************************************************

    if plot == True:

        # define cross-validation
        cv = LeaveOneGroupOut()

        # ==> Block design
        events_fig = mne.viz.plot_events(events, show=False)  # plot the block distribution (study design)
        if save_parameters == True:
            events_fig.savefig(report_folder + "/blocks-design.png")
        plt.close(events_fig)

        # ==> Cross-Validation
        cv_fig = plotcv(cv=cv, psd_data=psd_data, epochs=epochs)  # plot the CV for checking & clarification purposes
        if save_parameters == True:
            cv_fig.savefig(report_folder + "/CV-check.png")

        # ==> Decoding probability
        y_pred, proba_fig = compute_probas(psd_data, epochs, subject_id,plot=True)  # Plot the decoding performance over time
        if save_parameters == True:
            proba_fig.savefig(report_folder + "/decoded-proba.png")

        # ==> Spatial Patterns
        try:
            raw.plot_sensors()
            plt.close()
        except:
            print('No sensors founds, applying the info montage : {}'.format(raw.info['montage']))
            raw.set_montage(montage=raw.info['montage'])  # if pos info not present the raw.info['chs']

        if len(used_eeg) > 100:  # don't plot the names if 128 or 256 montage
            show_names = False
        else:
            show_names = True

        sp_fig = plot_sp(psd_data, epochs, raw, show_names=show_names)
        if save_parameters == True:
            sp_fig.savefig(report_folder + "/spatial-patterns.png")

    endrun = time.time()
    print('------------------------------------------')
    print('Computation over for {}.\nAUC : {:.2}\nTime spent : {}s'.format(subject_id, mean_score,
                                                                           int(endrun - startrun)))
    print('------------------------------------------')

    '''
    text_file = os.path.join(report_folder,"{}_parameters.txt".format(subject_id))
    os.makedirs(text_file, exist_ok=True)
    with open(text_file,'w') as f:
        f.write('AUC = %i'i(mean_score)))
        if pvalue !=None:
            f.write('pvalue : {}'.format(pvalue))
        f.write('Used epochs : {}\n'.format(epochs))
    f.close()
    '''
    del raw, epochs
    return scores, mean_score, pvalue if 'pvalue' in locals() else None




def run_all(multirun):
    results = dict()
    df_idx = {'%s' % multirun[i]: [] for i in range(len(multirun))}
    df_results = pd.DataFrame(index=df_idx, columns=['AUC', 'pvalue', 'lateralization'])

    for subject in range(len(multirun)):
        subject_id = multirun[subject]
        try:
            scores, mean_score, pvalue = run_MCP(subject_id=subject_id, plot=plot, lat=lat)
            df_results.loc[subject_id].AUC = mean_score
            if pvalue is not None:
                df_results.loc[subject_id].pvalue = pvalue
            df_results.loc[subject_id].lateralization = lat
        except:
            print("Error with {}".format(subject_id))
            continue
    print('Computation over for the {} files'.format(len(multirun)))

    return df_results


def run():
    while True:
        try:
            print(
                '\n Enter ID of the subject you want to run. \n To run all subjects, type "all". \n To run all subjects from a single center, enter the name of the center. \n To display the IDs, type "display".')
            subject_id = input(' ID : ')

            # ==> Run a single subject
            if subject_id == "display":
                print('IDs for {} : {}'.format(subject_type, df.index))
                continue
                # subject_id = input(' ID : ')

            # ==> Run all subjects in subject_type csv database
            elif subject_id == "all":
                multirun = df.index.tolist()  # will store list of all subjects' IDs if subject_id == "all"
                df_results = run_all(multirun)  # store the results in a DataFrame
                df_results.to_csv(p.datapath + '/' + 'results_for_{}_{}M.csv'.format(subject_type,
                                                                                     change_montage))  # p.datapath+'/'+'results_for_{}.csv'.format(subject_type))
                print(df_results)

            # ==> Run all subjects in subject_type csv database
            elif subject_id == "multifiles":
                df_results = run_all(multifiles)  # store the results in a DataFrame
                df_results.to_csv(p.datapath + '/' + 'results_for_{}_{}M_{}.csv'.format(subject_type, change_montage,
                                                                                        "multifiles"))  # p.datapath+'/'+'results_for_{}.csv'.format(subject_type))
                print(df_results)

            # ==> Run all subjects from a center (given a subject_type)
            elif subject_id in np.unique(df.Center.to_list()):
                multirun = [ID for ind, ID in enumerate(df.index.tolist()) if df.loc[ID]['Center'] == subject_id]
                df_results = run_all(multirun)  # store the results in a DataFrame
                df_results.to_csv(p.datapath + '/' + 'results_for_{}_{}_{}M.csv'.format(subject_type, subject_id,
                                                                                        change_montage))  # here subject_id == center
                print(df_results)

            elif subject_id == 'imagine':
                multirun = [ID for ind, ID in enumerate(df.index.tolist()) if df.loc[ID]['Condition'] == 'imagine']
                df_results = run_all(multirun)  # store the results in a DataFrame
                df_results.to_csv(p.datapath + '/' + 'results_for_{}_{}_{}M.csv'.format(subject_type, subject_id,
                                                                                        change_montage))  # here subject_id == center
                print(df_results)

            # ==> quit
            elif subject_id == "quit":
                sys.exit(0)

            # ==> Enter the ID directly
            else:
                subject_id = subject_id
                assert subject_id in ['display',
                                      'all'] + df.index.tolist(), 'Subject ID or command "{}" non recognized.'.format(
                    subject_id)  # check if ID parameter exist or correct
                try:
                    scores, mean_score, pvalue = run_MCP(subject_id, plot=plot, lat=lat)
                except:
                    print('Error running {}'.format(subject_id))
                    continue
        except AssertionError as msg:
            print(msg)
            continue  # return to the start of the loop to type it again
        else:
            break

run()
