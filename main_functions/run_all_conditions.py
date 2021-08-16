"""
Created by Romy Beautée
romybeaute.univ@gmail.com

Use it to automatically calculate all subjects (controls and patients) in ALL setups and ALL lateralisations that are entered in the following lists :
montages = [4,8,19,128,256,None]
lateralizations = [['right'],['left'],['right','left']]
subjects = ['controls','patients']
EGI_centers = ['Paris','Germany','Italy','Munich']
==> change any of these parameters if don't want to run everything
==> if you want to do only one subject : enter his ID in run_single_subject
"""
#python functions
import time
import sys
import os


# Self defined functions

sys.path.append('main_functions')

'''
#add the sys access to the 'main_functions' personnal folder with all the created functions
if 'romybeaute' in os.getcwd():
    sys.path.append('/Users/romybeaute/Desktop/CMDcode/main_functions') #MAC
elif 'romy.beaute' in os.getcwd():
    sys.path.append('/home/romy.beaute/Bureau/CMDcode-MAC/main_functions') #LINUX
elif 'dragana.manasova' in os.getcwd():
    sys.path.append('/home/dragana.manasova/Desktop/MCP_Romy/CMDcode-MAC/main_functions') #LINUX
'''

from mapping_functions import diy_change_montage
from automatized_MCP import *
from MCP_vars import *
import configs as cfg


plot = True
permutations = None
run_imaginev = False  # if True, calculate only imaginary events
run_activev = True  # if True, calculate only active events


montages = [4,8,19,128,256,None]
lateralizations = [['right'],['left'],['right','left']]
subjects = ['controls','patients']
EGI_centers = ['Paris','Germany','Italy']
run_single_subject = None # enter here the ID of the subject you want to run if you only want to run this one


'''

def run_MCP(subject_id, lat, plot=plot):
    startrun = time.time()

    # ************************************************************************
    # PROCESSING
    # ************************************************************************

    #center, setup, filename, filename2, datapath, filepath, preproc, is_imagine = load_setups(subject_id,df) 
    p.get_setups(subject_id)
    center = p.center
    setup = p.setup
    filename = p.filename
    filename2 = p.filename2
    datapath = p.datapath
    filepath = p.filepath
    preproc = p.preproc
    is_imagine = p.is_imagine
    bands = cfg.bands

    # ==> Load specific SETUPS
    if center == 'Italy':
        datapath = os.path.join(datapath,center,subject_type,subject_id)
    else:
        datapath = os.path.join(datapath,center,subject_type)
    

    print('------------------------------------------')
    print('Loading {}\n------------------------------------------'.format(filename))
    print('Preprocessing parameters : ', preproc)
    print('Setup : ', setup)

    # create folder for this subject
    global report_folder
    report_folder = os.path.join(p.datapath, "Analysis_reports/{}/{}".format(subject_type,subject_id))
    if change_montage is not None:
        report_folder = os.path.join(report_folder, "{}_montage".format(change_montage))
    if len(lat) != 2:
        report_folder = os.path.join(report_folder, "{}".format(lat[0]))
    if is_imagine:
        if run_imaginev:
            report_folder = os.path.join(report_folder, "Imaginary")
    if plot == True:
        os.makedirs(report_folder, exist_ok=True)



    # ==> Load & preprocess RAW data
    if str(filename2) != 'nan':
        files = list([filename, filename2])
        raw = concatenate_rec(datapath, files, preproc, subject_type, center)
    else:
        raw = read_raw(filepath,preproc)
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
            print('Applied a {} --> {} electrodes mapping'.format(n_eeg_chans,change_montage))
            print('---------------------------------')
        except:
            print('Cannot apply this montage for {}'.format(subject_id))
            pass

        if change_montage == 256:
            raw = diy_drop_bads(raw,preproc)

    # ==> no re-mapping
    else: 
        # ==> get rid of bad chans we defined
        raw.info['description'] = ""
        raw = diy_drop_bads(raw,preproc)
        

    used_eeg = mne.pick_types(raw.info, eeg=True, stim=False) #kept eeg chans after preprocessing & remapping
    print('Used chans : ', [raw.info['chs'][i]['ch_name'] for i,p in enumerate(raw.info['chs']) if raw.info['chs'][i]['kind']==2])
    
    
    print('Keeping {} channels over {} after preprocessing steps ({:.1f} %)'.format(len(used_eeg),n_eeg_chans,len(used_eeg)/n_eeg_chans*100))
    print('---------------------------------')

    # ==> Construct EPOCHS & define METADATA
    epochs, metadata = create_epochs(raw, events, events_info, event_dict)

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
        events_fig.savefig(report_folder + "/blocks-design.png")
        plt.close(events_fig)

        # ==> Cross-Validation
        cv_fig = plotcv(cv=cv, psd_data=psd_data, epochs=epochs)  # plot the CV for checking & clarification purposes
        cv_fig.savefig(report_folder + "/CV-check.png")

        # ==> Decoding probability
        y_pred, proba_fig = compute_probas(psd_data, epochs, subject_id, plot=True)  # Plot the decoding performance over time
        proba_fig.savefig(report_folder + "/decoded-proba.png")

        # ==> Spatial Patterns
        try:
            raw.plot_sensors()
            plt.close()
        except:
            print('No sensors founds, applying the info montage : {}'.format(raw.info['montage']))
            raw.set_montage(montage=raw.info['montage'])  # if pos info not present the raw.info['chs']

        if len(used_eeg) > 100: #don't plot the names if 128 or 256 montage
            show_names = False
        else: 
            show_names = True

        sp_fig = plot_sp(psd_data, epochs, raw, show_names=show_names)
        sp_fig.savefig(report_folder + "/spatial-patterns.png")

    endrun = time.time()
    print('------------------------------------------')
    print('Computation over for {}.\nAUC : {:.2}\nTime spent : {}s'.format(subject_id, mean_score, int(endrun - startrun)))
    print('------------------------------------------')


    del raw,epochs
    return scores, mean_score, pvalue if 'pvalue' in locals() else None



'''
def run_MCP(subject_id, lat, plot=plot):
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
    global report_folder
    report_folder = os.path.join(p.datapath, "Analysis_reports/{}/{}".format(subject_type, subject_id))
    if change_montage is not None:
        report_folder = os.path.join(report_folder, "{}_montage".format(change_montage))
    if len(lat) != 2:
        report_folder = os.path.join(report_folder, "{}".format(lat[0]))
    if is_imagine:
        if run_imaginev:
            report_folder = os.path.join(report_folder, "Imaginary")
    if plot == True:
        os.makedirs(report_folder, exist_ok=True)


    # ==> Load & preprocess RAW data
    if str(filename2) != 'nan':
        files = list([filename, filename2])
        raw = concatenate_rec(datapath, files, preproc, subject_id, center)
    else:
        raw = read_raw(filepath, preproc)
        print('Loading data from {}.'.format(filepath))

    # ==> Define INSTRUCTIONS & EVENTS (before an eventual remapping)
    events, events_info, instructions = get_instructions(raw, df, subject_id, run_imaginev=run_imaginev,run_activev=run_activev)

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

    else:
        # ==> no re-mapping
        # ==> get rid of bad chans we defined (ears,cheeks etc)

        raw.info['description'] = ""
        raw = diy_drop_bads(raw, preproc)

    used_eeg = mne.pick_types(raw.info, eeg=True, stim=False)  # kept eeg chans after preprocessing & remapping
    print('Used chans : ',[raw.info['chs'][i]['ch_name'] for i, p in enumerate(raw.info['chs']) if raw.info['chs'][i]['kind'] == 2])

    print('Keeping {} channels over {} after preprocessing steps ({:.1f} %)'.format(len(used_eeg), n_eeg_chans,
                                                                                    len(used_eeg) / n_eeg_chans * 100))
    print('---------------------------------')

    # ==> Construct EPOCHS & define METADATA
    epochs, metadata = create_epochs(raw, events, events_info, cfg.event_dict)

    #if save_parameters == True:
        #epochs.save('processed_epochs/{}_epo.fif'.format(subject_id))

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

        events_fig.savefig(report_folder + "/blocks-design.png")
        plt.close(events_fig)

        # ==> Cross-Validation
        cv_fig = plotcv(cv=cv, psd_data=psd_data, epochs=epochs)  # plot the CV for checking & clarification purposes
        cv_fig.savefig(report_folder + "/CV-check.png")

        # ==> Decoding probability
        y_pred, proba_fig = compute_probas(psd_data, epochs, subject_id,plot=True)  # Plot the decoding performance over time
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
    #df_results.to_csv()
    return df_results



def run_all_conds(multirun):
    results = dict()
    df_idx = {'%s' % multirun[i]: [] for i in range(len(multirun))}
    df_results = pd.DataFrame(index=df_idx, columns=['AUC', 'pvalue', 'lateralization'])  
    
    for subject in range(len(multirun)):
        subject_id = multirun[subject]
        try:
            scores, mean_score, pvalue = run_MCP(subject_id=subject_id, plot=plot, lat=lat)
            df_results.loc[subject_id].lateralization = lat
            df_results.loc[subject_id].AUC = mean_score
            if pvalue is not None:
                df_results.loc[subject_id].pvalue = pvalue
        except:
            print("Error with {}".format(subject_id))
            continue
    print('Computation over for the {} files'.format(len(multirun)))
    return df_results



for s in range(len(subjects)):
    subject_type = subjects[s]
    p = Params(subject_type)
    df = p.df
    if run_single_subject is not None:
        multirun = [run_single_subject]
    else:
        multirun = df.index.tolist()
    for m in range(len(montages)):
        change_montage = montages[m]
        if change_montage == 4:
            plot = False
        else:
            plot = True
        for l in range(len(lateralizations)):
            lat = lateralizations[l]

            df_results = run_all_conds(multirun)
            if run_single_subject is not None:
                csv_file_path = os.path.join(p.datapath,'results_for_{}_{}M_{}.csv'.format(run_single_subject,str(change_montage),lat))
            else:
                csv_file_path = os.path.join(p.datapath,'results_for_{}_{}M_{}.csv'.format(subject_type, str(change_montage), lat))
            df_results.to_csv(csv_file_path)
            #df_results.to_csv(p.datapath+'/'+'results_for_{}_{}M_{}.csv'.format(run_single_subject,change_montage,lat))
            print('Saved the df in ',csv_file_path)
