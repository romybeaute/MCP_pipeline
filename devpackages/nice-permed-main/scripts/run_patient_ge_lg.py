import os
import os.path as op
import time
import sys
import traceback

from argparse import ArgumentParser

import mne
from mne.utils import logger

from nice_ext.api import (utils, read, preprocess, fit, create_report, predict,
                          summarize_subject)

import nice_ext


# mne.viz.plot_montage(epochs.get_montage())

# default_path = '/media/data2/new_patients'
# default_path = '/Volumes/Databases/ICM/doc_stable/'
# default_path = '/Users/fraimondo/data/icm/test'
# default_predict_path = '/media/data/old_patients/group_results/baseline_brain'
# default_predict_path = ('/Volumes/Databases/ICM/doc_stable/group_results/'
#                         'baseline_stable_20210128')

countries = ['germany', 'italy_lg', 'israel_rs']

default_path = '/home/dragana.manasova/Documents/data/permed/' + countries[0]
default_predict_path = ('/home/dragana.manasova/Documents/data/permed/'+countries[0]+'/results'
                        '/baseline/baseline_stable_20210128')

permeddata = ['D-08pTZB_LG_1_20210416_110152_lg_patient_permed-lg-egi400.mff', # p02
              'D-09pTZB_LG_1_20210420_113254_lg_patient_permed-lg-egi400.mff', # p03
              'DZ_LocGlob_20210413_135123_lg_control_permed-lg-egi400.mff.zip', # c01
              'SM_LocalGlobal_20210413_150336_lg_control_permed-lg-egi400.mff.zip' # c02
    ]

permeddata = ['D-14pTZB_LocalGlobal1_20210609_105617_permed-lg-egi400.mff',  # p005_lg # 200mmV
              'D_10-KUM_LG1_20210506_105727_permed-lg-egi400.mff' # p004_lg
    ]

permeddata = ['D-16pTZB_LG_T1_20210702_112952_permed-lg-egi400.mff',  # p006_lg
              'EF_LocGlob1_20210625_103734_permed-lg-egi400.mff',  # p_child_lg thresh=200
              'D-17pTZB_LG1_20210722_112354_permed-lg-egi400.mff' # p007_lg 
    ]

eegdata = permeddata[2]

ge_path = '/home/dragana.manasova/Documents/data/permed/germany/subjects/p007_lg/'


start_time = time.time()

parser = ArgumentParser(description='Run the pipeline on the selected suject')

parser.add_argument('--io', metavar='io', type=str, nargs='?',
                    default='permed/lg/mff/egi400', #'icm/lg/raw/egi',
                    help='IO to use (default = icm/lg/raw/egi)')
parser.add_argument('--preprocess', metavar='preprocess', type=str, nargs='?',
                    default='icm/lg/raw/egi',
                    help='Preprocessing to run (default = icm/lg/raw/egi)')
parser.add_argument('--fit', metavar='fit', type=str, nargs='?',
                    default='icm/lg',
                    help='Fit to run (default = icm/lg)')
parser.add_argument('--report', metavar='report', type=str, nargs='?',
                    default='icm/lg',
                    help='Report to create (default = icm/lg)')
parser.add_argument('--path', metavar='path', nargs=1, type=str,
                    help='Path with the database.',
                    default=default_path)
parser.add_argument('--subject', metavar='subject', nargs=1, type=str,
                    default=eegdata,
                    help='Subject name')#, required=True)
parser.add_argument('--predict-path', metavar='pred_path', nargs=1, type=str,
                    help='Path with the database. to predict',
                    default=default_predict_path)
parser.add_argument('--runid', metavar='runid', type=str, nargs='?',
                    default=None,
                    help='Run id (default = generated)')

args = parser.parse_args()
db_path = args.path
subject = args.subject
io_config = args.io
fit_config = args.fit
report_config = args.report
preprocess_config = args.preprocess
predict_path = args.predict_path

if isinstance(db_path, list):
    db_path = db_path[0]

if isinstance(subject, list):
    subject = subject[0]

if isinstance(predict_path, list):
    predict_path = predict_path[0]

if args.runid is not None:
    run_id = args.runid
    if isinstance(run_id, list):
        run_id = run_id[0]
else:
    run_id = utils.get_run_id()
s_path = op.join(db_path, 'subjects', subject)

if not op.exists(op.join(db_path, 'results')):
    os.mkdir(op.join(db_path, 'results'))

results_dir = op.join(db_path, 'results', run_id)
if not op.exists(results_dir):
    os.mkdir(results_dir)

if not op.exists(op.join(results_dir, subject)):
    os.mkdir(op.join(results_dir, subject))

now = time.strftime('%Y_%m_%d_%H_%M_%S')
log_suffix = '_{}.log'.format(now)
mne.utils.set_log_file(op.join(results_dir,
                               subject,
                               subject + log_suffix))
utils.configure_logging()
utils.log_versions()

logger.info('Running {}'.format(subject))
report = None
# try:
if True:

    # # Read 
    # io_params, _ = utils.parse_params_from_config(io_config)
    # data = read(s_path, io_config, io_params)

    # # Preprocess
    # preprocess_params, preprocess_config = \
    #     utils.parse_params_from_config(preprocess_config)
    # preprocess_params.update({'summary': True})
    # epochs, summary = preprocess(data, preprocess_config, preprocess_params)

    # # Preprocess Report
    # report = mne.report.Report(title=subject)
    # create_report(epochs, config='icm/preprocess',
    #               config_params=dict(summary=summary), report=report)

    # Read
    data = nice_ext.api.read(
                            ge_path,
                            'permed/lg/mff/egi400')
    
    # ev = mne.find_events(data)
    # mne.viz.plot_events(ev)
#%% 
    # Preprocess & Report
    epochs, summary = nice_ext.api.preprocess(data, config='icm/lg/raw/egi',
                                                  config_params={'summary': True, 'reject': 250e-6,
                                                                  'hpass': 0.5, 'lpass': 45.}) #, 
                                                                  # 'min_channels': 0.4})
#%%
    
    report = nice_ext.api.create_report(epochs, title='Preprocessing', config='icm/preprocess',
                                        config_params={'summary': summary})
    # report_sname = '/home/dragana.manasova/Documents/data/permed/germany_lg/results/report01.html'
    # report.save(report_sname, overwrite=True)
    

    # Fit
    fit_params, fit_config = utils.parse_params_from_config(fit_config)
    fc = fit(epochs, config=fit_config)
    out_fname = '{}_{}_markers.hdf5'.format(subject, now)
    # fc.save(op.join(results_dir, subject, out_fname), overwrite=True)

    # Fit report
    report_params, report_config = utils.parse_params_from_config(report_config)
    report_params.update(dict(epochs=epochs))
    create_report(fc, config=report_config,
                  config_params=report_params,
                  report=report)

    # # Summarize
    summary = summarize_subject(
        fc,
        reductions=[
            '{}/egi256/trim_mean80'.format(fit_config),
            '{}/egi256/std'.format(fit_config),
            '{}/egi256gfp/trim_mean80'.format(fit_config),
            '{}/egi256gfp/std'.format(fit_config)
        ],
        out_path=op.join(results_dir, subject))

    # Predict
    predictions = predict(op.join(results_dir, subject),
                          predict_path, config='icm/lg',
                          config_params={'summary': True})

    # Predict report
    create_report(predictions, config='icm/predict', report=report)

# except Exception as err:
#     msg = traceback.format_exc()
#     logger.info(msg + '\nRunning subject failed ("%s")' % subject)
#     sys.exit(-4)
# finally:
    # if report is not None:
    out_fname = '{}_{}_full_report.html'.format(subject, now)
    report.save(op.join(results_dir, subject, out_fname),
                overwrite=True, open_browser=False)

    elapsed_time = time.time() - start_time
    logger.info('Elapsed time {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))
    logger.info('Running pipeline done')
    utils.remove_file_logging()
