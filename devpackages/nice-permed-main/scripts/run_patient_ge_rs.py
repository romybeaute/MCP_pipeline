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

countries = ['germany', 'italy', 'israel']

default_path = '/home/dragana.manasova/Documents/data/permed/germany/subjects/p007_rs_3'
default_predict_path = ('/home/dragana.manasova/Documents/data/permed/'+countries[0]+'/results'
                        '/baseline/baseline_stable_20210419')

permeddata = ['D-08pTZB_RS_1_20210416_113738_rs_patient_permed-rs-egi400.mff', # p002_01
              'D-08pTZB_RS_2_20210416_114801_rs_patient_permed-rs-egi400.mff', # p002_02   # very noisy, thresh=300
              'D-09pTZB_RS_1_20210420_121231_rs_patient_permed-rs-egi400.mff', # p003_01  # thresh=400
              'D-09pTZB_RS_2_20210420_122256_rs_patient_permed-rs-egi400.mff', # p003_02 # thresh=400
              'DZ_LocGlob_20210413_135123_lg_control_permed-lg-egi400.mff.zip', # c01
              'SM_LocalGlobal_20210413_150336_lg_control_permed-lg-egi400.mff.zip' # c02
    ]

permeddata = ['D_10-KUM_RS_1_2_permed-rs-egi400.mff', # p004_rs_2_5
              'D-14pTZB_RestingState1a_20210604_105508_permed-rs-egi400.mff', # p005_rs_1a # thresh=100 too small
              'D-14pTZB_RestingState1b_20210604_110048_permed-rs-egi400.mff', # p005_rs_1b 
              'D-14pTZB_RestingState_1c_20210604_110630_permed-rs-egi400.mff', # p005_rs_1c
              'D-14pTZB_RestingState1d_20210604_111210_permed-rs-egi400.mff', # p005_rs_1d
              'D-14pTZB_1_2_permed-rs-egi400.mff' # p005_rs_1a_1b
              ]

permeddata = ['D-16pTZB_RS_T1_20210702_120402_permed-rs-egi400.mff', # p006_rs_1
              'D-16pTZB_RS_T1_b_20210702_121051_permed-rs-egi400.mff', # p006_rs_2
              'D-16pTZB_RS_T1_c_20210702_121637_permed-rs-egi400.mff', # p006_rs_3
              'EF_RS1_20210625_110841_permed-rs-egi400.mff',  # p_child_rs
              'D-17pTZB_RS1_20210720_142641_permed-rs-egi400.mff', # p007_rs_1
              'D-17pTZB_RS2_20210720_143946_permed-rs-egi400.mff', # p007_rs_2
              'D-17pTZB_RS3_20210721_110938_permed-rs-egi400.mff' # p007_rs_3
    ]

eegdata = permeddata[6]

# ge_path = '/home/dragana.manasova/Documents/data/permed/germany/subjects/p002_01_rs/'


start_time = time.time()

parser = ArgumentParser(description='Run the pipeline on the selected suject')

parser.add_argument('--io', metavar='io', type=str, nargs='?',
                    default='permed/rs/mff/egi400', #'icm/lg/raw/egi',
                    help='IO to use (default = icm/lg/raw/egi)')
parser.add_argument('--preprocess', metavar='preprocess', type=str, nargs='?',
                    default='icm/rs/raw/egi',
                    help='Preprocessing to run (default = icm/rs/raw/egi)')
parser.add_argument('--fit', metavar='fit', type=str, nargs='?',
                    default='icm/rs',
                    help='Fit to run (default = icm/lg)')
parser.add_argument('--report', metavar='report', type=str, nargs='?',
                    default='icm/rs',
                    help='Report to create (default = icm/rs)')
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
s_path = op.join(db_path)  # changed

# if not op.exists(op.join(db_path, 'results')):
#     os.mkdir(op.join(db_path, 'results'))

# results_dir = op.join(db_path, 'results', run_id)
results_dir = op.join('/home/dragana.manasova/Documents/data/permed/germany', 
                      'results', subject) 
if not op.exists(results_dir):
    os.mkdir(results_dir)
if not op.exists(op.join(results_dir, run_id)):
    os.mkdir(op.join(results_dir, run_id))

now = time.strftime('%Y_%m_%d_%H_%M_%S')
log_suffix = '_{}.log'.format(now)
mne.utils.set_log_file(op.join(results_dir,
                               run_id,
                               subject + log_suffix))

utils.configure_logging()
utils.log_versions()

logger.info('Running {}'.format(subject))
report = None
# try:
if True:
    
    # Read
    io_params, _ = utils.parse_params_from_config(io_config)
    data = read(s_path, io_config, io_params)

    # Preprocess
    preprocess_params, preprocess_config = \
        utils.parse_params_from_config(preprocess_config)
    preprocess_params.update({'summary': True, 'reject': 100e-6, 'min_events': 130})
    epochs, summary = preprocess(data, preprocess_config, preprocess_params)

    # Preprocess Report
    report = mne.report.Report(title=subject)
    create_report(epochs, config='icm/preprocess',
                  config_params=dict(summary=summary), report=report)
   

    # Fit
    fit_params, fit_config = utils.parse_params_from_config(fit_config)
    fc = fit(epochs, config=fit_config)
    # out_fname = '{}_{}_markers.hdf5'.format(subject, now)
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
        out_path=op.join(results_dir))

    # Predict
    predictions = predict(op.join(results_dir),
                          predict_path, config='icm/rs',
                          config_params={'summary': True, 'target':'Label'})

    # Predict report
    create_report(predictions, config='icm/predict', report=report)

# except Exception as e:
    # msg = traceback.format_exc()
    # logger.info(str(e) + '\nRunning subject failed ("%s")' % subject)
    # sys.exit(-4)
# finally:
if report is not None:
    out_fname = 'permed_germany_rs_patient_{}_{}_full_report.html'.format(subject.split('_')[0], now)
    report.save(op.join(results_dir, out_fname),
                overwrite=True, open_browser=True)

elapsed_time = time.time() - start_time
logger.info('Elapsed time {}'.format(
    time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))
logger.info('Running pipeline done')
utils.remove_file_logging()




#%%
## Read
# data = nice_ext.api.read(
                        # ge_path,
                        # 'permed/rs/mff/egi400')

# ev = mne.find_events(data)
# mne.viz.plot_events(ev)
# Preprocess & Report
# epochs, summary = nice_ext.api.preprocess(data, config='icm/rs/raw/egi',
                                              # config_params={'summary': True, 'reject': 100e-6,
                                                              # 'hpass': 0.5, 'lpass': 45., 
                                                              # 'min_channels': 0.4})