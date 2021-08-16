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

#
# raw= mne.io.read_raw_bdf('/home/dragana.manasova/Documents/data/permed/israel_rs/Testdata-1-DGD_permed-rs-biosemi.bdf', 
#                         preload=True, verbose=True)
#
# default_path = '/media/data2/new_patients'
# default_path = '/Volumes/Databases/ICM/doc_stable/'
# default_path = '/Users/fraimondo/data/icm/test'
# default_predict_path = '/media/data/old_patients/group_results/baseline_brain'
# default_predict_path = ('/Volumes/Databases/ICM/doc_stable/group_results/'
#                         'baseline_stable_20210128')

fsubs = ['p006_01_rs', 
         'p006_02_rs', 
         'p006_03_rs'
         ]

default_path = '/home/dragana.manasova/Documents/data/permed/israel/subjects/' + fsubs[1]
default_predict_path = ('/home/dragana.manasova/Documents/data/permed/israel/results'
                        '/baseline/baseline_stable_20210419')

subs = ['YM_20210418_rs_patient_permed-rs-biosemi.bdf', # p006_01_rs
        'YM_20210428_rs_patient_permed-rs-biosemi.bdf', # p006_02_rs
        'YM_20210503_rs_patient_permed-rs-biosemi.bdf'
        ]

start_time = time.time()

parser = ArgumentParser(description='Run the pipeline on the selected suject')

parser.add_argument('--io', metavar='io', type=str, nargs='?',
                    default='permed/rs/bdf/biosemi', #'icm/lg/raw/egi',
                    help='IO to use (default = )')
parser.add_argument('--preprocess', metavar='preprocess', type=str, nargs='?',
                    default='permed/rs/raw/biosemi',
                    help='Preprocessing to run (default = permed/rs/raw/biosemi)')
parser.add_argument('--fit', metavar='fit', type=str, nargs='?',
                    default='icm/rs',   # 'icm/rs' works
                    help='Fit to run (default = icm/rs)')
parser.add_argument('--report', metavar='report', type=str, nargs='?',
                    default='icm/rs',
                    help='Report to create (default = icm/rs)')
parser.add_argument('--path', metavar='path', nargs=1, type=str,
                    help='Path with the database.',
                    default=default_path)
parser.add_argument('--subject', metavar='subject', nargs=1, type=str,
                    default=subs[1],
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
s_path = op.join(db_path)

# if not op.exists(op.join(db_path, 'results')):
#     os.mkdir(op.join(db_path, 'results'))

# results_dir = op.join(db_path, 'results', run_id)
results_dir = op.join('/home/dragana.manasova/Documents/data/permed/israel', 
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
    preprocess_params.update({'summary': True, 'reject': 100e-6})
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
    redu_params = {'reduction_rs': 'permed/rs/biosemi64/trim_mean80'}
    report_params.update(redu_params)
    create_report(fc, config=report_config,
                  config_params=report_params,
                  report=report)
    # report_sname = '/home/dragana.manasova/Documents/data/permed/israel/results/' \
    #                 +os.path.splitext(subject)[0]+'_report_markers.html'
    # report.save(report_sname, overwrite=True)
    # # Final report for Israel

if report is not None:
    out_fname = 'permed_israel_rs_patient_{}_{}_full_report.html'.format(subject.split('_')[0], now)
    report.save(op.join(results_dir, out_fname),
                overwrite=True, open_browser=True)

elapsed_time = time.time() - start_time
logger.info('Elapsed time {}'.format(
    time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))
logger.info('Running pipeline done')
utils.remove_file_logging()
#%% until here it works -- prediction not possible for a different setup


    # # Read
    # data = nice_ext.api.read(
    #                         '/home/dragana.manasova/Documents/data/permed/israel/', 
    #                         'permed/rs/bdf/biosemi')
    
    # # Preprocess & Report
    # epochs, summary = nice_ext.api.preprocess(data, config='permed/rs/raw/biosemi',
    #                                               config_params={'summary': True, 'reject': 100e-6,
    #                                                               'hpass': 0.5, 'lpass': 45., 
    #                                                               'min_channels': 0.4})
    
    # report = nice_ext.api.create_report(epochs, title='Preprocessing', config='icm/preprocess',
    #                                     config_params={'summary': summary})
    # # This report is contained in the following one. 
    # # report_sname = '/home/dragana.manasova/Documents/data/permed/' \
    #                 # 'israel_rs/results/'+os.path.splitext(subject)[0]+'_report_preprocessing.html'
    # # report.save(report_sname, overwrite=True)
"""
    summary = summarize_subject(
        fc,
        reductions=[
            'permed/rs/biosemi64/trim_mean80',
            'permed/rs/biosemi64/std', 
            'permed/rs/biosemi64gfp/trim_mean80',
            'permed/rs/biosemi64gfp/std'
        ],
        out_path=op.join(results_dir, subject))

# code works until here
    # Predict
    predictions = predict(op.join(results_dir, subject),
                          predict_path, config='icm/rs',
                          config_params={'summary': True, 
                                         'target':'Label', 
                                         'reductions': 
                                           ['permed/rs/biosemi64/trim_mean80', 'permed/rs/biosemi64/std',
                                            'permed/rs/biosemi64gfp/trim_mean80', 'permed/rs/biosemi64gfp/std']})
        
                                         # , 'reductions': 'permed/rs/biosemi64/trim_mean80'}) # , 
# 
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
"""
