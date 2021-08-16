import nice
import nice_ext

# raw = nice_ext.api.read(
#     'C:/Users/Dragana/Documents/data/germany_lg/', 
#     'permed/lg/mff/egi400')

countries = ['germany_lg', 'italy_lg', 'israel_rs']

raw = nice_ext.api.read(
    '/home/dragana.manasova/Documents/data/permed/'+countries[2]+'/', 
    'permed/rs/bdf/biosemi')
    #'permed/lg/mff/egi400')


#%%
# epochs = nice_ext.api.preprocess(raw, 'icm/lg/raw/egi', config_params={'summary': False, 'reject': 200e-6,
#                                                              'hpass': 0.5, 'lpass': 45., 
#                                                              'min_channels': 0.4})

#%% w/ report
epochs, summary = nice_ext.api.preprocess(raw, config='icm/lg/raw/egi',
                                              config_params={'summary': True, 'reject': 200e-6,
                                                              'hpass': 0.5, 'lpass': 45., 
                                                              'min_channels': 0.4})

report = nice_ext.api.create_report(epochs, title='Preprocessing', config='icm/preprocess',
                                    config_params={'summary': summary})
# report_sname = 'C:/Users/Dragana/Documents/data/germany_lg/results/report.html'
report_sname = '/home/dragana.manasova/Documents/data/permed/germany_lg/results/report01.html'
report.save(report_sname, overwrite=True)
#%%
fc = nice_ext.api.fit(epochs, 'icm/lg')
# fc_path = 'C:/Users/Dragana/Documents/data/germany_lg/results/'
fc_path = '/home/dragana.manasova/Documents/data/permed/germany_lg/results/'
fc.save(fc_path + 'germany_lg_features-markers.hdf5')

#%%
from argparse import ArgumentParser

parser = ArgumentParser(description='Run the pipeline on the selected suject')

parser.add_argument('--report', metavar='report', type=str, nargs='?',
                    default='icm/lg',
                    help='Report to create (default = icm/lg)')
args = parser.parse_args()

report_config = args.report

# Fit report
report_params, report_config = nice_ext.api.utils.parse_params_from_config(report_config)
report_params.update(dict(epochs=epochs))
report02 = nice_ext.api.create_report(fc, config=report_config,
                  config_params=report_params)#,
                  #report=report) 

# report02_sname = 'C:/Users/Dragana/Documents/data/germany_lg/results/report02.html'
report02_sname = '/home/dragana.manasova/Documents/data/permed/germany_lg/results/report02.html'
report02.save(report_sname, overwrite=True)
#%%

reductions = ['icm/lg/egi256/trim_mean80', 'icm/lg/egi256/std',
              'icm/lg/egi256gfp/trim_mean80', 'icm/lg/egi256gfp/std']

nice_ext.api.summarize_subject(fc_path, reductions, out_path=fc_path)

#%%

# pred_summary = ('D:/permed/baseline/baseline_stable_20210128')
pred_summary = ('/home/dragana.manasova/Documents/data/permed/germany_lg/results/baseline/baseline_stable_20210128')
pred_summary = nice_ext.api.summarize.read_summary(pred_summary)

# pred_summary = nice_ext.api.summarize.Summary() # I tried this but it didn't work 

predconfig = 'icm/lg'
predparams = {}
predparams.update({'reductions': reductions,
                   'summary': True}) # 'target': 'Label', 
#%%
predictions = nice_ext.api.predict(
    markers=fc_path,
    summary=pred_summary,
    config=predconfig,
    config_params=predparams)

report = nice_ext.api.create_report(predictions, config='icm/predict', report=report)
report.save(fc_path + 'germany_lg_report.html', overwrite=True, open_browser=False)