"""
Initializing the basics (default) parameters for the MCP

"""
import os
import pandas as pd
import configs as cfg

class Params:

    def __init__(self, subject_type):

        """ IN : subject_type = "controls" or "patients"""

        self.subject_type = subject_type

        self.fmin, self.fmax = 1.0, 30.0  # frequency cut-off, in Hz
        self.n_epo_segments = 5
        self.bands = cfg.bands
        self.is_control = self.subject_type == 'controls'

        # ==> Load the datapath with the whole MCP dataset (according to used computer)
        # ==> Define datapath according to used computer
        if 'romybeaute' in os.getcwd():
            # MAC
            self.datapath = '/Users/romybeaute/MOTOR'  # file with ALL the MCP datas
        elif 'romy.beaute' in os.getcwd():
            # LINUX
            self.datapath = '/home/romy.beaute/Bureau/CMP_romy/MOTOR'
        elif 'dragana.manasova' in os.getcwd():
            self.datapath = '/media/dragana.manasova/Romy_SSDT5/CMP_romy/MOTOR' #hard drive
            #'/home/dragana.manasova/Desktop/MCP_Romy/DATA'

        # ==> Load the parameters from the CSV file
        sheet_id = '1wvrkO8mHMx-fGP9HrwpBs57iQmEwBpcJ-DdtcfjV0JE'
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={subject_type}'
        df = pd.read_csv(url)
        if not df.index.name == 'ID':
            df.set_index('ID', inplace=True)  # change the index if not already done
        self.df = df


    def get_setups(self, subject_id):

        """ Get the setups for each subject (for a given instance of Params : "controls" or "patients") """

        self.subject_id = subject_id
        self.center = self.df.loc[subject_id]['Center']
        self.setup = self.df.loc[subject_id]['Setup']
        self.filename = self.df.loc[subject_id]['Filename']
        self.filename2 = self.df.loc[subject_id]['Filename2']
        self.is_imagine = self.df.loc[subject_id]['Condition'] == 'imagine'
        self.preproc = self.df.loc[subject_id]['Preproc']            

        
        # ==> Load the filepath & the preproc
        if self.center not in ['Munich', 'Italy']:  # vhdr files, stored in a different way
            self.filepath = os.path.join(self.datapath, self.center, self.subject_type,self.filename)
            if self.center == 'Paris':
                self.preproc = cfg.preproc  # ==> basic preprocessing parameters : ['filt', 'csd','drop_ears','chan_to_drop','bios']
            elif self.center == 'Germany':
                self.preproc = cfg.preproc
            elif self.center == 'Columbia':
                self.preproc = cfg.std_preproc
        else:  # ==> .vhdr files
            self.filepath = os.path.join(self.datapath, self.center, self.subject_type, self.subject_id)
            if self.center == 'Munich':
                self.preproc = cfg.mun_preproc
            elif self.center == 'Italy':
                self.preproc = cfg.it_preproc


        """
        if str(self.filename2) != 'nan':
            self.filepath2 = os.path.join(self.datapath, self.center, self.subject_type, self.subject_id, self.filename2)
        else:
            self.filepath2 = None
        """


    def get_badchans(self):

        """ Get the self defined bad chans lists, stored in the configs file """

        drop_bads = list()
        dropped_bads = list()
        print("Preprocessing parameters: ", self.preproc)
        if 'drop_outer' in self.preproc:
            drop_bads += cfg.outer_outlines
        if 'chan_to_drop' in self.preproc:
            drop_bads += cfg.chan_to_drop
        if 'drop_ears' in self.preproc:
            drop_bads += cfg.ears_outlines
        if 'bios' in self.preproc:  # bio chans (ECG,respi,EMG)
            drop_bads += cfg.bios
        if 'bad_it_chans' in self.preproc:  # chans to drop for Italy
            drop_bads += cfg.bad_it_chans
        if 'bad_mun_chans' in self.preproc:  # chans to drop for Munich
            drop_bads += cfg.bad_mun_chans

        self.drop_bads = drop_bads

# ==> to write var in dic

def load_params(subject_type,subject_id):
    p = Params(subject_type)
    p.get_setups(subject_id)
    p.get_badchans()

    # create folder for this subject
    report_folder = os.path.join(p.datapath, "Analysis_reports/{}".format(subject_id))
    os.makedirs(report_folder, exist_ok=True)
    textfile = os.path.join(report_folder, "running_parameters.txt")
    f = open(textfile, "w")

    valdict = dict()
    for i, v in enumerate(vars(p)):
        if v != 'df':
            f.write("{} : {} \n".format(v,vars(p)[v]))
            print(v, ": ", vars(p)[v])
    f.close()