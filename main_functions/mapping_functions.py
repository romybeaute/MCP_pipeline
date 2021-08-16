"""
Created by Romy Beauté
romybeaute.univ@gmail.com
Function to map EGI setups in < setups (lower density EEG)

Functions used ONLY for EGI

Functions defined here :
 - change_montage(Raw/Epoch/Evok,int) -> list
 - chan_reject_zscore_var(Raw, list, int, float) -> list
 - split_channel_8_reg(int, list) -> dict 

Montage with 4 electrodes : keep ['F3', 'F4', 'P3', 'P4']
Montage with 8 electrodes : keep ['Fp1', 'Fp2', 'T7', 'C3', 'C4', 'T8', 'O1', 'O2']
Montage with 19 electrodes : keep ['C3', 'C4', 'O1', 'O2', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2', 'P3', 'P4', 'Pz', 'T7', 'T8', 'P7', 'P8']
Montage with 128 electrodes : keep ['E3', 'E7', 'E9', 'E10', 'E13', 'E14', 'E18', 'E19', 'E21', 'E22', 'E24', 'E25', 'E26', 'E28', 'E30', 'E31', 'E32', 'E33', 'E36', 'E37', 'E39', 'E41', 'E42', 'E43', 'E45', 'E46', 'E47', 'E48', 'E55', 'E56', 'E57', 'E59', 'E60', 'E61', 'E62', 'E63', 'E65', 'E66', 'E68', 'E69', 'E72', 'E73', 'E75', 'E78', 'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E89', 'E90', 'E93', 'E95', 'E97', 'E98', 'E100', 'E101', 'E105', 'E106', 'E108', 'E109', 'E112', 'E113', 'E118', 'E123', 'E125', 'E129', 'E130', 'E132', 'E133', 'E135', 'E137', 'E138', 'E140', 'E146', 'E151', 'E152', 'E154', 'E155', 'E157', 'E158', 'E161', 'E162', 'E164', 'E169', 'E171', 'E173', 'E174', 'E176', 'E177', 'E178', 'E179', 'E180', 'E182', 'E183', 'E186', 'E188', 'E191', 'E197', 'E201', 'E202', 'E203', 'E204', 'E206', 'E207', 'E210', 'E211', 'E212', 'E214', 'E215', 'E217', 'E218', 'E220', 'E221', 'E222', 'E224', 'E226', 'E231', 'E234', 'E235', 'E238', 'E239', 'E241', 'E242', 'E244', 'E249', 'E252']
 
 """

import numpy as np
import mne
from mne.channels._standard_montage_utils import _safe_np_loadtxt
import scipy.stats



def diy_change_montage(Inst, nb_eletrodes):
    """ Raw/Epoch/Evoke -> liste
    Modifies the number of eletrodes of Inst by choosing the closest eletrodes to a referent montage
    
    Values for nb_eletrodes : 4, 8, 19 (the new object uses a 10-05 montage ), 128 (the new object uses a egi-256 montage)"""
    
    montage_folder = '../devpackages/pycsd-master/pycsd/templates/'

    '''
    if 'romybeaute' in os.getcwd():
        montage_folder = '/Users/romybeaute/Desktop/CMDcode/devpackages/pycsd-master/pycsd/templates/'  # MAC
    elif 'romy.beaute' in os.getcwd():
        montage_folder = '/home/romy.beaute/Bureau/CMDcode-MAC/devpackages/pycsd-master/pycsd/templates/' # LINUX
    elif 'dragana.manasova' in os.getcwd():
        montage_folder = '/home/dragana.manasova/Desktop/MCP_Romy/CMDcode-MAC/devpackages/pycsd-master/pycsd/templates/'
    '''

    if nb_eletrodes in (4, 8, 19):

        # loading 1005 montage
        fname_1005 = montage_folder + 'standard_10-5.csd'  
        _str = 'U100'

        options = dict(comments='//',
                       dtype=(_str, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
        ch_names_1005, _, _, _, xs, ys, zs, _ = _safe_np_loadtxt(fname_1005, **options)
        pos_1005 = np.stack([xs, ys, zs], axis=-1)

        # loading 256 montage
        fname_256 = montage_folder + 'EGI_256.csd'  
        _str = 'U100'

        options = dict(comments='//',
                       dtype=(_str, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
        ch_names_256, _, _, _, xs, ys, zs, _ = _safe_np_loadtxt(fname_256, **options)
        pos_256 = np.stack([xs, ys, zs], axis=-1)


        ### Finding closest positions of 256 eletrodes corresponding in 20 model
        if nb_eletrodes == 19:
            use_ch =['C3', 'C4', 'O1', 'O2', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2', 'P3', 'P4', 'Pz', 'T7', 'T8', 'P7', 'P8']
        elif nb_eletrodes == 8: 
            use_ch = ['Fp1', 'Fp2', 'T7', 'C3', 'C4', 'T8', 'O1', 'O2']  
        elif nb_eletrodes == 4: 
            use_ch = ['F3', 'F4', 'P3', 'P4']
        elif nb_eletrodes == 2: 
            use_ch = ['C3','C4']

        len_1005 = len(ch_names_1005)
        dict_name_new_old = {}  # in this idct: {ol_name:new_name}

        # finding closest positions for each channel of 1005 montage
        for i in np.arange(len_1005):
            if ch_names_1005[i] in use_ch: # on ne garde que 20 életrodes sur les 200 du modèle 10/05

                liste_dist = []  # liste contenant les distances de l'eletrode du modèle 1005 par rapport
                                # à celles du modèle 156
                for ii in np.arange(256):
                    liste_dist.append(np.sqrt((pos_256[ii, 0]-pos_1005[i, 0])**2+\
                            (pos_256[ii, 1]-pos_1005[i, 1])**2+(pos_256[ii, 2]-pos_1005[i, 2])**2))
                old_name = ch_names_256[np.argmin(liste_dist)]  # on choisit l'eletrode la plus proche
                new_name = ch_names_1005[i]
                dict_name_new_old[old_name]=new_name    

        # Giving up chanels not kept
        stim_ch_names = [ch for ch in Inst.ch_names if not ch.startswith('E')]
        ch_kept = list(dict_name_new_old.keys()) + stim_ch_names
        chan_gave_up = [chan for chan in Inst.ch_names if chan not in ch_kept]
        Inst.drop_channels(chan_gave_up)

        # adding stim channel to channels kept
        for chan in stim_ch_names:
            dict_name_new_old[chan] = chan

        ### Adaptating model
        Inst.rename_channels(dict_name_new_old)
        Inst.reorder_channels(use_ch+stim_ch_names)

        Inst.info['description'] = 'standard/1020'
        Inst.info['montage'] = 'standard_1020'
        montage = mne.channels.make_standard_montage('standard_1020')
        Inst.set_montage(montage,on_missing='wan')


    ### Choosing 128 eletrodes (with egi 256  montage)
    elif nb_eletrodes in (128, 256):

        # loading egi 256 montage
        fname_256 = montage_folder + 'EGI_256.csd'  
        _str = 'U100'

        options = dict(comments='//',
                    dtype=(_str, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
        ch_names_256, _, _, _, xs, ys, zs, _ = _safe_np_loadtxt(fname_256, **options)
        pos_256 = np.stack([xs, ys, zs], axis=-1)

        dict_256_name_pos = {ch_names_256[i] :pos_256[i] for i in range(len(ch_names_256))} 

        if nb_eletrodes == 128:

            # loading egi 128 montage
            fname_128 = montage_folder + 'EGI_128.csd'  
            _str = 'U100'

            options = dict(comments='//',
                       dtype=(_str, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
            ch_names_128, _, _, _, xs, ys, zs, _ = _safe_np_loadtxt(fname_128, **options)
            pos_128 = np.stack([xs, ys, zs], axis=-1)

            dict_128_name_pos = {ch_names_128[i] :pos_128[i] for i in range(len(ch_names_128))} 


            ### Finding closest positions of 256 eletrodes corresponding in 20 model

            use_ch = ch_names_128
            len_128 = len(ch_names_128)
            dict_name_new_old = {}  # in this idct: {ol_name:new_name}

            # finding closest positions for each channel of 1005 montage
            for ch_128, pos_128 in dict_128_name_pos.items():
                if ch_128 in use_ch: # si on ne garde par toutes les eletrodes du modèle

                    liste_dist = []  # liste contenant les distances de l'eletrode du modèle 128 par rapport
                                # à celles du modèle 256
                    for ch_256, pos_256 in dict_256_name_pos.items():
                        liste_dist.append(np.sqrt((pos_256[0]-pos_128[0])**2+\
                            (pos_256[1]-pos_128[1])**2+(pos_256[2]-pos_128[2])**2))

                    old_name = list(dict_256_name_pos.keys())[np.argmin(liste_dist)]  # on choisit l'eletrode la plus proche
                    del dict_256_name_pos[old_name]
                    new_name = ch_128
                    dict_name_new_old[old_name]=new_name   


            # Giving up chanels not kept
            chan_gave_up = [chan for chan in Inst.ch_names if chan not in dict_name_new_old.keys()]
            Inst.drop_channels(chan_gave_up)
            
        Inst.info['description'] = 'egi/256'
        montage = mne.channels.make_standard_montage('EGI_256')
        Inst.set_montage(montage,on_missing='warn')
        use_ch = Inst.info['ch_names']
		
    else:
        raise RuntimeError('Valid values for nb_eletrodes are 4, 8, 19, 128, 256')
        
    return Inst, use_ch
	
	
def chan_reject_zscore_var(Inst, use_ch, nb_iter, z_max):
    """Raw*list*int*float -> list
	
    Returns channels marked as bad  that have a variance with a score > z_max, nb_iter times """
	
    ch_names = list(use_ch)
    nb_channels = len(ch_names)
    Inst_array, _ = Inst[:nb_channels]
    Inst_dict = {ch_names[n]:Inst_array[n] for n in np.arange(nb_channels)}
    bads = []
	
    for i in np.arange(nb_iter):
        Var_chan = [np.var(chan) for chan in Inst_dict.values()]
        Zscore_chan = {ch_names[n]:scipy.stats.zscore(Var_chan)[n] for n in np.arange(nb_channels)}

        for chan_name in ch_names:

            if np.abs(Zscore_chan[chan_name]) > 4:
                bads.append(chan_name)
                ch_names.remove(chan_name)
                del Inst_dict[chan_name]
                nb_channels-=1
	
    return bads
	
	
def split_channel_8_reg(nb_channels, use_ch):
    """int*list -> dict
    
    Returns a dict object with the channels contained in each 8 region :
        -Segmentation across x axes = left -> right (L - R)
        -Segmentation across y axes = posterior -> anterior (P - A)
        -Segmentation across z axes = foot -> head (F - H) """
    
    if nb_channels == 128 or nb_channels == 256:
        montage_folder = '../templates/'  # Must be changed depending on path of pycsd module

        # loading egi 256 montage
        fname_256 = montage_folder + 'EGI_256.csd'  
        _str = 'U100'

        options = dict(comments='//',
                   dtype=(_str, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
        ch_names_256, _, _, _, xs, ys, zs, _ = _safe_np_loadtxt(fname_256, **options)
        pos_256 = np.stack([xs, ys, zs], axis=-1)
        dict_256_name_pos = {ch_names_256[i] :pos_256[i] for i in range(len(ch_names_256))} 

        use_ch_LPF = []
        use_ch_LPH = []
        use_ch_LAF = []
        use_ch_LAH = []
        use_ch_RPF = []
        use_ch_RPH = []
        use_ch_RAF = []
        use_ch_RAH = []
        for ch, (x,y,z) in dict_256_name_pos.items():
            if ch in use_ch:
                if x<=0 and y<=0 and z<= 0:  # left posterior foot
                    use_ch_LPF.append(ch)
                if x<=0 and y<=0 and z>= 0:  # left posterior head
                    use_ch_LPH.append(ch)
                if x<=0 and y>=0 and z<= 0:  # left anterior foot
                    use_ch_LAF.append(ch)
                if x<=0 and y>=0 and z>= 0:  # left anterior head
                    use_ch_LAH.append(ch)

                if x>=0 and y<=0 and z<= 0:  # right posterior foot
                    use_ch_RPF.append(ch)
                if x>=0 and y<=0 and z>= 0:  # right posterior head
                    use_ch_RPH.append(ch)
                if x>=0 and y>=0 and z<= 0:  # right anterior foot
                    use_ch_RAF.append(ch)
                if x>=0 and y>=0 and z>= 0:  # right anterior head
                    use_ch_RAH.append(ch)
					
        return {'LPF':use_ch_LPF, 'LPH':use_ch_LPH, 'LAF':use_ch_LAF, 'LAH':use_ch_LAH,\
                'RPF':use_ch_RPF, 'RPH':use_ch_RPH, 'RAF':use_ch_RAF, 'RAH':use_ch_RAH}
    return
