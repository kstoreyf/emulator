import numpy as np


pickle_dir = f'../products/dynesty_results'

cosmo_params = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
hod_params = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f']
ab_params = ['f_env', 'delta_env', 'sigma_env']
hod_interest = ['M_sat', 'M_cut', 'alpha', 'f_env', 'delta_env']
all_interest = ['Omega_m', 'sigma_8', 'M_sat', 'f_env', 'delta_env']
key_params = ['Omega_m', 'sigma_8', 'M_sat', 'v_bc', 'v_bs', 'f', 'f_env']
all_params = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
bi_params = ['Omega_b', 'sigma_8', 'h', 'w', 'delta_env']
biplus_params = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'w', 'M_sat', 'delta_env']

cwp = 'deepskyblue'
cupf = 'green'
cmcf = 'magenta'

param_labels = {'Omega_m': '\Omega_m', 
                'Omega_b': '\Omega_b', 
                'sigma_8': '\sigma_8', 
                'h': 'h', 
                'n_s': 'n_s',
                'N_eff': 'N_{eff}', 
                'w': 'w', 
                'M_sat': 'M_{sat}', 
                'alpha': r'\alpha', 
                'M_cut': 'M_{cut}', 
                'sigma_logM': '\sigma_{logM}', 
                'v_bc': 'v_{bc}', 
                'v_bs': 'v_{bs}', 
                'c_vir': 'c_{vir}', 
                'f': 'f', 
                'f_env': 'f_{env}', 
                'delta_env': '\delta_{env}', 
                'sigma_env': '\sigma_{env}'}

nbins = 9
rbins = np.logspace(np.log10(0.1), np.log10(50), nbins + 1) # Note the + 1 to nbins
rlog = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))
rlin = np.linspace(5, 45, 9)
r_dict = {'wp': rlog, 'xi': rlog, 'upf': rlin, 'mcf': rlog}
scale_dict = {'wp': ('log', 'log'), 'xi': ('log', 'log'), 'upf': ('linear', 'log'), 'mcf': ('log', 'linear')} #x, y
