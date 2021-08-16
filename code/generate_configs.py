import numpy as np


def main():
    ntotal = 21
    ncosmos = 7
    statistics = ['wp']
    stat_str = '_'.join(statistics)
    for n in range(ntotal):
        hoddigit = int(n/ncosmos)
        cosmo = n%ncosmos
        hod = cosmo*10 + hoddigit
        contents = populate_config(statistics, cosmo, hod)
        config_fn = f'/home/users/ksf293/emulator/chain_configs/chains_{stat_str}_c{cosmo}h{hod}_covsg1.cfg'
        print(config_fn)
        with open(config_fn, 'w') as f:
            f.write(contents)

def populate_config(statistics, cosmo, hod):
    stat_str = '_'.join(statistics)
    n_stats = len(statistics)
    kernel_dict = {'wp': 'M32ExpConst2',
                   'xi': 'M32ExpConst',
                   'upf': 'M32ExpConst',
                   'mcf': 'M32ExpConst'}
    contents = \
f"""---
save_fn: '/home/users/ksf293/emulator/chains/chains_{stat_str}_c{cosmo}h{hod}_all_dy_covsmoothgauss1.h5'

emu:
    statistic: {statistics}
    traintag: {['_nonolap']*n_stats}
    testtag: {['_mean_test0']*n_stats}
    errtag: {['_hod3_test0']*n_stats}
    tag: {[f'_log_k{kernel_dict[s]}_100hod' for s in statistics]}
    kernel_name: {[kernel_dict[s] for s in statistics]} 
    nhod: {[100]*n_stats}
    log: {[True]*n_stats}
    mean: {[False]*n_stats}

chain:
    multi: True
    dlogz: 1e-2
    cov_fn: '/home/users/ksf293/clust/covariances/cov_smoothgauss1_emuperf_{stat_str}_nonolap_hod3_test0_mean_test0.dat'
    #seed: 12 # If not defined, code chooses random int in [0, 1000)
    param_names: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env'] #all

data:
    cosmo: {cosmo}
    hod: {hod}
"""
    return contents

def populate_config_static(cosmo, hod):
    contents = \
f"""---
save_fn: '/home/users/ksf293/emulator/chains/chains_wp_xi_upf_mcf_c{cosmo}h{hod}_all_dy_covsmoothgauss1.h5'

emu:
    statistic: ['wp', 'xi', 'upf', 'mcf']
    traintag: ['_nonolap', '_nonolap', '_nonolap', '_nonolap']
    testtag: ['_mean_test0', '_mean_test0', '_mean_test0', '_mean_test0'] # for emu covariance
    errtag: ['_hod3_test0', '_hod3_test0', '_hod3_test0', '_hod3_test0']
    tag: ['_log_kM32ExpConst2_100hod', '_log_kM32ExpConst_100hod', '_log_kM32ExpConst_100hod', '_log_kM32ExpConst_100hod']
    kernel_name: ['M32ExpConst2', 'M32ExpConst', 'M32ExpConst', 'M32ExpConst']
    nhod: [100, 100, 100, 100]
    log: [True, True, True, True]
    mean: [False, False, False, False]

chain:
    multi: True
    dlogz: 1e-2
    cov_fn: '/home/users/ksf293/clust/covariances/cov_smoothgauss1_emuperf_wp_xi_upf_mcf_nonolap_hod3_test0_mean_test0.dat'
    #seed: 12 # If not defined, code chooses random int in [0, 1000)
    param_names: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env'] #all

data:
    cosmo: {cosmo}
    hod: {hod}
"""
    return contents


if __name__=='__main__':
    main()
