import numpy as np


def main():
    ntotal = 21
    ncosmos = 7
    for n in range(ntotal):
        hoddigit = int(n/ncosmos)
        cosmo = n%ncosmos
        hod = cosmo*10 + hoddigit
        contents = populate_config(cosmo, hod)
        config_fn = f'/home/users/ksf293/emulator/chain_configs/chains_wp_xi_upf_mcf_c{cosmo}h{hod}_covsg1.cfg'
        print(config_fn)
        with open(config_fn, 'w') as f:
            f.write(contents)


def populate_config(cosmo, hod):
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
