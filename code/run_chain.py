import time
import numpy as np
import scipy
import emcee
import h5py 

import chain
import emulator
import initialize_chain


def main():
    #chaintag = 'upf_c4h4_fenv_sigma8_long'
    #chaintag = 'upf_c4h4_fenv_med_nolog'
    #chain_fn = f'../chains/chains_{chaintag}.h5'
    config_fn = f'../chains/chains_wp_config.cfg'
    chain_fn = initialize_chain.main(config_fn)
    run(chain_fn)

def run(chain_fn, mode='chain'):

    f = h5py.File(chain_fn, 'r')

    ### data params
    # required
    cosmo = f.attrs['cosmo']
    hod = f.attrs['hod']

    ### emu params
    # required
    statistic = f.attrs['statistic']
    traintag = f.attrs['traintag']
    testtag = f.attrs['testtag']
    errtag = f.attrs['errtag']
    tag = f.attrs['tag']
    # optional
    log = f.attrs['log']
    #log = True if log is None else log
    mean = f.attrs['mean']
    #mean = False if mean is None else mean

    ### chain params
    # required
    nwalkers = f.attrs['nwalkers']
    nburn = f.attrs['nburn']
    nsteps = f.attrs['nsteps']
    param_names = f.attrs['param_names']
    # optional
    multi = f.attrs['multi']
    #multi = True if multi is None else multi

    f.close()

    # Set file and directory names
    gptag = traintag + errtag + tag
    res_dir = '../../clust/results_{}/'.format(statistic)
    gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
    training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
    testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, testtag)
    hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)
    #plot_dir = '../plots/plots_2020-01-27'
    #plot_fn = f'{plot_dir}/prob_{statistic}{gptag}{savetag}.png'
    #chain_fn = f'../chains/chains_{statistic}{gptag}{savetag}.png'

    # actual calculated stat
    if 'parammean' in testtag:
        rad, y = np.loadtxt(f'../testing_results/{statistic}_parammean.dat', delimiter=',', unpack=True)
    else:
        rad, y = np.loadtxt(testing_dir+'{}_cosmo_{}_HOD_{}_mean.dat'
                            .format(statistic, cosmo, hod))
    print('y:', y.shape, y)

    # number of parameters, ex 8 hod + 7 cosmo
    num_params = len(param_names)
    #means = np.random.rand(ndim)
    cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    cosmos_truth = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')

    hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    hods_truth = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')
    hods_truth[:, 0] = np.log10(hods_truth[:, 0])
    hods_truth[:, 2] = np.log10(hods_truth[:, 2])

    fixed_params = {}
    if 'parammean' in testtag:
        names = cosmo_names + hod_names
        params_mean = np.loadtxt("../testing_results/parammean.dat")
        for (name, pm) in zip(names, params_mean):
            fixed_params[name] = pm
    else:
        cosmo_truth = cosmos_truth[cosmo]
        hod_truth = hods_truth[hod]
        for (cn, ct) in zip(cosmo_names, cosmo_truth):
            fixed_params[cn] = ct
        for (hn, ht) in zip(hod_names, hod_truth):
            fixed_params[hn] = ht

    # remove params that we want to vary from fixed param dict and add true values
    truth = {}
    for pn in param_names:
        truth[pn] = fixed_params[pn]
        fixed_params.pop(pn)

    print("Stat:", statistic)
    print("True values:")
    print(truth)

    print("Building emulator")
    emu = emulator.Emulator(statistic, training_dir,  fixed_params=fixed_params, gperr=gperr, hyperparams=hyperparams, log=log, mean=mean)
    emu.build()
    print("Emulator built")

    #diagonal covariance matrix from error
    cov = np.diag(emu.gperr)
    start = time.time()
    if mode=='chain':
        res = chain.run_mcmc([emu], param_names, [y], [cov], fixed_params=fixed_params, truth=truth, nwalkers=nwalkers,
            nsteps=nsteps, nburn=nburn, multi=multi, chain_fn=chain_fn)
    elif mode=='minimize':
        res = chain.run_minimizer([emu], param_names, [y], [cov], fixed_params=fixed_params, truth=truth, nwalkers=nwalkers,
                nsteps=nsteps, nburn=nburn, multi=multi, chain_fn=chain_fn)
    else:
        raise ValueError(f"Mode {mode} not recognized!")

    end = time.time()
    print(f"Time: {(end-start)/60.0} min")

    return res

if __name__=='__main__':
    main()