import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import scipy
import emcee
import h5py 
from scipy.linalg import block_diag
import argparse 

import chain
import emulator
import initialize_chain


def main(config_fn):
    #config_fn = f'../chain_configs/chains_wp.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi.cfg'
    #config_fn = f'../chain_configs/chains_wp_upf.cfg'
    #config_fn = f'../chain_configs/chains_wp_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_upf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_upf_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_upf_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_upf_mcf_c1h1.cfg'
    
    chain_fn = initialize_chain.main(config_fn)
    run(chain_fn, overwrite=True, mode='dynesty')
    #run(chain_fn, overwrite=True, mode='emcee')


def run(chain_fn, mode='dynesty', overwrite=False):

    # TODO: this overwrite check doesn't work bc initialize_chain
    # creates chain_fn either way. fix!
    if not overwrite and os.path.exists(chain_fn):
        raise ValueError(f"ERROR: File {chain_fn} already exists! Set overwrite=True to overwrite.")
    f = h5py.File(chain_fn, 'r+')

    ### data params
    # required
    cosmo = f.attrs['cosmo']
    hod = f.attrs['hod']

    ### emu params
    # required
    statistics = f.attrs['statistic']
    traintags = f.attrs['traintag']
    testtags = f.attrs['testtag']
    errtags = f.attrs['errtag']
    tags = f.attrs['tag']
    kernel_names = f.attrs['kernel_name']
    # optional
    logs = f.attrs['log']
    means = f.attrs['mean']
    nhods = f.attrs['nhod']

    ### chain params
    # required
    param_names = f.attrs['param_names']
    # optional
    multi = f.attrs['multi']
    nwalkers = f.attrs['nwalkers']
    nburn = f.attrs['nburn']
    nsteps = f.attrs['nsteps']
    dlogz = float(f.attrs['dlogz'])
    print('dlogz:', f.attrs['dlogz'], dlogz)
    seed = f.attrs['seed']
    nbins = f.attrs['nbins']
    cov_fn = f.attrs['cov_fn']
    icov_fn = f.attrs['icov_fn']
    msg = 'Cannot give both cov_fn and icov_fn in config file! But must have one of them'
    using_cov = isinstance(cov_fn, str)
    using_icov = isinstance(icov_fn, str)
    #assert using_cov ^ using_icov, msg
    
    # Set file and directory names
    nstats = len(statistics)
    training_dirs = [None]*nstats
    testing_dirs = [None]*nstats
    hyperparams = [None]*nstats
    acctags = [None]*nstats
    gperrs = [None]*nstats
    ys = []
    cov_dir = '../../clust/covariances/'
    print("WARNING: train tag and test tag no longer used!")
    for i, statistic in enumerate(statistics):
        gptag = traintags[i] + errtags[i] + tags[i]
        acctags[i] = gptag + testtags[i]
        gperrs[i] = np.loadtxt(cov_dir+"error_aemulus_{}{}.dat".format(statistic, errtags[i]))
        training_dirs[i] = f'../../clust/results_aemulus_train/results_{statistic}/'
        testing_dirs[i] = f'../../clust/results_aemulus_test_mean/results_{statistic}/'
        hyperparams[i] = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

        # actual calculated stat
        _, y = np.loadtxt(testing_dirs[i]+'{}_cosmo_{}_HOD_{}_mean.dat'.format(statistic, cosmo, hod), delimiter=',', unpack=True)
        y = y[:nbins]
        ys.extend(y)
    f.attrs['ys'] = ys

    # number of parameters, out of 11 hod + 7 cosmo
    num_params = len(param_names)
    cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    cosmos_truth = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')

    hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    hods_truth = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')
    hods_truth[:, 0] = np.log10(hods_truth[:, 0])
    hods_truth[:, 2] = np.log10(hods_truth[:, 2])

    fixed_params = {}
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
    #can't store dicts in h5py, so make sure truths (for variable params) are in same order as param names 
    truths = [truth[pname] for pname in param_names]
    f.attrs['true_values'] = truths
    if len(fixed_params)>0:
        fixed_param_names = list(fixed_params.keys())
        fixed_param_values = [fixed_params[fpn] for fpn in fixed_param_names]
    else:
        fixed_param_names = []
        fixed_param_values = []
    f.attrs['fixed_param_names'] = fixed_param_names
    f.attrs['fixed_param_values'] = fixed_param_values
    # print chain file info
    print(f"h5 file attributes for chain_fn: {chain_fn}")
    for k, v in f.attrs.items():
        print(f'{k}: {v}')

    # Set up Covariance matrix
    #if using_cov:
    icov = None
    if True:
        if os.path.exists(cov_fn):
            cov = np.loadtxt(cov_fn)
        else:
            raise ValueError(f"Path to covmat {cov_fn} doesn't exist!")

        nbins_tot = 9 #the covmat should have been constructed with 9 bins per stat
        err_message = f"Cov bad shape! {cov.shape}, but nbins_tot={nbins_tot} and nstats={nstats}"
        assert cov.shape[0] == nstats*nbins_tot and cov.shape[0] == nstats*nbins_tot, err_message
        # delete rows/cols of covmat we don't want to use
        #nbins_tot = cov.shape[0]/nstats
        if nbins < nbins_tot:
            print(f"nbins={nbins}, while total in cov nbins_tot={nbins_tot}; removing {nbins_tot-nbins} bins")
        idxs_toremove = np.array([np.arange(nbins_tot-1, nbins-1, -1)+(ns*nbins_tot) for ns in range(nstats)]).flatten()
        # remove both row and col
        cov = np.delete(cov, idxs_toremove, axis=0)
        cov = np.delete(cov, idxs_toremove, axis=1)
        print("Covariance matrix:")
        print(cov)    
        print(cov.shape)
        print("Condition number:", np.linalg.cond(cov))
        f.attrs['covariance_matrix'] = cov

    #elif using_icov:
    if using_icov:
    #if True:
        if os.path.exists(icov_fn):
            icov = np.loadtxt(icov_fn)
        else:
            raise ValueError(f"Path to inverse covmat {icov_fn} doesn't exist!")
        print("Inverse covariance matrix:")
        print(icov)    
        print(icov.shape)
        f.attrs['inverse_covariance_matrix'] = icov
        #cov = icov # calling it cov so can pass this variable (this is bad i know)

    ### Set up chain datasets ###
    dsetnames = ['chain', 'lnprob', 'lnweight', 'lnevidence', 'varlnevidence']
    #for now will overwrite
    for dsn in dsetnames:
        if dsn in f.keys():
            del f[dsn]
    f.create_dataset('chain', (0, 0, len(param_names)), chunks = True, compression = 'gzip', maxshape = (None, None, len(param_names)))
    f.create_dataset('lnprob', (0, 0,) , chunks = True, compression = 'gzip', maxshape = (None, None, ))
    f.create_dataset('lnweight', (0, 0,), chunks = True, compression = 'gzip', maxshape = (None, None, ))
    f.create_dataset('lnevidence', (0, 0,), chunks = True, compression = 'gzip', maxshape = (None, None, ))
    f.create_dataset('varlnevidence', (0, 0,), chunks = True, compression = 'gzip', maxshape = (None, None, ))
    f.close()

    print("Building emulators")
    emus = [None]*nstats
    for i, statistic in enumerate(statistics):
        emu = emulator.Emulator(statistic, training_dirs[i], nbins=nbins,  fixed_params=fixed_params, gperr=gperrs[i], hyperparams=hyperparams[i], log=logs[i], mean=means[i], nhod=nhods[i], kernel_name=kernel_names[i])
        emu.build()
        emus[i] = emu
        print(f"Emulator for {statistic} built")

    ### Set up covariance matrix ###
    # stat_str = '_'.join(statistics)
    # #If all of the tags for the stats are the same, just use one of them (usually doing this now!)
    # if len(set(traintags))==1 and len(set(errtags))==1 and len(set(testtags))==1:
    #     tag_str = traintags[0] + errtags[0] + testtags[0]
    # else:
    #     # Otherwise will need to join them all
    #     print("Using acctags joined for emu")
    #     tag_str.join(acctags)
    # # for now, use performance covariance on aemulus test set (see emulator/words/error.pdf)
    # cov_emuperf_fn = f"{cov_dir}cov_emuperf_{stat_str}{tag_str}.dat"
    # if os.path.exists(cov_emuperf_fn):
    #     cov = np.loadtxt(cov_emuperf_fn)
    # else:
    #     raise ValueError(f"Path to covmat {cov_emuperf_fn} doesn't exist!")

    start = time.time()
    if mode=='emcee':
        res = chain.run_mcmc_emcee(emus, param_names, ys, cov, fixed_params=fixed_params, 
                             truth=truth, nwalkers=nwalkers, nsteps=nsteps, 
                             nburn=nburn, multi=multi, chain_fn=chain_fn)
    elif mode=='dynesty':
        res = chain.run_mcmc_dynesty(emus, param_names, ys, cov, icov=icov, using_icov=using_icov, fixed_params=fixed_params, 
                                     truth=truth, multi=multi, chain_fn=chain_fn,
                                     dlogz=dlogz, seed=seed)
    elif mode=='minimize':
        res = chain.run_minimizer([emu], param_names, [y], [cov], fixed_params=fixed_params, 
                                  truth=truth, chain_fn=chain_fn)
    else:
        raise ValueError(f"Mode {mode} not recognized!")
    end = time.time()
    print(f"Time: {(end-start)/60.0} min ({(end-start)/3600.} hrs) [{(end-start)/(3600.*24.)} days]")

    return res


if __name__=='__main__':

    #main()
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fn', type=str,
                        help='name of config file')
    args = parser.parse_args()
    main(args.config_fn)

