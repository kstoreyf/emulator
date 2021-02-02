import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import scipy
import emcee
import h5py 
from scipy.linalg import block_diag

import chain
import emulator
import initialize_chain


def main():
    #config_fn = f'../chain_configs/chains_wp.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi.cfg'
    #config_fn = f'../chain_configs/chains_wp_upf.cfg'
    #config_fn = f'../chain_configs/chains_wp_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_upf_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_upf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_mcf.cfg'
    config_fn = f'../chain_configs/chains_wp_xi_upf_mcf.cfg'
    
    chain_fn = initialize_chain.main(config_fn)
    run(chain_fn, overwrite=True, mode='dynesty')
    #run(chain_fn, overwrite=True, mode='emcee')


def run(chain_fn, mode='dynesty', overwrite=False):

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
    
    # print chain file info
    for k, v in hf.attrs.items():
        print(f'{k}: {v}')

    # Set file and directory names
    nstats = len(statistics)
    training_dirs = [None]*nstats
    testing_dirs = [None]*nstats
    hyperparams = [None]*nstats
    acctags = [None]*nstats
    gperrs = [None]*nstats
    ys = []
    cov_dir = '../../clust/covariances/'
    for i, statistic in enumerate(statistics):
        gptag = traintags[i] + errtags[i] + tags[i]
        acctags[i] = gptag + testtags[i]
        res_dir = '../../clust/results_{}/'.format(statistic)
        gperrs[i] = np.loadtxt(cov_dir+"error_aemulus_{}{}.dat".format(statistic, errtags[i]))
        training_dirs[i] = '{}training_{}{}/'.format(res_dir, statistic, traintags[i])
        testing_dirs[i] = '{}testing_{}{}/'.format(res_dir, statistic, testtags[i])
        hyperparams[i] = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

        # actual calculated stat
        _, y = np.loadtxt(testing_dirs[i]+'{}_cosmo_{}_HOD_{}_mean.dat'.format(statistic, cosmo, hod))
        ys.extend(y)

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

    # full dict of true values
    f.attrs['true_params_all'] = fixed_params
    # remove params that we want to vary from fixed param dict and add true values
    truth = {}
    for pn in param_names:
        truth[pn] = fixed_params[pn]
        fixed_params.pop(pn)
    truths = [truth[pname] for pname in param_names]
    f.attrs['fixed_params'] = fixed_params
    f.attrs['variable_params'] = truth
    #only those not fixed, aka this mirrors param_names
    f.attrs['true_values'] = truths 
    print("True values:")
    print(truth)
    
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
        emu = emulator.Emulator(statistic, training_dirs[i],  fixed_params=fixed_params, gperr=gperrs[i], hyperparams=hyperparams[i], log=logs[i], mean=means[i], nhod=nhods[i], kernel_name=kernel_names[i])
        emu.build()
        emus[i] = emu
        print(f"Emulator for {statistic} built")

    ### Set up covariance matrix ###
    stat_str = '_'.join(statistics)
    cov_minerva_fn = f'../../clust/covariances/cov_minerva_{stat_str}.dat'
    #If all of the tags for the stats are the same, just use one of them (usually doing this now!)
    if len(set(traintags))==1 and len(set(errtags))==1 and len(set(testtags))==1:
        tag_str = traintags[0] + errtags[0] + testtags[0]
    else:
        # Otherwise will need to join them all
        print("Using acctags joined for emu")
        tag_str.join(acctags)
    # for now, use performance covariance on aemulus test set (see emulator/words/error.pdf)
    cov_emuperf_fn = f"{cov_dir}cov_emuperf_{stat_str}{tag_str}.dat"
    if os.path.exists(cov_emuperf_fn):
        cov = np.loadtxt(cov_emuperf_fn)
    else:
        raise ValueError(f"Path to covmat {cov_emuperf_fn} doesn't exist!")

    # eventually will combine emu covariance with some test covariance, e.g. from minerva
    #cov_emu_fn = f"../testing_results/cov_emu_{stat_str}{tag_str}.dat"
    #if os.path.exists(cov_minerva_fn) and os.path.exists(cov_emu_fn):
        # cov_minerva = np.loadtxt(cov_minerva_fn)
        # cov_minerva *= (1.5/1.05)**3
        # cov_emu = np.loadtxt(cov_emu_fn)
        # cov = cov_emu + cov_minerva
    # else:
    #     print("No combined covmat exists, making diagonal")
    #     covs = []
    #     for i, statistic in enumerate(statistics):
    #         cov_minerva = np.loadtxt(f'../../clust/covariances/cov_minerva_{statistic}.dat')
    #         cov_minerva *= 1./5. * (1.5/1.05)**3
    #         cov_emu = np.loadtxt(f"../testing_results/cov_emu_{statistic}{acctags[i]}.dat")
    #         cov_perf = cov_emu + cov_minerva
    #         covs.append(cov_perf)
    #     cov = block_diag(*covs)

    print("Covariance matrix:")
    print(cov)    
    print("Condition number:", np.linalg.cond(cov))
    
    start = time.time()
    if mode=='emcee':
        res = chain.run_mcmc_emcee(emus, param_names, ys, cov, fixed_params=fixed_params, 
                             truth=truth, nwalkers=nwalkers, nsteps=nsteps, 
                             nburn=nburn, multi=multi, chain_fn=chain_fn)
    elif mode=='dynesty':
        res = chain.run_mcmc_dynesty(emus, param_names, ys, cov, fixed_params=fixed_params, 
                                     truth=truth, multi=multi, chain_fn=chain_fn,
                                     dlogz=dlogz, seed=seed)
    elif mode=='minimize':
        res = chain.run_minimizer([emu], param_names, [y], [cov], fixed_params=fixed_params, 
                                  truth=truth, chain_fn=chain_fn)
    else:
        raise ValueError(f"Mode {mode} not recognized!")
    end = time.time()
    print(f"Time: {(end-start)/60.0} min ({(end-start)/3600.} hrs) [{(end-start)/(3600.*24.)} hrs]")

    return res


if __name__=='__main__':
    main()
