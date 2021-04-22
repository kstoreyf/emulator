import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import scipy
import emcee
import h5py 
from scipy.linalg import block_diag

import emulator
import initialize_chain


def main():
    #config_fn = f'../chain_configs/chains_wp.cfg'
    config_fn = f'../chain_configs/chains_wp_xi.cfg'
    #config_fn = f'../chain_configs/chains_wp_upf.cfg'
    #config_fn = f'../chain_configs/chains_wp_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_upf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_upf_mcf.cfg'
    #config_fn = f'../chain_configs/chains_wp_xi_upf_mcf.cfg'
    
    chain_fn = initialize_chain.main(config_fn)
    run(chain_fn, overwrite=True)
    #run(chain_fn, overwrite=True, mode='emcee')


def run(chain_fn, overwrite=False):

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
    assert len(param_names)==2, "exactly 2 free params needed!"
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

    print("Building emulators")
    emus = [None]*nstats
    for i, statistic in enumerate(statistics):
        emu = emulator.Emulator(statistic, training_dirs[i], nbins=nbins,  fixed_params=fixed_params, gperr=gperrs[i], hyperparams=hyperparams[i], log=logs[i], mean=means[i], nhod=nhods[i], kernel_name=kernel_names[i])
        emu.build()
        emus[i] = emu
        print(f"Emulator for {statistic} built")

    # Set up grid of parameters 
    n_grid = 10
    lnlike_grid = np.empty((n_grid, n_grid))
    bounds0 = emus[0].get_param_bounds(param_names[0])
    bounds1 = emus[0].get_param_bounds(param_names[1])
    p0 = np.linspace(bounds0[0], bounds0[1], n_grid)
    p1 = np.linspace(bounds1[0], bounds1[1], n_grid)
    #pp0, pp1 = np.meshgrid(p0, p1, sparse=False, indexing='ij')
    
    for i in range(len(p0)):
        for j in range(len(p1)):
            theta = [p0[i], p1[j]]
            lnlike_grid[i,j] = lnlike(theta, param_names, fixed_params, ys, cov, emus)
    
    save_lnlike_fn = '../products/likelihood_maps/like_test.npy'
    save_lnlikeparams_fn = '../products/likelihood_maps/like_params.npy'
    np.save(save_lnlike_fn, lnlike_grid)
    np.save(save_lnlikeparams_fn, [p0, p1])    

    start = time.time()

    end = time.time()
    print(f"Time: {(end-start)/60.0} min ({(end-start)/3600.} hrs) [{(end-start)/(3600.*24.)} days]")

    #return res


def lnlike(theta, param_names, fixed_params, ys, cov, emus):
    s = time.time()
    theta = np.array(theta).flatten() #theta looks like [[[p]]] for some reason
    param_dict = dict(zip(param_names, theta)) #weirdly necessary for Powell minimization
    param_dict.update(fixed_params)
    emu_preds = []
    for emu in emus:
        pred = emu.predict(param_dict)
        emu_preds.append(pred)

    emu_pred = np.hstack(emu_preds)
    diff = (np.array(emu_pred) - np.array(ys))/np.array(ys) #fractional error
    diff = diff.flatten()

    # the solve is a better way to get the inverse
    like = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    e = time.time()
    print("like call: theta=", theta, "; time=", e-s, "s; like =", like)

    return like

if __name__=='__main__':
    main()
