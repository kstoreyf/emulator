import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
from scipy.linalg import block_diag, sqrtm
from scipy import optimize
import scipy
import time
import pickle

from schwimmbad import MultiPool
import emcee
import dynesty
from dynesty.dynamicsampler import stopping_function, weight_function

import hypercube_prior


def lnprob(theta, *args):
    #lp = lnprior(theta, *args)
    lp = lnprior_hypercube(theta, *args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args)

# use flat prior on bounds emu is built from. 0 if all params in prior (ln 1), -inf if out (ln 0)
def lnprior(theta, param_names, *args):    
    for pname, t in zip(param_names, theta):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if pname=='M_cut':
            # ADDING STRICTER PRIOR ON M_CUT
            low = 11.5
        if np.isnan(t) or t<low or t>high:
            return -np.inf
    return 0


def lnprior_hypercube(theta, param_names, fixed_params, *args):
    # indices in param_names/theta that are hod params
    idxs_hod = [i for i in range(len(param_names)) if param_names[i] in _param_names_hod]
    params_hod = param_names[idxs_hod]
    theta_hod = theta[idxs_hod]

    for pname, t in zip(params_hod, theta_hod):
        low, high = _emus[0].get_param_bounds(pname)
        # ADD PRIOR ON M_CUT
        if pname=='M_cut':
            low = 11.5
        if np.isnan(t) or t<low or t>high:
            return -np.inf   

    # need to do this to get in proper order
    # indices in param_names/theta that are cosmo params
    theta = np.array(theta).flatten()
    param_dict = dict(zip(param_names, theta))
    param_dict.update(fixed_params)

    # hypercube prior
    theta_cosmo = [param_dict[pn] for pn in _param_names_cosmo]
    in_prior = is_in_hprior(theta_cosmo)
    if not in_prior:
        return -np.inf

    return 0


def prior_transform(u, param_names):
    v = np.array(u)
    for i, pname in enumerate(param_names):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if pname=='M_cut':
            # ADDING STRICTER PRIOR ON M_CUT
            low = 11.5
        v[i] = u[i]*(high-low)+low
    return v


def prior_transform_hypercube(u, param_names):
    v = np.array(u)
    # the indices of u / param_names that are cosmo
    idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    if len(idxs_cosmo)>0:
        params_cosmo = param_names[idxs_cosmo]
        dist = scipy.stats.norm.ppf(u[idxs_cosmo])  # convert to standard normal
        v[idxs_cosmo] = np.dot(_hprior_cov_sqrt, dist) + _hprior_means
    
    idxs_hod = [i for i in range(len(param_names)) if param_names[i] in _param_names_hod]
    params_hod = param_names[idxs_hod]
    for i, pname in zip(idxs_hod, params_hod):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if pname=='M_cut':
            low = 11.5
        v[i] = u[i]*(high-low)+low

    return v


def lnlike(theta, param_names, fixed_params, ys, cov):
    s = time.time()
    theta = np.array(theta).flatten() #theta looks like [[[p]]] for some reason
    param_dict = dict(zip(param_names, theta)) #weirdly necessary for Powell minimization
    param_dict.update(fixed_params)
    emu_preds = []
    for emu in _emus:
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


def lnlike_hypercube(theta, param_names, fixed_params, ys, cov):#, prior_getter):
    s = time.time()
    theta = np.array(theta).flatten() #theta looks like [[[p]]] for some reason
    param_dict = dict(zip(param_names, theta)) #weirdly necessary for Powell minimization
    param_dict.update(fixed_params)
    
    # hypercube prior
    theta_cosmo = [param_dict[pn] for pn in _param_names_cosmo]
    in_prior = is_in_hprior(theta_cosmo)
    if not in_prior:
        return -np.inf

    # make emu predictions
    emu_preds = []
    for emu in _emus:
        pred = emu.predict(param_dict)
        emu_preds.append(pred)

    emu_pred = np.hstack(emu_preds)
    diff = (np.array(emu_pred) - np.array(ys))/np.array(ys) #fractional error
    diff = diff.flatten()

    # the solve is a better way to get the inverse
    like = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    e = time.time()

    print("lnlike_hypercube: theta=", theta, "; time=", e-s, "s; like =", like)
    return like


# to test constant priors
def lnlike_consthypercube(theta, param_names, fixed_params, ys, cov):#, prior_getter):
    theta = np.array(theta).flatten() #theta looks like [[[p]]] for some reason
    param_dict = dict(zip(param_names, theta)) #weirdly necessary for Powell minimization
    param_dict.update(fixed_params)
    
    # hypercube prior
    theta_cosmo = [param_dict[pn] for pn in _param_names_cosmo]
    in_prior = is_in_hprior(theta_cosmo)
    if not in_prior:
        return -np.inf

    return 0

def lnlike_const(theta, param_names, fixed_params, ys, cov):
    return 0

def get_hprior_icov_center():
    cosmo = np.loadtxt("/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/hod_file/cosmology_camb_full.dat")
    Cosmo = np.delete(cosmo, 23, axis=0)
    center = np.mean(Cosmo, axis=0)
    cov = np.cov(Cosmo.T)
    icov = np.linalg.inv(cov)
    return icov, center

def get_hprior_cov_means():
    cosmo = np.loadtxt("/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/hod_file/cosmology_camb_full.dat")
    Cosmo = np.delete(cosmo, 23, axis=0)
    means = np.mean(Cosmo, axis=0)
    cov = np.cov(Cosmo.T)
    return cov, means

def is_in_hprior(position, CUT=12):
    dif = position - _hprior_center
    t = np.dot(dif, np.dot(_hprior_icov, dif))
    if t<CUT:
        return True
    else:
        return False

def _random_initial_guess(param_names, fixed_params, nwalkers, num_params):
    pos0 = np.zeros((nwalkers, num_params))
    # enumerate adds an index just based on array pos
    for idx, pname in enumerate(param_names):
        low, high = _emus[0].get_param_bounds(pname)
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0
    return pos0

def _random_initial_guess_hypercube(param_names, fixed_params, nwalkers, num_params):
    pos0 = np.zeros((nwalkers, num_params))

    for nw in range(nwalkers):
        # stop when in_prior
        in_prior = False
        while not in_prior:
            pos = np.zeros(num_params)
            for idx, pname in enumerate(param_names):
                low, high = _emus[0].get_param_bounds(pname)

                sigma = np.abs(high - low) / 6.0
                mu = (low + high) / 2.0
                pos[idx] = sigma*np.random.randn() + mu 

            param_dict = dict(zip(param_names, pos))
            param_dict.update(fixed_params)
            pos_cosmo = [param_dict[pn] for pn in _param_names_cosmo]
            in_prior = is_in_hprior(pos_cosmo)

        pos0[nw,:] = pos

    return pos0


def run_minimizer(emus, param_names, ys, cov, fixed_params={}, truth={}, chain_fn=None):

    global _emus
    _emus = emus

    num_params = len(param_names)

    truths = [truth[pname] for pname in param_names]
    f = h5py.File(chain_fn, 'r+')
    f.attrs['true_values'] = truths

    args = (param_names, fixed_params, ys, cov)
    
    p0 = _random_initial_guess(param_names, 1, num_params)
    bounds = [_emus[0].get_param_bounds(pname) for pname in param_names]

    def neglnlike(*args):
        return -lnlike(*args)

    #res = optimize.minimize(neglnlike, p0, args=args, method='L-BFGS-B', bounds=bounds)
    res = optimize.minimize(neglnlike, p0, args=args, method='Powell')
    print(res)
    f.attrs['minimizer_results'] = res.x
    print("resx:",res.x)
    print(truth)
    f.close()
    return res



def run_mcmc_emcee(emus, param_names, ys, cov, fixed_params={}, truth={}, nwalkers=24,
        nsteps=100, nburn=20, plot_fn=None, multi=True, chain_fn=None):

    global _emus, _param_names_cosmo, _param_names_hod, _hprior_icov, _hprior_center
    _emus = emus
    
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    _hprior_icov, _hprior_center = get_hprior_icov_center()

    num_params = len(param_names)
    args = [param_names, fixed_params, ys, cov]
    
    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")

    print('truth:', truth)
    with mp.Pool() as pool:

        if multi:
            print("multi")
            pool = pool
        else:
            pool = None
            print("serial")

        sampler = emcee.EnsembleSampler(nwalkers, num_params, lnprob, args=args, pool=pool)

        #p0 = _random_initial_guess(param_names, nwalkers, num_params)
        p0 = _random_initial_guess_hypercube(param_names, fixed_params, nwalkers, num_params)

        print(param_names)
        print("Initial:", p0)

        if nburn==0:
            pos = p0
        else:
            pos, prob, state = sampler.run_mcmc(p0, nburn)
            sampler.reset()

        itsave = 100
        chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
        lnprob_chunk = np.empty((nwalkers, itsave)) 
        for it, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False)):
            iti = it + 1
            rem = iti % itsave

            chain_chunk[:,rem-1,:] = result[0]
            lnprob_chunk[:,rem-1] = result[1]

            if rem != 0:
                continue
            
            print("saving!")
            print('iti:', iti)
            f = h5py.File(chain_fn, 'r+')

            chain_dset = f['chain']
            lnprob_dset = f['lnprob']
            chain_dset.resize((nwalkers, iti, len(param_names)))
            lnprob_dset.resize((nwalkers, iti))
        
            chain_dset[:,iti-itsave:iti,:] = chain_chunk
            lnprob_dset[:,iti-itsave:iti] = lnprob_chunk

            f.close()

            chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
            lnprob_chunk = np.empty((nwalkers, itsave)) 


# NONGEN
def run_mcmc_dynesty(emus, param_names, ys, cov, fixed_params={}, truth={}, 
                     plot_fn=None, multi=True, chain_fn=None, dlogz=0.01, seed=None):

    print("Dynesty sampling (static) - nongen")
    global _emus, _param_names_cosmo, _param_names_hod, _hprior_cov_sqrt, _hprior_means
    #global _prior_cut, _prior_getter, _hprior_icov, _hprior_center, _hprior_cov, _hprior_means, _hprior_cov_sqrt
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    # _prior_cut = 12
    # #_prior_getter = hypercube_prior.GET_PriorND()
    # _hprior_icov, _hprior_center = get_hprior_icov_center()

    idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    if len(idxs_cosmo)>0:
        hcov, hmeans = get_hprior_cov_means()
        hcov_idxs = hcov[idxs_cosmo][:,idxs_cosmo] 
        _hprior_cov_sqrt = sqrtm(hcov_idxs)
        _hprior_means = hmeans[idxs_cosmo]
    else:
        _hprior_cov_sqrt = None
        _hprior_means = None

    nwalkers = 1 #make this have a dimension to line up with emcee chains
    num_params = len(param_names)

    args = [param_names, fixed_params, ys, cov]
    prior_args = [param_names]

    f = h5py.File(chain_fn, 'r+')
    if 'dlogz' not in f.attrs:
        f.attrs['dlogz'] = dlogz
    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")

    # "The rule of thumb I use is N^2 * a few" (https://github.com/joshspeagle/dynesty/issues/208) 
    nlive = num_params**2 * 3
    #nlive = max(num_params**2 * 3, 100)
    f.attrs['nlive'] = nlive
    #nlive = 500 
    #sample_method = 'rslice'
    sample_method = 'rwalk' #default
    slices = 5 #default = 5
    walks = 25 #default = 25
    #bound = 'single'
    bound = 'multi' #default
    # FOR MULTI
    vol_dec = 0.5 #default = 0.5, smaller more conservative
    vol_check = 2.0 #default = 2.0, larger more conservative
    f.close()
    
    print("nlive:", nlive)
    print("sample_method:", sample_method)
    print("walks:", walks)
    print("bound:", bound)
    print("seed:", seed)
    print("vol_dec:", vol_dec, "vol_check:", vol_check)
    print("slices:", slices)

    if np.isnan(seed):
        seed = np.random.randint(low=0, high=1000)
        #rstate = None #default global random state will be used
    rstate = RandomState(seed)

    result_keys_2_save_keys = {'samples': 'chain', 'logl': 'lnprob', 'logwt':'lnweight', 
                               'logz':'lnevidence', 'logzerr':'varlnevidence'}
    print('truth:', truth)
    with mp.Pool() as pool:
        print("pool")
        if multi:
            print("multi")
            pool = pool
            queue_size = ncpu
        else:
            print("serial")
            pool = None
            queue_size = None

        print("initialize sampler")
        sampler = dynesty.NestedSampler(
            lnlike,
            prior_transform_hypercube, 
            num_params, logl_args=args, nlive=nlive,
            ptform_args=prior_args, rstate=rstate,
            pool=pool, queue_size=queue_size,
            sample=sample_method, walks=walks,
            bound=bound, vol_dec=vol_dec, vol_check=vol_check,
            slices=slices)

        sampler.run_nested(dlogz=dlogz)
        res = sampler.results
        print(res.summary())

        # save with pickle
        print("saving results obejct with pickle")
        chain_main = chain_fn.split('/')[-1]
        chaintag = chain_main.split('chains_')[-1].split('.h5')[0]
        pickle_dir = f'../products/dynesty_results'
        os.makedirs(pickle_dir, exist_ok=True)
        pickle_fn = f'{pickle_dir}/results_{chaintag}.pkl'
        with open(pickle_fn, 'wb') as pickle_file:
            pickle.dump(res, pickle_file)

        print("save to h5 file")
        f = h5py.File(chain_fn, 'r+')
        ncall = len(res['ncall'])
        
        for reskey, savekey in result_keys_2_save_keys.items():
            dset = f[savekey]
            if reskey=='samples':
                dset.resize((nwalkers, ncall, num_params))
                dset[0,:,:] = res[reskey]
            else:
                dset.resize((nwalkers, ncall))
                dset[0,:] = res[reskey]
            
        f.close()