import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
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

# use flat prior on bounds emu is built from. 0 if in (ln 1), -inf if out (ln 0)
# if proposal is outside of any parameter's prior, returns -inf; only 0 if inside all
# theta: params proposed by sampler
def lnprior(theta, param_names, *args):    
    for pname, t in zip(param_names, theta):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        # ADD PRIOR ON M_CUT
        if pname=='M_cut':
            #print("ADDING STRICTER PRIOR ON M_CUT")
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
    #print("checking hypercube prior!")
    #print(theta_cosmo, in_prior)
    if not in_prior:
        return -np.inf

    return 0

def prior_transform(u, param_names):
    v = np.array(u)
    for i, pname in enumerate(param_names):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if pname=='M_cut':
            #print("ADDING STRICTER PRIOR ON M_CUT")
            low = 11.5
        v[i] = u[i]*(high-low)+low
    return v

def prior_transform_hypercube(u, param_names):
    v = np.array(u)
    # the indices of u / param_names that are cosmo
    idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    if len(idxs_cosmo)>0:
        params_cosmo = param_names[idxs_cosmo]
        #print("params")
        #print(params_cosmo)
        dist = scipy.stats.norm.ppf(u[idxs_cosmo])  # convert to standard normal
        v[idxs_cosmo] = np.dot(_hprior_cov_sqrt, dist) + _hprior_means
        #print(v[idxs_cosmo])
    
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
    #print("lnlike_hypercube; in_prior=",in_prior)#, "prior time:", pe-ps)
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

def lnlike_consthypercube(theta, param_names, fixed_params, ys, cov):#, prior_getter):
    theta = np.array(theta).flatten() #theta looks like [[[p]]] for some reason
    param_dict = dict(zip(param_names, theta)) #weirdly necessary for Powell minimization
    param_dict.update(fixed_params)
    
    # hypercube prior
    theta_cosmo = [param_dict[pn] for pn in _param_names_cosmo]
    in_prior = is_in_hprior(theta_cosmo)
    #print("lnlike_hypercube; in_prior=",in_prior)#, "prior time:", pe-ps)
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
    global _emus, _param_names_cosmo, _param_names_hod
    global _prior_cut, _prior_getter, _hprior_icov, _hprior_center, _hprior_cov, _hprior_means, _hprior_cov_sqrt
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    _prior_cut = 12
    #_prior_getter = hypercube_prior.GET_PriorND()
    _hprior_icov, _hprior_center = get_hprior_icov_center()

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
    f.close()

    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")
    # "The rule of thumb I use is N^2 * a few" (https://github.com/joshspeagle/dynesty/issues/208) 
    nlive = max(num_params**2 * 4, 100)
    #nlive = 500 
    #sample_method = 'rslice'
    sample_method = 'rwalk'
    slices = 5 #default = 5
    walks = 25 #default = 25
    #bound = 'single'
    bound = 'multi'
    # FOR MULTI
    vol_dec = 0.25 #default = 0.5, smaller more conservative
    vol_check = 4.0 #default = 2.0, larger more conservative
    
    print("nlive:", nlive)
    print("sample_method:", sample_method)
    print("walks:", walks)
    print("bound:", bound)
    print("seed:", seed)
    print("vol_dec:", vol_dec, "vol_check:", vol_check)
    print("slices:", slices)

    if np.isnan(seed):
        rstate = None #default global random state will be used
    else:
        from numpy.random import RandomState
        rstate = RandomState(seed)

    #result_keys_tosave = ['samples', 'logl', 'logwt', 'logz', 'logzerr']
    result_keys_2_save_keys = {'samples': 'chain', 'logl': 'lnprob', 'logwt':'lnweight', 
                               'logz':'lnevidence', 'logzerr':'varlnevidence'}
    print('truth:', truth)
    with mp.Pool() as pool:
        print("pool")
        if multi:
            print("multi")
            pool = pool
            queue_size = ncpu
            print("queue_size")
        else:
            print("serial")
            pool = None
            queue_size = None

        print("initialize sampler")
        sampler = dynesty.NestedSampler(
            lnlike,
            #lnlike_const,
            #lnlike_consthypercube,
            #lnlike_hypercube, 
            #prior_transform,
            prior_transform_hypercube, 
            num_params, logl_args=args, nlive=nlive,
            ptform_args=prior_args, rstate=rstate,
            pool=pool, queue_size=queue_size,
            sample=sample_method, walks=walks,
            bound=bound, vol_dec=vol_dec, vol_check=vol_check,
            slices=slices)

        sampler.run_nested(dlogz=dlogz)#, maxiter=10)
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



def run_mcmc_dynesty_gen(emus, param_names, ys, cov, fixed_params={}, truth={}, 
                     plot_fn=None, multi=True, chain_fn=None, dlogz=0.01, seed=None):

    print("Dynesty sampling (static, generator)")
    global _emus, _param_names_cosmo, _param_names_hod
    global _prior_cut, _prior_getter, _hprior_icov, _hprior_center, _hprior_cov, _hprior_means, _hprior_cov_sqrt
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

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
    f.close()

    ncpu = mp.cpu_count()
    nlive = max(num_params**2 * 4, 100)
    print(f"{ncpu} CPUs")

    print("seed:", seed)
    if np.isnan(seed):
        rstate = None #default global random state will be used
    else:
        from numpy.random import RandomState
        rstate = RandomState(seed)
    print("rstate:", rstate)

    result_info_dict = {'lnprob': 3, 'lnweight': 5, 'lnevidence': 6, 'varlnevidence': 7}

    print('truth:', truth)
    with mp.Pool() as pool:

        print("pool")
        if multi:
            print("multi")
            pool = pool
            queue_size = ncpu
            print("queue_size")
        else:
            print("serial")
            pool = None
            queue_size = None
        
        print("initialize sampler")
        sampler = dynesty.NestedSampler(
            lnlike,
            #lnlike_const,
            #lnlike_hypercube, 
            #prior_transform,
            prior_transform_hypercube, 
            num_params, logl_args=args, nlive=nlive, 
            ptform_args=prior_args, rstate=rstate,
            pool=pool, queue_size=queue_size)

        itsave = 100
        
        print("create empty chunk arrays")
        chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
        res_chunks = {}
        for reskey in result_info_dict.keys():
            res_chunks[reskey] = np.empty((nwalkers, itsave)) 

        print("begin sampling via generator")
        for it, result in enumerate(sampler.sample(dlogz=dlogz)):
            pass

            # unindent this!
            # print("adding final live points")
            # for it_final, result in enumerate(sampler.add_live_points()):
            pass
            iti = it + 1
            rem = iti % itsave
            print('iti:', iti)
            print(result)

            chain_chunk[:,rem-1,:] = result[2]
            for reskey, resval in result_info_dict.items():
                res_chunks[reskey][:,rem-1] = result[resval]
            
            if rem != 0:
                continue

            # only happens if at an iteration that's a multiple of itsave!
            print("SAVING")
            print('save iti:', iti)
            f = h5py.File(chain_fn, 'r+')

            chain_dset = f['chain']
            chain_dset.resize((nwalkers, iti, len(param_names)))
            chain_dset[:,iti-itsave:iti,:] = chain_chunk

            for reskey in result_info_dict.keys():
                dset = f[reskey]
                dset.resize((nwalkers, iti))
                dset[:,iti-itsave:iti] = res_chunks[reskey]

            f.close()

            # TOCHECK: need to empty chain_chunk??? -> added
            chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
            # empty res_chunk arrs (intermediate products)
            for reskey in result_info_dict.keys():
                res_chunks[reskey] = np.empty((nwalkers, itsave)) 

        # add final batch even if not a multiple of itsave
        print("saving final bit")
        f = h5py.File(chain_fn, 'r+')

        chain_dset = f['chain']
        chain_dset.resize((nwalkers, iti, len(param_names)))
        chain_dset[:,iti-itsave:iti,:] = chain_chunk[:iti] #cut off the empty samples

        for reskey in result_info_dict.keys():
            dset = f[reskey]
            dset.resize((nwalkers, iti))
            dset[:,iti-itsave:iti] = res_chunks[reskey][:iti] #cut off the empty samples

        f.close()

        # save with pickle
        res = sampler.results
        print(res.summary())

        print("saving results obejct with pickle")
        chain_main = chain_fn.split('/')[-1]
        chaintag = chain_main.split('chains_')[-1].split('.h5')[0]
        pickle_dir = f'../products/dynesty_results'
        os.makedirs(pickle_dir, exist_ok=True)
        pickle_fn = f'{pickle_dir}/results_{chaintag}.pkl'
        with open(pickle_fn, 'wb') as pickle_file:
            pickle.dump(res, pickle_file)



# dynamic!
def run_mcmc_dynesty_dynamic_generator(emus, param_names, ys, cov, fixed_params={}, truth={}, 
                     plot_fn=None, multi=True, chain_fn=None, dlogz=0.01, seed=None):

    print("Dynesty MCMC")
    global _emus, _param_names_cosmo, _param_names_hod, _prior_cut
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    _prior_cut = 12

    nwalkers = 1 #make this have a dimension to line up with emcee chains
    num_params = len(param_names)

    prior_getter = hypercube_prior.GET_PriorND()
    args = [param_names, fixed_params, ys, cov]
    #args = [param_names, fixed_params, ys, cov, prior_getter] # if using lnlike_hypercube
    prior_args = [param_names]

    f = h5py.File(chain_fn, 'r+')
    if 'dlogz' not in f.attrs:
        f.attrs['dlogz'] = dlogz
    f.close()

    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")

    print("seed:", seed)
    if np.isnan(seed):
        rstate = None #default global random state will be used
    else:
        from numpy.random import RandomState
        rstate = RandomState(seed)
    print("rstate:", rstate)

    result_info_dict = {'lnprob': 3, 'lnweight': 5, 'lnevidence': 6, 'varlnevidence': 7}

    
    print('truth:', truth)

    print("DYNAMIC SAMPLING!")
    with mp.Pool() as pool:
        
        if multi:
            print("multi")
            pool = pool
            queue_size = ncpu
        else:
            pool = None
            queue_size = None
        
        sampler = dynesty.DynamicNestedSampler(lnlike, prior_transform, 
                                        num_params, logl_args=args, 
                                        ptform_args=prior_args, rstate=rstate,
                                        pool=pool, queue_size=queue_size)

        print("initial sampling!")
        dlogz_init = dlogz
        nlive_init = 10
        for it, result in enumerate(sampler.sample_initial(nlive=nlive_init,
                                                           maxiter=100, # FOR TESTING
                                                           dlogz=dlogz_init)):
            print('it:', it)
            print(result)
            pass
        
        itsave = 100
        #chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
        chain_chunk = []
        res_chunks = {}
        for reskey in result_info_dict.keys():
            #res_chunks[reskey] = np.empty((nwalkers, itsave)) 
            res_chunks[reskey] = []

        print("main batch sampling!")
        while True:
            stop = stopping_function(sampler.results)  # evaluate stop
            print("stop:", stop)
            if not stop:
                logl_bounds = weight_function(sampler.results)  # derive bounds
                print("logl_bounds:", logl_bounds)
                for it, result in enumerate(sampler.sample_batch(logl_bounds=logl_bounds)):

                    # ok so it's the same kind of single result dict as static case.
                    # but the batch will terminate at some point. so could append to array in this loop
                    # and then save after this loop? and/or at every X iters, bc batches can be large/slow?
                    # the counting does get weird but it does too w static case
                    iti = it + 1
                    rem = iti % itsave
                    print('iti:', iti)
                    print(result)

                    # chain_chunk[:,rem-1,:] = result[2]
                    # for reskey, resval in result_info_dict.items():
                    #     res_chunks[reskey].append(result[resval])
                    #     #res_chunks[reskey][:,rem-1] = result[resval]
                    
                    # # if rem != 0:
                    # #     continue

                    # print("SAVING")
                    # f = h5py.File(chain_fn, 'r+')

                    # chain_dset = f['chain']
                    # chain_dset.resize((nwalkers, iti, len(param_names)))
                    # chain_dset[:,iti-itsave:iti,:] = chain_chunk

                    # for reskey in result_info_dict.keys():
                    #     dset = f[reskey]
                    #     dset.resize((nwalkers, iti))
                    #     dset[:,iti-itsave:iti] = res_chunks[reskey]

                    # f.close()

                sampler.combine_runs()  # add new samples to previous results
            else:
                break


# dynamic! non-generator
def run_mcmc_dynesty_dynamic(emus, param_names, ys, cov, fixed_params={}, truth={}, 
                     plot_fn=None, multi=True, chain_fn=None, dlogz=0.01, seed=None):

    print("Dynesty MCMC")
    global _emus, _param_names_cosmo, _param_names_hod, _prior_cut
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    _prior_cut = 12

    nwalkers = 1 #make this have a dimension to line up with emcee chains
    num_params = len(param_names)

    prior_getter = hypercube_prior.GET_PriorND()
    args = [param_names, fixed_params, ys, cov]
    #args = [param_names, fixed_params, ys, cov, prior_getter] # if using lnlike_hypercube
    prior_args = [param_names]

    f = h5py.File(chain_fn, 'r+')
    if 'dlogz' not in f.attrs:
        f.attrs['dlogz'] = dlogz
    f.close()

    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")

    print("seed:", seed)
    if np.isnan(seed):
        rstate = None #default global random state will be used
    else:
        from numpy.random import RandomState
        rstate = RandomState(seed)
    print("rstate:", rstate)

    #result_info_dict = {'lnprob': 3, 'lnweight': 5, 'lnevidence': 6, 'varlnevidence': 7}
    result_info_dict = {'lnprob': 'logl', 'lnweight': 'logwt', 'lnevidence': 'logz', 'varlnevidence': 'logzerr'}

    print('truth:', truth)

    print("DYNAMIC SAMPLING! no gen!")
    with mp.Pool() as pool:
        
        if multi:
            print("multi")
            pool = pool
            queue_size = ncpu
        else:
            pool = None
            queue_size = None
        
        sampler = dynesty.DynamicNestedSampler(lnlike, prior_transform, 
                                        num_params, logl_args=args, 
                                        ptform_args=prior_args, rstate=rstate,
                                        pool=pool, queue_size=queue_size)

        print("initial sampling!")
        dlogz_init = dlogz
        nlive_init = 1000
        
        #sampler.run_nested(dlogz_init=dlogz_init, nlive_init=nlive_init)#, maxiter=30)
        #result = sampler.results

        # save results after each batch, a la https://github.com/joshspeagle/dynesty/issues/209#issuecomment-700354928

        chaintag = chain_fn.split('/')[-1].split('.')[0]
        pickle_dir = f'../products/dynesty_results/results_{chaintag}'
        os.makedirs(pickle_dir, exist_ok=True)

        print("Saving in batches!")
        n_batch = 10
        #for i in range(1, n):
        i = 0
        stop = False
        while not stop:
            i += 1
            print("ITER:", i)
            sampler.run_nested(dlogz_init=dlogz_init, nlive_init=nlive_init, maxiter_init=100,
                               maxbatch=1) #maxiter=i*n_batch)
            result = sampler.results
            print("RESULT:", result)
            
            pickle_fn = f'{pickle_dir}/result_batch{i}.pkl'
            with open(pickle_fn, 'wb') as pickle_file:
                pickle.dump(result, pickle_file)
            stop = stopping_function(sampler.results)
            print("STOP?:", stop)

        print("SAVING")
        f = h5py.File(chain_fn, 'r+')

        chain_dset = f['chain']
        chain_arr = result['samples']
        print(chain_arr)
        N_chain, N_dim = chain_arr.shape
        print('Nchain, Ndim:', N_chain, N_dim)
        chain_dset.resize((nwalkers, N_chain, N_dim))
        chain_dset[:,:,:] = chain_arr

        for reskey, resval in result_info_dict.items():
            res_arr = result[resval]
            N_arr = len(res_arr)
            dset = f[reskey]
            dset.resize((nwalkers, N_arr))
            dset[:,:] = res_arr

        f.close()

#from git commit on oct 1 - https://github.com/kstoreyf/emulator/blob/master/code/chain.py
def run_mcmc_dynesty_orig(emus, param_names, ys, cov, fixed_params={}, truth={}, 
                     plot_fn=None, multi=True, chain_fn=None, dlogz=0.5):

    global _emus
    _emus = emus

    nwalkers = 1 #make this have a dimension to line up with emcee chains
    num_params = len(param_names)
    args = [param_names, fixed_params, ys, cov]
    prior_args = [param_names]

    f = h5py.File(chain_fn, 'r+')
    if 'dlogz' not in f.attrs:
        f.attrs['dlogz'] = dlogz
    f.close()

    result_info_dict = {'lnprob': 3, 'lnweight': 5, 'lnevidence': 6, 'varlnevidence': 7}

    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")

    print('truth:', truth)
    with mp.Pool() as pool:
        if multi:
            print("multi")
            sampler = dynesty.NestedSampler(lnlike, prior_transform, num_params, logl_args=args, 
                                            ptform_args=prior_args, pool=pool, queue_size=ncpu)
        else:
            print("serial")
            sampler = dynesty.NestedSampler(lnlike, prior_transform, num_params, logl_args=args,
                                            ptform_args=prior_args)

        itsave = 100
        chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
        res_chunks = {}
        for reskey in result_info_dict.keys():
            res_chunks[reskey] = np.empty((nwalkers, itsave)) 

        for it, result in enumerate(sampler.sample(dlogz=dlogz)):
            iti = it + 1
            rem = iti % itsave
            print('iti:', iti)
            print(result)
            chain_chunk[:,rem-1,:] = result[2]
            for reskey, resval in result_info_dict.items():
                res_chunks[reskey][:,rem-1] = result[resval]

            if rem != 0:
                continue

            print("SAVING")
            print('iti:', iti)
            f = h5py.File(chain_fn, 'r+')

            chain_dset = f['chain']
            chain_dset.resize((nwalkers, iti, len(param_names)))
            chain_dset[:,iti-itsave:iti,:] = chain_chunk

            for reskey in result_info_dict.keys():
                dset = f[reskey]
                dset.resize((nwalkers, iti))
                dset[:,iti-itsave:iti] = res_chunks[reskey]

            f.close()

            # empty res_chunk arrs (intermediate products)
            chain_chunk = np.empty((nwalkers, itsave, len(param_names)))       
            for reskey in result_info_dict.keys():
                res_chunks[reskey] = np.empty((nwalkers, itsave)) 

        # not orig here!
        # add final batch even if not a multiple of itsave
        print("saving final bit")
        f = h5py.File(chain_fn, 'r+')

        chain_dset = f['chain']
        chain_dset.resize((nwalkers, iti, len(param_names)))
        chain_dset[:,iti-itsave:iti,:] = chain_chunk[:iti] #cut off the empty samples

        for reskey in result_info_dict.keys():
            dset = f[reskey]
            dset.resize((nwalkers, iti))
            dset[:,iti-itsave:iti] = res_chunks[reskey][:iti] #cut off the empty samples

        f.close()



def run_mcmc_dynesty_genres(emus, param_names, ys, cov,         fixed_params={}, truth={}, 
                        plot_fn=None, multi=True, chain_fn=None, dlogz=0.01, seed=None):

    print("Dynesty sampling (genres)")
    global _emus, _param_names_cosmo, _param_names_hod
    global _prior_cut, _prior_getter, _hprior_icov, _hprior_center, _hprior_cov, _hprior_means, _hprior_cov_sqrt
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

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
    f.close()

    ncpu = mp.cpu_count()
    nlive = max(num_params**2 * 4, 100)
    print(f"{ncpu} CPUs")

    print("seed:", seed)
    if np.isnan(seed):
        rstate = None #default global random state will be used
    else:
        from numpy.random import RandomState
        rstate = RandomState(seed)
    print("rstate:", rstate)

    result_info_dict = {'lnprob': 3, 'lnweight': 5, 'lnevidence': 6, 'varlnevidence': 7}

    print('truth:', truth)
    with mp.Pool() as pool:

        print("pool")
        if multi:
            print("multi")
            pool = pool
            queue_size = ncpu
            print("queue_size")
        else:
            print("serial")
            pool = None
            queue_size = None
        
        print("initialize sampler")
        sampler = dynesty.NestedSampler(
            lnlike,
            #lnlike_const,
            #lnlike_hypercube, 
            #prior_transform,
            prior_transform_hypercube, 
            num_params, logl_args=args, nlive=nlive, 
            ptform_args=prior_args, rstate=rstate,
            pool=pool, queue_size=queue_size)

        print("begin sampling via generator")
        for it, result in enumerate(sampler.sample(dlogz=dlogz)):
            pass

        print("adding final live points")
        for it_final, res in enumerate(sampler.add_live_points()):
            pass
        #sampler.run_nested(dlogz=dlogz)#, maxiter=10)

        print("loops done")
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


def run_mcmc_dynesty_genmatch(emus, param_names, ys, cov, fixed_params={}, truth={}, 
                     plot_fn=None, multi=True, chain_fn=None, dlogz=0.01, seed=None):

    print("Dynesty sampling (static, generator)")
    global _emus, _param_names_cosmo, _param_names_hod
    global _prior_cut, _prior_getter, _hprior_icov, _hprior_center, _hprior_cov, _hprior_means, _hprior_cov_sqrt
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

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
    f.close()

    ncpu = mp.cpu_count()
    nlive = max(num_params**2 * 4, 100)
    print(f"{ncpu} CPUs")

    print("seed:", seed)
    if np.isnan(seed):
        rstate = None #default global random state will be used
    else:
        from numpy.random import RandomState
        rstate = RandomState(seed)
    print("rstate:", rstate)

    result_info_dict = {'lnprob': 3, 'lnweight': 5, 'lnevidence': 6, 'varlnevidence': 7}

    print('truth:', truth)
    with mp.Pool() as pool:

        print("pool")
        if multi:
            print("multi")
            pool = pool
            queue_size = ncpu
            print("queue_size")
        else:
            print("serial")
            pool = None
            queue_size = None
        
        print("initialize sampler")
        sampler = dynesty.NestedSampler(
            #lnlike,
            lnlike_const,
            #lnlike_hypercube, 
            prior_transform,
            #prior_transform_hypercube, 
            num_params, logl_args=args, nlive=nlive, 
            ptform_args=prior_args, rstate=rstate,
            pool=pool, queue_size=queue_size)

        itsave = 100
        
        print("create empty chunk arrays")
        chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
        res_chunks = {}
        for reskey in result_info_dict.keys():
            res_chunks[reskey] = np.empty((nwalkers, itsave)) 

        print("begin sampling via generator")
        for it, result in enumerate(sampler.sample(dlogz=dlogz)):

            pass
            # iti = it + 1
            # rem = iti % itsave
            # print('iti:', iti)
            # print(result)

            # chain_chunk[:,rem-1,:] = result[2]
            # for reskey, resval in result_info_dict.items():
            #     res_chunks[reskey][:,rem-1] = result[resval]
            
            # if rem != 0:
            #     continue

            # only happens if at an iteration that's a multiple of itsave!
            # print("SAVING")
            # print('save iti:', iti)
            # f = h5py.File(chain_fn, 'r+')

            # chain_dset = f['chain']
            # chain_dset.resize((nwalkers, iti, len(param_names)))
            # chain_dset[:,iti-itsave:iti,:] = chain_chunk

            # for reskey in result_info_dict.keys():
            #     dset = f[reskey]
            #     dset.resize((nwalkers, iti))
            #     dset[:,iti-itsave:iti] = res_chunks[reskey]

            # f.close()

            # # TOCHECK: need to empty chain_chunk???
            # chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
            # # empty res_chunk arrs (intermediate products)
            # for reskey in result_info_dict.keys():
            #     res_chunks[reskey] = np.empty((nwalkers, itsave)) 

        # add final batch even if not a multiple of itsave
        # print("saving final bit")
        # f = h5py.File(chain_fn, 'r+')

        # chain_dset = f['chain']
        # chain_dset.resize((nwalkers, iti, len(param_names)))
        # chain_dset[:,iti-itsave:iti,:] = chain_chunk[:iti] #cut off the empty samples

        # for reskey in result_info_dict.keys():
        #     dset = f[reskey]
        #     dset.resize((nwalkers, iti))
        #     dset[:,iti-itsave:iti] = res_chunks[reskey][:iti] #cut off the empty samples

        # f.close()