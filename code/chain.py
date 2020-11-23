import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
from scipy.linalg import block_diag
from scipy import optimize
import scipy
import time

from schwimmbad import MultiPool
import emcee
import dynesty

import hypercube_prior


def lnprob(theta, *args):
    lp = lnprior(theta, *args)
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
            print("ADDING STRICTER PRIOR ON M_CUT")
            low = 11.5
        if np.isnan(t) or t<low or t>high:
            return -np.inf
    return 0

# not using this rn bc putting prior in likelihood
def lnprior_hypercube(theta, param_names, *args):

    theta_hod = [theta[i] for i in range(len(theta)) if param_names[i] in _param_names_hod]
    for pname, t in zip(param_names, theta):
        low, high = _emus[0].get_param_bounds(pname)
        # ADD PRIOR ON M_CUT
        if pname=='M_cut':
            low = 11.5
        if np.isnan(t) or t<low or t>high:
            return -np.inf   

    # need to do this to get in proper order
    theta_cosmo = []
    for i in range(len(_param_names_cosmo)):
        t = theta[param_names.index(_param_names_cosmo[i])]
        theta_cosmo.append(t)

    # hypercube prior
    GG = hypercube_prior.GET_PriorND()
    CUT=12
    in_prior = GG.isinornot(theta_cosmo, CUT) 
    print("checking hypercube prior!")
    print(theta_cosmo)
    print(in_prior)
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
    params_cosmo = param_names[idxs_cosmo]
    print("idxs, params")
    print(idxs_cosmo)
    print(params_cosmo)
    dist = scipy.stats.norm.ppf(u[idxs_cosmo])  # convert to standard normal
    
    GG = hypercube_prior.GET_PriorND()
    cov = GG.get_cov(params_cosmo)
    print("cov")
    print(cov)
    means = GG.get_means(params_cosmo)
    #print(np.dot(np.sqrt(cov), dist))
    print(means)
    #v[idxs_cosmo] = scipy.stats.multivariate_normal.cdf(u[idxs_cosmo], mean=means, cov=cov)
    #v[idxs_cosmo] = 1./v[idxs_cosmo]
    icov = np.linalg.inv(cov)
    v[idxs_cosmo] = np.dot(icov, dist) + means
    print(v[idxs_cosmo])
    
    idxs_hod = [i for i in range(len(param_names)) if param_names[i] in _param_names_hod]
    params_hod = param_names[idxs_hod]
    for i, pname in zip(idxs_hod, params_hod):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if pname=='M_cut':
            print("Mcut prior")
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


def lnlike_hypercube(theta, param_names, fixed_params, ys, cov):
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

    # hypercube prior
    theta_cosmo = [param_dict[pn] for pn in _param_names_cosmo]
    GG = hypercube_prior.GET_PriorND()
    CUT=12
    in_prior = GG.isinornot(theta_cosmo, CUT)
    print("like call: theta=", theta, "; time=", e-s, "s; like =", like, '; in_prior=',in_prior)
    if not in_prior:
        return -np.inf

    return like


def _random_initial_guess(param_names, nwalkers, num_params):
    pos0 = np.zeros((nwalkers, num_params))
    # enumerate adds an index just based on array pos
    for idx, pname in enumerate(param_names):
        low, high = _emus[0].get_param_bounds(pname)
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0
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



def run_mcmc(emus, param_names, ys, cov, fixed_params={}, truth={}, nwalkers=24,
        nsteps=100, nburn=20, plot_fn=None, multi=True, chain_fn=None):

    global _emus
    _emus = emus

    num_params = len(param_names)
    args = [param_names, fixed_params, ys, cov]
    
    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")

    print('truth:', truth)
    with mp.Pool() as pool:
    #with MultiPool() as pool:
        if multi:
            print("multi")
            sampler = emcee.EnsembleSampler(nwalkers, num_params, lnprob, args=args, pool=pool)
            #sampler = emcee.EnsembleSampler(nwalkers, num_params, lnprob, args=args, threads=4)
        else:
            print("serial")
            sampler = emcee.EnsembleSampler(nwalkers, num_params, lnprob, args=args)

        p0 = _random_initial_guess(param_names, nwalkers, num_params)
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


def run_mcmc_dynesty(emus, param_names, ys, cov, fixed_params={}, truth={}, 
                     plot_fn=None, multi=True, chain_fn=None, dlogz=0.01, seed=None):

    global _emus, _param_names_cosmo, _param_names_hod
    _emus = emus
    _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

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

    print("seed:", seed)
    if np.isnan(seed):
        rstate = None #default global random state will be used
    else:
        from numpy.random import RandomState
        rstate = RandomState(seed)
    print("rstate:", rstate)

    print('truth:', truth)
    with mp.Pool() as pool:
        if multi:
            print("multi")
            sampler = dynesty.NestedSampler(lnlike_hypercube, prior_transform, 
                                            num_params, logl_args=args, 
                                            ptform_args=prior_args, rstate=rstate,
                                            pool=pool, queue_size=ncpu)
        else:
            print("serial")
            sampler = dynesty.NestedSampler(lnlike_hypercube, prior_transform,
                                            num_params, logl_args=args,
                                            ptform_args=prior_args, rstate=rstate)

        itsave = 100
        chain_chunk = np.empty((nwalkers, itsave, len(param_names))) 
        lnprob_chunk = np.empty((nwalkers, itsave)) 
        lnweight_chunk = np.empty((nwalkers, itsave))
        lnevidence_chunk = np.empty((nwalkers, itsave))
        varlnevidence_chunk = np.empty((nwalkers, itsave))
        for it, result in enumerate(sampler.sample(dlogz=dlogz)):
            iti = it + 1
            rem = iti % itsave
            print('iti:', iti)
            print(result)
            chain_chunk[:,rem-1,:] = result[2]
            lnprob_chunk[:,rem-1] = result[3]
            lnweight_chunk[:,rem-1] = result[5]
            lnevidence_chunk[:,rem-1] = result[6]
            varlnevidence_chunk[:,rem-1] = result[7]

            if rem != 0:
                continue

            print("SAVING")
            print('iti:', iti)
            f = h5py.File(chain_fn, 'r+')

            chain_dset = f['chain']
            lnprob_dset = f['lnprob']
            lnweight_dset = f['lnweight']
            lnevidence_dset = f['lnevidence']
            varlnevidence_dset = f['varlnevidence']

            chain_dset.resize((nwalkers, iti, len(param_names)))
            lnprob_dset.resize((nwalkers, iti))
            lnweight_dset.resize((nwalkers, iti))
            lnevidence_dset.resize((nwalkers, iti))
            varlnevidence_dset.resize((nwalkers, iti))
        
            chain_dset[:,iti-itsave:iti,:] = chain_chunk
            lnprob_dset[:,iti-itsave:iti] = lnprob_chunk
            lnweight_dset[:,iti-itsave:iti] = lnweight_chunk
            lnevidence_dset[:,iti-itsave:iti] = lnevidence_chunk
            varlnevidence_dset[:,iti-itsave:iti] = varlnevidence_chunk

            f.close()
