import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
from scipy.linalg import block_diag
from scipy import optimize
import time

from schwimmbad import MultiPool
import emcee
import corner


def lnprob(theta, *args):
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args)

# use flat prior on bounds emu is built from. 0 if in (ln 1), -inf if out (ln 0)
# theta: params proposed by sampler
def lnprior(theta, param_names, *args):
    for pname, t in zip(param_names, theta):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if np.isnan(t) or t<low or t>high:
            return -np.inf
    return 0


def lnlike(theta, param_names, fixed_params, ys, cov):

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