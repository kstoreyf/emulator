import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
from scipy import optimize

import emcee
import corner


def lnprob(theta, *args):
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args)
    # to test likelihood issues
    #return lp

# use flat prior on bounds emu is built from. 0 if in (ln 1), -inf if out (ln 0)
# x/theta: params proposed by sampler
def lnprior(theta, param_names, *args):
    #print(theta)
    for pname, t in zip(param_names, theta):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if np.isnan(t) or t<low or t>high:
            return -np.inf
    return 0


def lnlike(theta, param_names, fixed_params, ys, combined_inv_cov):
    param_dict = dict(zip(param_names, theta))
    param_dict.update(fixed_params)
    emu_preds = []
    for emu in _emus:
        pred = emu.predict(param_dict)
        emu_preds.append(pred)
    emu_pred = np.hstack(emu_preds)
    diff = np.array(emu_pred) - np.array(ys)
    # TODO: sean doesn't have the 1/2 factor?
    like = -np.dot(diff, np.dot(combined_inv_cov, diff.T).T) / 2.0
    return like


def _random_initial_guess(param_names, nwalkers, num_params):
    pos0 = np.zeros((nwalkers, num_params))
    # enumerate adds an index just based on array pos
    for idx, pname in enumerate(param_names):
        low, high = _emus[0].get_param_bounds(pname)
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0
        # TODO variable with of the initial guess

    return pos0


def run_minimizer(emus, param_names, ys, covs, fixed_params={}, truth={}, nwalkers=1000,
        nsteps=100, nburn=20, plot_fn=None, multi=True, chain_fn=None):

    global _emus
    _emus = emus

    num_params = len(param_names)

    # todo: not sure how to deal w multiple cov mats?!
    combined_inv_cov = []
    for cov in covs:
        combined_inv_cov.append(np.linalg.inv(cov))
    combined_inv_cov = np.array(combined_inv_cov)

    truths = [truth[pname] for pname in param_names]
    f = h5py.File(chain_fn, 'r+')
    f.attrs['true_values'] = truths

    args = (param_names, fixed_params, ys, combined_inv_cov)
    
    p0 = _random_initial_guess(param_names, 1, num_params)
    bounds = [_emus[0].get_param_bounds(pname) for pname in param_names]
    #p0 = [truth[pn] for pn in param_names]
    print(param_names)
    print(p0)
    print(bounds)
    def neglnlike(*args):
        return -lnlike(*args)

    # ngrid = 10
    # grid = np.array([np.linspace(bounds[i][0], bounds[i][1], ngrid) for i in range(len(param_names))])
    # print(grid.shape)
    # print("go")
    # grid = grid.reshape(ngrid, -1, len(param_names))
    # for theta in grid:
    #     val = lnlike(theta, *args)
    #     print(theta, val)
    res = optimize.minimize(neglnlike, p0, args=args, method='L-BFGS-B', bounds=bounds)
    print(res)
    print("resx:",res.x)
    print(truth)
    return res


def run_mcmc(emus, param_names, ys, covs, fixed_params={}, truth={}, nwalkers=1000,
        nsteps=100, nburn=20, plot_fn=None, multi=True, chain_fn=None):

    global _emus
    _emus = emus

    num_params = len(param_names)

    # todo: not sure how to deal w multiple cov mats?!
    combined_inv_cov = []
    for cov in covs:
        combined_inv_cov.append(np.linalg.inv(cov))
    combined_inv_cov = np.array(combined_inv_cov)

    args = [param_names, fixed_params, ys, combined_inv_cov]
    
    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")
    with mp.Pool() as pool:
        if multi:
            sampler = emcee.EnsembleSampler(nwalkers, num_params, lnprob, args=args, pool=pool)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, num_params, lnprob, args=args)
        # lnprob(p, means, icov)
        p0 = _random_initial_guess(param_names, nwalkers, num_params)
        print("Initial:", p0)

        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()

        sampler.run_mcmc(pos, nsteps)

    for i in range(num_params):
        plt.figure()
        plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
        if param_names[i] in truth:
            true = truth[param_names[i]]
            plt.axvline(true,color='magenta')
        plt.title("Dimension {0:d}".format(i))
        plt.xlabel(param_names[i])
        #if plot_fn:
        #    plt.savefig(plot_fn)
        #    print(f'Saved {param_names[i]} probability plot to {plot_fn}')

    #plt.show()
    #flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    flat_samples = sampler.flatchain
    truths = [truth[pname] for pname in param_names]
    fig = corner.corner(flat_samples, labels=param_names, truths=truths)
    if plot_fn:
        plt.savefig(plot_fn)

    f = h5py.File(chain_fn, 'r+')
    f.attrs['true_values'] = truths

    #for now will overwrite
    if 'chain' in f.keys():
        del f['chain']
    if 'lnprob' in f.keys():
        del f['lnprob']

    f.create_dataset('chain', (0, 0, len(param_names)), chunks = True, compression = 'gzip', maxshape = (None, None, len(param_names)))
    f.create_dataset('lnprob', (0, 0,) , chunks = True, compression = 'gzip', maxshape = (None, None, ))

    chain_dset = f['chain']
    lnprob_dset = f['lnprob']

    chain_dset.resize((nwalkers, nsteps, len(param_names)))
    lnprob_dset.resize((nwalkers, nsteps))
 
    chain_dset[:,:,:] = np.array(sampler.chain)
    lnprob_dset[:,:] = np.array(sampler.lnprobability)


    print("Mean acceptance fraction: {0:.3f}"
          .format(np.mean(sampler.acceptance_fraction)))

    tol = 1
    print("Mean autocorrelation time: {0:.3f} steps"
     .format(np.mean(sampler.get_autocorr_time(c=tol))))

    f.attrs['mean_acceptance_fraction'] = np.mean(sampler.acceptance_fraction)
    f.attrs['mean_autocorr_time'] = np.mean(sampler.get_autocorr_time(c=tol))

    f.close()

    return 0
