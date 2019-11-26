import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

import emcee
import corner


def lnprob(theta, *args):
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args)

# use flat prior on bounds emu is built from. 0 if in (ln 1), -inf if out (ln 0)
# x/theta: params proposed by sampler
def lnprior(theta, param_names, *args):

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
        emu_preds.append(emu.predict(param_dict))
    emu_pred = np.hstack(emu_preds)
    diff = np.array(emu_pred) - np.array(ys)
    # TODO: sean doesn't have the 1/2 factor?
    return -np.dot(diff, np.dot(combined_inv_cov, diff.T).T) / 2.0


def _random_initial_guess(param_names, nwalkers, num_params):
    pos0 = np.zeros((nwalkers, num_params))
    # enumerate adds an index just based on array pos
    for idx, pname in enumerate(param_names):
        low, high = _emus[0].get_param_bounds(pname)
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0
        # TODO variable with of the initial guess

    return pos0


def run_mcmc(emus, param_names, ys, covs, fixed_params={}, truth={}, nwalkers=1000,
        nsteps=100, nburn=20, plot_fn=None, multi=False):

    global _emus
    _emus = emus

    num_params = len(param_names)

    # todo: not sure how to deal w multiple cov mats?!
    combined_inv_cov = []
    for cov in covs:
        combined_inv_cov.append(np.linalg.inv(cov))
    combined_inv_cov = np.array(combined_inv_cov)

    print(combined_inv_cov.shape)
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
    fig = corner.corner(flat_samples, labels=param_names, truths=truths);
    if plot_fn:
        plt.savefig(plot_fn)

    print("Mean acceptance fraction: {0:.3f}"
          .format(np.mean(sampler.acceptance_fraction)))

    tol = 1
    print("Mean autocorrelation time: {0:.3f} steps"
     .format(np.mean(sampler.get_autocorr_time(c=tol))))
