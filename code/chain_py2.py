import numpy as np
#TODO: figure out what this is
from itertools import izip
import emcee
import matplotlib.pyplot as plt


def lnprob(theta, *args):
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args)

# use flat prior on bounds emu is built from. 0 if in (ln 1), -inf if out (ln 0)
# Omega_m
# x/theta: params proposed by sampler
def lnprior(theta, param_names, *args):

    for pname, t in izip(param_names, theta):
        # all emus should have same bounds, so just get first
        low, high = _emus[0].get_param_bounds(pname)
        if np.isnan(t) or t<low or t>high:
            return -np.inf
    return 0

# looks like multivariate gaussian, where x is emu prediction for given parameter values
# and mean y is "measured value of observables" - or, "data to constrain against"
# sean loads in mean from file, for each emu observable. looks like for each box number and realization?
# ok so the y is the calculated stat value from the testbox. size nbins. if multiple (m) emus, m x nbins
# mm and we're trying to minimize this difference?
def lnlike(theta, param_names, fixed_params, r_bin_centers, ys, combined_inv_cov):
    # don't want this!!! we are varying the theta (param vals), that's the point!!
    # so we need the actual emulator
    #emu_pred = np.loadtxt('../emulator/testing_results/predictions_{}_{}/{}_cosmo_{}_HOD_{}_mean.dat'
    #                      .format(stat, tag, stat, cosmo, hod))
    param_dict = dict(izip(param_names, theta))
    param_dict.update(fixed_params)

    print("ys:", np.array(ys).shape, ys)

    emu_preds = []
    for emu in _emus:
        print(emu.predict(param_dict), emu.predict(param_dict).shape)
        emu_preds.append(emu.predict(param_dict))
    emu_pred = np.hstack(emu_preds)
    print('emus:', emu_pred.shape, emu_pred)

    diff = emu_pred - ys
    # TODO: sean doesn't have the 1/2 factor?
    print(diff.shape)
    print(diff)
    print(combined_inv_cov.shape)
    combined_inv_cov = [combined_inv_cov]

    print(np.dot(combined_inv_cov, diff))
    return -np.dot(diff, np.dot(combined_inv_cov, diff)) / 2.0


def _random_initial_guess(param_names, nwalkers, num_params):
    pos0 = np.zeros((nwalkers, num_params))
    # enumerate adds an index just based on array pos
    for idx, pname in enumerate(param_names):
        low, high = _emus[0].get_param_bounds(pname)
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0
        # TODO variable with of the initial guess

    return pos0


def run_mcmc(emus, param_names, ys, covs, rpoints, fixed_params={}, nwalkers=1000,
        nsteps=100, nburn=20):

    global _emus
    _emus = emus

    num_params = len(param_names)

    # todo: not sure how to deal w multiple cov mats?!
    if len(covs)==1:
        covs = covs[0]
        combined_inv_cov = np.linalg.inv(covs)

    print(combined_inv_cov.shape)
    args = [param_names, fixed_params, rpoints, ys, combined_inv_cov]
    sampler = emcee.EnsembleSampler(nwalkers, num_params, lnprob, args=args)

    # lnprob(p, means, icov)
    p0 = _random_initial_guess(param_names, nwalkers, num_params)

    pos, prob, state = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    sampler.run_mcmc(pos, nsteps)


    for i in range(num_params):
        plt.figure()
        plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
        plt.title("Dimension {0:d}".format(i))

    plt.show()

    print("Mean acceptance fraction: {0:.3f}"
          .format(np.mean(sampler.acceptance_fraction)))


# ok so walkers are exploring the parameter space aka hod params and cosmo params, and passing
# them to probability functions and evaluating probability at those values

# we do a recovery test on a randomly chosen cosmology.

#zhongxu paper: x is emu prediction of stat, mu is calculated stat from testbox (or survey)
# but emu predic can't be based off of the actual values of params bc we shouldn't know them -
# so i guess that's what the mcmc is doing?

#covariance: corr bw r-bins as well as stats themselves (ex wp, monopole, quadrupole...)
#minerva: 100 nbody sims. all at same cosmology i think? and with a particular hod?
# calc corrfunc, estimate correlation matrix by normalizing cov mat of gal
# corrfuncs (??). error of corr func is sum in quadrature of input training error and emulator uncertainty
# this error estimate is diag elements, combined w correlation matrix to pop cov mat
# cov mat is how parameters vary?? size of params/dims? yes ok
# but wait still confused bc usually cov mat has dims of number of bins
