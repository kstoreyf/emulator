import time
import numpy as np
import multiprocessing as mp
from schwimmbad import MultiPool

import emcee


def log_prob(x, mu, cov):
    diff = x - mu
    time.sleep(0.001)
    like = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    return like

    
def main():

    #multi = True
    nwalkers = 12
    print("nwalkers:", nwalkers)
    nburn = 10
    nsteps = 1000
    
    ndim = 1
    np.random.seed(42)
    means = np.random.rand(ndim)

    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)
    
    args = [means, cov]
    p0 = np.random.rand(nwalkers, ndim)

    ncpu = mp.cpu_count()
    print(f"{ncpu} CPUs")

    # serial
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args)
        
    # pos, prob, state = sampler.run_mcmc(p0, nburn)
    # sampler.reset()

    # start = time.time()
    # sampler.run_mcmc(pos, nsteps)
    # end = time.time()
    # print(f"Time serial: {end-start} s")

    # multi:
    with mp.Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args, pool=pool)

        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()

        start = time.time()
        sampler.run_mcmc(pos, nsteps)
        end = time.time()
        print(f"Time multi: {end-start} s")
        

###########
# def lnprob(theta, *args):
#     lp = lnprior(theta, *args)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + lnlike(theta, *args)

# def lnprior(theta, param_names, *args):
#     for pname, t in zip(param_names, theta):
#         low, high = 0, 1
#         if np.isnan(t) or t<low or t>high:
#             return -np.inf
#     return 0

# def lnlike(theta, param_names, fixed_params, ys, cov):
#     print("guess:", theta)
#     theta = np.array(theta).flatten()


########

if __name__=='__main__':
    main()