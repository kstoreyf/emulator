import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np

import emcee


def log_prob(theta):
    # t = time.time() + np.random.uniform(0.005, 0.008)
    # while True:
    #     if time.time() >= t:
    #         break
    # like = -0.5 * np.sum(theta ** 2)

    #like = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    #
    time.sleep(0.0001)
    like = np.random.random()
    return like


nwalkers = 32
ndim = 5
np.random.seed(42)
initial = np.random.randn(nwalkers, ndim)
nsteps = 100

from multiprocessing import cpu_count

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
start = time.time()
sampler.run_mcmc(initial, nsteps)
end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds".format(serial_time))

from multiprocessing import Pool

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    start = time.time()
    sampler.run_mcmc(initial, nsteps)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    print("{0:.1f} times faster than serial".format(serial_time / multi_time))