import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np

import emcee
import george
from george import kernels


def log_prob(theta):
    xval = np.random.random()*10 # to be in same xrange
    s = time.time()
    pred, pred_var = gp.predict(y, xval, return_var=True)
    e = time.time()
    print("GP predict time:", e-s)
    like = pred
    return like

nwalkers = 24
ndim = 2
np.random.seed(42)
initial = np.random.randn(nwalkers, ndim)
nsteps = 20

## Build GP (from george tutorial)

N = 5000
np.random.seed(1234)
x = 10 * np.sort(np.random.rand(N))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

kernel = np.var(y) * kernels.ExpSquaredKernel(0.5)
gp = george.GP(kernel)
gp.compute(x, yerr)

## Serial
from multiprocessing import cpu_count

predict_times = []
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
start = time.time()
sampler.run_mcmc(initial, nsteps)
end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds".format(serial_time))
#print(len(predict_times))
#print("Serial avg predict time: {0:.1f} seconds".format(np.mean(predict_times)))

## Parallel
from multiprocessing import Pool

predict_times = []
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    start = time.time()
    sampler.run_mcmc(initial, nsteps)
    end = time.time()
    multi_time = end - start
    print("Parallel multiprocessing took {0:.1f} seconds".format(multi_time))
    print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    #print(len(predict_times))
    #print("Parallel avg predict time: {0:.1f} seconds".format(np.mean(predict_times)))


