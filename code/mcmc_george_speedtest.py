import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np

import emcee
import george
from george import kernels
import dynesty

from multiprocessing import Pool


def log_prob(theta):
    xval = np.random.random()*10 # to be in same xrange
    s = time.time()
    pred, pred_var = gp.predict(y, xval, return_var=True)
    e = time.time()
    #print("GP predict time:", e-s, pred)
    like = pred[0]
    return like

def ptform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""
    return 10*u
    #return 10. * (2. * u - 1.)

#engine = 'dynesty' #can be dynesty, emcee
engine = 'emcee'

nwalkers = 24
ndim = 2
np.random.seed(42)
initial = np.random.randn(nwalkers, ndim)
nsteps = 100

## Build GP (from george tutorial) 
# https://george.readthedocs.io/en/latest/tutorials/first/
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

if engine=='dynesty':
    sampler = dynesty.NestedSampler(log_prob, ptform, ndim)
    start = time.time()
    sampler.run_nested(maxiter=nsteps)
    end = time.time()
    serial_time = end - start
    print("DYNESTY: Serial took {0:.1f} seconds".format(serial_time))
    sresults = sampler.results
    print(sresults.summary())

    with Pool() as pool:
        sampler = dynesty.NestedSampler(log_prob, ptform, ndim, pool=pool, queue_size=ncpu)
        start = time.time()
        sampler.run_nested()
        end = time.time()
        multi_time = end - start
        print("DYNESTY: Parallel multiprocessing took {0:.1f} seconds".format(multi_time))
        print("DYNESTY: {0:.1f} times faster than serial".format(serial_time / multi_time))
        sresults = sampler.results
        print(sresults.summary())
        
elif engine=='emcee':
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    start = time.time()
    sampler.run_mcmc(initial, nsteps)
    end = time.time()
    serial_time = end - start
    print("EMCEE: Serial took {0:.1f} seconds".format(serial_time))
    #print(len(predict_times))
    #print("Serial avg predict time: {0:.1f} seconds".format(np.mean(predict_times)))

    ## Parallel
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


