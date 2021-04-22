import numpy as np
import h5py
import pickle
from matplotlib import pyplot as plt

import dynesty 
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import emulator
from chain_variables import *


def covariance(arrs, zeromean=False):
    arrs = np.array(arrs)
    N = arrs.shape[0]

    if zeromean:
        w = arrs
    else:
        w = arrs - arrs.mean(0)

    outers = np.array([np.outer(w[n], w[n]) for n in range(N)])
    covsum = np.sum(outers, axis=0)
    cov = 1.0/float(N-1.0) * covsum
    return cov


# aka Correlation Matrix
def reduced_covariance(cov):
    cov = np.array(cov)
    Nb = cov.shape[0]
    reduced = np.zeros_like(cov)
    for i in range(Nb):
        ci = cov[i][i]
        for j in range(Nb):
            cj = cov[j][j]
            reduced[i][j] = cov[i][j]/np.sqrt(ci*cj)
    return reduced


# The prefactor unbiases the inverse; see e.g. Pearson 2016
def inverse_covariance(cov, N):
    inv = np.linalg.inv(cov)
    Nb = cov.shape[0]
    prefac = float(N - Nb - 2)/float(N - 1)
    return prefac * inv


def get_emulator_bounds():
    bounds = {}
    cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

    cosmos_train = np.loadtxt('../tables/cosmology_camb_full.dat') # 40
    hods_train = np.loadtxt('../tables/HOD_design_np11_n5000_new_f_env.dat') # 5000
    hods_train[:, 0] = np.log10(hods_train[:, 0])
    hods_train[:, 2] = np.log10(hods_train[:, 2])

    for pname in cosmo_names:
        pidx = cosmo_names.index(pname)
        vals = cosmos_train[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        buf = (pmax-pmin)*0.1
        bounds[pname] = [pmin-buf, pmax+buf]

    for pname in hod_names:
        pidx = hod_names.index(pname)
        vals = hods_train[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        buf = (pmax-pmin)*0.1
        bounds[pname] = [pmin-buf, pmax+buf]

    return bounds


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def load_chains(chaintag, show_walkers=False, show_corner=True, show_params=None, figure=None, nsteps=None, color='blue'): 
    #chaintag = 'wp_c3h3_Msat_fenv_xlong_diag'
    chain_fn = f'../chains/chains_{chaintag}.h5'
    fw = h5py.File(chain_fn, 'r')

    chain_dset = fw['chain']
    print(chain_dset)
    chain = np.array(chain_dset)
    lnprob_dset = fw['lnprob']
    param_names = fw.attrs['param_names']
    true_values = fw.attrs['true_values']
    if nsteps:
        chain = chain[:,:nsteps,:]
    nwalkers, nchain, ndim = chain.shape
    fw.close()
    
    if show_params:    
        idxs = []
        for sp in show_params:
            idxs.append(np.where(param_names == sp))
        idxs = np.array(idxs).flatten()
        chain = chain[:,:,idxs]
        param_names = show_params
        true_values = true_values[idxs]

    return chain


def build_emus(f):
    
    #chain_fn = f'../chains/chains_{chaintag}.h5'
    #f = h5py.File(chain_fn, 'r')
    
    ### data params
    cosmo = f.attrs['cosmo']
    hod = f.attrs['hod']
    
    ### emu params
    statistics = f.attrs['statistic']
    traintags = f.attrs['traintag']
    testtags = f.attrs['testtag']
    errtags = f.attrs['errtag']
    tags = f.attrs['tag']
    kernel_names = f.attrs['kernel_name']
    logs = f.attrs['log']
    means = f.attrs['mean']
    nhods = f.attrs['nhod']

    ### chain params
    param_names = f.attrs['param_names']
    
    # Set file and directory names
    nstats = len(statistics)
    training_dirs = [None]*nstats
    testing_dirs = [None]*nstats
    hyperparams = [None]*nstats
    acctags = [None]*nstats
    gperrs = [None]*nstats
    ys = []
    cov_dir = '../../clust/covariances/'
    for i, statistic in enumerate(statistics):
        gptag = traintags[i] + errtags[i] + tags[i]
        acctags[i] = gptag + testtags[i]
        res_dir = '../../clust/results_{}/'.format(statistic)
        gperrs[i] = np.loadtxt(cov_dir+"error_aemulus_{}{}.dat".format(statistic, errtags[i]))
        training_dirs[i] = '{}training_{}{}/'.format(res_dir, statistic, traintags[i])
        testing_dirs[i] = '{}testing_{}{}/'.format(res_dir, statistic, testtags[i])
        hyperparams[i] = "../training_results/{}_training_results{}.dat".format(statistic, gptag)
    
    # number of parameters, out of 11 hod + 7 cosmo
    num_params = len(param_names)
    cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    cosmos_truth = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')

    hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    hods_truth = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')
    hods_truth[:, 0] = np.log10(hods_truth[:, 0])
    hods_truth[:, 2] = np.log10(hods_truth[:, 2])

    fixed_params = {}
    cosmo_truth = cosmos_truth[cosmo]
    hod_truth = hods_truth[hod]
    for (cn, ct) in zip(cosmo_names, cosmo_truth):
        fixed_params[cn] = ct
    for (hn, ht) in zip(hod_names, hod_truth):
        fixed_params[hn] = ht

    # remove params that we want to vary from fixed param dict and add true values
    truths = f.attrs['true_values']
    for pn in param_names:
        fixed_params.pop(pn)
        
    print("Building emulators")
    emus = [None]*nstats
    for i, statistic in enumerate(statistics):
        print(f"Rebuilding emulator for {statistic}")
        emu = emulator.Emulator(statistic, training_dirs[i], testing_dir=testing_dirs[i], fixed_params=fixed_params, 
                                gperr=gperrs[i], hyperparams=hyperparams[i], log=logs[i], 
                                mean=means[i], nhod=nhods[i], kernel_name=kernel_names[i])
        emu.build()
        emus[i] = emu
        
        
    #return emus, statistics, param_names, fixed_params, truths, cosmo, hod
    return emus, fixed_params, gperrs

def get_maxweight_params(res):
    samples = res['samples']
    weights = res['logwt']
    argmax_weight = np.argmax(weights)
    maxweight_params = samples[argmax_weight]
    return maxweight_params

def get_minlogl_params(res, index=-1):
    samples = res['samples']
    minlogl_params = samples[index]
    return minlogl_params

def get_mean_params(res):
    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    return mean

def get_emu_predictions(emu, params_topredict, param_names, fixed_params):
    param_dict_pred = dict(zip(param_names, params_topredict))
    param_dict_pred.update(fixed_params)
    emu_preds = emu.predict(param_dict_pred)
    return emu_preds

def load_res(chaintag):
    pickle_fn = f'{pickle_dir}/results_{chaintag}.pkl'
    with open(pickle_fn, 'rb') as pf:
        res = pickle.load(pf)
    return res

def get_true_params(chaintag):
    chain_fn = f'../chains/chains_{chaintag}.h5'
    f = h5py.File(chain_fn, 'r')
    truths = f.attrs['true_values']
    return truths

def get_fits(chaintag, param_arr):
    
    chain_fn = f'../chains/chains_{chaintag}.h5'
    f = h5py.File(chain_fn, 'r')
    
    param_names = f.attrs['param_names']
    statistics = f.attrs['statistic']
    cosmo = f.attrs['cosmo']
    hod = f.attrs['hod']
    
    emus, fixed_params, gperrs = build_emus(f)
    
    vals_arr_all = []
    vals_true_all = []

    for i, statistic in enumerate(statistics):
        vals_arr = []
        for params in param_arr:
            vals_arr.append(get_emu_predictions(emus[i], params, param_names, fixed_params))
        
        vals_arr_all.append(vals_arr)
        vals_true_all.append(emus[i].testing_data[(cosmo, hod)])

    return statistics, np.array(vals_arr_all), np.array(vals_true_all), gperrs

def get_cov(statistics, cov_tag, tag_str='', cov_dir='/home/users/ksf293/clust/covariances'):
    stat_str = '_'.join(statistics)
    cov_fn = f"{cov_dir}/cov_{cov_tag}_{stat_str}{tag_str}.dat"
    cov = np.loadtxt(cov_fn)
    return cov

