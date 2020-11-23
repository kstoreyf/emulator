import numpy as np
import h5py
from matplotlib import pyplot as plt

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


def plot_autocorr(chain, color='steelblue'):
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
    ndim = chain.shape[2]
    auto = np.empty((len(N), ndim))
    for i, n in enumerate(N):
        for d in range(ndim):
            auto[i][d] = autocorr_new(chain[:, :n, d])
    
    # Plot the comparisons
    for d in range(ndim):
        plt.loglog(N, auto, "o-", color=color)
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    #plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)


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
