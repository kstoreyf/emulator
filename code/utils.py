import numpy as np


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
