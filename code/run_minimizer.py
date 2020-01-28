import time
import numpy as np
import scipy
import emcee
import h5py 
from scipy import optimize
import emulator


def main():
    chaintag = 'upf_c4h4_fenv_med_nolog'
    chain_fn = f'../chains/chains_{chaintag}.h5'
    f = h5py.File(chain_fn, 'r')

    ### data params
    # required
    cosmo = f.attrs['cosmo']
    hod = f.attrs['hod']

    ### emu params
    # required
    statistic = f.attrs['statistic']
    traintag = f.attrs['traintag']
    testtag = f.attrs['testtag']
    errtag = f.attrs['errtag']
    tag = f.attrs['tag']
    # optional
    log = f.attrs['log']
    #log = True if log is None else log
    mean = f.attrs['mean']
    #mean = False if mean is None else mean

    ### chain params
    # required
    nwalkers = f.attrs['nwalkers']
    nburn = f.attrs['nburn']
    nsteps = f.attrs['nsteps']
    param_names = f.attrs['param_names']
    # optional
    multi = f.attrs['multi']
    #multi = True if multi is None else multi

    f.close()

    # Set file and directory names
    gptag = traintag + errtag + tag
    res_dir = '../../clust/results_{}/'.format(statistic)
    gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
    training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
    testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, testtag)
    hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)
    #plot_dir = '../plots/plots_2020-01-27'
    #plot_fn = f'{plot_dir}/prob_{statistic}{gptag}{savetag}.png'
    #chain_fn = f'../chains/chains_{statistic}{gptag}{savetag}.png'

    # actual calculated stat
    if 'parammean' in testtag:
        rad, y = np.loadtxt(f'../testing_results/{statistic}_parammean.dat', delimiter=',', unpack=True)
    else:
        rad, y = np.loadtxt(testing_dir+'{}_cosmo_{}_HOD_{}_mean.dat'
                            .format(statistic, cosmo, hod))
    print('y:', y.shape, y)

    # number of parameters, ex 8 hod + 7 cosmo
    num_params = len(param_names)
    #means = np.random.rand(ndim)
    cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
    cosmos_truth = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')

    hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    hods_truth = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')
    hods_truth[:, 0] = np.log10(hods_truth[:, 0])
    hods_truth[:, 2] = np.log10(hods_truth[:, 2])

    fixed_params = {}
    if 'parammean' in testtag:
        names = cosmo_names + hod_names
        params_mean = np.loadtxt("../testing_results/parammean.dat")
        for (name, pm) in zip(names, params_mean):
            fixed_params[name] = pm
    else:
        cosmo_truth = cosmos_truth[cosmo]
        hod_truth = hods_truth[hod]
        for (cn, ct) in zip(cosmo_names, cosmo_truth):
            fixed_params[cn] = ct
        for (hn, ht) in zip(hod_names, hod_truth):
            fixed_params[hn] = ht

    # remove params that we want to vary from fixed param dict and add true values
    truth = {}
    for pn in param_names:
        truth[pn] = fixed_params[pn]
        fixed_params.pop(pn)

    print("Stat:", statistic)
    print("True values:")
    print(truth)

    print("Building emulator")
    emu = emulator.Emulator(statistic, training_dir,  fixed_params=fixed_params, gperr=gperr, hyperparams=hyperparams, log=log, mean=mean)
    emu.build()
    print("Emulator built")

    #diagonal covariance matrix from error
    cov = np.diag(emu.gperr)
    #cov *= 10
    print(cov)
    print(cov.shape)
    start = time.time()


def minimize(emu, param_names, y, cov):
    optimize.minimize(lnprob, )




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



if __name__=='__main__':
    main()