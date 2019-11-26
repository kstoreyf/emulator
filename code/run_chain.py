import time
import numpy as np
import scipy
import emcee

import chain
import emulator

# Parameters
cosmo, hod = 2, 2
statistic = 'upf'
traintag = '_nonolap'
testtag = '_mean_test0'
errtag = '_10hod_test0'
#tag = '_log'
tag = ''
log = True
multi = False
#multi = True
param_names = ['f_env']
#savetag = '_sigma8_nchain2000'
savetag = '_f_env_nchain2000'
plot_dir = '../plots/plots_2019-11-26'

#nwalkers = 10
#nburn = 50
#nsteps = 100
nwalkers = 250
nburn = 100
nsteps = 2000

#param_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
#param_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
#param_names = ['c_vir', 'f']

# Set file and directory names
gptag = traintag + errtag + tag
res_dir = '../../clust/results_{}/'.format(statistic)
ytag = traintag + errtag + testtag
gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, testtag)
hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)
plot_fn = f'{plot_dir}/prob_{statistic}{gptag}{savetag}.png'

# actual calculated stat
rad, y = np.loadtxt('{}/{}_cosmo_{}_HOD_{}_mean.dat'
                          .format(testing_dir, statistic, cosmo, hod))
print('y:', y.shape, y)

# number of parameters, ex 8 hod + 7 cosmo
num_params = len(param_names)
#means = np.random.rand(ndim)
cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
cosmos_truth = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')
cosmo_truth = cosmos_truth[cosmo]

hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
hods_truth = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')
hod_truth = hods_truth[hod]

fixed_params = {}
for (cn, ct) in zip(cosmo_names, cosmo_truth):
    fixed_params[cn] = ct
for (hn, ht) in zip(hod_names, hod_truth):
    fixed_params[hn] = ht

# remove params that we want to vary from fixed param dict and add true values
truth = {}
for pn in param_names:
    truth[pn] = fixed_params[pn]
    fixed_params.pop(pn)

print("Building emulator")
emu = emulator.Emulator(statistic, training_dir,  fixed_params=fixed_params, gperr=gperr, hyperparams=hyperparams, log=log)
emu.build()
print("Emulator built")

#diagonal covariance matrix from error
cov = np.diag(emu.gperr)

# sean loads in his cov mat from file, for a given stat (ex xi_gg)
# sum of measuremnt cov mat and mat from emulator (indep of emu params, can be precomputed)
#cov = 0.5 - np.random.rand(nbins ** 2).reshape((nbins, nbins))
#cov = np.triu(cov)
#cov += cov.T - np.diag(cov.diagonal())
#cov = np.dot(cov,cov)
#print(cov.shape)
# do this in chain.y
#cov = scipy.linalg.inv(cov)
print(cov)
print(cov.shape)
start = time.time()
chain.run_mcmc([emu], param_names, [y], [cov], fixed_params=fixed_params, truth=truth, nwalkers=nwalkers,
        nsteps=nsteps, nburn=nburn, plot_fn=plot_fn, multi=multi)
end = time.time()
print(f"Time: {(end-start)/60.0} min")


