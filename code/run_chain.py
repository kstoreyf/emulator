import numpy as np
import scipy
import emcee

import chain
import emulator

cosmo, hod = 0, 0
statistic = 'upf'
res_dir = '../../clust/results_{}/'.format(statistic)
plot_dir = '../plots/plots_2019-10-10'
ytag = '_nonolap_10hod_test0_mean_test0'
#savetag = '_msat_fenv'
savetag = '_sigmalogM_mcut'
traintag = '_nonolap'
nhodpercosmo = 50

errtag = '_10hod_test0'
gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))

tag = ''

training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
gptag = traintag + errtag + tag
hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

plot_fn = f'{plot_dir}/prob{gptag}{savetag}.png'

# actual calculated stat
rad, y = np.loadtxt('../testing_results/tests_{}{}/{}_cosmo_{}_HOD_{}_mean.dat'
                          .format(statistic, ytag, statistic, cosmo, hod))
print('y:', y.shape, y)

# number of parameters, ex 8 hod + 7 cosmo
#param_names = ['Omega_m']
#param_names = ['M_sat', 'f_env']
param_names = ['sigma_logM', 'M_cut']
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


rbins = range(5, 50, 5) # 5 to 45, 9 bins w 5 Mpc/h spacing
nbins = len(rbins)

print("Building emulator")
emu = emulator.Emulator(statistic, training_dir, hyperparams, fixed_params=fixed_params, gperr=gperr)
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
#nwalkers = 250
#nburn = 100
#nsteps = 200
nwalkers = 30
nburn = 50
nsteps = 300
#nsteps = 10
chain.run_mcmc([emu], param_names, [y], [cov], rbins, fixed_params=fixed_params, truth=truth, nwalkers=nwalkers,
        nsteps=nsteps, nburn=nburn, plot_fn=plot_fn)


