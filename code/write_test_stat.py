import numpy as np

import emulator


statistic = 'upf'
traintag = '_nonolap'
testtag = '_mean_test0'
errtag = '_100hod_test0'
log = True
mean = False
nhod = 100
kernel_name = 'M32ExpConst'
tag = f'_log_k{kernel_name}_{nhod}hod'

hods_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat")
nhodparams_test = hods_test.shape[1]
hods_test[:,0] = np.log10(hods_test[:,0])
hods_test[:,2] = np.log10(hods_test[:,2])
cosmos_test = np.loadtxt("../CMASS/Gaussian_Process/hod_file/cosmology_camb_test_box_full.dat")
ncosmoparams_test = cosmos_test.shape[1]
nparams_test = nhodparams_test + ncosmoparams_test

hod_mean = np.mean(hods_test, axis=0)
print(hod_mean.shape)
cosmo_mean = np.mean(cosmos_test, axis=0)
print(cosmo_mean.shape)
params_mean = np.concatenate((cosmo_mean, hod_mean))
print(params_mean.shape)

# Set file and directory names
gptag = traintag + errtag + tag
res_dir = '../../clust/results_{}/'.format(statistic)
gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
testing_dir = '../testing_results/'
hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

print("Building emulator")
emu = emulator.Emulator(statistic, training_dir, gperr=gperr, hyperparams=hyperparams, log=log, mean=mean, nhod=nhod, kernel_name=kernel_name)
emu.build()
print("Emulator built")

y_pred = emu.predict(params_mean)

if statistic=='wp':
    rmin = 0.1
    rmax = 50
    nbins = 9
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1) # Note the + 1 to nbins
    r = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))
elif statistic=='upf':
    r = np.linspace(5, 45, 9)

pred_fn = f"{testing_dir}/{statistic}_parammean.dat"
results = np.array([r, y_pred])
np.savetxt(pred_fn, results.T, delimiter=',', fmt=['%f', '%e']) 

param_fn = f"{testing_dir}/parammean.dat"
np.savetxt(param_fn, params_mean.T, fmt='%f')
