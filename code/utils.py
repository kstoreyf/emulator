import numpy as np

cosmos = range(40)
# cosmos = [0]
hods = range(1000, 1010)
rps = []
wps = []
labels = []
for cosmo in cosmos:
    for hod in hods:
        fn = "../wp_covar_results/wp_covar_cosmo_{}_HOD_{}_test_0.dat" \
            .format(cosmo, hod)
        rp, wp = np.loadtxt(fn, unpack=True, usecols=(0, 1))
        labels.append(cosmo)
        rps.append(rp)
        wps.append(wp)

wpmean = np.mean(wps, axis=0)

np.savetxt("../wp_covar_results/wp_covar_mean.dat", [rps[0], wpmean])