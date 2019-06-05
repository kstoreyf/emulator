# This code is used for prediction. The training process is performed in another code.
# So the output from the training process: the optimized hyper parameters of the Gaussian
# process, is now the input. Note the kernels in this prediction code and in the graining
# process should be exactly the same, as well as the input dataset.


import numpy as np
import george
from george import kernels
import os

from time import time

t1 = time()


statistic = 'upf'
#statistic = 'upf'

#traintag = '_cos0'
traintag = '_nonolap'
nhodpercosmo = 50
#traintag = '_sample50v4'
#testtag = '_hod1_mean'
#testtag = '_cos0_mean'
testtag = '_mean_test0'
#hod = 0
hod = None
#errtag = ''
errtag = '_10hod_test0'
tag = ''

testmean = False
if 'mean' in testtag:
    testmean = True

#subsample = 50
#version = 4

nbins = 9

testid = 0
boxid = 0

res_dir = '../../clust/results_{}/'.format(statistic)
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
#if testmean:
#    testing_dir = '{}testing_{}{}_mean/'.format(res_dir, statistic, testtag)
#else:
testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, testtag)

gptag = traintag + errtag + tag
acctag = gptag + testtag

test_savedir = "../testing_results/tests_{}{}/".format(statistic, acctag)
predict_savedir = "../testing_results/predictions_{}{}/".format(statistic, acctag)
if not os.path.exists(test_savedir):
    os.makedirs(test_savedir)
if not os.path.exists(predict_savedir):
    os.makedirs(predict_savedir)

fixed_cosmo = False
fixed_hod = False
if type(hod)==int:
    fixed_hod = True
if "cos" in traintag:
    fixed_cosmo = True

log = False
if 'log' in tag:
    log = True

# hod parameters (5000 rows, 8 cols)
#OLD hods = np.loadtxt("../CMASS_BIAS/COS_2000HOD/HOD_design_np8_n5000.dat")
#NEW
hods = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/galaxy_mocks/HOD_design_np11_n5000_new_f_env.dat")

hods[:,0] = np.log10(hods[:,0])
hods[:,2] = np.log10(hods[:,2])
nhodparams = hods.shape[1]

# cosmology params (40 rows, 7 cols)
cosmos = np.loadtxt("../CMASS/Gaussian_Process/hod_file/cosmology_camb_full.dat")
ncosmoparams = cosmos.shape[1]


if fixed_hod:
    CC = range(0, 40)
    HH = range(0, 1)
    HH = np.atleast_2d(HH)
    nparams = ncosmoparams
elif fixed_cosmo:
    CC = range(0, 1)
    HH = range(0, 200)
    HH = np.atleast_2d(HH)
    nparams = nhodparams
else:
    CC = range(0, 40)
    nhodnonolap = 100
    HH = np.array(range(0,len(CC)*nhodnonolap))
    HH = HH.reshape(len(CC), nhodnonolap)
    HH = HH[:,0:nhodpercosmo]
    #HH = np.loadtxt("../CMASS/Gaussian_Process/GP/HOD_random_subsample_{}_version_{}.dat".format(subsample, version))
    nparams = nhodparams + ncosmoparams

rr = np.empty((HH.shape[1] * len(CC), nparams))
YY = np.empty((nbins, HH.shape[1] * len(CC)))

pp = np.loadtxt("../training_results/{}_training_results{}.dat".format(statistic, gptag))


##################   find the mean of the data  #############

s2 = 0
for CID in CC:
    if fixed_hod or fixed_cosmo:
        HH_set = HH[0]
    else:
        HH_set = HH[CID]
    for HID in HH_set:
        HID = int(HID)
        rad, vals = np.loadtxt(training_dir + "{}_cosmo_{}_HOD_{}_test_0.dat".format(statistic, CID, HID),
                             delimiter=',', unpack=True)
        rad = rad[:nbins]
        vals = vals[:nbins]

        YY[:, s2] = vals
        s2 += 1

# mean of values in each bin (Ymean has length nbins)
Ymean = np.mean(YY, axis=1)
Ystd = np.std(YY, axis=1)

##################  found the mean of the data ################

GP_error = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
print "GP error:", GP_error

gps = []
y2 = np.empty((len(rr) * nbins))
ss2 = 0
for j in range(nbins):
    Ym = Ymean[j]
    ss = 0
    yerr = np.zeros((len(rr)))
    y = np.empty((len(rr)))
    for CID in CC:
        if fixed_hod or fixed_cosmo:
            HH_set = HH[0]
        else:
            HH_set = HH[CID]
        for HID in HH_set:
            HID = int(HID)

            # seems silly to load this every bin loop but makes some sense, otherwise would have to store
            rad, vals = np.loadtxt(training_dir+"{}_cosmo_{}_HOD_{}_test_0.dat".format(statistic, CID, HID),
                                 delimiter=',', unpack=True)
            rad = rad[:nbins]
            vals = vals[:nbins]
            # the cosmology and HOD values used for this data

            if fixed_hod:
                rr[ss, 0:ncosmoparams] = cosmos[CID,:]
            elif fixed_cosmo:
                rr[ss, 0:nhodparams] = hods[HID, :]
            else:
                rr[ss, 0:ncosmoparams] = cosmos[CID,:]
                rr[ss, ncosmoparams:ncosmoparams+nhodparams] = hods[HID,:]

            val = vals[j]
            if log:
                y[ss] = np.log10(val / Ym)
            else:
                y[ss] = val / Ym
                #y[ss] = val
            yerr[ss] = GP_error[j]
            #yerr[ss] = Ystd[j]
            #yerr[ss] = GP_error[j]/2.303
            #yerr[ss] = np.log10(GP_error[j])
            y2[ss2] = y[ss]

            ss += 1
            ss2 += 1

            ######

    # 15 initial values for the 7 hod and 8 cosmo params
    p0 = np.full(nparams, 0.1)

    k1 = kernels.ExpSquaredKernel(p0, ndim=len(p0))
    k2 = kernels.Matern32Kernel(p0, ndim=len(p0))
    k3 = kernels.ConstantKernel(0.1, ndim=len(p0))
    #k4 = kernels.WhiteKernel(0.1, ndim=len(p0))
    k5 = kernels.ConstantKernel(0.1, ndim=len(p0))

    kernel = k2 + k5
    #kernel = np.var(y)*k1

    ppt = pp[j]

    gp = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
    #gp = george.GP(kernel, solver=george.BasicSolver)

    gp.compute(rr, yerr)
    #gp.kernel.vector = ppt
    gp.set_parameter_vector(ppt)
    gp.compute(rr, yerr)

    gps.append(gp)



if fixed_hod:
    #HH_test = range(hod, hod+1)
    CC_test = range(0, 7)
    # TODO: add more tests, for now just did first 10 hod
    HH_test = range(0, 10)
elif fixed_cosmo:
    CC_test = range(0, 1)
    HH_test = range(0, 7)
else:
    CC_test = range(0, 7)
    # TODO: add more tests, for now just did first 10 hod
    HH_test = range(0, 10)
    #HH_test = [0, 6, 10, 11, 14, 16, 19, 20, 23, 24]
  
ss = 0
#OLD (still using the good ones for now)
#hods_test = np.loadtxt("../CMASS_BIAS/GP_Test_BOX/HOD_test_np8_n1000.dat")
#FIRST NEW
#hods_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks/HOD_test_np11_n1000.dat")
# NEW NEW
hods_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat")

hods_test[:,0] = np.log10(hods_test[:,0])
hods_test[:,2] = np.log10(hods_test[:,2])


#cosmos_test = np.loadtxt("../CMASS/GP_Test_Box/cosmology_camb_test_box.dat")
cosmos_test = np.loadtxt("../CMASS/Gaussian_Process/hod_file/cosmology_camb_test_box_full.dat")


frac_rms = np.empty((len(HH_test) * len(CC_test), nbins))

for CID_test in CC_test:
    for HID_test in HH_test:
        print('CID, HID:', CID_test, HID_test)
        hods_test_hid = hods_test[HID_test,:]

        if testmean:
            idtag = "cosmo_{}_HOD_{}_mean".format(CID_test, HID_test)
            rad, vals_test = np.loadtxt(testing_dir + "{}_{}.dat".format(statistic, idtag))
        else:
            idtag = "cosmo_{}_Box_{}_HOD_{}_test_{}".format(CID_test, boxid, HID_test, testid)
            rad, vals_test = np.loadtxt(testing_dir + "{}_{}.dat".format(statistic, idtag),
                                          delimiter=',', unpack=True)

        rad = rad[:nbins]
        vals_test = vals_test[:nbins]

        if fixed_hod:
            params_test = np.atleast_2d(cosmos_test[CID_test])
        elif fixed_cosmo:
            params_test = np.atleast_2d(hods_test_hid)
        else:
            params_test = np.atleast_2d(np.hstack((cosmos_test[CID_test], hods_test_hid)))

        loc = len(CC) * HH.shape[1]
        mus = np.zeros(nbins)
        for bb in range(nbins):
            # predict on all the statistic values in the bin
            mu, cov = gps[bb].predict(y2[loc*bb : loc*(bb+1)], params_test)
            mus[bb] = mu


        if log:
            vals_predict = np.array(10**(mus*Ymean))
        else:
            vals_predict = mus*Ymean
            #vals_predict = mus

        frac_err = (vals_predict - vals_test) / vals_test
        frac_rms[ss, :] = frac_err
        print 'Results:'
        print vals_predict
        print vals_test
        print frac_err

        ss += 1

        np.savetxt(test_savedir+"{}_{}.dat".format(statistic, idtag), [rad, vals_test])
        np.savetxt(predict_savedir+"{}_{}.dat".format(statistic, idtag), [rad, vals_predict])


np.savetxt("../testing_results/{}_testing_results{}.dat".format(statistic, acctag), frac_rms)

