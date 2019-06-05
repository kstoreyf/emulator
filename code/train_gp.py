import numpy as np
from george.kernels import *
import sys

sys.path.insert(0, '../../../CMASS/Gaussian_Process/GP/')

import gp_trainer as trainer
import george


statistic = 'upf'
#traintag = '_sample50v4'
traintag = '_nonolap'
nhodpercosmo = 50
#traintag = '_hod0'
#traintag = '_cos0'
tag = ''
errtag = '_10hod_test0'


#subsample = 50
#version = 4

nbins = 9
testid = 0

res_dir = '../../clust/results_{}/'.format(statistic)
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)

gptag = traintag + errtag + tag

print "Training, savetag={}".format(gptag)

fixed_cosmo = False
fixed_hod = False
if "hod" in traintag:
    fixed_hod = True
elif "cos" in traintag:
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
    #CC = range(0, 1)
    nhodnonolap = 100
    HH = np.array(range(0,len(CC)*nhodnonolap))
    HH  = HH.reshape(len(CC), nhodnonolap)
    HH = HH[:,0:nhodpercosmo]
    #HH = np.loadtxt("../CMASS/Gaussian_Process/GP/HOD_random_subsample_{}_version_{}.dat".format(subsample, version))
    nparams = nhodparams + ncosmoparams


rr = np.empty((HH.shape[1] * len(CC), nparams))
YY = np.empty((nbins, HH.shape[1] * len(CC)))


##################   find the mean of the data  #############

s2 = 0
for CID in CC:
    if fixed_hod or fixed_cosmo:
        HH_set = HH[0]
    else:
        HH_set = HH[CID]
    for HID in HH_set:
        HID = int(HID)
        rad, vals = np.loadtxt(training_dir+"{}_cosmo_{}_HOD_{}_test_{}.dat".format(statistic, CID, HID, testid),
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

pp = []
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
            rad, vals = np.loadtxt(training_dir+"{}_cosmo_{}_HOD_{}_test_{}.dat".format(statistic, CID, HID, testid),
                                 delimiter=',', unpack=True)
            rad = rad[:nbins]
            vals = vals[:nbins]

            # the cosmology and HOD values used for this data
            if fixed_hod:
                rr[ss, 0:ncosmoparams] = cosmos[CID,:]
            elif fixed_cosmo:
                rr[ss, 0:nhodparams] = hods[HID,:]
            else:
                rr[ss, 0:ncosmoparams] = cosmos[CID,:]
                rr[ss, ncosmoparams:ncosmoparams+nhodparams] = hods[HID,:]

            val = vals[j]
            if log:
                y[ss] = np.log10(val/Ym)
                # y[ss] = np.nan_to_num(np.log10(val / Ym))
            else:
                y[ss] = val / Ym
                #y[ss] = val
            yerr[ss] = GP_error[j]
            #yerr[ss] = 0.0
            #yerr[ss] = Ystd[j]
            #yerr[ss] = np.nan_to_num(np.log10(Ystd[j]))
            #yerr[ss] = GP_error[j]/2.303
            #yerr[ss] = np.log10(GP_error[j])
            #yerr[ss] = np.nan_to_num(np.log10(GP_error[j]))

            ss += 1
            #########

    # 15 initial values for the 8 hod and 7 cosmo params
    p0 = np.full(nparams, 0.1)

    k1 = ExpSquaredKernel(p0, ndim=len(p0))
    k2 = Matern32Kernel(p0, ndim=len(p0))
    k3 = ConstantKernel(0.1, ndim=len(p0))
    #k4 = WhiteKernel(0.1, ndim=len(p0))
    k5 = ConstantKernel(0.1, ndim=len(p0))

    kernel = k2 + k5
    #kernel = np.var(y) * k1

    gp = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
    #gp = george.GP(kernel, solver=george.BasicSolver)

    print "computing bin:", j
    gp.compute(rr, yerr)
    print "training"
    ppt = trainer.gp_tr(rr, y, yerr, gp, optimize=True).p_op
    #ppt = trainer.gp_tr(rr, y, yerr, gp, MCMC=True).p_op

    # gp.kernel.vector = ppt
    #
    # print "second compute"
    # # is this even doing anything?
    # gp.compute(rr, yerr)
    # print "ppt:", ppt
    pp.append(ppt)


np.savetxt("../training_results/{}_training_results{}.dat".format(statistic, gptag), pp, fmt='%.7f')
