import numpy as np
import george
from george.kernels import *
import scipy.optimize as op
from scipy.linalg import cholesky, cho_solve
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, '/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/GP/')
import gp_training
from gp_training import *

print("Starting training")
tag = '_kM32ExpConst2'
#tag = '_kM32Exp'
#tag = ''
data = 'wp_covar'

x = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/galaxy_mocks/HOD_design_np11_n5000_new_f_env.dat")

x[:,0] = np.log10(x[:,0])
x[:,2] = np.log10(x[:,2])

xc = np.loadtxt("/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/hod_file/cosmology_camb_full.dat")


NewKernel = False
HODLR = False

Nsize1 = 0
Nsize2 = 50
    
N_hod_up = 0 # the parameter to change HODs: e.g. 0~4000 -> 1000~5000 when this number is 1000

HH = np.array(range(0,4000))
HH  = HH.reshape(40, 100)
HH = HH + N_hod_up
HH = HH[:,Nsize1:Nsize2]
print(HH.shape)
CC = range(40)
rr = np.empty((HH.shape[1]*len(CC), x.shape[1]+xc.shape[1]))
YY = np.empty((9, HH.shape[1]*len(CC)))

##################   find the mean of the data  #############
print("Finding mean")
s2 = 0
for CID in CC:
    for HID in HH[CID]:
        HID = int(HID)

        d = np.loadtxt("/home/users/ksf293/clust/results_wp/training_wp_nonolap/wp_cosmo_"+str(CID)+"_HOD_"+str(HID)+"_test_0.dat", delimiter=',')
        YY[:,s2] = d[:,1]
        s2 = s2+1

Ymean = np.mean(YY, axis=1)

##################  found the mean of the data ################

GP_error = np.loadtxt("/home/users/ksf293/clust/results_wp/wp_error_100hod_test0.dat")

GP_err = GP_error

pp = []
for j in range(9):
    print("Bin",j)
    DC = j
    Ym = Ymean[DC]
    ss = 0
    yerr = np.zeros((len(rr)))
    y = np.empty((len(rr)))
    for CID in CC:
        for HID in HH[CID]:
            HID = int(HID)
            
            d = np.loadtxt("/home/users/ksf293/clust/results_wp/training_wp_nonolap/wp_cosmo_"+str(CID)+"_HOD_"+str(HID)+"_test_0.dat", delimiter=',')

            rr[ss,0:7]=xc[CID, :]
            rr[ss,7:18]=x[HID, :]
                
            d = d[:,1]
            d1 = d[DC]
            y[ss] = np.log10(d1/Ym)

            yerr[ss] = GP_err[j]/2.303
            ss = ss+1
#########

    p0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    k1 = ExpSquaredKernel(p0, ndim=len(p0))
    k2 = Matern32Kernel(p0, ndim=len(p0))
    k3 = ConstantKernel(0.1, ndim=len(p0))
    k4 = WhiteKernel(0.1, ndim=len(p0))
    k5 = ConstantKernel(0.1, ndim=len(p0))
    #if NewKernel == False:
    #kernel = k1*k5+k2+k3+k4 # tag = '' (was default)
    #elif k12:
    kernel = k1+k2 # tag = '_kM32Exp' 
    #kernel = k1*k5+k2 #tag='_kM32ExpConst'
    kernel = k1*k5+k2+k3 #tag='_kM32ExpConst2'
    #else:
    #    kernel = k2+k5


    if HODLR == True:
        gp = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
    else:
        gp = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
    gp.compute(rr, yerr)

    ppt = gp_tr(rr, y, yerr, gp, optimize=True).p_op
    gp.kernel.vector = ppt
    gp.compute(rr, yerr)
    print(ppt)
    pp.append(ppt)

print("Saving")
np.savetxt("wp_9bins_pp_Nsize_"+str(Nsize1)+"_"+str(Nsize2)+"+N_hod_up"+str(N_hod_up)+tag+".dat", pp, fmt='%.7f')

