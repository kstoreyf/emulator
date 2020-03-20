#   This code is used for prediction. The training process is performed in another code. So the output from the training process: the optimized hyper parameters of the Gaussian process, is now the input. Note the kernels in this prediction code and in the graining process should be exactly the same, as well as the input dataset.

import numpy as np
import scipy as sp
import george
from george.kernels import *
import sys
sys.path.append('../')
sys.path.insert(0, '/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/GP/')
from gp_training import Invdisttree
from gp_training import *
from time import time
import scipy.interpolate

#tag = '_kM32ExpConst'
tag = '_kM32ExpConst2'
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
CC = range(40)
rr = np.empty((HH.shape[1]*len(CC), x.shape[1]+xc.shape[1]))
YY = np.empty((9, HH.shape[1]*len(CC)))

pp = np.loadtxt("wp_9bins_pp_Nsize_"+str(Nsize1)+"_"+str(Nsize2)+"+N_hod_up"+str(N_hod_up)+tag+".dat")

##################   find the mean of the data  #############

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

y2 = np.empty((len(rr)*9))
ss2 = 0
for j in range(9):
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
            y2[ss2] = y[ss]
            ss = ss+1
            ss2 = ss2+1

######

    p0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    k1 = ExpSquaredKernel(p0, ndim=len(p0))
    k2 = Matern32Kernel(p0, ndim=len(p0))
    k3 = ConstantKernel(0.1, ndim=len(p0))
    k4 = WhiteKernel(0.1, ndim=len(p0))
    k5 = ConstantKernel(0.1, ndim=len(p0))

    if NewKernel == False:
        kernel = k1*k5+k2+k3+k4
    else:
        kernel = k2+k5

    ppt = pp[j]

    if j == 0:
        if HODLR == True:
            gp0 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp0 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp0.compute(rr, yerr)

        gp0.kernel.vector = ppt
        print(ppt)
        print(rr.shape)
        #for ree in range(len(rr)):
        #    print(rr[ree])
        print(yerr)
        gp0.compute(rr, yerr)

    if j == 1:
        if HODLR == True:
            gp1 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp1 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp1.compute(rr, yerr)
        
        gp1.kernel.vector = ppt
        gp1.compute(rr, yerr)

    if j == 2:
        if HODLR == True:
            gp2 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp2 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp2.compute(rr, yerr)
        
        gp2.kernel.vector = ppt
        gp2.compute(rr, yerr)

    if j == 3:
        if HODLR == True:
            gp3 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp3 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp3.compute(rr, yerr)

        gp3.kernel.vector = ppt
        gp3.compute(rr, yerr)

    if j == 4:
        if HODLR == True:
            gp4 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp4 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp4.compute(rr, yerr)
        
        gp4.kernel.vector = ppt
        gp4.compute(rr, yerr)

    if j == 5:
        if HODLR == True:
            gp5 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp5 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp5.compute(rr, yerr)
        
        gp5.kernel.vector = ppt
        gp5.compute(rr, yerr)

    if j == 6:
        if HODLR == True:
            gp6 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp6 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp6.compute(rr, yerr)
        
        gp6.kernel.vector = ppt
        gp6.compute(rr, yerr)

    if j == 7:
        if HODLR == True:
            gp7 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp7 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp7.compute(rr, yerr)
        
        gp7.kernel.vector = ppt
        gp7.compute(rr, yerr)

    if j == 8:
        if HODLR == True:
            gp8 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp8 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp8.compute(rr, yerr)
        
        gp8.kernel.vector = ppt
        gp8.compute(rr, yerr)


ran_pre = range(0, 100)
ss = 0

GP_test_HOD = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat")

GP_test_HOD[:,0] = np.log10(GP_test_HOD[:,0])
GP_test_HOD[:,2] = np.log10(GP_test_HOD[:,2])


GP_mean = True

GP_test_COS = np.loadtxt("/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/hod_file/cosmology_camb_test_box_full.dat")

GP_test_COS_id = range(0,7)

frac_rms = np.empty((len(ran_pre)*len(GP_test_COS_id), 9))
weigh_cov = np.empty((len(ran_pre)*len(GP_test_COS_id), 9))

cc = ['r', 'b', 'k', 'y', 'm', 'c', 'g']
for pre_CID in GP_test_COS_id:
    for pre_HID in ran_pre:
        print(pre_CID, pre_HID)
        t = np.atleast_2d(GP_test_HOD[pre_HID,:])
        
        if GP_mean == False:
            wp_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/wp_output/wp_covar_cosmo_"+str(pre_CID)+"_Box_0_HOD_"+str(pre_HID)+"_test_X.dat")
        
        elif GP_mean == True:
            wp_test = np.loadtxt("/home/users/ksf293/clust/results_wp/testing_wp_mean_test0/wp_cosmo_"+str(pre_CID)+"_HOD_"+str(pre_HID)+"_mean.dat")
        
        wp_test = wp_test.T

        tc = np.atleast_2d(np.hstack((GP_test_COS[pre_CID], t[0,0:11])))

        mu0, cov0 = gp0.predict(y2[len(CC)*HH.shape[1]*0:len(CC)*HH.shape[1]*1], tc)
        mu1, cov1 = gp1.predict(y2[len(CC)*HH.shape[1]*1:len(CC)*HH.shape[1]*2], tc)
        mu2, cov2 = gp2.predict(y2[len(CC)*HH.shape[1]*2:len(CC)*HH.shape[1]*3], tc)
        mu3, cov3 = gp3.predict(y2[len(CC)*HH.shape[1]*3:len(CC)*HH.shape[1]*4], tc)
        mu4, cov4 = gp4.predict(y2[len(CC)*HH.shape[1]*4:len(CC)*HH.shape[1]*5], tc)
        mu5, cov5 = gp5.predict(y2[len(CC)*HH.shape[1]*5:len(CC)*HH.shape[1]*6], tc)
        mu6, cov6 = gp6.predict(y2[len(CC)*HH.shape[1]*6:len(CC)*HH.shape[1]*7], tc)
        mu7, cov7 = gp7.predict(y2[len(CC)*HH.shape[1]*7:len(CC)*HH.shape[1]*8], tc)
        mu8, cov8 = gp8.predict(y2[len(CC)*HH.shape[1]*8:len(CC)*HH.shape[1]*9], tc)
        mu = list(mu0)+list(mu1)+list(mu2)+list(mu3)+list(mu4)+list(mu5)+list(mu6)+list(mu7)+list(mu8)
        mu = np.array(mu)
        pre=10**mu*Ymean
    
        frac_err = (np.array(pre).flatten()-np.array(wp_test[:,1]))/np.array(wp_test[:,1])
        frac_rms[ss,:] = frac_err
        print(frac_err)

        cov_list = list(cov0)+list(cov1)+list(cov2)+list(cov3)+list(cov4)+list(cov5)+list(cov6)+list(cov7)+list(cov8)
        cov_list = np.array(cov_list)
        weigh_cov[ss,:] = cov_list.T
        
        ss = ss+1

np.savetxt("fractional_error/wp_frac_rms_9bins_cos_GP_mean_"+str(GP_mean)+"_Nsize_"+str(Nsize1)+"_"+str(Nsize2)+"+N_hod_up"+str(N_hod_up)+tag+".dat", frac_rms)
np.savetxt("fractional_error/GPcov_wp_frac_rms_9bins_cos_GP_mean_"+str(GP_mean)+"_Nsize_"+str(Nsize1)+"_"+str(Nsize2)+"+N_hod_up"+str(N_hod_up)+tag+".dat", weigh_cov)

