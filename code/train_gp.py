import numpy as np
from george.kernels import *
import sys

import george

import gp_trainer as trainer
import emulator

sys.path.insert(0, '../../../CMASS/Gaussian_Process/GP/')


statistic = 'wp'
#statistic = 'upf'
traintag = '_nonolap'
#tag = '_emuobj'
#tag = ''
tag = '_log_kM32ExpConst1_50hod'
nhod = 50
log = True
mean = False
errtag = '_100hod_test0'
gptag = traintag + errtag + tag

nthreads = 9
nbins = 9

res_dir = '../../clust/results_{}/'.format(statistic)
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
save_hyperparams_fn = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

print("Training, savetag={}".format(gptag))

emu = emulator.Emulator(statistic, training_dir=training_dir, 
                            nbins=nbins, gperr=gperr, log=log, mean=mean, nhod=nhod)
emu.train(save_hyperparams_fn, nthreads=nthreads)
