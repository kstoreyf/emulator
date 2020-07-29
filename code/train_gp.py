import numpy as np
from george.kernels import *
import sys

import george

import gp_trainer as trainer
import emulator

sys.path.insert(0, '../../../CMASS/Gaussian_Process/GP/')


#statistic = 'wp'
#statistic = 'upf'
statistic = 'mcf'
savetag = '_fstar8.0_p1.0'
traintag = savetag+'_nonolap'
nhod = 100
#nhod = 3
kernel_name = 'M32Const'
tag = '_log_k{}_{}hod'.format(kernel_name, nhod)
log = True
mean = False
#errtag = '_100hod_test0'
errtag = '_hod3_test0'
gptag = traintag + errtag + tag

nthreads = 9
nbins = 9

res_dir = '../../clust/results_{}/'.format(statistic)
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
gperr = np.loadtxt("../../clust/covariances/error_aemulus_{}{}{}.dat".format(statistic, errtag, savetag))
save_hyperparams_fn = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

print("Training, savetag={}".format(gptag))

emu = emulator.Emulator(statistic, training_dir=training_dir, 
                            nbins=nbins, gperr=gperr, log=log, mean=mean, nhod=nhod, kernel_name=kernel_name)
emu.train(save_hyperparams_fn, nthreads=nthreads)
