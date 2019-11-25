import numpy as np
from george.kernels import *
import sys

import george

import gp_trainer as trainer
import emulator

sys.path.insert(0, '../../../CMASS/Gaussian_Process/GP/')


statistic = 'wp'
traintag = '_nonolap'
#tag = '_emuobj'
tag = ''
errtag = '_10hod_test0'
gptag = traintag + errtag + tag

nbins = 9

res_dir = '../../clust/results_{}/'.format(statistic)
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
save_hyperparams_fn = "../training_results/{}_training_results{}.dat".format(statistic, gptag)


print("Training, savetag={}".format(gptag))

emu = emulator.Emulator(statistic, training_dir=training_dir, 
                                    nbins=nbins, gperr=gperr)
emu.train(save_hyperparams_fn)
