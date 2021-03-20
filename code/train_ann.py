import numpy as np
import sys

import emulator


statistic = 'xi'
traintag = '_nonolap'
nhod = 100

tag = '_log_{}hod_ann_aemuluserr_lr5e-4_3layer_validfix'.format(nhod)
log = True 
mean = False
meansub = False
xrsq = False
errtag = '_hod3_test0'
anntag = traintag + errtag + tag

nthreads = 9
nbins = 9

res_dir = '../../clust/results_{}/'.format(statistic)
training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
gperr = np.loadtxt("../../clust/covariances/error_aemulus_{}{}.dat".format(statistic, errtag))
save_model_dir = "../training_ann_results/{}_training_results{}".format(statistic, anntag)

print("Training, savetag={}".format(anntag))

emu = emulator.Emulator(statistic, training_dir=training_dir, nbins=nbins, 
                            gperr=gperr, log=log, mean=mean, meansub=meansub, xrsq=xrsq,
                            nhod=nhod,)
#emu.train_ann_bybin(save_model_dir)
emu.train_ann(save_model_dir)
