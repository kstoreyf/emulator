import numpy as np
import os

import emulator


statistic = 'xi'
savetag = ''
traintag = savetag+'_nonolap'
#testtag = savetag
testtag = savetag+'_mean_test0'
errtag = '_hod3_test0'
testmean = True
testsavetag = ''

nhod_test = 100
nhod = 100
tag = '_log_{}hod_ann_aemuluserr_lr5e-4_3layer_validfix'.format(nhod)

log = True
mean = False
meansub = False
xrsq = False
anntag = traintag + errtag + tag
acctag = anntag + testtag + testsavetag

res_dir = '../../clust/results_{}/'.format(statistic)
gperr = np.loadtxt("../../clust/covariances/error_aemulus_{}{}{}.dat".format(statistic, errtag, savetag))

training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
model_dir = "../training_ann_results/{}_training_results{}".format(statistic, anntag)

testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, testtag)
predict_savedir = f"../testing_results/predictions_{statistic}{acctag}"
os.makedirs(predict_savedir, exist_ok=True)

emu = emulator.Emulator(statistic, training_dir, testing_dir=testing_dir, 
            gperr=gperr, testmean=testmean, log=log, 
            mean=mean, meansub=meansub, xrsq=xrsq, nhod=nhod, nhod_test=nhod_test, model_dir=model_dir)
emu.build_ann()
emu.test_ann(predict_savedir)
#emu.build_ann_bybin()
#emu.test_ann_bybin(predict_savedir)