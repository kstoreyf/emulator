import numpy as np
import os

import emulator


#statistic = 'upf'
#statistic = 'wp'
statistic = 'mcf'
savetag = '_fstar8.0_p1.0'
traintag = savetag+'_nonolap'
testtag = savetag+'_mean_test0'
#errtag = '_100hod_test0'
errtag = '_hod3_test0'
testmean = True
testsavetag = ''

nhod_test = 100
nhod = 100
kernel_name = 'M32Const'
tag = '_log_k{}_{}hod'.format(kernel_name, nhod)
log = True
mean = False
gptag = traintag + errtag + tag
acctag = gptag + testtag + testsavetag

res_dir = '../../clust/results_{}/'.format(statistic)
gperr = np.loadtxt("../../clust/covariances/error_aemulus_{}{}{}.dat".format(statistic, errtag, savetag))

training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, testtag)
predict_savedir = f"../testing_results/predictions_{statistic}{acctag}/"
os.makedirs(predict_savedir, exist_ok=True)

emu = emulator.Emulator(statistic, training_dir, testing_dir=testing_dir, gperr=gperr, testmean=testmean, hyperparams=hyperparams, log=log, mean=mean, nhod=nhod, nhod_test=nhod_test, kernel_name=kernel_name)
emu.build()
emu.test(predict_savedir)
