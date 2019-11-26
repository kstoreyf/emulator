import numpy as np
import os

import emulator


statistic = 'upf'
#statistic = 'wp'
traintag = '_nonolap'
testtag = '_mean_test0'
errtag = '_10hod_test0'
testmean = True

#tag = '_emuobj'
tag = '_log'
gptag = traintag + errtag + tag
acctag = gptag + testtag

res_dir = '../../clust/results_{}/'.format(statistic)
gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))

training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, testtag)
predict_savedir = f"../testing_results/predictions_{statistic}{acctag}/"
os.makedirs(predict_savedir, exist_ok=True)

emu = emulator.Emulator(statistic, training_dir=training_dir, testing_dir=testing_dir, gperr=gperr, testmean=testmean, hyperparams=hyperparams)
emu.build()
emu.test(predict_savedir)