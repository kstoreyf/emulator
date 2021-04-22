import numpy as np
import os

import emulator

statistic = 'wp'
#statistic = 'xi'
#statistic = 'upf'
#statistic = 'mcf'
#statistic = 'xi2'
#savetag = '_fstar8.0_p1.0'
savetag = ''
traintag = savetag+'_nonolap'
testtag = savetag+'_glam'
errtag = '_hod3_test0' #used for emulator error
testmean = False
testsavetag = ''

nhod_test = 100
nhod = 100
kernel_name = 'M32ExpConst' # xi, upf, mcf
#kernel_name = 'M32ExpConst2' # wp
tag = '_log_k{}_{}hod'.format(kernel_name, nhod)
#tag = '_meansub_xrsq_k{}_{}hod'.format(kernel_name, nhod)
#log = True
log = True
mean = False
meansub = False
xrsq = False
gptag = traintag + errtag + tag
acctag = gptag + testtag + testsavetag

res_dir = '../../clust/results_{}/'.format(statistic)
gperr = np.loadtxt("../../clust/covariances/error_aemulus_{}{}{}.dat".format(statistic, errtag, savetag))

training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

predict_savedir = f"../testing_results/predictions_{statistic}{acctag}/"
os.makedirs(predict_savedir, exist_ok=True)

print("Initializing emu")
emu = emulator.Emulator(statistic, training_dir,
            gperr=gperr, testmean=testmean, hyperparams=hyperparams, log=log, 
            mean=mean, meansub=meansub, xrsq=xrsq, nhod=nhod, nhod_test=nhod_test, kernel_name=kernel_name)
print("Building emu")
emu.build()
print("Testing emu")
emu.test_glam(predict_savedir)
print("Done!")
