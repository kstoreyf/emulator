import numpy as np

import emulator


traintag = '_nonolap'
nhodpercosmo = 50

errtag = '_10hod_test0'
gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))

tag = ''

training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
gptag = traintag + errtag + tag
hyperparams = "../training_results/{}_training_results{}.dat".format(statistic, gptag)

testing_dir = '{}testing_{}{}/'.format(res_dir, statistic, traintag)


emu = emulator.Emulator(statistic, training_dir=training_dir, testing_dir=testing_dir, fixed_params=fixed_params, gperr=gperr)
emu.set_hyperparams(hyperparams)
emu.build()
emu.test()