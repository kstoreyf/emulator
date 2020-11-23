import numpy as np

import utils


def main():

    compute_instrinsic_emu_error = False
    #statistics = ['wp','upf','mcf']
    #statistics = ['wp', 'mcf']
    statistics = ['wp', 'xi', 'mcf']
    #savetag = '_fstar8.0_p1.0'
    #traintags = ['_nonolap', '_nonolap']#, f'{savetag}_nonolap']
    traintag = '_nonolap'
    testtag = '_mean_test0'
    errtag = '_hod3_test0'
    #testtags = ['_mean_test0','_mean_test0']#,f'{savetag}_mean_test0']
    #errtags = ['_hod3_test0','_hod3_test0']#, '_hod3_test0']
    tags = ['_log_kM32ExpConst2_100hod','_log_kM32ExpConst_100hod','_log_kM32ExpConst_100hod']#, '_log_kM32ExpConst_100hod']
    #tags = ['_log_kM32ExpConst2_100hod']
   
    # statistics = ['mcf']
    # savetag = '_fstar8.0_p2.0'
    # traintags = [f'{savetag}_nonolap']
    # testtags = [f'{savetag}_mean_test0']
    # errtags = ['_hod3_test0']
    # tags = ['_log_kM32ExpConst_100hod']

    nhod_test = 100
    CC_test = range(0, 7)
    HH_test = range(0, nhod_test)
    stat_str = '_'.join(statistics)
    cov_dir = '../../clust/covariances'.format(stat_str)

    acctags = []
    fracerr_arrs = []
    for i, statistic in enumerate(statistics):
        #gptag = traintags[i] + errtags[i] + tags[i]
        #acctag = gptag + testtags[i]
        gptag = traintag + errtag + tags[i]
        acctag = gptag + testtag
        acctags.append(acctag)

        print("Computing emu error for", statistic, testtag, acctag)
        ptests, ppredicts = load_data(statistic, testtag, acctag, CC_test, HH_test)
        fracerr_arrs.append((ptests-ppredicts)/ptests)

    fracerrs = np.concatenate(fracerr_arrs, axis=1)
    cov_perf = utils.covariance(fracerrs, zeromean=True)

    tag_str = traintag + errtag + testtag
    #tag_str += ''.join(tags)
    #tag_str = ''
    #acc_str = ''.join(acctags)
    save_fn_perf = f"{cov_dir}/cov_emuperf_{stat_str}{tag_str}.dat"
    print('Saving cov_perf to', save_fn_perf)
    np.savetxt(save_fn_perf, cov_perf)

    # subtract test cov in order to isolate emulator error
    # MINERVA as test
    if compute_instrinsic_emu_error:
        cov_minerva = np.loadtxt(f"{cov_dir}/cov_minerva_{stat_str}.dat")
        L_minerva = 1.5 #Gpc
        L_aemulus = 1.05 #Gpc
        cov_test = cov_minerva*(L_minerva/L_aemulus)**3
        if 'mean' in testtags[0]:
            cov_test *= 1./5. #because using the mean of 5 boxes

        #because cov_performance = cov_emu + cov_test
        cov_emu = cov_perf - cov_test
        save_fn = f"{cov_dir}/cov_emu_{stat_str}{tag_str}.dat"
        print('Saving cov_emu to', save_fn)
        np.savetxt(save_fn, cov_emu)


def load_data(statistic, testtag, acctag, CC_test, HH_test):

    res_dir = '../../clust/results_{}/'.format(statistic)
    ptests = []
    ppredicts = []
    for cosmo in CC_test:
        for hod in HH_test:
            hod = int(hod)

            if "mean" in acctag:
                idtag = '{}_cosmo_{}_HOD_{}_mean'.format(statistic, cosmo, hod)
            else:
                idtag = '{}_cosmo_{}_Box_0_HOD_{}_test_0'.format(statistic, cosmo, hod)

            fnt = '{}testing_{}{}/{}.dat'.format(res_dir, statistic, testtag, idtag)
            ntest, ptest = np.loadtxt(fnt)
            fnp = '../testing_results/predictions_{}{}/{}.dat'.format(statistic, acctag, idtag)
            npredict, ppredict = np.loadtxt(fnp, delimiter=',', unpack=True)

            ptests.append(ptest)
            ppredicts.append(ppredict)
    
    return np.array(ptests), np.array(ppredicts)


def compute_rmse(ys, y_preds):
    ys = np.array(ys)
    y_preds = np.array(y_preds)
    return np.sqrt(np.mean(np.square(ys - y_preds), axis=0))


if __name__=='__main__':
    main()
