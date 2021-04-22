import numpy as np

import utils


def main():

    #statistics = ['wp']
    #statistics = ['wp','upf','mcf']
    #statistics = ['wp', 'mcf']
    statistics = ['wp', 'xi', 'upf', 'mcf']
    #savetag = '_fstar8.0_p1.0'
    #traintags = ['_nonolap', '_nonolap']#, f'{savetag}_nonolap']
    traintag = '_nonolap'
    #testtag = '_mean_test0'
    testtag = '_glam'
    errtag = '_hod3_test0'
    savetag = '_residuals'
    #testtags = ['_mean_test0','_mean_test0']#,f'{savetag}_mean_test0']
    #errtags = ['_hod3_test0','_hod3_test0']#, '_hod3_test0']
    tags = ['_log_kM32ExpConst2_100hod','_log_kM32ExpConst_100hod','_log_kM32ExpConst_100hod', '_log_kM32ExpConst_100hod']
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
        if 'glam' in testtag:
            fracerrs = load_fracerrs_glam(statistic, testtag, acctag)
            print("Using residual fractional errors!")
            fracerrs -= np.mean(fracerrs, axis=0)
            fracerr_arrs.append(fracerrs)
        else:
            fracerrs = load_fracerrs_aemulus(statistic, testtag, acctag, CC_test, HH_test)
            fracerr_arrs.append(fracerrs)

    fracerrs = np.concatenate(fracerr_arrs, axis=1)
    cov_perf = utils.covariance(fracerrs, zeromean=True)

    tag_str = traintag + errtag + testtag + savetag
    #tag_str += ''.join(tags)
    #tag_str = ''
    #acc_str = ''.join(acctags)
    save_fn_perf = f"{cov_dir}/cov_emuperf_{stat_str}{tag_str}.dat"
    print('Saving cov_perf to', save_fn_perf)
    np.savetxt(save_fn_perf, cov_perf)
    
    p16 = np.percentile(fracerrs, 16, axis=0)
    p84 = np.percentile(fracerrs, 84, axis=0)
    save_fn_p16_perf = f"{cov_dir}/p16_emuperf_{stat_str}{tag_str}.dat"
    save_fn_p84_perf = f"{cov_dir}/p84_emuperf_{stat_str}{tag_str}.dat"
    np.savetxt(save_fn_p16_perf, p16)
    np.savetxt(save_fn_p84_perf, p84)

    cov_perf_nonzeromean = utils.covariance(fracerrs)
    save_fn_perf_nonzeromean = f"{cov_dir}/cov_emuperf_nonzeromean_{stat_str}{tag_str}.dat"
    np.savetxt(save_fn_perf_nonzeromean, cov_perf_nonzeromean)


def load_fracerrs_glam(statistic, testtag, acctag, N_mocks=986, nbins=9):
    
    # Load emu prediction
    predict_savedir = f"../testing_results/predictions_{statistic}{acctag}"
    pred_fn = f"{predict_savedir}/{statistic}_glam.dat"
    r_pred, vals_pred = np.loadtxt(pred_fn, delimiter=',', unpack=True)
    
    # Load all observations
    vals_all_obs = np.zeros((N_mocks, nbins))
    for n in range(N_mocks):
        testing_dir = f'/home/users/ksf293/clust/results_glam/results_glam_{statistic}'
        fnt = f"{testing_dir}/{statistic}_glam_n{n}.dat"
        r_obs, vals_obs = np.loadtxt(fnt, delimiter=',', unpack=True)
        vals_all_obs[n,:] = vals_obs

    # Compute fractional errors
    fracerrs = (vals_all_obs - vals_pred)/vals_all_obs
    return fracerrs


def load_fracerrs_aemulus(statistic, testtag, acctag, CC_test, HH_test):

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
    
    fracerrs = (np.array(ptests)-np.array(ppredicts))/np.array(ptests)
    return fracerrs


def compute_rmse(ys, y_preds):
    ys = np.array(ys)
    y_preds = np.array(y_preds)
    return np.sqrt(np.mean(np.square(ys - y_preds), axis=0))


if __name__=='__main__':
    main()
