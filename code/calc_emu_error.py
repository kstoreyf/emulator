import numpy as np


def main():

    statistic='wp'
    traintag = '_nonolap'
    testtag = '_mean_test0'
    #errtag = '_100hod_test0' # for emu
    errtag = '_hod3_test0'
    tag = '_log_kM32ExpConst2_100hod'
    gptag = traintag + errtag + tag
    acctag = gptag + testtag

    nhod_test = 100
    CC_test = range(0, 7)
    HH_test = range(0, nhod_test)

    print("Computing emu error for", statistic, testtag, acctag)
    ptests, ppredicts = load_data(statistic, testtag, acctag, CC_test, HH_test)
    #emu_performance = compute_rmse(ptests, ppredicts)
    fracerrs = (ptests-ppredicts)/ptests #consider denom here
    #emu_performance = np.std(fracerrs)
    cov_perf = covariance(fracerrs, zeromean=True)
    cov_perf *= 5 #bc using mean of 5 testboxes

    res_dir = '../../clust/results_{}/'.format(statistic)
    errtag = '_hod3_test0' # for cov test
    cov_test = np.loadtxt(res_dir+"{}_cov{}.dat".format(statistic, errtag))

    #cov_performance = cov_emu + cov_test
    cov_emu = cov_perf - cov_test
    #emu_err = np.sqrt(emu_performance**2 - testset_error**2)
    print(cov_emu)
    save_fn = f"../testing_results/{statistic}_emu_cov{acctag}.dat"
    print('Saving to', save_fn)
    np.savetxt(save_fn, cov_emu)

    save_fn_perf = f"../testing_results/{statistic}_emuperf_cov{acctag}.dat"
    np.savetxt(save_fn_perf, cov_perf)


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



def covariance(arrs, zeromean=False):
    N = arrs.shape[0]

    if zeromean:
        w = arrs
    else:
        w = arrs - arrs.mean(0)

    outers = np.array([np.outer(w[n], w[n]) for n in range(N)])
    covsum = np.sum(outers, axis=0)
    cov = 1.0/float(N-1.0) * covsum
    return cov



def compute_rmse(ys, y_preds):
    ys = np.array(ys)
    y_preds = np.array(y_preds)
    return np.sqrt(np.mean(np.square(ys - y_preds), axis=0))

if __name__=='__main__':
    main()
