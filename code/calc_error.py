import numpy as np




def main():

    statistic = 'wp'

    testtag = ''
    errtag = '_hod3_test0'

    res_dir = '../../clust/results_{}/'.format(statistic)
    testing_dir = '../../clust/results_{}/testing_{}{}/'.format(statistic, statistic, testtag)

    # choose a middle-of-the-road hod
    hod = 3

    nbins = 9
    ncosmos = 7
    nboxes = 5
    ntests = 1 #eventually this should be 10

    cosmos = list(range(ncosmos))
    boxes = list(range(nboxes))
    tests = list(range(ntests)) 


    devmeans = []
    for cosmo in cosmos:

        ys_box = [] #this will contain 5 statistics
        for box in boxes:
            
            # Compute the average over tests for a single box & model
            ys_test = []
            for test in tests:
                fn = testing_dir+'{}_cosmo_{}_Box_{}_HOD_{}_test_{}.dat'\
                    .format(statistic, cosmo, box, hod, test)
                rad, y = np.loadtxt(fn, delimiter=',', unpack=True)
                ys_test.append(y)
            y_box = np.mean(ys_test, axis=0) #mean is our estimate for the statistic of the box with the given model

            ys_box.append(y_box)

        #The mean of the 5 boxes, for a given model (cosmo & HOD)
        y_mean = np.mean(ys_box, axis=0) 

        for y in ys_box:
            devmean = (y-y_mean)/y_mean
            devmeans.append(devmean)
            
    #compute covariance assuming the mean is zero, as that is the expectation value (should be unbiased)
    devmeans = np.array(devmeans)
    cov = covariance(devmeans, zeromean=True)

    np.savetxt(res_dir+"{}_cov{}.dat".format(statistic, errtag), cov)

    err = np.std(devmeans, axis=0)
    np.savetxt(res_dir+"{}_error{}.dat".format(statistic, errtag), err)
    print("err:", err)
    
    p16 = np.percentile(devmeans, 16, axis=0)
    p84 = np.percentile(devmeans, 84, axis=0)
    #std_obs = np.std(vals_all, axis=0)
    #save to both the directory and the mean, same error for both
    # 
    np.savetxt(res_dir+"{}_p16{}.dat".format(statistic, errtag), p16)
    np.savetxt(res_dir+"{}_p84{}.dat".format(statistic, errtag), p84)
    # np.savetxt(res_dir+"{}_std{}.dat".format(statistic, errtag), std_obs)


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

if __name__=="__main__":
    main()
