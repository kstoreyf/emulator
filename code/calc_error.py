import numpy as np




def main():

    statistic = 'wp'

    fixed_hod = False
    testtag = ''
    errtag = '_100hod_test0'
    nhods = 100

    res_dir = '../../clust/results_{}/'.format(statistic)
    testing_dir = '../../clust/results_{}/testing_{}{}/'.format(statistic, statistic, testtag)

    if fixed_hod:
        hods = [1]
        cosmos = list(range(7))
    else:
        hods = list(range(nhods))
        cosmos = list(range(7))

    ncosmos = len(cosmos)
    nboxes = 5
    boxes = list(range(nboxes))
    tests = list(range(1))
    ntests = len(tests)
    rads = []
    vals = []

    nbins = 9

    wps_grid = np.zeros((ncosmos, nboxes, nbins))
    shots = []

    devmeans = []
    vals_all = []
    for hod in hods:
        for cosmo in cosmos:
            cosmo_avg = np.zeros(nbins)
            print(cosmo)
            box_avgs = []
            for box in boxes:
                box_avg = np.zeros(nbins)
                vpfs_box = np.zeros(nbins)
                for test in tests:
                    fn = testing_dir+'{}_cosmo_{}_Box_{}_HOD_{}_test_{}.dat'\
                        .format(statistic, cosmo, box, hod, test)
                    #fn = "../results_wp_mean/wp_{}_cosmo_{}_HOD_{}_mean.dat" \
                    #    .format(tag, cosmo, hod)
                    rad, val = np.loadtxt(fn, delimiter=',', unpack=True)
                    rads.append(rad)
                    vals.append(val)
                    vals_all.append(val)
                    box_avg += val
                    vpfs_box += val
                    #wps_cosmo.append(wp)
                    print("vals:", val)

                #turn sum into average
                box_avg /= ntests
                print("boxavg:", box_avg)
                box_avgs.append(box_avg)
                cosmo_avg += box_avg

            cosmo_avg /= nboxes
            print("cosmoavg", cosmo_avg)

            for boxmean in box_avgs:
                devmean = np.nan_to_num((boxmean - cosmo_avg)/cosmo_avg)
                #devmean = np.nan_to_num(boxmean - cosmo_avg)

                #print devmean
                devmeans.append(devmean)


   #shots.append(np.var(wps_hod, axis=0))

    #err = np.var(np.array(devmeans), axis=0)
    devmeans = np.array(devmeans)
    err = np.std(devmeans, axis=0)
    p16 = np.percentile(devmeans, 16, axis=0)
    p84 = np.percentile(devmeans, 84, axis=0)
    print(len(devmeans))
    print("err:", err)
    
    std_obs = np.std(vals_all, axis=0)

    #err *= 100
    #save to both the directory and the mean, same error for both
    np.savetxt(res_dir+"{}_error{}.dat".format(statistic, errtag), err)
    np.savetxt(res_dir+"{}_p16{}.dat".format(statistic, errtag), p16)
    np.savetxt(res_dir+"{}_p84{}.dat".format(statistic, errtag), p84)
    np.savetxt(res_dir+"{}_std{}.dat".format(statistic, errtag), std_obs)

    #wps_box_avg /= len(hods)
    #        wps_grid[cosmo][box] += wps_box_avg


if __name__=="__main__":
    main()
