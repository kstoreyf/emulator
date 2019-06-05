import numpy as np




def main():

    statistic = 'upf'

    #datatag = '_cos0'
    #datatag = '_hod1'
    #fixed_hod = True
    fixed_hod = False
    #testtag = '_10hod_old'
    testtag = ''
    #errtag = '_errx100'
    errtag = '_10hod_test0'

    res_dir = '../../clust/results_{}/'.format(statistic)
    testing_dir = '../../clust/results_{}/testing_{}{}/'.format(statistic, statistic, testtag)
    #testmean_dir = '../../clust/results/testing_{}{}_mean/'.format(statistic, testtag)

    #hods = range(123, 135)

    if fixed_hod:
        hods = [1]
        cosmos = range(7)
    else:
        hods = range(10)
        #hods = [0, 6, 10, 11, 14, 16, 19, 20, 23, 24]#, 25, 26, 27, 28, 29, 32, 33, 38, 41, 43, 44, 47, 50, 52, 57, 58, 59, 60, 64, 68, 71, 72, 74, 75, 77, 81, 82, 83, 84, 85, 87, 91, 97, 99] #good hods
        cosmos = range(7)

    ncosmos = len(cosmos)
    nboxes = 5
    boxes = range(nboxes)
    tests = range(1)
    ntests = len(tests)
    rads = []
    vals = []

    nbins = 9

    wps_grid = np.zeros((ncosmos, nboxes, nbins))
    shots = []

    devmeans = []
    for hod in hods:
        for cosmo in cosmos:
            cosmo_avg = np.zeros(nbins)
            print cosmo
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
                    box_avg += val
                    vpfs_box += val
                    #wps_cosmo.append(wp)
                    print "vals:", val

                #turn sum into average
                box_avg /= ntests
                print "boxavg:", box_avg
                box_avgs.append(box_avg)
                cosmo_avg += box_avg

            cosmo_avg /= nboxes
            print "cosmoavg", cosmo_avg

            for boxmean in box_avgs:
                devmean = np.nan_to_num((boxmean - cosmo_avg)/cosmo_avg)
                #devmean = np.nan_to_num(boxmean - cosmo_avg)

                #print devmean
                devmeans.append(devmean)


   #shots.append(np.var(wps_hod, axis=0))

    #err = np.var(np.array(devmeans), axis=0)
    err = np.std(np.array(devmeans), axis=0)
    print len(devmeans)
    print "err:", err

    #err *= 100
    #save to both the directory and the mean, same error for both
    np.savetxt(res_dir+"{}_error{}.dat".format(statistic, errtag), err)

    #wps_box_avg /= len(hods)
    #        wps_grid[cosmo][box] += wps_box_avg


if __name__=="__main__":
    main()
