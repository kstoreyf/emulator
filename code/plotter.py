from matplotlib import pyplot as plt
import numpy as np




def main():
    #plot_test()
    #plot_predic()
    #plot_error()
    #plot_error_analytic()
    #plot_analytic_train()
    plot_error_sim()


def plot_wp_simple(rs, vals, saveto=None):
    plt.figure()
    rs = np.array(rs)
    vals = np.array(vals)
    if len(rs.shape)==1:
        rs = [rs]
        vals = [vals]
    for i in range(len(rs)):
        plt.plot(rs[i], vals[i], color='blue')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel(r'$r_p$ (Mpc/h)')
    plt.ylabel(r'$w_p$($r_p$)')
    if saveto:
        plt.savefig(saveto)


def plot_predic_statistic():

    statistic = 'vpf'

    if statistic=='vpf':
        plotfunc = plot_vpf



def plot_analytic_train():

    # cosmos = range(40)
    # hods = range(400)
    cosmos = range(40)
    hods = range(40)
    rps = []
    wps = []
    labels = []
    for cosmo in cosmos:
        for hod in hods:
            dir = '../CMASS_Analytic_EH/Gaussian_Process/wp_clustering_emu/'
            fn = dir+'cosmo_{}_HOD_{}_emu.clustering'.format(cosmo, hod)
            rp, wp = np.loadtxt(fn, unpack=True, usecols=(0,1))
            labels.append('{}_{}'.format(cosmo,hod))
            rps.append(rp)
            wps.append(wp)

    rpmean = np.mean(rps, axis=0)
    wpmean = np.mean(wps, axis=0)
    rps.append(rpmean)
    wps.append(wpmean)
    labels.append('mean')
    plot_wprp(rps, wps, labels)


def plot_test():

    cosmos = range(7)
    hods = range(10)
    rps = []
    wps = []
    labels = []
    for cosmo in cosmos:
        for hod in hods:
            fn = "../Test_Box_wp_covar_mean/wp_covar_cosmo_{}_HOD_{}_mean.dat"\
                .format(cosmo, hod)
            rp, wp = np.loadtxt(fn, unpack=True, usecols=(0,1))
            labels.append(cosmo)
            rps.append(rp)
            wps.append(wp)

    fnmean = "../wp_covar_results/wp_covar_mean.dat"
    rp, wpmean = np.loadtxt(fnmean, unpack=False)
    rps.append(rp)
    wps.append(wpmean)
    plot_wprp(rps, wps, labels, wp_tocompare=wpmean)


def plot_predic():

    cosmos = range(2)

    #cosmos = [0]
    #hods = range(1000, 1010)
    #hods = range(100)
    hods = np.random.randint(0, 100, 5)
    rps = []
    wps = []
    labels = []
    tags = ['testbox', 'empredic']
    nmodels = 10
    for nn in range(nmodels):
        cosmo = np.random.randint(0, 7)
        hod = np.random.randint(0, 100)
        for tag in tags:
            fn = "../results_wp_mean/wp_{}_cosmo_{}_HOD_{}_mean.dat"\
                .format(tag, cosmo, hod)
            #fn = "../wp_covar_results/wp_covar_cosmo_{}_HOD_{}_test_0.dat"\
            #    .format(cosmo, hod)
            #rp, wp = np.loadtxt(fn, unpack=True, usecols=(0,1))
            rp, wp = np.loadtxt(fn)
            labels.append('{}_cosmo_{}_HOD_{}'.format(tag, cosmo, hod))
            rps.append(rp)
            wps.append(wp)

    fnmean = "../wp_covar_results/wp_covar_mean.dat"
    rp, wpmean = np.loadtxt(fnmean, unpack=False)
    #rps.append(rp)
    #wps.append(wpmean)
    #labels.append('mean')
    print(labels)
    plot_wprp(rps, wps, labels, wp_tocompare=wpmean)



def plot_wprp(rps, wprps, labels, colors=None, wp_tocompare=None):
    # if np.array(rps).ndim==1:
    #     rps = [rps]
    #     wprps = [wprps]
    #     labels = [labels]
    if not colors:
        color_idx = np.linspace(0, 1, len(wprps)/2)
        colors = [plt.cm.rainbow(c) for c in color_idx]

    if type(wp_tocompare) is np.ndarray:
        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
    else:
        fig = plt.figure()
        ax0 = fig.gca()

    for i in range(len(rps)):
        rp = rps[i]
        wprp = np.array(wprps[i])
        label = labels[i]
        #
        if "testbox" in label:
            ax0.loglog(rp, wprp, marker='o', ls='none', mfc='none', color=colors[i/2])
        elif "empredic" in label:
            ax0.loglog(rp, wprp, marker=None, ls='-', color=colors[i/2])
        else:
            #plot mean
            if i == len(rps) - 1:
                ax0.loglog(rp, wprp, label='mean', color='black', marker=None, ls='--')
            else:
                ax0.loglog(rp, wprp, color='red', marker=None, alpha=0.05)
        #ax0.semilogx(rp, wprp, label=label, color=color, marker='o')

        plt.xlabel(r'$r_p$ (Mpc/h)')
        plt.xlim(0.1, 30.)
        ax0.set_ylim(5, 10**4)
        ax0.set_ylabel(r'$w_p$($r_p$)')

        if type(wp_tocompare) is np.ndarray:
            # if wp.all() != wp_tocompare.all():
            #wpcomp = wprps[compidx]

            if len(wprp)==len(wp_tocompare):# and i!=len(wprps)-1:
                #ax1.semilogx(rp, (np.log10(wprp)-np.log10(wpcomp)) / np.log10(wpcomp), color=colors[label])
                if "testbox" in label:
                    ax1.semilogx(rp, wprp/wp_tocompare, color=colors[i/2], marker='o', ls='None', mfc='none')
                elif "empredic" in label:
                    ax1.semilogx(rp, wprp/wp_tocompare, color=colors[i/2], marker=None, ls='-')

                #ax1.set_ylabel(r'$w_p$/$w_{{p,\mathrm{{{0}}}}}$'.format(wp_tocompare))
                ax1.set_ylabel(r'$w_p$/$w_{p,mean}$')


    ax0.legend(loc='best')

    plt.show()


def plot_error():

    ### Fractional error (predic) ###
    fn = 'fractional_error/wp_frac_rms_9bins_cos_HOD_Safe_False_50_version_4_7COS_Larger_err' \
          '_False_NewKernel_False_Err_Corr_False_GP_mean_True.dat'
    #fn = '../CMASS_BIAS/Gaussian_Process/GP/RSD_multiple_mono_9bins_fixed_False_kernel_50_version_0_7COS_Err_Corr_False.dat'
    #fn = '../CMASS_BIAS/Gaussian_Process/GP/RSD_multiple_mono_9bins_fixed_False_50_version_4_7COS_Err_Corr_False.dat'
    ax = 1

    frac_rms = np.loadtxt(fn)
    mean_err = np.mean(frac_rms, axis=ax)
    sig1p = mean_err + np.std(frac_rms, axis=ax)
    sig1m = mean_err - np.std(frac_rms, axis=ax)
    sig2p = mean_err + 2*np.std(frac_rms, axis=ax)
    sig2m = mean_err - 2*np.std(frac_rms, axis=ax)

    #get rps from mean wp
    fnmean = "../wp_covar_results/wp_covar_mean.dat"
    rp, wpmean = np.loadtxt(fnmean, unpack=False)

    print(mean_err)
    c1 = 'grey'
    c2 = 'lightgrey'
    plt.semilogx(rp, sig1p, c=c1)
    plt.semilogx(rp, sig1m, c=c1)

    plt.semilogx(rp, sig2p, c=c2)
    plt.semilogx(rp, sig2m, c=c2)

    plt.fill_between(rp, sig2m, sig2p, color=c2)
    plt.fill_between(rp, sig1m, sig1p, color=c1)

    nbins = 9
    ### Test error ###
    cosmos = range(7)
    hods = range(1)
    #hods = range(10)
    boxes = range(5)
    seeds = range(10)
    rps = []
    wps = []

    wps_grid = np.zeros((len(cosmos), len(boxes), nbins))
    shots = []
    for cosmo in cosmos:
        #wps_cosmo_avg = []
        print(cosmo)
        for box in boxes:
            wps_box_avg = np.zeros(nbins)
            for hod in hods:
                wps_hod_avg = np.zeros(nbins)
                wps_hod = []
                for seed in seeds:
                    dir = "../CMASS_BIAS/Gaussian_Process/GP/GP_test_data/Test_Box_wp_covar/"
                    fn = dir + 'wp_covar_cosmo_{}_Box_{}_HOD_{}_test_{}.dat'\
                        .format(cosmo, box, hod, seed)
                    #fn = "../results_wp_mean/wp_{}_cosmo_{}_HOD_{}_mean.dat" \
                    #    .format(tag, cosmo, hod)
                    rp, wp = np.loadtxt(fn, usecols=(0,1), delimiter='\t', unpack=True)
                    rps.append(rp)
                    wps.append(wp)
                    wps_hod_avg += wp
                    wps_hod.append(wp)
                    #wps_cosmo.append(wp)
                #turn sum into average
                wps_hod_avg /= len(seeds)
                wps_box_avg += wps_hod_avg
                shots.append(np.var(wps_hod, axis=0))

            wps_box_avg /= len(hods)
            wps_grid[cosmo][box] += wps_box_avg
            #wps_cosmo_avg += wps_box_avg

        #wps_cosmo_avg /= len(boxes)


    diffmeans = []
    for cosmo in cosmos:
        wp_cosmo_avg = np.mean(wps_grid, axis=(0,1))
        for box in boxes:
            diffmeans.append((wp_cosmo_avg - wps_grid[cosmo][box])/wp_cosmo_avg)
    sample_var = np.var(diffmeans, axis=0)
    print(sample_var)

    print(shots)
    shot_noise = np.mean(shots, axis=0)
    print(shot_noise)

    # shot_noise.append(np.var(wps_cosmo, axis=0))
    #
    # diffmeans = [w-wpmean for w in wps]
    # print sample/wpmean
    #
    # print shot_noise
    # shot = np.mean(shot_noise, axis=0)
    # print shot/wpmean
    # test_err = shot**2 + sample**2
    #plt.semilogx(rp, sample_var, color='blue', ls='--')
    #plt.semilogx(rp, -sample_var, color='blue', ls='--')

    #plt.plot(rp, shot_noise, color='green', ls='--')

    #plt.plot(rp, test_err/wpmean)
    #plt.plot(rp, -test_err/wpmean)

    plt.xlabel(r'$r_p$ (Mpc/h)')
    plt.ylabel(r'$\Delta w_p$/$w_{p}$')

    plt.show()


def plot_error_analytic():

    fig = plt.figure()
    ### Fractional error (predic) ###
    #fn = '../CMASS_Analytic_EH/frac_rms_9bins_HOD_200_NewGaussian_' \
    #    'error_mag_1.0_version_0_preshift_0.0.dat'
    fn = '../CMASS_Analytic_EH/wp_error_frac_rms/frac_rms_cross_check_9bins_HOD_150_' \
           'NewGaussian_error_mag_1.0_version_0_preshift_0.0.dat'
    #fn = '../CMASS_Analytic_EH/wp_error_frac_rms/frac_rms_cross_check_9bins_HOD_50_NewGaussian_error_mag_0.0_version_0_preshift_0.0.dat'
    frac_rms = np.loadtxt(fn)
    frac_rms *= 100.
    mean_err = np.mean(frac_rms, axis=0)
    sig1p = mean_err + np.std(frac_rms, axis=0)
    #sig1m = mean_err - np.std(frac_rms, axis=0)
    #sig2p = mean_err + 2*np.std(frac_rms, axis=0)
    #sig2m = mean_err - 2*np.std(frac_rms, axis=0)

    #get rps from mean wp
    fnmean = "../wp_covar_results/wp_covar_mean.dat"
    rp, wpmean = np.loadtxt(fnmean, unpack=False)

    print(mean_err)
    plt.loglog(rp, sig1p, color='red', label='prediction error, 1 sigma noise')
    # plt.semilogx(rp, sig1m)
    # plt.semilogx(rp, sig2p)
    # plt.semilogx(rp, sig2m)

    plt.ylabel('fractional error (%)')
    plt.xlabel(r'$r_p$ (Mpc/h)')

    plt.xlim(0.1, 40)
    plt.ylim(0.1, 40)

    fig.savefig('plots_2019-02-06/fracerr_analytic_1sigma.png')

    plt.show()


def plot_error_sim():

    numhod = 50
    #versions = [0, 1, 2, 3, 4]
    #versions = [0]
    Versions = [5000, 400]
    #numhods = [50, 80, 100]
    version = 0

    fig = plt.figure()
    ### Fractional error ###

    #for version in versions:
    for Version in Versions:
    #for numhod in numhods:
        dir = '../CMASS_BIAS/Gaussian_Process/GP/fractional_error/'
        fn = dir+'wp_frac_rms_9bins_cos_HOD_Safe_False_{}_version_{}_7COS_' \
                 'Larger_err_False_NewKernel_False_Err_Corr_False_Version_{}' \
                 '_subsample_{}_GP_mean_True_ksf.dat'\
            .format(numhod, version, Version, numhod)
        frac_rms = np.loadtxt(fn)
        #frac_rms *= 100.
        #frac_rms = abs(frac_rms)
        mean_err = np.mean(frac_rms, axis=0)
        sig1p = mean_err + np.std(frac_rms, axis=0)
        #sig1m = mean_err - np.std(frac_rms, axis=0)
        #sig2p = mean_err + 2*np.std(frac_rms, axis=0)
        #sig2m = mean_err - 2*np.std(frac_rms, axis=0)

        #get rps from mean wp
        fnmean = "../wp_covar_results/wp_covar_mean.dat"
        rp, wpmean = np.loadtxt(fnmean, unpack=False)

        print(mean_err)
        #plt.semilogx(rp, mean_err, color='black', label='mean')
        plt.semilogx(rp, sig1p, label='Version {}, {} hods, version {}'
                     .format(Version, numhod, version))
        # plt.semilogx(rp, sig1m)
        # plt.semilogx(rp, sig2p)
        # plt.semilogx(rp, sig2m)

    sample_var, shot_noise, diffmeans = get_training_error()

    # diffmeans_mean = np.mean(diffmeans, axis=0)
    # diffmeans_sig1p = diffmeans_mean + np.std(diffmeans, axis=0)
    # print diffmeans_sig1p
    # plt.semilogx(rp, sample_var, color='blue', ls='--')
    #plt.semilogx(rp, diffmeans_sig1p, color='orange', ls='--')

    #plt.semilogx(rp, shot_noise, color='red', ls='--')




    plt.ylabel('fractional error')
    plt.xlabel(r'$r_p$ (Mpc/h)')

    plt.legend(loc='best')

    plt.xlim(0.1, 40)
    #plt.ylim(0.1, 40)

    fig.savefig('plots_2019-02-12/fracerr_sim_numhod50_versions.png')

    plt.show()


def get_training_error():
    nbins = 9
    ### Test error ###
    cosmos = range(7)
    hods = range(1)
    # hods = range(10)
    boxes = range(5)
    seeds = range(10)
    rps = []
    wps = []

    wps_grid = np.zeros((len(cosmos), len(boxes), nbins))
    shots = []
    for cosmo in cosmos:
        # wps_cosmo_avg = []
        print(cosmo)
        for box in boxes:
            wps_box_avg = np.zeros(nbins)
            for hod in hods:
                wps_hod_avg = np.zeros(nbins)
                wps_hod = []
                for seed in seeds:
                    dir = "../CMASS_BIAS/Gaussian_Process/GP/GP_test_data/Test_Box_wp_covar/"
                    fn = dir + 'wp_covar_cosmo_{}_Box_{}_HOD_{}_test_{}.dat' \
                        .format(cosmo, box, hod, seed)
                    # fn = "../results_wp_mean/wp_{}_cosmo_{}_HOD_{}_mean.dat" \
                    #    .format(tag, cosmo, hod)
                    rp, wp = np.loadtxt(fn, usecols=(0, 1), delimiter='\t', unpack=True)
                    rps.append(rp)
                    wps.append(wp)
                    wps_hod_avg += wp
                    wps_hod.append(wp)
                    # wps_cosmo.append(wp)
                # turn sum into average
                wps_hod_avg /= len(seeds)
                wps_box_avg += wps_hod_avg
                shots.append(np.var(wps_hod, axis=0))

            wps_box_avg /= len(hods)
            wps_grid[cosmo][box] += wps_box_avg
            # wps_cosmo_avg += wps_box_avg

            # wps_cosmo_avg /= len(boxes)

    diffmeans = []
    for cosmo in cosmos:
        wp_cosmo_avg = np.mean(wps_grid, axis=(0, 1))
        for box in boxes:
            diffmeans.append((wp_cosmo_avg - wps_grid[cosmo][box]) / wp_cosmo_avg)
    sample_var = np.var(diffmeans, axis=0)
    print(sample_var)

    print(shots)
    shot_noise = np.mean(shots, axis=0)
    print(shot_noise)

    return sample_var, shot_noise, diffmeans

def plot_training(statistic, res_dir, data_dir, errtag='', subsample=None, version=None, nbins=9, test=False):
    plt.figure(figsize=(10,8)) 
    ps = []

    CC = range(0, 40)
    #CC = range(0,1)
    #HH = np.loadtxt("../CMASS/Gaussian_Process/GP/HOD_random_subsample_{}_version_{}.dat".format(subsample, version))
    #HH = np.atleast_2d(HH[0][:3])
    nhodnonolap = 100
    nhodpercosmo = 100
    #nhodpercosmo = 1
    HH = np.array(range(0,len(CC)*nhodnonolap))
    HH  = HH.reshape(len(CC), nhodnonolap)
    HH = HH[:,0:nhodpercosmo]
    
    if errtag:
        GP_error = np.loadtxt(f"{res_dir}/{statistic}_error{errtag}.dat")
    
    #color_idx = np.linspace(0, 1, np.max(HH)+1)
    color_idx = np.linspace(0, 1, len(CC))

    for cosmo in CC:
        HH_set = HH[cosmo]
        for hod in HH_set:
            zz = np.random.randint(len(HH.flatten()))
            hod = int(hod)
            color=plt.cm.rainbow(color_idx[cosmo])
            if test:
                for box in range(0,5):
                    fn = '{}/{}_cosmo_{}_Box_{}_HOD_{}_test_0.dat'.format(data_dir, statistic, cosmo, box, hod)
                    r, p = np.loadtxt(fn, delimiter=',',unpack=True)
                    if errtag:
                        plt.errorbar(r[:nbins], p[:nbins], yerr=GP_error[:nbins], lw=0.5, elinewidth=1, capsize=1, color=color, 
                                     zorder=zz)
                    else:
                        plt.plot(r[:nbins], p[:nbins], color=color, lw=0.5, zorder=zz)
            else:
                fn = '{}/{}_cosmo_{}_HOD_{}_test_0.dat'.format(data_dir, statistic, cosmo, hod)
                r, p = np.loadtxt(fn, delimiter=',',unpack=True)
                if errtag:
                    plt.errorbar(r[:nbins], p[:nbins], yerr=GP_error[:nbins], lw=0.5, elinewidth=1, capsize=1, color=color, 
                                 zorder=zz)
                else:
                    plt.plot(r[:nbins], p[:nbins], color=color, lw=0.5, zorder=zz)
                
    plt.yscale("log")
    plt.xlabel("r (Mpc/h)") #is it? are positions in Mpc? not h?
    
    if statistic == 'upf':
        plt.ylabel(r"P$_U$(r)")
        
    elif statistic == 'wp':
        plt.ylabel(r'$w_p$($r_p$)')
        plt.xscale('log')
    elif statistic == 'mcf':
        plt.ylabel(r'$M$(r)')
   

def plot_testing(statistic, testtag, errtag='', nbins=9, onehod=None, nboxes=5):
    plt.figure(figsize=(10,8)) 
    ax = plt.gca()

    ncosmos = 7
    CC_test = range(0, ncosmos)
    if onehod is not None:
        HH_test = [onehod]
        color_idx = np.linspace(0, 1, ncosmos)
    else:
        HH_test = range(0, 100)
        color_idx = np.linspace(0, 1, len(HH_test))

    res_dir = '../../clust/results_{}/'.format(statistic)
    if errtag:
        GP_error = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
   
    print(HH_test) 
    boxes = range(nboxes)

    if onehod is not None:
        colidx = 0
    for cosmo in CC_test:
        if onehod is None:
            colidx = 0
        for hod in HH_test:
            for box in boxes:
                hod = int(hod)
                
                color=plt.cm.rainbow(color_idx[colidx])

                idtag = '{}_cosmo_{}_Box_{}_HOD_{}_test_0.dat'.format(statistic, cosmo, box, hod)
                fnt = '{}testing_{}{}/{}'.format(res_dir, statistic, testtag, idtag)
                #fnt = '../testing_results/tests_{}{}/{}.dat'.format(statistic, acctag, idtag)

                ntest, ptest = np.loadtxt(fnt, delimiter=',', unpack=True)
                if errtag:
                    plt.errorbar(ntest[:nbins], ptest[:nbins], yerr=GP_error[:nbins], lw=0.5, elinewidth=1, capsize=1, color=color)
                else:
                    plt.plot(ntest[:nbins], ptest[:nbins], color=color, lw=1)
            
            if not onehod:
                colidx += 1
        
        if onehod:
            colidx += 1
                   
    plt.yscale("log")
    plt.xlabel(r"r ($h^{-1}$Mpc)") #is it? are positions in Mpc? not h?
    ax.legend()
    
    if statistic == 'upf':
        plt.ylabel(r"P$_U$(r)")
    elif statistic == 'wp':
        plt.ylabel(r'$w_p$($r_p$)')
        plt.xscale('log')
    elif statistic == 'mcf':
        plt.ylabel(r'$M$(r)')
        plt.yscale("linear")

if __name__=="__main__":
    main()
