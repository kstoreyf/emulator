from matplotlib import pyplot as plt
import numpy as np
import h5py
import pickle
from getdist import plots, MCSamples

import utils
from chain_variables import *

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

def plot_training(statistic, res_dir, data_dir, errtag='', subsample=None, version=None, nbins=9, test=False, xrsq=False):
    plt.figure(figsize=(10,8)) 
    ps = []

    CC = range(0, 40)
    nhodnonolap = 100
    nhodpercosmo = 100
    HH = np.array(range(0,len(CC)*nhodnonolap))
    HH  = HH.reshape(len(CC), nhodnonolap)
    HH = HH[:,0:nhodpercosmo]
    
    if errtag:
        gperr = np.loadtxt(f"{res_dir}/{statistic}_error{errtag}.dat")
    
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
                    if xrsq:
                        pplot = r[:nbins]**2 * p[:nbins]
                    else:
                        pplot = p[:nbins]
                    if errtag:
                        plt.errorbar(r[:nbins], pplot, yerr=gperr[:nbins], lw=0.5, elinewidth=1, capsize=1, color=color, zorder=zz)
                    else:
                        plt.plot(r[:nbins], pplot, color=color, lw=0.5, zorder=zz)
            else:
                fn = '{}/{}_cosmo_{}_HOD_{}_test_0.dat'.format(data_dir, statistic, cosmo, hod)
                r, p = np.loadtxt(fn, delimiter=',',unpack=True)
                if xrsq:
                    pplot = r[:nbins]**2 * p[:nbins]
                else:
                    pplot = p[:nbins]
                if errtag:
                    plt.errorbar(r[:nbins], pplot, yerr=gperr[:nbins], lw=0.5, elinewidth=1, capsize=1, color=color, zorder=zz)
                else:
                    plt.plot(r[:nbins], pplot, color=color, lw=0.5, zorder=zz)
    
    plt.xlabel(r"r ($h^{-1}$Mpc)")
    stat_labels = {'upf':r"P$_U$(r)", 'wp':r'$w_p$($r_p$)', 'mcf':"M(r)", 'xi':r"$\xi_0$(r)", 'xi2':r"$\xi_2$(r)"}    
    if xrsq:
        plt.ylabel(r'r$^2$'+stat_labels[statistic])
    else:
        plt.ylabel(stat_labels[statistic])
    if statistic in ['wp', 'upf', 'xi']:
        plt.yscale("log")
    if statistic in ['wp', 'mcf', 'xi', 'xi2']:
        plt.xscale('log')


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
        gperr = np.loadtxt(res_dir+"{}_error{}.dat".format(statistic, errtag))
   
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

                rtest, ptest = np.loadtxt(fnt, delimiter=',', unpack=True)
                if errtag:
                    plt.errorbar(rtest[:nbins], ptest[:nbins], yerr=gperr[:nbins], lw=0.5, elinewidth=1, capsize=1, color=color)
                else:
                    plt.plot(rtest[:nbins], ptest[:nbins], color=color, lw=1)
            
            if not onehod:
                colidx += 1
        
        if onehod:
            colidx += 1
                   
    ax.legend()
    plt.xlabel(r"r ($h^{-1}$Mpc)")
    stat_labels = {'upf':r"P$_U$(r)", 'wp':r'$w_p$($r_p$)', 'mcf':"M(r)", 'xi':r"$\xi_0$(r)", 'xi2':r"$\xi_2$(r)"}
    if xrsq:
        plt.ylabel(r'r$^2$'+stat_labels[statistic])
    else:
        plt.ylabel(stat_labels[statistic])
    if statistic in ['wp', 'upf', 'xi']:
        plt.yscale("log")
    if statistic in ['wp', 'mcf', 'xi', 'xi2']:
        plt.xscale('log')


def plot_accuracy(statistic, testtag, acctag, errtag, hod=None, nbins=9, nhods=100, 
                    xrsq=False, err_as_percentiles=False, sample_var='aemulus'):
    
    ncols = 3
    fig, ax = plt.subplots(ncols, 1, figsize=(10,15), gridspec_kw={'height_ratios': [1]*ncols})
    plt.subplots_adjust(hspace=0.15)

    CC_test = range(0, 7)
    HH_test = range(0, nhods)
    res_dir = '../../clust/results_{}/'.format(statistic)

    stat_mean = np.zeros(nbins)
        
    if sample_var=='minerva':
        minerva_dir = f'../../clust/covariances'
        cov_minerva = np.loadtxt(f"{minerva_dir}/cov_minerva_{statistic}.dat")
        L_minerva = 1.5 #Gpc
        L_aemulus = 1.05 #Gpc
        cov_test = cov_minerva*(L_minerva/L_aemulus)**3 
    elif sample_var=='aemulus':
        cov_test = np.loadtxt("../../clust/covariances/cov_aemulus_{}{}.dat".format(statistic, errtag))
    
    #if 'mean' in testtag:
    #     cov_test *= 1./5. 
    err_test = np.diag(cov_test)

    for cosmo in CC_test:
        for hod in HH_test:
            hod = int(hod)
            if "mean" in acctag:
                idtag = '{}_cosmo_{}_HOD_{}_mean'.format(statistic, cosmo, hod)
            else:
                idtag = '{}_cosmo_{}_Box_0_HOD_{}_test_0'.format(statistic, cosmo, hod)
            fnt = '{}testing_{}{}/{}.dat'.format(res_dir, statistic, testtag, idtag)

            rtest, ptest = np.loadtxt(fnt)
            #rtest, ptest = np.loadtxt(fnt, delimiter=',', unpack=True)
            stat_mean += ptest[:nbins]
        
    stat_mean /= len(CC_test)*len(HH_test)
    color_idx = np.linspace(0, 1, len(HH_test))

    i = 0
    fracerrs = []

    for cosmo in CC_test:
        colidx = 0
        for hod in HH_test:
            hod = int(hod)
            color=plt.cm.rainbow(color_idx[colidx])
            colidx += 1
            if "mean" in acctag:
                idtag = '{}_cosmo_{}_HOD_{}_mean'.format(statistic, cosmo, hod)
            else:
                idtag = '{}_cosmo_{}_Box_0_HOD_{}_test_0'.format(statistic, cosmo, hod)
            
            lw = 1
            alpha = 1
            fnt = '{}testing_{}{}/{}.dat'.format(res_dir, statistic, testtag, idtag)        
            rtest, ptest = np.loadtxt(fnt)
            if xrsq:
                ptplot = rtest[:nbins]**2 * ptest[:nbins]
            else:
                ptplot = ptest[:nbins]
            #rtest, ptest = np.loadtxt(fnt, delimiter=',', unpack=True)
            if i==0:
                ax[0].plot(rtest[:nbins], ptplot, marker='o', ls='None', markerfacecolor='None', 
                               markeredgecolor=color, lw=lw, alpha=alpha, label='truth')
            else:
                ax[0].plot(rtest[:nbins], ptplot, marker='o', ls='None', markerfacecolor='None', 
                               markeredgecolor=color, lw=lw, alpha=alpha)

            
            fnp = '../testing_results/predictions_{}{}/{}.dat'.format(statistic, acctag, idtag)
            rpredic, ppredic = np.loadtxt(fnp, delimiter=',', unpack=True)
            if xrsq:
                ppplot = rpredic[:nbins]**2 * ppredic[:nbins]
            else:
                ppplot = ppredic[:nbins]
            if i==0:
                ax[0].plot(rpredic[:nbins], ppplot, marker=None, ls='-', color=color, lw=lw, 
                               alpha=alpha, label='emulator prediction')
            else: 
                ax[0].plot(rpredic[:nbins], ppplot, marker=None, ls='-', color=color, lw=lw, 
                               alpha=alpha)
            

            fracerr = (ppredic-ptest)/ptest
            #print(fracerr)
            if fracerr[5]>10:
                print(fracerr)
                #print(ptest)
                #print(ppredic)
                continue
            fracerrs.append(fracerr)
            ax[1].plot(rtest[:nbins], fracerr[:nbins], color=color, lw=lw, alpha=alpha)
            i += 1

    ax[2].axhline(0, color='k', ls=':')
    print(np.sqrt(np.diag(utils.covariance(fracerrs, zeromean=True))))
    print(np.sqrt(np.diag(utils.covariance(fracerrs, zeromean=False))))
    print(np.std(fracerrs, axis=0)) 
    #cov_fracerrs = utils.covariance(fracerrs, zeromean=True)
    #err_fracerrs = np.diag(cov_fracerrs)
    err_fracerrs = np.var(fracerrs, axis=0)
    ax[2].plot(rtest[:nbins], np.sqrt(err_fracerrs[:nbins]), color='b', lw=2, ls='-', label='error (RMS of fractional error)')
    ax[2].plot(rtest[:nbins], np.sqrt(err_test[:nbins]), color='r', lw=2, label='sample variance')

    ax[2].set_ylabel("error")
    ax[2].legend()
    
    ax[0].legend()
    
    ax[2].set_xlabel(r"r ($h^{-1}$Mpc)")
    stat_labels = {'upf':r"P$_U$(r)", 'wp':r'$w_p$($r_p$)', 'mcf':"M(r)", 'xi':r"$\xi_0$(r)", 'xi2':r"$\xi_2$(r)"}
    if xrsq:
        ax[0].set_ylabel(r'r$^2$'+stat_labels[statistic])
    else:
        ax[0].set_ylabel(stat_labels[statistic])
    if statistic in ['wp', 'upf', 'xi']:
        ax[0].set_yscale('log')
    if statistic in ['wp', 'mcf', 'xi', 'xi2']:
        for nc in range(ncols):
            ax[nc].set_xscale('log')
              
    ax[1].set_ylabel("fractional error")

    return fracerrs


def plot_accuracy_fromparams(statistic, testtag, acctag, errtag, params_zipped, hod=None, nbins=9, nhods=100, 
                    xrsq=False, err_as_percentiles=False, sample_var='aemulus'):
    

    ncols = 3
    fig, ax = plt.subplots(ncols, 1, figsize=(10,15), gridspec_kw={'height_ratios': [1]*ncols})
    plt.subplots_adjust(hspace=0.15)

    res_dir = '../../clust/results_{}/'.format(statistic)

    stat_mean = np.zeros(nbins)
        
    if sample_var=='minerva':
        minerva_dir = f'../../clust/covariances'
        cov_minerva = np.loadtxt(f"{minerva_dir}/cov_minerva_{statistic}.dat")
        L_minerva = 1.5 #Gpc
        L_aemulus = 1.05 #Gpc
        cov_test = cov_minerva*(L_minerva/L_aemulus)**3 
    elif sample_var=='aemulus':
        cov_test = np.loadtxt("../../clust/covariances/cov_aemulus_{}{}.dat".format(statistic, errtag))
    
    if 'mean' in testtag:
         cov_test *= 1./5. 
    err_test = np.diag(cov_test)

    for (cosmo, hod) in params_zipped:
        hod = int(hod)
        if "mean" in acctag:
            idtag = '{}_cosmo_{}_HOD_{}_mean'.format(statistic, cosmo, hod)
        else:
            idtag = '{}_cosmo_{}_Box_0_HOD_{}_test_0'.format(statistic, cosmo, hod)
        fnt = '{}testing_{}{}/{}.dat'.format(res_dir, statistic, testtag, idtag)

        rtest, ptest = np.loadtxt(fnt)
        #rtest, ptest = np.loadtxt(fnt, delimiter=',', unpack=True)
        stat_mean += ptest[:nbins]
        
    stat_mean /= len(list(params_zipped))
    #color_idx = np.linspace(0, 1, len(set(params_zipped[:,0])))
    color_idx = np.linspace(0, 1, len(list(params_zipped)))

    i = 0
    fracerrs = []
    colidx = 0
    for cosmo, hod in params_zipped:
        #colidx = cosmo
        #color=plt.cm.rainbow(color_idx[colidx])
        color=plt.cm.rainbow(color_idx[colidx])
        colidx += 1
        if "mean" in acctag:
            idtag = '{}_cosmo_{}_HOD_{}_mean'.format(statistic, cosmo, hod)
        else:
            idtag = '{}_cosmo_{}_Box_0_HOD_{}_test_0'.format(statistic, cosmo, hod)
        
        lw = 1
        alpha = 1
        fnt = '{}testing_{}{}/{}.dat'.format(res_dir, statistic, testtag, idtag)        
        rtest, ptest = np.loadtxt(fnt)
        if xrsq:
            ptplot = rtest[:nbins]**2 * ptest[:nbins]
        else:
            ptplot = ptest[:nbins]
        #rtest, ptest = np.loadtxt(fnt, delimiter=',', unpack=True)
        if i==0:
            ax[0].plot(rtest[:nbins], ptplot, marker='o', ls='None', markerfacecolor='None', 
                            markeredgecolor=color, lw=lw, alpha=alpha, label='truth')
        else:
            ax[0].plot(rtest[:nbins], ptplot, marker='o', ls='None', markerfacecolor='None', 
                            markeredgecolor=color, lw=lw, alpha=alpha)

        
        fnp = '../testing_results/predictions_{}{}/{}.dat'.format(statistic, acctag, idtag)
        rpredic, ppredic = np.loadtxt(fnp, delimiter=',', unpack=True)
        if xrsq:
            ppplot = rpredic[:nbins]**2 * ppredic[:nbins]
        else:
            ppplot = ppredic[:nbins]
        if i==0:
            ax[0].plot(rpredic[:nbins], ppplot, marker=None, ls='-', color=color, lw=lw, 
                            alpha=alpha, label='emulator prediction')
        else: 
            ax[0].plot(rpredic[:nbins], ppplot, marker=None, ls='-', color=color, lw=lw, 
                            alpha=alpha)
        

        fracerr = (ppredic-ptest)/ptest
        fracerrs.append(fracerr)
        ax[1].plot(rtest[:nbins], fracerr[:nbins], color=color, lw=lw, alpha=alpha)
        i += 1

        if fracerr[-2]>10:
            print(cosmo, hod, fracerr)

    ax[2].axhline(0, color='k', ls=':')
    
    cov_fracerrs = utils.covariance(fracerrs, zeromean=True)
    err_fracerrs = np.diag(cov_fracerrs)
    ax[2].plot(rtest[:nbins], np.sqrt(err_fracerrs[:nbins]), color='b', lw=2, ls='-', label='error (RMS of fractional error)')
    ax[2].plot(rtest[:nbins], np.sqrt(err_test[:nbins]), color='r', lw=2, label='sample variance')

    ax[2].set_ylabel("error")
    ax[2].legend()
    
    ax[0].legend()
    
    ax[2].set_xlabel(r"r ($h^{-1}$Mpc)")
    stat_labels = {'upf':r"P$_U$(r)", 'wp':r'$w_p$($r_p$)', 'mcf':"M(r)", 'xi':r"$\xi_0$(r)", 'xi2':r"$\xi_2$(r)"}
    if xrsq:
        ax[0].set_ylabel(r'r$^2$'+stat_labels[statistic])
    else:
        ax[0].set_ylabel(stat_labels[statistic])
    if statistic in ['wp', 'upf', 'xi']:
        ax[0].set_yscale('log')
    if statistic in ['wp', 'mcf', 'xi', 'xi2']:
        for nc in range(ncols):
            ax[nc].set_xscale('log')
              
    ax[1].set_ylabel("fractional error")

    return fracerrs


def compare_emulators(statistic, testtags, acctags, errtag, savetags, labels=None, subsample=None, nbins=9, remove=None,
                 nhods=100):
    
    if labels==None:
        labels = acctags
    
    ncols = 2
    fig, ax = plt.subplots(ncols, 1, figsize=(10,10), gridspec_kw={'height_ratios': [1]*ncols})

    if statistic == 'upf':
        plt.title('UPF')
    elif statistic == 'wp':
        plt.title(r'$w_p$($r_p$)')
    
    CC_test = range(0, 7)
    HH_test = range(0, nhods)
    if remove:
        for rval in remove:
            #HH_test.remove(rval)
            CC_test.remove(rval)
 
    upf_mean = np.zeros(nbins)
    
    res_dir = '../../clust/results_{}/'.format(statistic)

    Nemu = len(testtags)
    color_idx = np.linspace(0, 1, Nemu)

    for ee in range(Nemu):
        testtag = testtags[ee]
        acctag = acctags[ee]
        color=plt.cm.cool(color_idx[ee])
    
        i = 0
        fracerrs = []

        for cosmo in CC_test:
            for hod in HH_test:
                hod = int(hod)
                if "mean" in acctag:
                    idtag = '{}_cosmo_{}_HOD_{}_mean'.format(statistic, cosmo, hod)
                else:
                    idtag = '{}_cosmo_{}_Box_0_HOD_{}_test_0'.format(statistic, cosmo, hod)
                fnt = '{}testing_{}{}/{}.dat'.format(res_dir, statistic, testtag, idtag)
                
                lw =1
                rtest, ptest = np.loadtxt(fnt)

                fnp = '../testing_results/predictions_{}{}/{}.dat'.format(statistic, acctag, idtag)
                rpredic, ppredic = np.loadtxt(fnp, delimiter=',', unpack=True)

                fracerr = (ppredic-ptest)/ptest
                fracerrs.append(fracerr)

                i += 1

        fracerrs = np.array(fracerrs)

        std = np.std(fracerrs, axis=0)
        p16 = np.percentile(fracerrs, 16, axis=0)
        p84 = np.percentile(fracerrs, 84, axis=0)
        pavg = (np.percentile(fracerrs, 84, axis=0)-np.percentile(fracerrs, 16, axis=0))/2.0
        p68 = np.percentile(fracerrs, 68, axis=0)

        ax[0].plot(rtest[:nbins], std[:nbins], color=color, ls='-', label=labels[ee])
        ax[1].plot(rtest[:nbins], p16[:nbins], color=color, ls='-')
        ax[1].plot(rtest[:nbins], p84[:nbins], color=color, ls='-')
        #ax[1].plot(rtest[:nbins], p68[:nbins], color='limegreen', ls='-', label='p68')
        #ax[1].plot(rtest[:nbins], pavg[:nbins], color='orange', ls='-', label='pavg')

    # if multiple savetags, aka they are different:
    single_savetag = False
    if type(savetags) is str:
        savetags = [savetags]
        single_savetag = True
    
    #lss = ['-', '--', ':']
    for ee, savetag in enumerate(savetags):
        stat_str = statistic
        err_str = errtag
        cov_dir = "../../clust/covariances/"
        #gperr = np.loadtxt(cov_dir+"error_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))
        gpcov = np.loadtxt(cov_dir+"cov_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))
        gperr = np.sqrt(1./5.*np.diag(gpcov))
        gpp16 = np.loadtxt(cov_dir+"p16_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))
        gpp84 = np.loadtxt(cov_dir+"p84_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))

        color=plt.cm.cool(color_idx[ee])
        ls = '--'
        if single_savetag:
            color='r'
            ls='-'
        ax[0].plot(rtest[:nbins], gperr[:nbins], color=color, ls=ls, label=f'Aemulus error, {savetag[1:]}')
        ax[1].plot(rtest[:nbins], gpp16[:nbins], color=color, ls=ls)
        ax[1].plot(rtest[:nbins], gpp84[:nbins], color=color, ls=ls)
        ax[1].axhline(0, color='k', ls=':')

        ax[0].set_ylabel("fractional error (std)")
        ax[1].set_ylabel("fractional error (16/84 percentile)")

    ax[0].legend()
    
    plt.xlabel(r"r ($h^{-1}$Mpc)")
    if statistic in ['wp', 'mcf', 'xi', 'xi2']:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')


def plot_autocorr(chain, color='steelblue'):
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
    ndim = chain.shape[2]
    auto = np.empty((len(N), ndim))
    for i, n in enumerate(N):
        for d in range(ndim):
            auto[i][d] = utils.autocorr_new(chain[:, :n, d])

    # Plot the comparisons
    for d in range(ndim):
        plt.loglog(N, auto, "o-", color=color)
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    #plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)


def plot_contours_dynesty(chaintags, legend_labels=None, params_toplot=None, colors=None, 
                  legend_loc='upper center', legend_fontsize=20, 
                  vertical_markers=None, vertical_marker_color='grey'): 
    bounds = utils.get_emulator_bounds()
    sample_arr = []
    for i, chaintag in enumerate(chaintags):

        chain_fn = f'../chains/chains_{chaintag}.h5'
        fw = h5py.File(chain_fn, 'r')
        param_names = fw.attrs['param_names']
        if vertical_markers is None:
            vertical_markers_toplot = fw.attrs['true_values']
        else:
            vertical_markers_toplot = vertical_markers
        fw.close()
        
        pickle_fn = f'{pickle_dir}/results_{chaintag}.pkl'
        with open(pickle_fn, 'rb') as pf:
            res = pickle.load(pf)
            samples = res['samples']
            lnweight = np.array(res['logwt'])
            lnevidence = np.array(res['logz'])

        if params_toplot is not None:    
            idxs = []
            for pm in params_toplot:
                idxs.append(np.where(param_names == pm))
            idxs = np.array(idxs).flatten()
            samples = samples[:,idxs]
            param_names = params_toplot
            vertical_markers_toplot = vertical_markers_toplot[idxs]
        
        labels = [param_labels[pn] for pn in param_names]
        ranges = [bounds[pn] for pn in param_names]

        #[-1] bc just care about final evidence value
        weights = np.exp(lnweight - lnevidence[-1]) 
        weights = weights.flatten()

        samps = MCSamples(names=param_names, labels=labels)
        samps.setSamples(samples, weights=weights)
        sample_arr.append(samps)

    marker_args = {'color': vertical_marker_color}
        
    g = plots.get_subplot_plotter()
    g.settings.alpha_filled_add=0.4
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = legend_fontsize
    g.settings.axis_marker_lw = 1.0
    g.settings.axis_marker_color = 'dimgrey'
    g.triangle_plot(sample_arr, filled=True, contour_colors=colors, names=param_names,
                   legend_labels=legend_labels, markers=vertical_markers_toplot, title_limit=1, legend_loc=legend_loc,
                    marker_args=marker_args, axis_marker_color='red')
    return g


def plot_fits(statistics, vals_arr_all, vals_true_all, gperrs, fit_labels, fit_colors, title=''):
    lss = ['-', '--', ':', '-.']
    vals_arr_all = np.array(vals_arr_all)
    vals_true_all = np.array(vals_true_all)
    
    for i, statistic in enumerate(statistics):
        fig, axarr = plt.subplots(2, 1, figsize=(7,7), sharex=True)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax0, ax1 = axarr
        
        
        ax0.plot(r_dict[statistic], vals_true_all[i], label='Truth', marker='o', ls='None', color='black')
        #ax0.errorbar(r_dict[statistic], vals_true_all[i], yerr=10*gperrs[i]*vals_true_all[i], #fmt='o')
        #             label='Truth', ls='None', color='black')
        
        for j in range(vals_arr_all.shape[1]):
            ax0.plot(r_dict[statistic], vals_arr_all[i][j], label=fit_labels[j], ls=lss[j], color=fit_colors[j])
            frac = (vals_arr_all[i][j] - vals_true_all[i])/vals_true_all[i]
            #frac = (vals_arr_all[i][j] - vals_true_all[i])/(gperrs[i]*vals_true_all[i])
            ax1.plot(r_dict[statistic], frac, label=fit_labels[j], ls=lss[j], color=fit_colors[j])
            
        # dummy to get this into legend
        ax0.plot(0, 0, ls='-', marker='None', label='Aemulus 1-box sample variance', color='plum')
        ax1.plot(r_dict[statistic], gperrs[i], label='Aemulus 1-box sample variance', ls='-', color='plum')
        
            
        ax0.set_xscale(scale_dict[statistic][0])
        ax0.set_yscale(scale_dict[statistic][1])
        ax0.set_ylabel(statistic)
        ax0.legend(fontsize=10, loc='best')
        
        ax1.axhline(0, color='darkgrey')
        ax1.set_xscale(scale_dict[statistic][0])
        ax1.set_xlabel(r"$r$")
        ax1.set_ylabel("fractional error")


def plot_contours(chaintags, legend_labels=None, params_toplot=None, nsteps=None, colors=None, 
                  legend_loc='upper center', legend_fontsize=20, weight_with_dynesty=False): 
   
    bounds = utils.get_emulator_bounds() 
    sample_arr = []
    for i, chaintag in enumerate(chaintags):
        
        resample_chains = False
        if 'dynesty' in chaintag:
            resample_chains = True
            assert not (nsteps!=None and resample_chains), "if resampling, shouldn't be selecting nsteps!"
    
        chain_fn = f'../chains/chains_{chaintag}.h5'
        fw = h5py.File(chain_fn, 'r')

        chain_dset = fw['chain']
        chain = np.array(chain_dset)
        print(chaintag, ':', chain.shape)
        if nsteps:
            chain = chain[:,:nsteps,:]
   
        lnprob_dset = fw['lnprob']  
        param_names = fw.attrs['param_names']
        true_values = fw.attrs['true_values']

        if params_toplot is not None:    
            idxs = []
            for pm in params_toplot:
                idxs.append(np.where(param_names == pm))
            idxs = np.array(idxs).flatten()
            chain = chain[:,:,idxs]
            param_names = params_toplot
            true_values = true_values[idxs]

        nwalkers, nchain, ndim = chain.shape
        
        samples = chain.reshape(-1, chain.shape[-1])
        labels = [param_labels[pn] for pn in param_names]
        ranges = [bounds[pn] for pn in param_names]

        if resample_chains:
            lnweight_dset = fw['lnweight']
            lnevidence_dset = fw['lnevidence']
            #[-1] bc just care about final evidence value
            lnweight = np.array(lnweight_dset)[0] #[0] bc an extra array dim
            lnevidence = np.array(lnevidence_dset)[0]
            weights = np.exp(lnweight - lnevidence[-1]) 
            weights = weights.flatten()
            
            samples = np.empty((nchain, ndim))
            for nd in range(ndim):
                cn = chain[:,:,nd].flatten()
                if weight_with_dynesty:
                    samples[:,nd] = dynesty.utils.resample_equal(cn, weights)
                else:
                    samples[:,nd] = cn
        else:
            weights=None
            
        fw.close()
        
        if weight_with_dynesty:
            weights = None
        
        # have not gotten ranges to work
        samps = MCSamples(names=param_names, labels=labels, ranges=ranges)
        # for some reason get slightly diff answer if don't use setSamples and pass them straight to MCSamples!
        samps.setSamples(samples, weights=weights)
        sample_arr.append(samps)

    g = plots.get_subplot_plotter()
    g.settings.alpha_filled_add=0.4
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = legend_fontsize
    g.settings.axis_marker_lw = 1.0
    g.settings.axis_marker_color = 'dimgrey'
    g.triangle_plot(sample_arr, filled=True, contour_colors=colors, names=param_names,
                   legend_labels=legend_labels, markers=true_values, title_limit=1, legend_loc=legend_loc,
                   #marker_args={'color': 'orange', 'lw':1.5}
                   )
    return g


def plot_correlation_matrix(corr, statistics):
    
    nstats = len(statistics)
    plt.figure(figsize=(2.5*nstats,2.5*nstats))
    tick_labels = np.concatenate([np.round(r_dict[stat], 2) for stat in statistics])
    im = plt.imshow(corr, origin='lower left', cmap='bwr_r', vmin=-1, vmax=1)
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=90)
    plt.yticks(ticks=range(len(tick_labels)), labels=tick_labels)

    # Label statistics
    for i, statistic in enumerate(statistics):
        plt.text(9*i+3.5, -6, statistic, fontsize=20)
        plt.text(-6, 9*i+3.5, statistic, fontsize=20, rotation=90)
        if i==0:
            continue
        plt.axvline(9*i-0.5, color='k')
        plt.axhline(9*i-0.5, color='k')

    plt.xlabel(r"r ($h^{-1}$Mpc)", labelpad=40)
    plt.ylabel(r"r ($h^{-1}$Mpc)", labelpad=40)

    plt.colorbar(im, fraction=0.046, pad=0.04)



if __name__=="__main__":
    main()
