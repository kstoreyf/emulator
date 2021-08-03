import numpy as np
from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel


import utils


def main():
    
    #statistics = ['wp']
    #statistics = ['wp', 'xi']
    #statistics = ['wp', 'xi', 'upf']
    statistics = ['wp', 'xi', 'upf', 'mcf']
    cov_tag = 'emuperf'
    tag_str = '_nonolap_hod3_test0_mean_test0'

    cov_dir = '/home/users/ksf293/clust/covariances'    
    stat_str = '_'.join(statistics)
    cov_fn = f"{cov_dir}/cov_{cov_tag}_{stat_str}{tag_str}.dat"
    cov_smooth_fn = f"{cov_dir}/cov_smoothgauss1_{cov_tag}_{stat_str}{tag_str}.dat"

    cov = np.loadtxt(cov_fn)
    print(cov)
    print(f"Smoothing {cov_fn}...")
    corr = utils.reduced_covariance(cov)
    print(corr)
    corr_convolved = smooth_corr_gaussian(corr, statistics, width=1)
    # normalize corr
    print(corr_convolved)
    corr_convolved_norm = utils.reduced_covariance(corr_convolved)
    print(corr_convolved_norm)
    cov_smooth = utils.correlation_to_covariance(corr_convolved_norm, cov)
    print(cov_smooth)
    np.savetxt(cov_smooth_fn, cov_smooth)
    print(f"Successfully saved to {cov_smooth_fn}!")
    


def smooth_corr_gaussian(corr_orig, statistics, nbins=9, width=2):
    corr = np.copy(corr_orig)
    
    # replace diags with avg of 4 neighbors
    nstats = len(statistics)
    for ii in range(nstats*nbins):
        if ii==0:
            corr[ii,ii] = np.mean([corr[ii,ii+1], corr[ii+1,ii]])
        elif ii==nstats*nbins-1:
            corr[ii,ii] = np.mean([corr[ii,ii-1], corr[ii-1,ii]])
        else:
            corr[ii,ii] = np.mean([corr[ii,ii+1], corr[ii,ii-1], corr[ii+1,ii], corr[ii-1,ii]])
    
    # smooth with Gaussian kernel
    kernel = Gaussian2DKernel(width)
    #corr_convolved = np.zeros_like(corr)
    for i in range(nstats):
        for j in range(nstats):
            corr_sub = corr[i*nbins:(i+1)*nbins, j*nbins:(j+1)*nbins]
            corr_sub_convolved = convolve(corr_sub, kernel, boundary='extend')
            corr[i*nbins:(i+1)*nbins, j*nbins:(j+1)*nbins] = corr_sub_convolved
            
    # replace diags
    diags_orig = np.diag(corr_orig)
    for ii in range(nstats*nbins):
        corr[ii,ii] = diags_orig[ii]
    return corr


if __name__=='__main__':
    main()
