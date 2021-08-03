import numpy as np
from astropy.convolution import convolve, Box2DKernel


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
    cov_smooth_fn = f"{cov_dir}/cov_smooth_{cov_tag}_{stat_str}{tag_str}.dat"

    cov = np.loadtxt(cov_fn)
    print(cov)
    print(f"Smoothing {cov_fn}...")
    corr = utils.reduced_covariance(cov)
    print(corr)
    corr_convolved = smooth_corr(corr, statistics)
    # normalize corr
    print(corr_convolved)
    corr_convolved_norm = utils.reduced_covariance(corr_convolved)
    print(corr_convolved_norm)
    cov_smooth = utils.correlation_to_covariance(corr_convolved_norm, cov)
    print(cov_smooth)
    np.savetxt(cov_smooth_fn, cov_smooth)
    print(f"Successfully saved to {cov_smooth_fn}!")
    
    icov_fn = f"{cov_dir}/icov_smooth_{cov_tag}_{stat_str}{tag_str}.dat"
    #icov = np.linalg.inv(corr_convolved)
    icov = compute_icov_norm(corr_convolved)
    np.savetxt(icov_fn, icov)
    print(f"Successfully saved smooth icov to {icov_fn}!")

def compute_icov_norm(cov):
    corr = utils.reduced_covariance(cov)
    u1, s1, vh1 = np.linalg.svd(corr)
    v1 = vh1.T
    s1_inv = 1./s1
    s1_new = np.diag(s1_inv)
    icov = np.dot(v1, np.dot(s1_new, u1.T)) # inverse matrix
    return icov


def smooth_corr(corr, statistics, nbins=9, width=3):
    nstats = len(statistics)
    kernel = Box2DKernel(width=width)
    corr_convolved = np.zeros_like(corr)
    for i in range(nstats):
        for j in range(nstats):
            corr_sub = corr[i*nbins:(i+1)*nbins, j*nbins:(j+1)*nbins]
            corr_sub_convolved = convolve(corr_sub, kernel, boundary='extend')
            corr_convolved[i*nbins:(i+1)*nbins, j*nbins:(j+1)*nbins] = corr_sub_convolved
    return corr_convolved


if __name__=='__main__':
    main()
