import numpy as np

import utils


def main():
    
    #statistics = ['wp']
    statistics = ['wp', 'xi']
    #statistics = ['wp', 'xi', 'upf']
    #statistics = ['wp', 'xi', 'upf', 'mcf']
    #cov_tag = 'glam'
    #Nmocks = 986
    #tag_str = ''
    #cov_tag = 'aemulus'
    #tag_str = '_hod3_test0'
    #Nmocks = 35
    #cov_tag = 'emuaem'
    #Nmocks = 700
    #tag_str = ''
    #cov_tag = 'final'
    cov_tag = 'glamemudiagnoB'
    Nmocks = 986 # the smaller of 700*5 (c_emu) and 986 (c_glam) to be conservative
    tag_str = ''

    cov_dir = '/home/users/ksf293/clust/covariances'    
    stat_str = '_'.join(statistics)
    cov_fn = f"{cov_dir}/cov_{cov_tag}_{stat_str}{tag_str}.dat"
    icov_svd_fn = f"{cov_dir}/icov_svd_{cov_tag}_{stat_str}{tag_str}.dat"

    cov = np.loadtxt(cov_fn)
    print(f"Denoising {cov_fn} with svd...")
    icov = compute_icov_svd(cov, Nmocks)
    np.savetxt(icov_svd_fn, icov)
    print(f"Successfully saved to {icov_svd_fn}!")

    # also saving the straight inverse as a check
    icov_fn = f"{cov_dir}/icov_{cov_tag}_{stat_str}{tag_str}.dat"
    icov_norm = compute_icov_norm(cov)
    np.savetxt(icov_fn, icov_norm)
    print(f"Successfully saved reg icov to {icov_fn}!")


def compute_icov_svd(cov, Nmocks):
    corr = utils.reduced_covariance(cov)
    u1, s1, vh1 = np.linalg.svd(corr)
    v1 = vh1.T
    ids_small = np.where(s1<np.sqrt(2./Nmocks))[0] # Eq (20) of https://arxiv.org/pdf/astro-ph/0501637.pdf
    print(f"Removing {len(ids_small)} smallest modes")
    s1_inv = 1./s1
    s1_inv[ids_small] = 0.0
    s1_new = np.diag(s1_inv)
    icov = np.dot(v1, np.dot(s1_new, u1.T)) # inverse matrix
    return icov


def compute_icov_norm(cov):
    corr = utils.reduced_covariance(cov)
    u1, s1, vh1 = np.linalg.svd(corr)
    v1 = vh1.T
    s1_inv = 1./s1
    s1_new = np.diag(s1_inv)
    icov = np.dot(v1, np.dot(s1_new, u1.T)) # inverse matrix
    return icov


if __name__=='__main__':
    main()
