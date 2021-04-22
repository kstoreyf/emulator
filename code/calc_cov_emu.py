import numpy as np

import utils


def main():
    statistics = ['xi']
    #statistics = ['wp', 'xi']
    #statistics = ['wp', 'xi', 'upf']
    #statistics = ['wp', 'xi', 'upf', 'mcf']
    cov_tag = 'emuaem'
    tag_str = ''

    cov_dir = '/home/users/ksf293/clust/covariances'    
    stat_str = '_'.join(statistics)
    cov_emu_fn = f"{cov_dir}/cov_{cov_tag}_{stat_str}{tag_str}.dat"
    cov_final_fn = f"{cov_dir}/cov_glamemudiag_{stat_str}{tag_str}.dat"
    tag_str_perf = '_nonolap_hod3_test0_mean_test0'
    tag_str_aem = '_hod3_test0'

    L_glam = 1000.
    L_aemulus = 1050.

    cov_emuperf_glam = utils.get_cov(statistics, 'emuperf', tag_str='_nonolap_hod3_test0_glam')
    cov_emuperf = utils.get_cov(statistics, 'emuperf', tag_str=tag_str_perf)
    cov_glam = utils.get_cov(statistics, 'glam')
    cov_aemulus = utils.get_cov(statistics, 'aemulus', tag_str=tag_str_aem)

    #cov_glam_scaled = cov_glam*(1/5)*(L_glam/L_aemulus)**3
    #cov_emu = cov_emuperf - cov_glam_scaled

    cov_aemulus_5box = cov_aemulus*(1/5)
    cov_emu = cov_emuperf - cov_aemulus_5box

    np.savetxt(cov_emu_fn, cov_emu)
    print(f"Successfully saved to {cov_emu_fn}!")

    cov_glam_scaled = cov_glam*(1/5)*(L_glam/L_aemulus)**3
    #cov_glam_scaled = cov_glam*(L_glam/L_aemulus)**3
    cov_data = cov_glam_scaled
    #cov_final = cov_emu + cov_data
    cov_final = np.diag(np.diag(cov_emu)) + cov_data
    np.savetxt(cov_final_fn, cov_final)
    print(f"Successfully saved to {cov_final_fn}!")

    # all-glam
    #cov_emuperf_glam_scaled = diag(cov_emuperf)*diag(cov_aemulus_5box) 
    #cov_emu_glam = cov_emuperf_glam - cov_aemulus_5box
    #cov_final = np.diag(np.diag(cov_emu)) + cov_data

    # Percentiles
    save_fn_p16_perf = f"{cov_dir}/p16_emuperf_{stat_str}{tag_str_perf}.dat"
    save_fn_p84_perf = f"{cov_dir}/p84_emuperf_{stat_str}{tag_str_perf}.dat"
    p16_perf = np.loadtxt(save_fn_p16_perf)
    p84_perf = np.loadtxt(save_fn_p84_perf)
    err68_perf = 0.5*(p84_perf - p16_perf)
    
    save_fn_p16_aem = f"{cov_dir}/p16_aemulus_{stat_str}{tag_str_aem}.dat"
    save_fn_p84_aem = f"{cov_dir}/p84_aemulus_{stat_str}{tag_str_aem}.dat"
    p16_aem = np.loadtxt(save_fn_p16_aem)
    p84_aem = np.loadtxt(save_fn_p84_aem)
    err68_aem = 0.5*(p84_aem - p16_aem)

    print(err68_perf)
    print(err68_aem)

    save_fn_p16_emuaem = f"{cov_dir}/p16_emuaem_{stat_str}{tag_str}.dat"
    save_fn_p84_emuaem = f"{cov_dir}/p84_emuaem_{stat_str}{tag_str}.dat"
    save_fn_err68_emuaem = f"{cov_dir}/err68_emuaem_{stat_str}{tag_str}.dat"
    #p16_emuaem = p16_perf - p16_aem
    #p84_emuaem = p84_perf - p84_aem
    #print(p84_emuaem)
    #print(p16_emuaem)
    err68_emuaem = np.sqrt(err68_perf**2 - (1/np.sqrt(5)*err68_aem)**2)

    #np.savetxt(save_fn_p16_emuaem, p16_emuaem)
    #np.savetxt(save_fn_p84_emuaem, p84_emuaem)
    np.savetxt(save_fn_err68_emuaem, err68_emuaem)


if __name__=='__main__':
    main()
