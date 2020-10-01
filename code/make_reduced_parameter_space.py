import numpy as np


cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

def main():
    CC_train, HH_train, cosmos_train, hods_train = get_training_space()
    CC_test, HH_test, cosmos_test, hods_test = get_testing_space()
    bounds = get_bounds(cosmos_train, hods_train)
    good_params = get_central_test(CC_test, HH_test, cosmos_test, hods_test, bounds)
    save_fn = '../tables/params_test_central.dat'
    save_params(good_params, save_fn)


def save_params(params, save_fn):
    np.savetxt(save_fn, params, fmt='%d')


def get_bounds(cosmos_train, hods_train):

    bounds = {}
    for pname in cosmo_names:
        pidx = cosmo_names.index(pname)
        vals = cosmos_train[:,pidx]
        bounds[pname] = [np.min(vals), np.max(vals)]
    for pname in hod_names:
        pidx = hod_names.index(pname)
        vals = hods_train[:,pidx]
        bounds[pname] = [np.min(vals), np.max(vals)]

    return bounds


def remove_edge_params_training(CC, HH, bounds, cut_frac=0.1, mode='train'):

    hods = np.loadtxt("../tables/HOD_design_np11_n5000_new_f_env.dat")
    cosmos = np.loadtxt("../tables/cosmology_camb_full.dat")

    for CID in CC:
        HH_set = HH[CID]
        for HID in HH_set:
            HID = int(HID)
            cosmo_params = cosmos[CID,:]
            hod_params = hods[HID,:]

    
def get_central_test(CC_test, HH_test, cosmos_test, hods_test, bounds, cut_frac=0.15):

    print(len(CC_test)*len(HH_test))
    good_params = []
    for CID_test in CC_test:
        for HID_test in HH_test:
            nbad = 0
            cs = cosmos_test[CID_test,:]
            hs = hods_test[HID_test,:]

            for i, c in enumerate(cs):
                pmin, pmax = bounds[cosmo_names[i]]
                buf = (pmax-pmin)*cut_frac
                pmin_cut = pmin + buf
                pmax_cut = pmax + buf
                #print(c, pmin_cut, pmax_cut)
                if c<pmin_cut or c>pmax_cut:
                    nbad += 1
                    
            for i, h in enumerate(hs):
                pmin, pmax = bounds[hod_names[i]]
                buf = (pmax-pmin)*cut_frac
                pmin_cut = pmin + buf
                pmax_cut = pmax + buf
                if h<pmin_cut or h>pmax_cut:
                    nbad += 1
            
            if nbad<=0:
                good_params.append((CID_test, HID_test))

    print(len(good_params))
    return good_params

    
def get_training_space():
    hods = np.loadtxt("../tables/HOD_design_np11_n5000_new_f_env.dat")

    hods[:, 0] = np.log10(hods[:, 0])
    hods[:, 2] = np.log10(hods[:, 2])
    nhodparams = hods.shape[1]
    nhodnonolap = 100
    # cosmology params (40 rows, 7 cols)
    cosmos = np.loadtxt("../tables/cosmology_camb_full.dat")
    ncosmoparams = cosmos.shape[1]

    CC = range(0, cosmos.shape[0])
    nhodpercosmo = 100

    HH = np.array(range(0, len(CC) * nhodpercosmo))
    HH = HH.reshape(len(CC), nhodpercosmo)
    HH = HH[:, 0:nhodnonolap]

    return CC, HH, cosmos, hods


def get_testing_space():
    hods_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat")
    nhodparams_test = hods_test.shape[1]
    hods_test[:,0] = np.log10(hods_test[:,0])
    hods_test[:,2] = np.log10(hods_test[:,2])
    cosmos_test = np.loadtxt("../CMASS/Gaussian_Process/hod_file/cosmology_camb_test_box_full.dat")
    ncosmoparams_test = cosmos_test.shape[1]

    CC_test = range(0, 7)
    # TODO: add more tests, for now just did first 10 hod
    HH_test = range(0, 100)
    return CC_test, HH_test, cosmos_test, hods_test


if __name__=='__main__':
    main()