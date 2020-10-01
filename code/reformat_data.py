import numpy as np

import emulator

def main():
    statistic = 'xi'
    traintag = '_nonolap'
    res_dir = '../../clust/results_{}/'.format(statistic)
    training_dir = '{}training_{}{}/'.format(res_dir, statistic, traintag)
    params, data, labels = load_training_data(statistic, training_dir, fixed_hod=3)
    write_data(params, data, labels)


def write_data(params, data, labels):
    data_dict = {}
    for i in range(params.shape[0]):
        info = {'x': params[i], 'y': data[i], 'labels': labels[i]}
        data_dict[i] = info
    return data_dict



def load_training_data(statistic, training_dir, nhod=100, nbins=9, fixed_hod=None):
    hods = np.loadtxt("../tables/HOD_design_np11_n5000_new_f_env.dat")

    hods[:, 0] = np.log10(hods[:, 0])
    hods[:, 2] = np.log10(hods[:, 2])
    nhodnonolap = nhod
    # cosmology params (40 rows, 7 cols)
    cosmos = np.loadtxt("../tables/cosmology_camb_full.dat")
    ncosmoparams = cosmos.shape[1]

    CC = range(0, cosmos.shape[0])
    nhodpercosmo = 100

    if fixed_hod is not None:
        #HH = np.array(range(0, len(CC)))
        #HH = HH.reshape(len(CC), 1)
        #HH = HH[:, fixed_hod]
        HH = np.full((len(CC), 1), fixed_hod)
        nhodparams = 0
    else:
        HH = np.array(range(0, len(CC) * nhodpercosmo))
        HH = HH.reshape(len(CC), nhodpercosmo)
        HH = HH[:, 0:nhodnonolap]
        nhodparams = hods.shape[1]

    print(HH)

    nparams = nhodparams + ncosmoparams
    print(f"Nparams: {nparams}")
    ndata = HH.shape[1] * cosmos.shape[0]

    training_params = np.empty((ndata, nparams))
    training_data = np.empty((ndata, nbins))
    training_labels = np.empty((ndata, nbins))

    idata = 0
    for CID in CC:
        HH_set = HH[CID]
        for HID in HH_set:
            HID = int(HID)
            # training data is all test0 (always)
            rs, vals = np.loadtxt(training_dir + "{}_cosmo_{}_HOD_{}_test_0.dat".format(statistic, CID, HID),
                                   delimiter=',', unpack=True)
            vals = vals[:nbins]
            training_labels[idata, :] = rs
            training_data[idata,:] = vals
            if fixed_hod:
                param_arr = cosmos[CID,:]
            else:
                param_arr = np.concatenate((cosmos[CID,:], hods[HID,:]))
            training_params[idata, :] = param_arr
            idata += 1

    print(training_params)
    print(training_data)
    return training_params, training_data, training_labels

if __name__=='__main__':
    main()
