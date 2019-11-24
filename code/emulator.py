import numpy as np
import george
from george import kernels

import gp_trainer as trainer


class Emulator:

    def __init__(self, statistic, training_dir=None, testing_dir=None, hyperparams=None, fixed_params={}, nbins=9, gperr=None, mode='train'):
        
        # set parameters
        self.statistic = statistic
        self.fixedparams = fixed_params
        self.nbins = nbins
        self.gps = np.empty(nbins)
        self.training_dir = training_dir
        self.testing_dir = testing_dir

        # load data
        #if mode=='train':
        if self.testing_dir:
            self.load_training_data()
        #elif mode=='test':
        if self.training_dir:
            self.load_testing_data()
        #else:
        #    raise ValueError(r"Mode {mode} not recognized! Use 'train' or 'test'")

        # initialize emulator
        self.set_kernel(np.full(self.nparams, 0.1))
        self.param_bounds = self.set_param_bounds()
        self.gperr = self.load_file_or_obj(gperr)
        if hyperparams:
            self.hyperparams = self.load_file_or_obj(hyperparams) #may still be None
        else:
            self.hyperparams = np.empty((nbins, self.nparams+1))


    def load_file_or_obj(self, name):
        if type(name)==str:
            return np.loadtxt(name)
        else:
            return name


    def predict(self, params_pred):

        if type(params_pred)==dict:
            params_arr = []
            param_names_ordered = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w',
                                    'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f',
                                   'f_env', 'delta_env', 'sigma_env']
            for pn in param_names_ordered:
                params_arr.append(params_pred[pn])
        elif type(params_pred)==list or type(params_pred)==np.ndarray:
            params_arr = params_pred
        else:
            raise ValueError("Params to predict at must be dict or array")

        params_arr = np.atleast_2d(params_arr)

        y_pred = np.zeros(self.nbins)
        for bb in range(self.nbins):
            # predict on all the training data in the bin - normalized by the mean as used for building emu
            training_data_pred = self.training_data_normmean[:,bb]
            mu, cov = self.gps[bb].predict(training_data_pred, params_arr)
            # multiply by mean to get back to original values
            y_pred[bb] = mu * self.training_mean[bb]

        return y_pred


    def build(self):
        print("Rebuilding emulators")
        for bb in range(self.nbins):
            
            # TODO: make sure this is doing same thing
            #training_data_normmean = self.training_data[:,bb]/self.training_mean[bb] # ndata x 1
            #mean = np.mean(training_data_normmean)
            # ok i think it's good
            mean = np.mean(self.training_data_normmean[:,bb])
            gp = self.init_gp(self.training_params, self.gperr[bb], mean=mean)
            self.gps[bb] = self.set_hyperparams(gp, self.training_params, 
                            self.gperr[bb], self.hyperparams[bb])

            print("Computed GP {}".format(bb))


    def train(self, save_hyperparams_fn):
        print("Training commences!")
        for bb in range(self.nbins):
            print(f"Training bin {bb}")
            #training_data_normmean = self.training_data[:,bb]/self.training_mean[bb] # ndata x 1
            #mean = np.mean(training_data_normmean)
            mean = np.mean(self.training_data_normmean[bb])
            gp = self.init_gp(self.training_params, self.gperr[bb], mean=mean)
            self.hyperparams[bb] = trainer.gp_tr(self.training_params[:,bb], 
                        self.training_data_normmean[:,bb], self.gperr[bb], 
                        gp, optimize=True).p_op
        
        print("Done training!")
        np.savetxt(save_hyperparams_fn, self.hyperparams, fmt='%.7f')
        print(f"Saved hyperparameters to {save_hyperparams_fn}")


    def test(self):
        for tparams in self.testing_params:
            vals_pred = self.predict(tparams)


    def load_training_data(self):
        print("Loading training data")
        # hod parameters (5000 rows, 8 cols)
        hods = np.loadtxt("../tables/HOD_design_np11_n5000_new_f_env.dat")

        hods[:, 0] = np.log10(hods[:, 0])
        hods[:, 2] = np.log10(hods[:, 2])
        nhodparams = hods.shape[1]

        # cosmology params (40 rows, 7 cols)
        cosmos = np.loadtxt("../tables/cosmology_camb_full.dat")
        ncosmoparams = cosmos.shape[1]

        CC = range(0, cosmos.shape[0])
        #CC = range(0, 1)
        nhodnonolap = 100
        nhodpercosmo = 50
        HH = np.array(range(0, len(CC) * nhodnonolap))
        HH = HH.reshape(len(CC), nhodnonolap)
        HH = HH[:, 0:nhodpercosmo]

        self.nparams = nhodparams + ncosmoparams
        print(f"Nparams: {self.nparams}")
        self.ndata = HH.shape[1] * cosmos.shape[0]

        self.training_params = np.empty((self.ndata, self.nparams))
        self.training_data = np.empty((self.ndata, self.nbins))

        idata = 0
        for CID in CC:
            HH_set = HH[CID]
            for HID in HH_set:
                HID = int(HID)
                # training data is all test0
                _, vals = np.loadtxt(self.training_dir + "{}_cosmo_{}_HOD_{}_test_0.dat".format(self.statistic, CID, HID),
                                       delimiter=',', unpack=True)
                vals = vals[:self.nbins]
                self.training_data[idata,:] = vals

                param_arr = np.concatenate((cosmos[CID,:], hods[HID,:]))
                self.training_params[idata, :] = param_arr 
                idata += 1

        # mean of values in each bin (training_mean has length nbins)
        self.training_mean = np.mean(self.training_data, axis=0)
        self.training_data_normmean = self.training_data/self.training_mean


    def load_testing_data():
        print("Loading testing data")
        
        hods_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat")
        nhodparams_test = hods.shape[1]
        hods_test[:,0] = np.log10(hods_test[:,0])
        hods_test[:,2] = np.log10(hods_test[:,2])
        cosmos_test = np.loadtxt("../CMASS/Gaussian_Process/hod_file/cosmology_camb_test_box_full.dat")
        ncosmoparams_test = cosmos_test.shape[1]

        CC_test = range(0, 7)
        # TODO: add more tests, for now just did first 10 hod
        HH_test = range(0, 10)

        self.nparams_test = nhodparams_test + ncosmoparams_test
        print(f"Nparams: {self.nparams}")
        self.ndata_test = HH.shape[1] * cosmos.shape[0]

        self.testing_params = np.empty((self.ndata_test, self.nparams_test))
        self.testing_data = np.empty((self.ndata_test, self.nbins))
        #self.testing_ids = np.

        idata = 0
        for CID_test in CC_test:
            for HID_test in HH_test:
                print('CID, HID:', CID_test, HID_test)
                hods_test_hid = hods_test[HID_test,:]

                if testmean:
                    idtag = "cosmo_{}_HOD_{}_mean".format(CID_test, HID_test)
                    _, vals_test = np.loadtxt(testing_dir + "{}_{}.dat".format(statistic, idtag))
                else:
                    idtag = "cosmo_{}_Box_{}_HOD_{}_test_{}".format(CID_test, boxid, HID_test, testid)
                    _, vals_test = np.loadtxt(testing_dir + "{}_{}.dat".format(statistic, idtag),
                                                  delimiter=',', unpack=True)

                vals_test = vals_test[:nbins]
                self.testing_data[idata,:] = vals_test

                param_arr = np.concatenate((cosmos_test[CID_test,:], hods_test[HID_test,:]))
                self.testing_params[idata, :] = param_arr
                idata += 1


    def init_gp(self, training_params, err, mean=None):
        gp = george.GP(self.kernel, mean=mean, solver=george.BasicSolver)
        #TODO: WHY DO I COMPUTE TWICE??
        gp.compute(training_params, err)
        return gp


    def set_hyperparams(self, gp, training_params, err, hyperparams):
        gp.set_parameter_vector(hyperparams)
        gp.compute(training_params, err)
        return gp


    # 15 initial values for the 7 hod and 8 cosmo params
    def set_kernel(self, p0):
        #k1 = kernels.ExpSquaredKernel(p0, ndim=len(p0))
        k2 = kernels.Matern32Kernel(p0, ndim=len(p0))
        #k3 = kernels.ConstantKernel(0.1, ndim=len(p0))
        # k4 = kernels.WhiteKernel(0.1, ndim=len(p0))
        k5 = kernels.ConstantKernel(0.1, ndim=len(p0))
        kernel = k2 + k5
        # kernel = np.var(y)*k1
        self.kernel = kernel


    def get_param_bounds(self, pname):
        return self.param_bounds[pname]


    def set_param_bounds(self):
        bounds = {}
        cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
        hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

        cosmos_truth = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')
        hods_truth = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')

        for pname in cosmo_names:
            pidx = cosmo_names.index(pname)
            vals = cosmos_truth[:,pidx]
            # should i add a buffer?
            pmin = np.min(vals)
            pmax = np.max(vals)
            buf = (pmax-pmin)*0.1
            bounds[pname] = [pmin-buf, pmax+buf]

        for pname in hod_names:
            pidx = hod_names.index(pname)
            vals = hods_truth[:,pidx]
            # should i add a buffer?
            pmin = np.min(vals)
            pmax = np.max(vals)
            buf = (pmax-pmin)*0.1
            bounds[pname] = [pmin-buf, pmax+buf]
        
        return bounds
