import time
import numpy as np
import george
from george import kernels
import multiprocessing as mp

import gp_trainer as trainer


class Emulator:

    def __init__(self, statistic, training_dir, testing_dir=None, hyperparams=None, fixed_params={}, nbins=9, gperr=None, testmean=True, log=False, mean=False):
        
        # set parameters
        self.statistic = statistic
        self.fixedparams = fixed_params
        self.nbins = nbins
        self.gps = [None]*nbins
        self.training_dir = training_dir
        self.testing_dir = testing_dir
        self.testmean = testmean # use the mean of the test boxes (recommended)
        self.log = log
        self.mean = mean

        # load data
        self.load_training_data()
        if self.testing_dir:
            self.load_testing_data()

        # initialize emulator
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

    def process_data(self, data_orig, bb):
        data = data_orig.copy()
        if self.log:
            data = np.log10(data)
        if self.mean:
            data /= self.training_mean[bb]
            #data -= 1
        return data

    # Make sure consistent with unprocess! 
    # [opposite order and operations]
    def unprocess_data(self, data_orig, bb):
        data = data_orig.copy()
        if self.mean:
            #data += 1
            data *= self.training_mean[bb]
        if self.log:
            data = 10**data
        return data

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
            #training_data_pred = self.training_data_normmean[:,bb]
            #training_data_pred = self.training_data[:,bb]
            training_data_pred = self.process_data(self.training_data[:,bb], bb)
            #if self.log:
            #    training_data_pred = np.log10(training_data_pred)
            val_pred, cov_pred = self.gps[bb].predict(training_data_pred, params_arr)
            val_pred = self.unprocess_data(val_pred, bb)
            #if self.log:
            #    val_pred = 10**val_pred
            #val_pred += 1
            # multiply by mean to get back to original values
            #y_pred[bb] = val_pred * self.training_mean[bb]
            y_pred[bb] = val_pred

        return y_pred


    def build(self):
        print("Rebuilding emulators")
        for bb in range(self.nbins):
            #training_data = self.training_data_normmean[:,bb]
            training_data = self.process_data(self.training_data[:,bb], bb)
            #if self.log:
            #    training_data = np.log10(training_data)
            #training_data /= np.mean(training_data)
            #training_data -= 1
            mean = np.mean(training_data)
            kernel = self.get_kernel(np.full(self.nparams, 0.1))
            gp = self.init_gp(self.training_params, self.gperr[bb], kernel, mean=mean)
            self.gps[bb] = self.set_hyperparams(gp, self.training_params, 
                            self.gperr[bb], self.hyperparams[bb])

            #print("Computed GP {}".format(bb))


    def train_serial(self, save_hyperparams_fn):
        start = time.time()
        print("Training commences!")
        for bb in range(self.nbins):
        #for bb in range(1,self.nbins):
            print(f"Training bin {bb}")
            #training_data = self.training_data_normmean[:,bb]
            training_data = self.process_data(self.training_data[:,bb], bb)            
            #if self.log:
            #    training_data = np.log10(training_data)
            #training_data /= np.mean(training_data)
            mean = np.mean(training_data)
            kernel = self.get_kernel(np.full(self.nparams, 0.1))
            gp = self.init_gp(self.training_params, self.gperr[bb], kernel, mean=mean)
            hyps = trainer.gp_tr(self.training_params, 
                        training_data, self.gperr[bb], 
                        gp, optimize=True).p_op
            self.hyperparams[bb, :] = hyps

        print("Done training!")
        np.savetxt(save_hyperparams_fn, self.hyperparams, fmt='%.7f')
        print(f"Saved hyperparameters to {save_hyperparams_fn}")
        end = time.time()
        print(f"Time: {(end-start)/60.0} min")

    def train(self, save_hyperparams_fn, nthreads=None):
        start = time.time()
        print("Training commences!")
        if not nthreads:
            nthreads = self.nbins
        print("Constructing pool")
        pool = mp.Pool(processes=nthreads)
        print("Mapping bins")
        res = pool.map(self.train_bin, range(self.nbins))
        print("Done training!")
        for bb in range(self.nbins):
            self.hyperparams[bb, :] = res[bb]
        np.savetxt(save_hyperparams_fn, self.hyperparams, fmt='%.7f')
        print(f"Saved hyperparameters to {save_hyperparams_fn}")
        end = time.time()
        print(f"Time: {(end-start)/60.0} min")

    def train_bin(self, bb):
        print(f"Training bin {bb}")
        #training_data = self.training_data_normmean[:,bb]
        training_data = self.process_data(self.training_data[:,bb], bb)
        #if self.log:
        #    training_data = np.log10(training_data)
        #training_data /= np.mean(training_data)
        #training_data -= 1
        mean = np.mean(training_data)
        kernel = self.get_kernel(np.full(self.nparams, 0.1))
        gp = self.init_gp(self.training_params, self.gperr[bb], kernel, mean=mean)
        hyps = trainer.gp_tr(self.training_params, 
                    training_data, self.gperr[bb], 
                    gp, optimize=True).p_op
        #self.hyperparams[bb, :] = hyps
        return hyps

    def test(self, predict_savedir):
        if not self.testing_dir:
            raise ValueError('Must provide testing directory in emulator constructor!')
        for pid, tparams in self.testing_params.items():
            vals_pred = self.predict(tparams)
            if self.testmean:
                idtag = "cosmo_{}_HOD_{}_mean".format(pid[0], pid[1])
            else:
                idtag = "cosmo_{}_Box_{}_HOD_{}_test_{}".format(pid[0], boxid, pid[1], testid)

            pred_fn = f"{predict_savedir}/{self.statistic}_{idtag}.dat"

            results = np.array([self.testing_radii, vals_pred])
            np.savetxt(pred_fn, results.T, delimiter=',', fmt=['%f', '%e']) 

    def load_training_data(self):
        print("Loading training data")
        # hod parameters (5000 rows, 8 cols)
        hods = np.loadtxt("../tables/HOD_design_np11_n5000_new_f_env.dat")

        hods[:, 0] = np.log10(hods[:, 0])
        hods[:, 2] = np.log10(hods[:, 2])
        nhodparams = hods.shape[1]
        nhodnonolap = 100

        # cosmology params (40 rows, 7 cols)
        cosmos = np.loadtxt("../tables/cosmology_camb_full.dat")
        ncosmoparams = cosmos.shape[1]

        CC = range(0, cosmos.shape[0])
        nhodpercosmo = 50
        #speedy params
        #CC = range(0, 1)
        #nhodpercosmo = 10

        HH = np.array(range(0, len(CC) * nhodnonolap))
        HH = HH.reshape(len(CC), nhodnonolap)
        HH = HH[:, 0:nhodpercosmo]

        self.nparams = nhodparams + ncosmoparams
        print(f"Nparams: {self.nparams}")
        self.ndata = HH.shape[1] * cosmos.shape[0]

        #self.training_params = {}
        #self.training_data = {}
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
        if self.log:
            self.training_mean = np.mean(np.log10(self.training_data), axis=0)
        else:
            self.training_mean = np.mean(self.training_data, axis=0)
        #self.training_data_normmean = self.training_data/self.training_mean


    def load_testing_data(self):
        print("Loading testing data")
        
        hods_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat")
        nhodparams_test = hods_test.shape[1]
        hods_test[:,0] = np.log10(hods_test[:,0])
        hods_test[:,2] = np.log10(hods_test[:,2])
        cosmos_test = np.loadtxt("../CMASS/Gaussian_Process/hod_file/cosmology_camb_test_box_full.dat")
        ncosmoparams_test = cosmos_test.shape[1]

        CC_test = range(0, 7)
        # TODO: add more tests, for now just did first 10 hod
        HH_test = range(0, 10)

        self.nparams_test = nhodparams_test + ncosmoparams_test
        print(f"Nparams: {self.nparams_test}")
        #self.ndata_test = HH_test.shape[1] * cosmos_test.shape[0]

        #self.testing_params = np.empty((self.ndata_test, self.nparams_test))
        #self.testing_data = np.empty((self.ndata_test, self.nbins))
        self.testing_params = {}
        self.testing_data = {}

        #idata = 0
        for CID_test in CC_test:
            for HID_test in HH_test:
                hods_test_hid = hods_test[HID_test,:]

                if self.testmean:
                    idtag = "cosmo_{}_HOD_{}_mean".format(CID_test, HID_test)
                    rads, vals_test = np.loadtxt(self.testing_dir + "{}_{}.dat".format(self.statistic, idtag))
                else:
                    idtag = "cosmo_{}_Box_{}_HOD_{}_test_{}".format(CID_test, boxid, HID_test, testid)
                    rads, vals_test = np.loadtxt(self.testing_dir + "{}_{}.dat".format(self.statistic, idtag),
                                                  delimiter=',', unpack=True)

                #self.testing_data[idata,:] = vals_test
                pid = (CID_test, HID_test)
                self.testing_data[pid] = vals_test
                param_arr = np.concatenate((cosmos_test[CID_test,:], hods_test[HID_test,:]))
                #self.testing_params[idata, :] = param_arr
                self.testing_params[pid] = param_arr
                #TODO: really only need to set this once
                self.testing_radii = rads
                #idata += 1


    def init_gp(self, training_params, err, kernel, mean=None):
        gp = george.GP(kernel, mean=mean, solver=george.BasicSolver)
        #TODO: WHY DO I COMPUTE TWICE??
        gp.compute(training_params, err)
        return gp


    def set_hyperparams(self, gp, training_params, err, hyperparams):
        gp.set_parameter_vector(hyperparams)
        gp.compute(training_params, err)
        return gp


    # 15 initial values for the 7 hod and 8 cosmo params
    def get_kernel(self, p0):
        #k1 = kernels.ExpSquaredKernel(p0, ndim=len(p0))
        k2 = kernels.Matern32Kernel(p0, ndim=len(p0))
        #k3 = kernels.ConstantKernel(0.1, ndim=len(p0))
        # k4 = kernels.WhiteKernel(0.1, ndim=len(p0))
        k5 = kernels.ConstantKernel(0.1, ndim=len(p0))
        kernel = k2 + k5
        # kernel = np.var(y)*k1
        #print(kernel)
        return kernel


    def get_param_bounds(self, pname):
        return self.param_bounds[pname]


    def set_param_bounds(self):
        bounds = {}
        cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
        hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

        # wait should this be the training set??
        # cosmos_truth = np.loadtxt('../tables/cosmology_camb_test_box_full.dat') # 7
        # hods_truth = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat') # 1000
        cosmos_train = np.loadtxt('../tables/cosmology_camb_full.dat') # 40
        hods_train = np.loadtxt('../tables/HOD_design_np11_n5000_new_f_env.dat') # 5000

        for pname in cosmo_names:
            pidx = cosmo_names.index(pname)
            vals = cosmos_train[:,pidx]
            pmin = np.min(vals)
            pmax = np.max(vals)
            buf = (pmax-pmin)*0.1
            bounds[pname] = [pmin-buf, pmax+buf]

        for pname in hod_names:
            pidx = hod_names.index(pname)
            vals = hods_train[:,pidx]
            pmin = np.min(vals)
            pmax = np.max(vals)
            buf = (pmax-pmin)*0.1
            bounds[pname] = [pmin-buf, pmax+buf]
        
        return bounds
