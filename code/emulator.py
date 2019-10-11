import numpy as np
import george
from george import kernels


class Emulator:

    def __init__(self, statistic, training_dir, hyperparams, fixed_params={}, nbins=9, gperr=None):
        self.statistic = statistic
        self.training_dir = training_dir
        self.hyperparams = self.load_file_or_obj(hyperparams)
        self.fixedparams = fixed_params
        self.nbins = nbins
        self.gperr = self.load_file_or_obj(gperr)
        #self.training_data = self.load_training_data()
        self.build_emulator()

        #TODO: do this properly (read from file?)
        #self.param_bounds = {'Omega_m': [0.2, 0.4], 'f_env': [-0.5, 1.0]}
        self.param_bounds = self.set_param_bounds()

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

        #loc = len(CC) * HH.shape[1]
        loc = len(self.ydata)/self.nbins
        mus = np.zeros(self.nbins)
        for bb in range(self.nbins):
            # predict on all the statistic values in the bin
            mu, cov = self.gps[bb].predict(self.ydata[int(loc * bb):int(loc * (bb + 1))], params_arr)
            mus[bb] = mu

        y_predicted = mus * self.training_mean
        return y_predicted


    ### TODO: break this up, major rewrite!!
    def build_emulator(self):

        # hod parameters (5000 rows, 8 cols)
        hods = np.loadtxt("../tables/HOD_design_np11_n5000_new_f_env.dat")

        hods[:, 0] = np.log10(hods[:, 0])
        hods[:, 2] = np.log10(hods[:, 2])
        nhodparams = hods.shape[1]

        # cosmology params (40 rows, 7 cols)
        cosmos = np.loadtxt("../tables/cosmology_camb_full.dat")
        ncosmoparams = cosmos.shape[1]

        CC = range(0, cosmos.shape[0])
        nhodnonolap = 100
        nhodpercosmo = 50
        HH = np.array(range(0, len(CC) * nhodnonolap))
        HH = HH.reshape(len(CC), nhodnonolap)
        HH = HH[:, 0:nhodpercosmo]
        nparams = nhodparams + ncosmoparams

        xdata = np.empty((HH.shape[1] * cosmos.shape[0], nparams))
        training_data = np.empty((self.nbins, HH.shape[1] * cosmos.shape[0]))

        ### calc mean
        s2 = 0
        for CID in CC:
            HH_set = HH[CID]
            for HID in HH_set:
                HID = int(HID)
                rad, vals = np.loadtxt(self.training_dir + "{}_cosmo_{}_HOD_{}_test_0.dat".format(self.statistic, CID, HID),
                                       delimiter=',', unpack=True)
                rad = rad[:self.nbins]
                vals = vals[:self.nbins]

                training_data[:, s2] = vals
                s2 += 1

        # mean of values in each bin (training_mean has length nbins)
        self.training_mean = np.mean(training_data, axis=1)
        training_std = np.std(training_data, axis=1)

        self.gps = []
        self.ydata = np.empty((len(xdata) * self.nbins))
        ss2 = 0
        for j in range(self.nbins):
            Ym = self.training_mean[j]
            ss = 0
            yerr = np.zeros((len(xdata)))
            y = np.empty((len(xdata)))
            for CID in CC:
                HH_set = HH[CID]
                for HID in HH_set:
                    HID = int(HID)

                    # seems silly to load this every bin loop but makes some sense, otherwise would have to store
                    rad, vals = np.loadtxt(self.training_dir + "{}_cosmo_{}_HOD_{}_test_0.dat".format(self.statistic, CID, HID),
                                           delimiter=',', unpack=True)
                    rad = rad[:self.nbins]
                    vals = vals[:self.nbins]
                    # the cosmology and HOD values used for this data

                    xdata[ss, 0:ncosmoparams] = cosmos[CID, :]
                    xdata[ss, ncosmoparams:ncosmoparams + nhodparams] = hods[HID, :]

                    val = vals[j]
                    y[ss] = val / Ym
                    yerr[ss] = self.gperr[j]
                    self.ydata[ss2] = y[ss]

                    ss += 1
                    ss2 += 1

                    ######

            # 15 initial values for the 7 hod and 8 cosmo params
            p0 = np.full(nparams, 0.1)

            k1 = kernels.ExpSquaredKernel(p0, ndim=len(p0))
            k2 = kernels.Matern32Kernel(p0, ndim=len(p0))
            k3 = kernels.ConstantKernel(0.1, ndim=len(p0))
            # k4 = kernels.WhiteKernel(0.1, ndim=len(p0))
            k5 = kernels.ConstantKernel(0.1, ndim=len(p0))

            kernel = k2 + k5
            # kernel = np.var(y)*k1

            ppt = self.hyperparams[j]

            gp = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
            # gp = george.GP(kernel, solver=george.BasicSolver)

            gp.compute(xdata, yerr)
            # gp.kernel.vector = ppt
            gp.set_parameter_vector(ppt)
            gp.compute(xdata, yerr)
            print("Computed GP {}".format(j))
            self.gps.append(gp)

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
            bounds[pname] = [pmin, pmax]

        for pname in hod_names:
            pidx = hod_names.index(pname)
            vals = hods_truth[:,pidx]
            # should i add a buffer?
            pmin = np.min(vals)
            pmax = np.max(vals)
            bounds[pname] = [pmin, pmax]
        
        return bounds