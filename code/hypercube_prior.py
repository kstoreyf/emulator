import numpy as np
import scipy as sp
from numpy.linalg import eig


class GET_Prior(object):
    def __init__(self):
        self.cosmo = np.loadtxt("/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/hod_file/cosmology_camb_full.dat")

    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    def get_lambda(self):
        NC = self.cosmo.shape[1]
        s = 0
        LL = np.zeros((NC*NC, 7))

        for i in range(NC):
            for j in range(NC):
                if i>j:
                    s = s+1
                    pass
                else:
                    s = s+1
                    continue

                tX = self.cosmo[:,j]
                tY = self.cosmo[:,i]
                CenterX = np.mean(tX)
                CenterY = np.mean(tY)
                cov = np.cov(tX, tY)
                eigvalue, eigvector = eig(cov)
                #lambda1 = np.max(eigvalue)*9
                #lambda2 = np.min(eigvalue)*9
        
                LL[s, 0] = j
                LL[s, 1] = i
                LL[s, 2] = CenterX
                LL[s, 3] = CenterY
                LL[s, 4] = np.max(eigvalue)
                LL[s, 5] = np.min(eigvalue)

                vals, vecs = self.eigsorted(cov)
                theta = np.arctan2(*vecs[:,0][::-1])
        
                LL[s, 6] = theta

        return LL

    def func(self, x, y, CX, CY, ll1, ll2, theta):
        value = ( np.cos(theta)*(x-CX) + np.sin(theta)*(y-CY) )**2.0/ll1 + ( np.sin(theta)*(x-CX) - np.cos(theta)*(y-CY) )**2.0/ll2
        return value


    def isinornot(self, position, LL, sigma=3):
        NC = self.cosmo.shape[1]
        vall = []
        s = 0
        for i in range(NC):
            for j in range(NC):
        
                Lin = LL[np.where((LL[:,0]==j) & (LL[:,1]==i))[0]].flatten()
                if i>j:
                    s = s+1
                    pass
                else:
                    s = s+1
                    continue

                vv = self.func(position[j], position[i], Lin[2], Lin[3], Lin[4]*sigma**2, Lin[5]*sigma**2, Lin[6])
                vall.append(vv)

        vall = np.array(vall)
        if vall.max()<1.0:
            return True
        else:
            return False



class GET_PriorND(object):
    def __init__(self):
        self.cosmo = np.loadtxt("/mount/sirocco1/zz681/emulator/CMASS/Gaussian_Process/hod_file/cosmology_camb_full.dat")
        self.Cosmo = np.delete(self.cosmo, 23, axis=0)
        self.center = np.mean(self.Cosmo, axis=0)
        self.cov = np.cov(self.Cosmo.T)
        
        self.icov = np.linalg.inv(self.cov)
        self.Cos_min = np.min(self.Cosmo, axis=0)
        self.Cos_max = np.max(self.Cosmo, axis=0)
        self.Cos_min[0:6] = self.Cos_min[0:6]*0.99
        self.Cos_max[0:6] = self.Cos_max[0:6]*1.01
        self.Cos_min[6] = self.Cos_min[6]*1.01
        self.Cos_max[6] = self.Cos_max[6]*0.99
        self.param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']    

    def isinornot(self, position, CUT=12):
        dif = position - self.center
        t = np.dot(dif, np.dot(self.icov, dif))
        if t<CUT:
            return True
        else:
            return False


    def get_cov(self, params_forcov):
        idxs = [self.param_names_cosmo.index(pn) for pn in params_forcov]
        cosmo_cut = self.Cosmo[:,idxs]

        return np.cov(cosmo_cut.T)

    def get_means(self, params_formean):
        idxs = [self.param_names_cosmo.index(pn) for pn in params_formean]
        cosmo_cut = self.Cosmo[:,idxs]

        return np.mean(cosmo_cut, axis=0) 
