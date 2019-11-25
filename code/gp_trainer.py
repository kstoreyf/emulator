import numpy as np
import scipy as sp
import george
from george.kernels import ExpSquaredKernel
from george import kernels
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as op
import emcee
from scipy.linalg import cholesky, cho_solve
from scipy.spatial import cKDTree as KDTree
from scipy import stats


class gp_tr(object):
    def __init__(self, x, y, yerr, gp, optimize=False, MCMC=False, save=False, savename='GP_mcmc'):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.gp = gp
        #self.p0 = gp.kernel.pars
        self.p0 = gp.kernel.get_parameter_vector() # is this the same thing in new george??
        # self.gp.compute(self.x)
        print(self.p0)
        if optimize == True and MCMC == True:
            print("Both optimization and MCMC are chosen, only optimization will run...")
            MCMC = False

        elif optimize == True and MCMC == False:
            self.bnd = [np.log((1e-6, 1e+6)) for i in range(len(self.p0))]
            self.results = op.minimize(self.nll, self.p0, jac=self.grad_nll, method='L-BFGS-B', bounds=self.bnd)
            #self.gp.kernel[:] = self.results.x
            self.gp.set_parameter_vector(self.results.x)
            self.p_op = self.results.x

        elif optimize == False and MCMC == False:
            self.p_op = self.gp.kernel.vector

        elif optimize == False and MCMC == True:
            ndim = len(self.p0)
            nwalkers = 2 * ndim
            Nstep = 5
            print((ndim, nwalkers))
            position = [self.p0 + 1e-3 * np.random.randn(len(self.p0)) for jj in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
            sampler.run_mcmc(position, Nstep)
            samples = sampler.chain[:, :, :].reshape((-1, ndim))
            chi2p = sampler.lnprobability[:, :].reshape(-1)

            mcmc_data = list(zip(-chi2p, samples))
            mcmc_data = np.array(mcmc_data)
            self.p_op1 = mcmc_data[np.where(mcmc_data[:, 0] == min(mcmc_data[:, 0]))]
            #self.p_op = self.p_op[0]
            #??
            print(self.p_op1)
            print(self.p_op1[0][1])
            self.p_op = self.p_op1[0][1]
            if save == True:
                np.savetxt(savename + "_mcmc.dat", mcmc_data, fmt='%.6e')

    def nll(self, p):
        #self.gp.kernel[:] = p
        self.gp.set_parameter_vector(p)
        ll = self.gp.lnlikelihood(self.y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, p):
        #self.gp.kernel[:] = p
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_lnlikelihood(self.y, quiet=True)

    def lnprob(self, p):
        if np.any((-5.0 > p) + (p > 5.0)):
            return -np.inf
        lnprior = 0.0
        # self.gp.compute(self.x)
        #self.gp.kernel[:] = p
        self.gp.set_parameter_vector(p)
        #print(self.gp.lnlikelihood(self.y, quiet=True))
        return lnprior + self.gp.lnlikelihood(self.y, quiet=True)


class gp_tr_new(object):
    def __init__(self, x, y, yerr, gp, optimize=False):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.gp = gp
        self.p0 = gp.kernel.pars
        # self.gp.compute(self.x)

        if optimize == True:
            self.bnd = [np.log((1e-6, 1e+6)) for i in range(len(self.p0))]
            self.results = op.minimize(self.nll, self.p0, method='L-BFGS-B', bounds=self.bnd)
            self.gp.kernel[:] = self.results.x
            self.p_op = self.results.x

        elif optimize == False:
            self.p_op = self.gp.kernel.vector

    def nll(self, p):
        self.gp.kernel[:] = p
        self.pre, self.cov = self.gp.predict(self.y, self.x * 1.0)
        ll = sum(((self.pre - self.y) / (self.yerr)) ** 2.0)
        # ll = sum(((self.pre-self.y)/(self.yerr+np.sqrt(np.diag(self.cov))))**2.0)
        return ll if np.isfinite(ll) else 1e25

    def grad_nll(self, p):
        self.gp.kernel[:] = p
        return -self.gp.grad_lnlikelihood(self.y, quiet=True)


class Invdisttree:
    # this module is from stackoverflow
    """ inverse-distance-weighted interpolation using KDTree:
        invdisttree = Invdisttree( X, z )  -- data points, values
        interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        interpolates z from the 3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        q may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        p: use 1 / distance**p
        weights: optional multipliers for 1 / distance**p, of the same shape as q
        stat: accumulate wsum, wn for average weights

        How many nearest neighbors should one take ?
        a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
        b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

        p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.

        Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .

        A nice property of IDW is that it's scale-free around query points:
        if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
        the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
        In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
        is exceedingly sensitive to distance and to h.

        """

    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, X, z, leafsize=10, stat=0):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__(self, q, nnear=6, eps=0, p=1, weights=None):
        # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist ** p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]


class INV_intp(object):
    def __init__(self, x, y, metric, leafsize=20, stats=1, optimize=False):
        self.x = x
        self.y = y
        self.metric = metric
        self.leafsize = leafsize
        self.stats = stats

        if optimize == True:
            self.bnd = [np.log((1e-6, 1e+6)) for i in range(len(self.metric))]
            self.results = op.minimize(self.nll, self.metric, method='L-BFGS-B', bounds=self.bnd)
            self.p_op = self.results.x

    def nll(self, p):
        self.x = self.x - np.min(self.x, axis=0)
        self.x = slef.x * self.metric
        self.invdisttree = Invdisttree(self.x, np.array(self.y), leafsize=self.leafsize, stat=self.stats)
        self.pre = self.invdisttree(self.t, nnear=14, eps=0, p=0)
        ll = sum(((self.pre - self.y) / (self.yerr)) ** 2.0)
        return ll if np.isfinite(ll) else 1e25


class HODspace:
    def __init__(self, x, FIXed=True, Scale=True):
        self.FIXed = FIXed
        self.x = x
        self.Scale = Scale
        self.xs = np.empty((self.x.shape[0], self.x.shape[1]))

        if self.Scale == True:

            if FIXed == True:
                self.x0m = 14.5
                self.x0n = 13.8
                self.x1m = 1.8
                self.x1n = 0.2
                self.x2m = 13.7
                self.x2n = 10.0
                self.x3m = 0.6
                self.x3n = 0.05
            else:
                self.x0m = max(self.x[:, 0])
                self.x0n = min(self.x[:, 0])
                self.x1m = max(self.x[:, 1])
                self.x1n = min(self.x[:, 1])
                self.x2m = max(self.x[:, 2])
                self.x2n = min(self.x[:, 2])
                self.x3m = max(self.x[:, 3])
                self.x3n = min(self.x[:, 3])

            self.xs[:, 0] = (self.x[:, 0] - self.x0n) / (self.x0m - self.x0n)
            self.xs[:, 1] = (self.x[:, 1] - self.x1n) / (self.x1m - self.x1n)
            self.xs[:, 2] = (self.x[:, 2] - self.x2n) / (self.x2m - self.x2n)
            self.xs[:, 3] = (self.x[:, 3] - self.x3n) / (self.x3m - self.x3n)

            self.mid = np.array([0.5, 0.5, 0.5, 0.5])

    def ISinPars(self, t):
        self.ts1 = list(t)
        if len(t) != len(self.x[0]):
            print("dimension dismatch, exit...")
            exit()
        self.ts1[0] = (t[0] - self.x0n) / (self.x0m - self.x0n)
        self.ts1[1] = (t[1] - self.x1n) / (self.x1m - self.x1n)
        self.ts1[2] = (t[2] - self.x2n) / (self.x2m - self.x2n)
        self.ts1[3] = (t[3] - self.x3n) / (self.x3m - self.x3n)
        if min(ts1) > 0.0 and max(ts1) < 1.0:
            return self.ts1
        else:
            print("outside the HOD parameter space")
            return self.mid

    def Shrink(self, t, pers=0.8, after=True):
        self.ts2 = list(t)
        if len(t) != len(self.x[0]):
            print("dimension dismatch, exit...")
            exit()
        self.ts2[0] = (t[0] - self.x0n) / (self.x0m - self.x0n)
        self.ts2[1] = (t[1] - self.x1n) / (self.x1m - self.x1n)
        self.ts2[2] = (t[2] - self.x2n) / (self.x2m - self.x2n)
        self.ts2[3] = (t[3] - self.x3n) / (self.x3m - self.x3n)
        if min(self.ts2) > (1.0 - pers) / 2.0 and max(self.ts2) < 1.0 - (1.0 - pers) / 2.0:
            return True
        else:
            return False


class Training_COS_HOD_fsat:
    # for training samples only
    def __init__(self, CID, HID):
        self.CID = CID
        self.HID = HID

    def value(self, name='fsat'):
        self.fold = open(
            "/export/sirocco1/zz681/emulator/CMASS/Gaussian_Process//CMASS_training_error/output_fsat/output_cosmo_" + str(
                self.CID) + "_Box_0_HOD_" + str(self.HID) + ".dat", "r")
        for line in self.fold:
            if name in line[:10]:
                newline = line.split()
                if name == 'M_min':
                    fsat = float(newline[1])
                else:
                    fsat = float(newline[-1])
                return fsat


class Training_wp_COS_HOD_error:
    # for training samples only
    def __init__(self, CID, HID, rpbin):
        self.CID = CID
        self.HID = HID
        self.rpbin = rpbin
        self.fsat = Training_COS_HOD_fsat(self.CID, self.HID).value('fsat')
        self.M_eff = Training_COS_HOD_fsat(self.CID, self.HID).value('M_eff')

        self.GP_error = np.loadtxt(
            "/export/sirocco1/zz681/emulator/CMASS/Gaussian_Process/wp_covar/wp_covar_data/data/Minerva_error.dat")

    def error(self):
        if self.rpbin == 0:
            if self.M_eff > 5.5e14:
                return 0.015
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([0.025, 0.005])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * np.log10(self.fsat) + coef[1]

        elif self.rpbin == 1:
            if self.M_eff > 5.5e14:
                return 0.015
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([0.018, 0.005])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * np.log10(self.fsat) + coef[1]

        elif self.rpbin == 2:
            if self.M_eff > 5.5e14:
                return 0.015
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([0.018, 0.007])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * np.log10(self.fsat) + coef[1]

        elif self.rpbin == 3:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.018, 0.06])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        elif self.rpbin == 4:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin] * 1.2
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.01, 0.06])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        elif self.rpbin == 5:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin] * 1.2
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.016, 0.036])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        elif self.rpbin == 6:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin] * 1.4
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.018, 0.025])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        else:
            return self.GP_error[self.rpbin]


class Training_mono_COS_HOD_error:
    # for training samples only
    def __init__(self, CID, HID, rpbin):
        self.CID = CID
        self.HID = HID
        self.rpbin = rpbin
        self.fsat = Training_COS_HOD_fsat(self.CID, self.HID).value('fsat')
        self.M_eff = Training_COS_HOD_fsat(self.CID, self.HID).value('M_eff')

        self.GP_error = np.loadtxt("../RSD_multiple/RSD_multiple_data/data/Cosmo_err.dat")
        self.GP_error = self.GP_error[1:10, 0]

    def error(self):
        if self.rpbin == 0:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([0.04, 0.01])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * np.log10(self.fsat) + coef[1]

        elif self.rpbin == 1:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([0.028, 0.006])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * np.log10(self.fsat) + coef[1]

        elif self.rpbin == 2:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([0.014, 0.004])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * np.log10(self.fsat) + coef[1]

        elif self.rpbin == 3:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.005, 0.016])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        elif self.rpbin == 4:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin] * 1.2
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.005, 0.016])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        elif self.rpbin == 5:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin] * 1.2
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.005, 0.016])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        elif self.rpbin == 6:
            if self.M_eff < 4.5e13:
                return self.GP_error[self.rpbin] * 1.4
            else:
                x2 = np.array([4.5, 8]) * (10 ** 13)
                y2 = np.array([0.01, 0.022])
                coef = np.polyfit(x2, y2, 1)
                return coef[0] * self.M_eff + coef[1]

        else:
            return self.GP_error[self.rpbin]


class Training_quad_COS_HOD_error:
    # for training samples only
    def __init__(self, CID, HID, rpbin):
        self.CID = CID
        self.HID = HID
        self.rpbin = rpbin
        self.fsat = Training_COS_HOD_fsat(self.CID, self.HID).value('fsat')
        self.M_eff = Training_COS_HOD_fsat(self.CID, self.HID).value('M_eff')

        self.GP_error = np.loadtxt("../RSD_multiple/RSD_multiple_data/data/Cosmo_err.dat")
        self.GP_error = self.GP_error[1:10, 1]

    def error(self):
        if self.rpbin == 0:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([-0.7, -1.7])
                coef = np.polyfit(x2, y2, 1)
                return 10 ** (coef[0] * np.log10(self.fsat) + coef[1])

        elif self.rpbin == 1:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([-1.0, -2.0])
                coef = np.polyfit(x2, y2, 1)
                return 10 ** (coef[0] * np.log10(self.fsat) + coef[1])

        elif self.rpbin == 2:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([-0.9, -2.3])
                coef = np.polyfit(x2, y2, 1)
                return 10 ** (coef[0] * np.log10(self.fsat) + coef[1])

        elif self.rpbin == 3:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin]
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([-0.9, -2.5])
                coef = np.polyfit(x2, y2, 1)
                return 10 ** (coef[0] * np.log10(self.fsat) + coef[1])

        elif self.rpbin == 4:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin] * 1.2
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([-1.15, -2.45])
                coef = np.polyfit(x2, y2, 1)
                return 10 ** (coef[0] * np.log10(self.fsat) + coef[1])

        elif self.rpbin == 5:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin] * 1.2
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([0.0, -2.2])
                coef = np.polyfit(x2, y2, 1)
                return 10 ** (coef[0] * np.log10(self.fsat) + coef[1])

        elif self.rpbin == 6:
            if self.M_eff > 5.5e14:
                return self.GP_error[self.rpbin] * 1.4
            else:
                x2 = np.array([-2, -0.5])
                y2 = np.array([-1.8, -1.0])
                coef = np.polyfit(x2, y2, 1)
                return 10 ** (coef[0] * np.log10(self.fsat) + coef[1])

        else:
            return self.GP_error[self.rpbin] * 1.2


class Test_COS_HOD_fsat:
    # for test cosmologies (and boxes) only
    def __init__(self, testCID, testHID):
        self.testCID = testCID
        self.testHID = testHID

    def value(self, name='fsat'):
        self.fold = open(
            "/Users/zhongxuzhai/Documents/work/research/LSS/emulator/CMASS/CMASS_error_analysis/test_HOD_1000/output_cosmo_" + str(
                self.testCID) + "_Box_0_HOD_" + str(self.testHID) + ".dat", "r")
        for line in self.fold:
            if name in line[:10]:
                newline = line.split()
                if name == 'M_min':
                    fsat = float(newline[1])
                else:
                    fsat = float(newline[-1])
                return fsat


class Test_wp_COS_error_COS_HOD:
    # for test cosmologies (and boxes) only
    def __init__(self, testCID, testHID):
        self.testCID = testCID
        self.testHID = testHID

        self.dd = np.empty((9, 10 * 5))
        self.dd_box = np.empty((9, 5))
        s = 0
        for BID in range(0, 5):
            for j in range(0, 10):
                d = np.loadtxt(
                    "/Users/zhongxuzhai/Documents/work/research/LSS/emulator/CMASS/GP/GP_test_data/Test_Box_wp_covar/wp_covar_cosmo_" + str(
                        self.testCID) + "_Box_" + str(BID) + "_HOD_" + str(self.testHID) + "_test_" + str(j) + ".dat")
                self.dd[:, s] = d[:, 1]
                s = s + 1
            self.dd_box[:, BID] = np.mean(self.dd[:, s - 10:s], axis=1)
        self.wp_mean = np.mean(self.dd, axis=1)

    def total_error(self):
        err = np.std(self.dd, axis=1)
        return err / self.wp_mean

    def box_error(self):
        err = np.std(self.dd_box, axis=1)
        return err / self.wp_mean

    def total_residual(self):
        self.res = np.empty((9, 10 * 5))
        for i in range(10 * 5):
            self.res[:, i] = self.dd[:, i] / self.wp_mean - 1.0
        return self.res

    def box_residual(self):
        self.res_box = np.empty((9, 5))
        for i in range(5):
            self.res_box[:, i] = self.dd_box[:, i] / self.wp_mean - 1.0
        return self.res_box

    def box_error_of_error(self):
        var = np.var(self.dd_box, axis=1)
        std = np.std(self.dd_box, axis=1)
        u2 = stats.moment(self.dd_box, moment=2, axis=1)
        u4 = stats.moment(self.dd_box, moment=4, axis=1)
        N = self.dd_box.shape[1]
        var_var = (N - 1.0) ** 2.0 / N ** 3.0 * u4 - (N - 1.0) * (N - 3.0) * u2 ** 2.0 / N ** 3.0
        std_var = np.sqrt(var_var)
        std_std = 0.5 / std * std_var
        return std_std / self.wp_mean


class Test_RSD_COS_error_COS_HOD:
    # for test cosmologies (and boxes) only
    def __init__(self, testCID, testHID):
        self.testCID = testCID
        self.testHID = testHID

        self.ddmono = np.empty((9, 10 * 5))
        self.ddmono_box = np.empty((9, 5))
        self.ddquad = np.empty((9, 10 * 5))
        self.ddquad_box = np.empty((9, 5))
        s = 0
        for BID in range(0, 5):
            for j in range(0, 10):
                d = np.loadtxt(
                    "/Users/zhongxuzhai/Documents/work/research/LSS/emulator/CMASS/GP/GP_test_data/Test_Box_RSD_multiple/RSD_multiple_cosmo_" + str(
                        self.testCID) + "_Box_" + str(BID) + "_HOD_" + str(self.testHID) + "_test_" + str(j) + ".dat")
                self.ddmono[:, s] = d[1:10, 1]
                self.ddquad[:, s] = d[1:10, 3]
                s = s + 1
            self.ddmono_box[:, BID] = np.mean(self.ddmono[:, s - 10:s], axis=1)
            self.ddquad_box[:, BID] = np.mean(self.ddquad[:, s - 10:s], axis=1)
        self.mono_mean = np.mean(self.ddmono, axis=1)
        self.quad_mean = np.mean(self.ddquad, axis=1)

    def total_mono_error(self):
        err = np.std(self.ddmono, axis=1)
        return err / self.mono_mean

    def box_mono_error(self):
        err = np.std(self.ddmono_box, axis=1)
        return err / self.mono_mean

    def total_quad_error(self):
        err = np.std(self.ddquad, axis=1)
        return err / self.quad_mean

    def box_quad_error(self):
        err = np.std(self.ddquad_box, axis=1)
        return err / self.quad_mean

    def total_mono_residual(self):
        self.res = np.empty((9, 10 * 5))
        for i in range(10 * 5):
            self.res[:, i] = self.ddmono[:, i] / self.mono_mean - 1.0
        return self.res

    def total_quad_residual(self):
        self.res = np.empty((9, 10 * 5))
        for i in range(10 * 5):
            self.res[:, i] = self.ddquad[:, i] / self.quad_mean - 1.0
        return self.res

    def box_mono_residual(self):
        self.res_box = np.empty((9, 5))
        for i in range(5):
            self.res_box[:, i] = self.ddmono_box[:, i] / self.mono_mean - 1.0
        return self.res_box

    def box_quad_residual(self):
        self.res_box = np.empty((9, 5))
        for i in range(5):
            self.res_box[:, i] = self.ddquad_box[:, i] / self.quad_mean - 1.0
        return self.res_box







