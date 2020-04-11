import numpy as np
from scipy.optimize import minimize

import george
from george import kernels


np.random.seed(1234)

x = 10 * np.sort(np.random.rand(15))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))




kernel = np.var(y) * kernels.ExpSquaredKernel(0.5)
gp = george.GP(kernel)
gp.compute(x, yerr)

#x_pred = np.linspace(0, 10, 500)
#pred, pred_var = gp.predict(y, x_pred, return_var=True)


def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)

result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)

gp.set_parameter_vector(result.x)
print(result.x)
np.savetxt('hyperparams_george_test.dat', gp.kernel.get_parameter_vector(), fmt='%.7f')

#pred, pred_var = gp.predict(y, x_pred, return_var=True)