from math import pi

M_1_PI = 1.0 / pi


class CubicSplineH1(object):
    r"""Cubic Spline Kernel with support-radius = 1 """

    def __init__(self, dim=1):
        self.radius_scale = 1.0
        self.dim = dim
        self.fkern = 0.5

        if dim == 3:
            self.fac = 8.0 * M_1_PI
        elif dim == 2:
            self.fac = 40.0 * M_1_PI / 7.0
        else:
            self.fac = 4.0 / 3.0

    def ndkernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
        if self.dim == 1:
            fac = h
        elif self.dim == 2:
            fac = h * h
        elif self.dim == 3:
            fac = h * h * h

        return fac * self.kernel(xij=xij, rij=rij, h=h)

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):

        h1 = 1. / h
        q = rij * h1

        fac = self.fac

        if self.dim == 1:
            fac *= h1
        elif self.dim == 2:
            fac *= h1 * h1
        elif self.dim == 3:
            fac *= h1 * h1 * h1

        tmp2 = 1. - q

        if q >= 1.0:
            val = 0.0
        elif q > 0.5:
            val = 2.0 * tmp2 * tmp2 * tmp2
        else:
            val = 1 - 6 * q * q * tmp2

        return val * fac

    def dwdq(self, rij=1.0, h=1.0):

        h1 = 1. / h
        q = rij * h1

        fac = self.fac

        if self.dim == 1:
            fac *= h1
        elif self.dim == 2:
            fac *= h1 * h1
        elif self.dim == 3:
            fac *= h1 * h1 * h1

        # compute sigma * dw_dq
        tmp2 = 1. - q
        if rij > 1e-12:
            if q >= 1.0:
                val = 0.0
            elif q > 0.5:
                val = -6 * tmp2 * tmp2
            else:
                val = -12 * q * tmp2 + 6 * q * q
        else:
            val = 0.0

        return val * fac

    def nddwdq(self, rij=1.0, h=1.0):

        if self.dim == 1:
            fac = h
        elif self.dim == 2:
            fac = h * h
        elif self.dim == 3:
            fac = h * h * h

        return fac * self.dwdq(rij=rij, h=h)

    def tilw(self, rij=1.0, h=1.0):

        h1 = 1. / h
        q = rij * h1

        if rij > 1e-12:
            return self.nddwdq(rij, h) / q
        else:
            return 0.0

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0, 0, 0]):
        h1 = 1. / h
        # compute the gradient.
        if (rij > 1e-12):
            wdash = self.dwdq(rij, h)
            tmp = wdash * h1 / rij
        else:
            tmp = 0.0

        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    def gradient_h(self, xij=[0., 0, 0], rij=1.0, h=1.0):
        h1 = 1. / h
        q = rij * h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # kernel and gradient evaluated at q
        tmp2 = 2. - q
        if (q > 2.0):
            w = 0.0
            dw = 0.0

        elif (q > 1.0):
            w = 2 * tmp2 * tmp2 * tmp2
            dw = -6 * tmp2 * tmp2
        else:
            w = 1 - 6 * q * q * tmp2
            dw = -12 * q * tmp2 + 6 * q * q

        return -fac * h1 * (dw * q + w * self.dim)
