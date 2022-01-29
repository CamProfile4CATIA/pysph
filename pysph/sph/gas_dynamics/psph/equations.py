from compyle.api import declare
from pysph.base.particle_array import get_ghost_tag

from pysph.sph.equation import Equation
from pysph.sph.wc.linalg import identity, gj_solve, augmented_matrix, mat_mult

GHOST_TAG = get_ghost_tag()


class PSPHSummationDensityAndPressure(Equation):
    def __init__(self, dest, sources, dim, gamma, density_iterations=False,
                 iterate_only_once=False, k=1.2, htol=1e-6):

        r"""Summation density with iterative solution of the smoothing lengths
        from pysph.sph.gas_dynamics.basic modified to use number density
        for calculation of grad-h terms.
        """

        self.density_iterations = density_iterations
        self.iterate_only_once = iterate_only_once
        self.dim = dim
        self.k = k
        self.htol = htol
        self.equation_has_converged = 1
        self.gamma = gamma
        self.gammam1 = gamma - 1.0

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_arho, d_n, d_dndh,
                   d_prevn, d_prevdndh, d_p, d_dpsumdh,
                   d_dprevpsumdh, d_an):

        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_prevn[d_idx] = d_n[d_idx]
        d_prevdndh[d_idx] = d_dndh[d_idx]
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0
        d_an[d_idx] = 0.0

        d_p[d_idx] = 0.0
        d_dprevpsumdh[d_idx] = d_dpsumdh[d_idx]
        d_dpsumdh[d_idx] = 0.0

        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_arho, s_m, VIJ, WI, DWI,
             GHI, d_n, d_dndh, d_h, d_prevn, d_prevdndh,
             s_e, d_p, d_dpsumdh, d_e, d_an):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]

        # density
        mj_wi = mj * WI
        d_rho[d_idx] += mj_wi
        d_p[d_idx] += self.gammam1 * s_e[s_idx] * mj_wi

        # number density accelerations
        hibynidim = d_h[d_idx] / (d_prevn[d_idx] * self.dim)
        inbrkti = 1 + d_prevdndh[d_idx] * hibynidim
        inprthsi = d_dpsumdh[d_idx] * hibynidim / (
                self.gammam1 * s_m[s_idx] * d_e[d_idx])
        fij = 1 - inprthsi / inbrkti
        vijdotdwij_fij = vijdotdwij * fij
        d_an[d_idx] += vijdotdwij_fij

        # density acceleration is not essential as such
        d_arho[d_idx] += mj * vijdotdwij_fij

        # gradient of kernel w.r.t h
        d_dpsumdh[d_idx] += mj * self.gammam1 * d_e[d_idx] * GHI
        d_n[d_idx] += WI
        d_dndh[d_idx] += GHI

    def post_loop(self, d_idx, d_rho,d_h0, d_h,
                  d_ah, d_converged, d_cs, d_p, d_n, d_dndh,
                  d_an):

        d_cs[d_idx] = sqrt(self.gamma * d_p[d_idx] / d_rho[d_idx])

        # iteratively find smoothing length consistent with the
        if self.density_iterations:
            if not (d_converged[d_idx] == 1):
                hi = d_h[d_idx]
                hi0 = d_h0[d_idx]

                # estimated, without summations
                ni = (self.k / hi) ** self.dim
                dndhi = - self.dim * d_n[d_idx] / hi

                # correct fi TODO: Remove if not required
                # if fi < 0:
                #     fi = 1.0

                # the non-linear function and it's derivative
                func = d_n[d_idx] - ni
                dfdh = d_dndh[d_idx] - dndhi

                # Newton Raphson estimate for the new h
                hnew = hi - func / dfdh

                # Nanny control for h TODO: Remove if not required
                if hnew > 1.2 * hi:
                    hnew = 1.2 * hi
                elif hnew < 0.8 * hi:
                    hnew = 0.8 * hi

                # overwrite if gone awry TODO: Remove if not required
                # if (hnew <= 1e-6) or (fi < 1e-6):
                #     hnew = self.k * (mi / d_rho[d_idx]) ** (1. / self.dim)

                # check for convergence
                diff = abs(hnew - hi) / hi0

                # if not ((diff < self.htol) and (fi > 0) or
                #         self.iterate_only_once):
                if not ((diff < self.htol) or
                        self.iterate_only_once):
                    # this particle hasn't converged. This means the
                    # entire group must be repeated until this fellow
                    # has converged, or till the maximum iteration has
                    # been reached.
                    self.equation_has_converged = -1

                    # set particle properties for the next
                    # iteration. For the 'converged' array, a value of
                    # 0 indicates the particle hasn't converged
                    d_h[d_idx] = hnew
                    d_converged[d_idx] = 0
                else:
                    d_ah[d_idx] = d_an[d_idx] / dndhi
                    d_converged[d_idx] = 1

    def converged(self):
        return self.equation_has_converged
