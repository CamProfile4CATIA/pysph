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
        self.gammam1= gamma-1.0

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_arho, d_drhosumdh, d_n, d_dndh,
                   d_prevn, d_prevdndh, d_prevdrhosumdh, d_p):

        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_prevn[d_idx] = d_n[d_idx]
        d_prevdrhosumdh[d_idx] = d_drhosumdh[d_idx]
        d_prevdndh[d_idx] = d_dndh[d_idx]

        d_drhosumdh[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0

        d_p[d_idx] = 0.0

        # set the converged attribute for the Equation to True. Within
        # the post-loop, if any particle hasn't converged, this is set
        # to False. The Group can therefore iterate till convergence.
        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_arho, d_drhosumdh, s_m, VIJ, WI, DWI,
             GHI, d_n, d_dndh, d_h, d_prevn, d_prevdndh, d_prevdrhosumdh,
             s_e, d_p):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]

        # density
        d_rho[d_idx] += mj * WI
        d_p[d_idx] += self.gammam1 * s_e[s_idx] * mj * WI
        # density accelerations
        hibynidim = d_h[d_idx] / (d_prevn[d_idx] * self.dim)
        inbrkti = 1 + d_prevdndh[d_idx] * d_h[d_idx] * hibynidim
        inprthsi = d_prevdrhosumdh[d_idx] * hibynidim
        fij = 1 - inprthsi / (s_m[s_idx] * inbrkti)
        d_arho[d_idx] += mj * vijdotdwij * fij

        # gradient of kernel w.r.t h
        d_drhosumdh[d_idx] += mj * GHI
        d_n[d_idx] += WI
        d_dndh[d_idx] += GHI

    def post_loop(self, d_idx, d_arho, d_rho, d_drhosumdh, d_h0, d_h, d_m,
                  d_ah, d_converged, d_cs, d_p):

        d_cs[d_idx] = sqrt(self.gamma * d_p[d_idx] / d_rho[d_idx])

        # iteratively find smoothing length consistent with the
        if self.density_iterations:
            if not (d_converged[d_idx] == 1):
                # current mass and smoothing length. The initial
                # smoothing length h0 for this particle must be set
                # outside the Group (that is, in the integrator)
                mi = d_m[d_idx]
                hi = d_h[d_idx]
                hi0 = d_h0[d_idx]

                # density from the mass, smoothing length and kernel
                # scale factor
                rhoi = mi / (hi / self.k) ** self.dim

                # using fi from density entropy formulation for convergence
                # related checks.
                dhdrhoi = -hi / (self.dim * d_rho[d_idx])
                obyfi = 1.0 - dhdrhoi * d_drhosumdh[d_idx]

                # correct fi TODO: Remove if not required
                if obyfi < 0:
                    obyfi = 1.0

                # kernel multiplier. These are the multiplicative
                # pre-factors, or the "grah-h" terms in the
                # equations. Remember that the equations use 1/omega
                fi = 1.0 / obyfi

                # the non-linear function and it's derivative
                func = d_rho[d_idx] - rhoi
                dfdh = d_drhosumdh[d_idx] - 1 / dhdrhoi

                # Newton Raphson estimate for the new h
                hnew = hi - func / dfdh

                # Nanny control for h TODO: Remove if not required
                if (hnew > 1.2 * hi):
                    hnew = 1.2 * hi
                elif (hnew < 0.8 * hi):
                    hnew = 0.8 * hi

                # overwrite if gone awry TODO: Remove if not required
                if ((hnew <= 1e-6) or (fi < 1e-6)):
                    hnew = self.k * (mi / d_rho[d_idx]) ** (1. / self.dim)

                # check for convergence
                diff = abs(hnew - hi) / hi0

                if not ((diff < self.htol) and (obyfi > 0) or
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
                    d_ah[d_idx] = d_arho[d_idx] * dhdrhoi
                    d_converged[d_idx] = 1

    def converged(self):
        return self.equation_has_converged

