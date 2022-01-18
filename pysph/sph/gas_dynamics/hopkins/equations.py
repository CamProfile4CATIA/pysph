from pysph.sph.equation import Equation
from pysph.base.particle_array import get_ghost_tag
from compyle.api import declare
from pysph.sph.wc.linalg import identity, gj_solve, augmented_matrix, mat_mult

GHOST_TAG = get_ghost_tag()


class SummationDensity(Equation):
    def __init__(self, dest, sources, dim, density_iterations=False,
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

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_grhox, d_grhoy, d_grhoz,
                   d_arho, d_drhosumdh, d_n, d_dndh, d_prevn, d_prevdndh,
                   d_prevdrhosumdh):

        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_prevn[d_idx] = d_n[d_idx]
        d_prevdrhosumdh[d_idx] = d_drhosumdh[d_idx]
        d_prevdndh[d_idx] = d_dndh[d_idx]

        d_drhosumdh[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0

        # set the converged attribute for the Equation to True. Within
        # the post-loop, if any particle hasn't converged, this is set
        # to False. The Group can therefore iterate till convergence.
        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_grhox, d_grhoy, d_grhoz, d_arho,
             d_drhosumdh, s_m, d_converged, VIJ, WI, DWI, GHI, d_n,
             d_dndh, d_h, t, d_prevn, d_prevdndh, d_prevdrhosumdh):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]

        # density
        d_rho[d_idx] += mj * WI

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

    def post_loop(self, d_idx, d_arho, d_rho, d_div, d_omega, d_drhosumdh,
                  d_h0, d_h, d_m, d_ah, d_converged, d_dndh, s_m):

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
                # related checks. TODO: remove these checks if not required.
                dhdrhoi = -hi / (self.dim * d_rho[d_idx])
                obyfi = 1.0 - dhdrhoi * d_drhosumdh[d_idx]

                # correct fi
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

                # Nanny control for h
                if (hnew > 1.2 * hi):
                    hnew = 1.2 * hi
                elif (hnew < 0.8 * hi):
                    hnew = 0.8 * hi

                # overwrite if gone awry
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


class TSPHAccelerations(Equation):
    def __init__(self, dest, sources, dim, fkern, beta=2.0, update_alpha1=False,
                 update_alpha2=False, alpha1_min=0.1, alpha2_min=0.1,
                 sigma=0.1):
        self.beta = beta
        self.sigma = sigma

        self.update_alpha1 = update_alpha1
        self.update_alpha2 = update_alpha2

        self.alpha1_min = alpha1_min
        self.alpha2_min = alpha2_min
        self.dim = dim
        self.fkern = fkern

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae, d_am,
                   d_aalpha1, d_aalpha2, d_del2e, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

        d_aalpha1[d_idx] = 0.0
        d_aalpha2[d_idx] = 0.0

        d_del2e[d_idx] = 0.0
        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs,
             d_e, s_e, d_rho, s_rho, d_au, d_av, d_aw, d_ae,
             d_omega, s_omega, XIJ, VIJ, DWI, DWJ, DWIJ, HIJ,
             d_del2e, d_alpha1, s_alpha1, d_alpha2, s_alpha2,
             EPS, RIJ, R2IJ, RHOIJ, d_dt_cfl, d_h, d_dndh, d_n,
             d_drhosumdh, s_h, s_dndh, s_n, s_drhosumdh, s_u, s_v, s_w, d_u,
             d_v, d_w):

        # particle pressure
        p_i = d_p[d_idx]
        pj = s_p[s_idx]

        # p_i/rhoi**2
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        pibrhoi2 = p_i / rhoi2

        # pj/rhoj**2
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]
        pjbrhoj2 = pj / rhoj2

        # averaged sound speed
        cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

        mj = s_m[s_idx]
        hij = self.fkern * HIJ
        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        # normalized interaction vector
        if RIJ < 1e-8:
            XIJ[0] = 0.0
            XIJ[1] = 0.0
            XIJ[2] = 0.0
        else:
            XIJ[0] /= RIJ
            XIJ[1] /= RIJ
            XIJ[2] /= RIJ

        # v_{ij} \cdot r_{ij} or vijdotxij
        dot = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        # compute the Courant-limited time step factor. TODO: Figure this out.
        d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], cij + self.beta * dot)

        # Artificial viscosity
        if dot <= 0.0:
            # viscosity
            alpha1 = 0.5 * (d_alpha1[d_idx] + s_alpha1[s_idx])
            muij = hij * vijdotxij / (R2IJ + 0.0001 * hij ** 2)
            common = alpha1 * muij * (cij - 2 * muij) * mj / (2 * RHOIJ)

            aaui = common * (DWI[0] + DWJ[0])
            aavi = common * (DWI[1] + DWJ[1])
            aawi = common * (DWI[2] + DWJ[2])
            d_au[d_idx] += aaui
            d_av[d_idx] += aavi
            d_aw[d_idx] += aawi

            # viscous contribution to the thermal energy
            d_ae[d_idx] -= 0.5 * (
                    VIJ[0] * aaui + VIJ[1] * aavi + VIJ[2] * aawi)

        # grad-h correction terms.
        hibynidim = d_h[d_idx] / (d_n[d_idx] * self.dim)
        inbrkti = 1 + d_dndh[d_idx] * d_h[d_idx] * hibynidim
        inprthsi = d_drhosumdh[d_idx] * hibynidim
        fij = 1 - inprthsi / (s_m[s_idx] * inbrkti)

        hjbynjdim = s_h[s_idx] / (s_n[s_idx] * self.dim)
        inbrktj = 1 + s_dndh[s_idx] * s_h[s_idx] * hjbynjdim
        inprthsj = s_drhosumdh[s_idx] * hibynidim
        fji = 1 - inprthsj / (d_m[d_idx] * inbrktj)

        # accelerations for velocity
        mmj_pibrhoi_fij = -mj * pibrhoi2 * fij
        mmj_pjbrhoj_fji = -mj * pjbrhoj2 * fji

        d_au[d_idx] += mmj_pibrhoi_fij * DWI[0] + mmj_pjbrhoj_fji * DWJ[0]
        d_av[d_idx] += mmj_pibrhoi_fij * DWI[1] + mmj_pjbrhoj_fji * DWJ[1]
        d_aw[d_idx] += mmj_pibrhoi_fij * DWI[2] + mmj_pjbrhoj_fji * DWJ[2]

        # accelerations for the thermal energy
        vijdotdwi = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]
        d_ae[d_idx] += mj * pibrhoi2 * fij * vijdotdwi

    def post_loop(self, d_idx, d_h, d_cs, d_alpha1, d_aalpha1, d_divv):

        hi = d_h[d_idx]
        tau = hi / (self.sigma * d_cs[d_idx])

        if self.update_alpha1:
            S1 = max(-d_divv[d_idx], 0.0)
            d_aalpha1[d_idx] = (self.alpha1_min - d_alpha1[d_idx]) / tau + S1


class VelocityGradient(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [augmented_matrix, gj_solve, identity, mat_mult]

    def initialize(self, d_gradv, d_idx, d_invcapr):
        rowcol, dim, indx, indx_rowcol, dimsq = declare('int', 5)
        dim = self.dim
        dimsq = dim * dim
        indx = 9 * d_idx
        for rowcol in range(dimsq):
            indx_rowcol = indx + rowcol
            d_gradv[indx_rowcol] = 0.0
            d_invcapr[indx_rowcol] = 0.0

    def loop(self, d_idx, d_invcapr, s_m, s_idx, VIJ, DWI, XIJ, d_gradv):
        row, col, dim, indx_rowcol, indx = declare('int', 5)
        dim = self.dim
        indx = d_idx * 9
        for row in range(dim):
            for col in range(dim):
                indx_rowcol = indx + dim * row + col
                d_invcapr[indx_rowcol] -= s_m[s_idx] * XIJ[row] * DWI[col]
                d_gradv[indx_rowcol] -= s_m[s_idx] * VIJ[row] * DWI[col]

    def post_loop(self, d_idx, d_invcapr, s_m, s_idx, VIJ, DWI, XIJ, d_gradv):
        row, col, dim, rowcol, indx, indx_rowcol = declare('int', 7)
        dim = self.dim
        gradv = declare('matrix(9)')
        gradvls = declare('matrix(9)')
        capr = declare('matrix(9)')
        invcapr = declare('matrix(9)')
        indx = d_idx * 9

        identity(capr, 3)
        for row in range(dim):
            for col in range(dim):
                rowcol = 3 * row + col
                indx_rowcol = indx + rowcol
                gradv[rowcol] = d_gradv[indx_rowcol]
                capr[rowcol] = d_invcapr[indx_rowcol]

        idmat = declare('matrix(9)')
        identity(idmat, 3)
        auga = declare('matrix(18)')
        augmented_matrix(A=capr, b=idmat, n=3, na=3, nmax=3, result=auga)
        gj_solve(m=auga, n=3, nb=3, result=invcapr)
        mat_mult(a=gradv, b=invcapr, n=3, result=gradvls)
        for row in range(dim):
            for col in range(dim):
                rowcol = 3 * row + col
                indx_rowcol = indx + rowcol
                d_gradv[indx_rowcol] = gradvls[rowcol]
                d_invcapr[indx_rowcol] = invcapr[rowcol]


class VelocityDivergence(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def post_loop(self, d_idx, d_divv, d_gradv):
        dim, i, indx_rowcol = declare('int', 3)
        dim = self.dim
        divv = 0.0
        for i in range(dim):
            indx_rowcol = d_idx * 9 + 3 * i + i
            divv += d_gradv[indx_rowcol]
        d_divv[d_idx] = divv
