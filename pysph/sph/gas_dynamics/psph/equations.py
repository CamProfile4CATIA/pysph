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

    def post_loop(self, d_idx, d_rho, d_h0, d_h,
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


class GradientKinsfolkC1(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [augmented_matrix, gj_solve, identity, mat_mult]

    def initialize(self, d_gradv, d_idx, d_invtt, d_divv, d_grada,
                   d_adivv, d_trssdsst):
        start_indx, i, dim = declare('int', 3)
        start_indx = 9 * d_idx
        for i in range(9):
            d_gradv[start_indx + i] = 0.0
            d_invtt[start_indx + i] = 0.0
            d_grada[start_indx + i] = 0.0

        d_divv[d_idx] = 0.0
        d_adivv[d_idx] = 0.0
        d_trssdsst[d_idx] = 0.0

    def loop(self, d_idx, d_invtt, s_m, s_idx, VIJ, DWI, XIJ, d_gradv,
             d_grada, d_au, s_au, d_av, s_av, d_aw, s_aw):
        start_indx, row, col, drowcol, dim = declare('int', 5)
        dim = self.dim
        start_indx = d_idx * 9

        aij = declare('matrix(3)')

        aij[0] = d_au[d_idx] - s_au[s_idx]
        aij[1] = d_av[d_idx] - s_av[s_idx]
        aij[2] = d_aw[d_idx] - s_aw[s_idx]

        for row in range(dim):
            for col in range(dim):
                drowcol = start_indx + row * 3 + col
                d_invtt[drowcol] -= s_m[s_idx] * XIJ[row] * DWI[col]
                d_gradv[drowcol] -= s_m[s_idx] * VIJ[row] * DWI[col]
                d_grada[drowcol] -= s_m[s_idx] * aij[row] * DWI[col]

    def post_loop(self, d_idx, d_gradv, d_invtt, d_divv, d_grada, d_adivv,
                  d_ss, d_trssdsst):
        tt = declare('matrix(9)')
        invtt = declare('matrix(9)')
        augtt = declare('matrix(18)')
        idmat = declare('matrix(9)')
        gradv = declare('matrix(9)')
        grada = declare('matrix(9)')

        start_indx, row, col, rowcol, drowcol, dim, colrow = declare('int', 7)
        ltstart_indx, dltrowcol = declare('int', 2)
        dim = self.dim
        start_indx = 9 * d_idx
        ltstart_indx = 6 * d_idx
        identity(idmat, 3)
        identity(tt, 3)

        for row in range(3):
            for col in range(3):
                rowcol = row * 3 + col
                drowcol = start_indx + rowcol
                gradv[rowcol] = d_gradv[drowcol]
                grada[rowcol] = d_grada[drowcol]

        for row in range(dim):
            for col in range(dim):
                rowcol = row * 3 + col
                drowcol = start_indx + rowcol
                tt[rowcol] = d_invtt[drowcol]

        augmented_matrix(tt, idmat, 3, 3, 3, augtt)
        gj_solve(augtt, 3, 3, invtt)
        gradvls = declare('matrix(9)')
        gradals = declare('matrix(9)')
        mat_mult(gradv, invtt, 3, gradvls)
        mat_mult(grada, invtt, 3, gradals)

        for row in range(dim):
            d_divv[d_idx] += gradvls[row * 3 + row]
            d_adivv[d_idx] += gradals[row * 3 + row]
            for col in range(dim):
                rowcol = row * 3 + col
                colrow = row + col * 3
                drowcol = start_indx + rowcol
                d_gradv[drowcol] = gradvls[rowcol]
                d_grada[drowcol] = gradals[rowcol]
                d_adivv[d_idx] -= gradals[rowcol] + gradals[colrow]

        # Traceless Symmetric Strain Rate
        divvbydim = d_divv[d_idx] / dim
        start_indx = d_idx * 9
        ltstart_indx = d_idx * 6
        for row in range(dim):
            col = row
            rowcol = start_indx + row * 3 + col
            dltrowcol = ltstart_indx + (row * (row + 1)) / 2 + col
            d_ss[dltrowcol] = d_gradv[rowcol] - divvbydim

        for row in range(1, dim):
            for col in range(row):
                rowcol = row * 3 + col
                colrow = row + col * 3
                dltrowcol = ltstart_indx + (row * (row + 1)) / 2 + col
                d_ss[dltrowcol] = 0.5 * (gradvls[rowcol] + gradvls[colrow])

        # Trace ( S dot S )
        for row in range(dim):
            for col in range(dim):
                dltrowcol = ltstart_indx + (row * (row + 1)) / 2 + col
                d_trssdsst[d_idx] += d_ss[dltrowcol] * d_ss[dltrowcol]


class SignalVelocity(Equation):
    def initialize(self, d_idx, d_vsig):
        d_vsig[d_idx] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, d_u, d_v, d_w, s_u,
                 s_v, s_w, d_cs, s_cs, d_vsig, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix(3)')
        vij = declare('matrix(3)')
        vijdotxij = 0.0
        cij = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            xij[2] = d_z[d_idx] - s_z[s_idx]

            vij[0] = d_u[d_idx] - s_u[s_idx]
            vij[1] = d_v[d_idx] - s_v[s_idx]
            vij[2] = d_w[d_idx] - s_w[s_idx]

            vijdotxij = vij[0] * xij[0] + vij[1] * xij[1] + vij[2] * xij[2]
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            d_vsig[d_idx] = max(d_vsig[d_idx], cij - min(0, vijdotxij))


class LimiterAndAlphas(Equation):
    def __init__(self, dest, sources, alphamin=0.02, alphamax=2.0, betac=0.7,
                 betad=0.05, betaxi=1.0, betab=1.0, fkern=1.0):
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.betac = betac
        self.betad = betad
        self.betaxi = betaxi
        self.betab = betab
        self.fkern = fkern
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_xi):
        d_xi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_xi, s_divv, WI):
        if abs(s_divv[s_idx]) < 1e-10:
            sign = 0.0
        elif s_divv[s_idx] < 0:
            sign = -1.0
        else:
            sign = 1.0

        d_xi[d_idx] += sign * s_m[s_idx] * WI

    def post_loop(self, d_idx, d_xi, d_rho, d_h, d_adivv, d_cs, d_alpha0,
                  d_vsig, dt, d_divv, d_trssdsst, d_alpha, d_alpha1):
        d_xi[d_idx] = 1.0 - d_xi[d_idx] / d_rho[d_idx]
        fhi = self.fkern * d_h[d_idx]

        if d_adivv[d_idx] >= 0 or d_divv[d_idx] >= 0:
            alphatmp = 0.0
        else:
            absadivv = abs(d_adivv[d_idx])
            csbyfhi = d_cs[d_idx] / fhi
            alphatmp = self.alphamax * absadivv / (
                    absadivv + self.betac * csbyfhi * csbyfhi)

        if alphatmp >= d_alpha0[d_idx]:
            d_alpha0[d_idx] = alphatmp
        elif alphatmp < d_alpha0[d_idx]:  # Isn't this obvious?
            epow = exp(-self.betad * dt * abs(d_vsig[d_idx]) * 0.5 / fhi)
            d_alpha0[d_idx] = alphatmp + (d_alpha0[d_idx] - alphatmp) * epow

        xip4 = d_xi[d_idx] * d_xi[d_idx] * d_xi[d_idx] * d_xi[d_idx]
        alnumtt = self.betaxi * xip4 * d_divv[d_idx]
        alnumt = alnumtt * alnumtt
        alnum = alnumt * d_alpha0[d_idx]
        alden = alnumt + d_trssdsst[d_idx]

        if alden < 1e-8:
            d_alpha[d_idx] = self.alphamin
        else:
            d_alpha[d_idx] = max(alnum/alden, self.alphamin)

class MomentumAndEnergy(Equation):
    def __init__(self, dest, sources, dim, fkern, beta=2.0):
        self.beta = beta
        self.dim = dim
        self.fkern = fkern
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs, d_rho, s_rho,
             d_au, d_av, d_aw, d_ae, XIJ, VIJ, DWI, DWJ, HIJ, d_alpha1,
             s_alpha1, RIJ, R2IJ, RHOIJ, d_dt_cfl, d_h, d_dndh, d_n,
             d_drhosumdh, s_h, s_dndh, s_n, s_drhosumdh):

        dim = self.dim

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

        if RIJ < 1e-8:
            vs = 2 * cij
        else:
            vs = 2 * cij - 3 * vijdotxij / RIJ


        # Is this really reqd?
        # # v_{ij} \cdot r_{ij} or vijdotxij
        # dot = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]
        #
        # # compute the Courant-limited time step factor.
        # d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], cij + self.beta * dot)

        # Artificial viscosity
        if vijdotxij <= 0.0:
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
        hibynidim = d_h[d_idx] / (d_n[d_idx] * dim)
        inbrkti = 1 + d_dndh[d_idx] * d_h[d_idx] * hibynidim
        inprthsi = d_drhosumdh[d_idx] * hibynidim
        fij = 1 - inprthsi / (s_m[s_idx] * inbrkti)

        hjbynjdim = s_h[s_idx] / (s_n[s_idx] * dim)
        inbrktj = 1 + s_dndh[s_idx] * s_h[s_idx] * hjbynjdim
        inprthsj = s_drhosumdh[s_idx] * hibynidim
        fji = 1 - inprthsj / (d_m[d_idx] * inbrktj)

        # accelerations for velocity
        mj_pibrhoi_fij = mj * pibrhoi2 * fij
        mj_pjbrhoj_fji = mj * pjbrhoj2 * fji

        d_au[d_idx] -= mj_pibrhoi_fij * DWI[0] + mj_pjbrhoj_fji * DWJ[0]
        d_av[d_idx] -= mj_pibrhoi_fij * DWI[1] + mj_pjbrhoj_fji * DWJ[1]
        d_aw[d_idx] -= mj_pibrhoi_fij * DWI[2] + mj_pjbrhoj_fji * DWJ[2]

        # accelerations for the thermal energy
        vijdotdwi = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]
        d_ae[d_idx] += mj * pibrhoi2 * fij * vijdotdwi