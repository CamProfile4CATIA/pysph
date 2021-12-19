from pysph.sph.equation import Equation
from math import exp, sqrt
from pysph.base.particle_array import get_ghost_tag

GHOST_TAG = get_ghost_tag()


class SummationDensity(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_hnurho, d_hnu, d_h):
        d_hnurho[d_idx] = 0.0
        d_hnu[d_idx] = d_h[d_idx] ** self.dim

    def loop(self, d_idx, s_idx, s_m, d_hnurho, WI, d_hnu):
        d_hnurho[d_idx] += s_m[s_idx] * WI * d_hnu[d_idx]

    def post_loop(self, d_idx, d_hnurho, d_rho, d_hnu):
        d_rho[d_idx] = d_hnurho[d_idx] / d_hnu[d_idx]


class Factorf(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_f):
        d_f[d_idx] = 0.0

    def loop(self, d_idx, RIJ, WDASHI, s_m, s_idx, d_f, d_h, d_hnu):
        if RIJ > 1e-12:
            qi = RIJ / d_h[d_idx]
            tilwij = WDASHI * d_hnu[d_idx] / qi
            d_f[d_idx] += s_m[s_idx] * qi * qi * tilwij

    def post_loop(self, d_idx, d_f, d_hnurho):
        dim = self.dim
        d_f[d_idx] = -dim * d_hnurho[d_idx] / d_f[d_idx]


class AdjustSmoothingLength(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def post_loop(self, d_idx, d_f, d_rho, d_h, d_ftil, d_m, SPH_KERNEL,
                  d_hnurho, d_hnu, WI, d_Mh):
        dim = self.dim

        if d_hnurho[d_idx] < d_Mh[d_idx]:
            wo = SPH_KERNEL.kernel(xij=[0.0, 0.0, 0.0], rij=0.0,
                                   h=d_h[d_idx]) * d_hnu[d_idx]
            d_ftil[d_idx] = (d_f[d_idx] * (d_hnurho[d_idx] - d_m[d_idx] * wo) /
                             d_hnurho[d_idx])
            pw = d_ftil[d_idx] / dim
            num = d_Mh[d_idx] - d_m[d_idx] * wo
            den = d_hnurho[d_idx] - d_m[d_idx] * wo
            d_h[d_idx] *= (num / den) ** pw
        else:
            pw = d_f[d_idx] / dim
            d_h[d_idx] *= (d_Mh[d_idx] / d_hnurho[d_idx]) ** pw

        d_hnu[d_idx] = d_h[d_idx] ** dim


class SmoothingLengthRate(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_ah, d_ahden):
        d_ah[d_idx] = 0.0
        d_ahden[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_h, RIJ, s_m, VIJ, XIJ, d_ah,
             d_ahden, R2IJ, WDASHI, d_hnu):
        if RIJ > 1e-12:
            vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]
            qi = RIJ / d_h[d_idx]
            tilwij = WDASHI * d_hnu[d_idx] / qi
            d_ah[d_idx] += s_m[s_idx] + vijdotxij * tilwij
            d_ahden[d_idx] += s_m[s_idx] + R2IJ * tilwij

    def post_loop(self, d_idx, d_ah, d_ahden):
        d_ah[d_idx] = d_ah[d_idx] / d_ahden[d_idx]


class VelocityGradient(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def initialize(
            self, d_idx, d_invT00, d_invT01, d_invT02, d_invT10, d_invT11,
            d_invT12, d_invT20, d_invT21, d_invT22, d_D00, d_D01, d_D02, d_D10,
            d_D11, d_D12, d_D20, d_D21, d_D22):
        d_D00[d_idx] = 0.0
        d_D01[d_idx] = 0.0
        d_D02[d_idx] = 0.0

        d_D10[d_idx] = 0.0
        d_D11[d_idx] = 0.0
        d_D12[d_idx] = 0.0

        d_D20[d_idx] = 0.0
        d_D21[d_idx] = 0.0
        d_D22[d_idx] = 0.0

        d_invT00[d_idx] = 0.0
        d_invT01[d_idx] = 0.0
        d_invT02[d_idx] = 0.0

        d_invT10[d_idx] = 0.0
        d_invT11[d_idx] = 0.0
        d_invT12[d_idx] = 0.0

        d_invT20[d_idx] = 0.0
        d_invT21[d_idx] = 0.0
        d_invT22[d_idx] = 0.0

    def loop(
            self, d_idx, s_idx, s_m, s_rho, VIJ, d_invT00, d_invT01,
            d_invT02, d_invT10, d_invT11, d_invT12, d_invT20, d_invT21,
            d_invT22, d_D00, d_D01, d_D02, d_D10, d_D11, d_D12, d_D20, d_D21,
            d_D22, RIJ, XIJ, d_h, WDASHI, d_hnu):
        Vb = s_m[s_idx] / s_rho[s_idx]
        qi = RIJ / d_h[d_idx]
        if RIJ > 1e-12:
            qi = RIJ / d_h[d_idx]
            tilwij = WDASHI * d_hnu[d_idx] / qi
            barwij = Vb * tilwij

            d_D00[d_idx] -= VIJ[0] * XIJ[0] * barwij
            d_D01[d_idx] -= VIJ[0] * XIJ[1] * barwij
            d_D02[d_idx] -= VIJ[0] * XIJ[2] * barwij

            d_D10[d_idx] -= VIJ[1] * XIJ[0] * barwij
            d_D11[d_idx] -= VIJ[1] * XIJ[1] * barwij
            d_D12[d_idx] -= VIJ[1] * XIJ[2] * barwij

            d_D20[d_idx] -= VIJ[2] * XIJ[0] * barwij
            d_D21[d_idx] -= VIJ[2] * XIJ[1] * barwij
            d_D22[d_idx] -= VIJ[2] * XIJ[2] * barwij

            d_invT00[d_idx] -= XIJ[0] * XIJ[0] * barwij
            d_invT01[d_idx] -= XIJ[0] * XIJ[1] * barwij
            d_invT02[d_idx] -= XIJ[0] * XIJ[2] * barwij

            d_invT10[d_idx] -= XIJ[1] * XIJ[0] * barwij
            d_invT11[d_idx] -= XIJ[1] * XIJ[1] * barwij
            d_invT12[d_idx] -= XIJ[1] * XIJ[2] * barwij

            d_invT20[d_idx] -= XIJ[2] * XIJ[0] * barwij
            d_invT21[d_idx] -= XIJ[2] * XIJ[1] * barwij
            d_invT22[d_idx] -= XIJ[2] * XIJ[2] * barwij

    def post_loop(
            self, d_idx, d_invT00, d_invT01, d_invT02, d_invT10, d_invT11,
            d_invT12, d_invT20, d_invT21, d_invT22, d_D00, d_D01, d_D02, d_D10,
            d_D11, d_D12, d_D20, d_D21, d_D22, d_gradv00, d_gradv01,
            d_gradv02, d_gradv10, d_gradv11, d_gradv12, d_gradv20,
            d_gradv21, d_gradv22):
        T00 = d_invT00[d_idx]
        T01 = d_invT01[d_idx]
        T02 = d_invT02[d_idx]

        T10 = d_invT10[d_idx]
        T11 = d_invT11[d_idx]
        T12 = d_invT12[d_idx]

        T20 = d_invT20[d_idx]
        T21 = d_invT21[d_idx]
        T22 = d_invT22[d_idx]

        if self.dim == 1:
            T11 = 1.0
            T22 = 1.0
        elif self.dim == 2:
            T22 = 1.0

        det = (T00 * T11 * T22 - T00 * T12 * T21 - T01 * T10 * T22 +
               T01 * T12 * T20 + T02 * T10 * T21 - T02 * T11 * T20)

        d_invT00[d_idx] = (T11 * T22 - T12 * T21) / det
        d_invT01[d_idx] = (T10 * T22 - T12 * T20) / det
        d_invT02[d_idx] = (T10 * T21 - T11 * T20) / det
        d_invT10[d_idx] = (T01 * T22 - T02 * T21) / det
        d_invT11[d_idx] = (T00 * T22 - T02 * T20) / det
        d_invT12[d_idx] = (T00 * T21 - T01 * T20) / det
        d_invT20[d_idx] = (T01 * T12 - T02 * T11) / det
        d_invT21[d_idx] = (T00 * T12 - T02 * T10) / det
        d_invT22[d_idx] = (T00 * T11 - T01 * T10) / det

        d_gradv00[d_idx] = (d_D00[d_idx] * d_invT00[d_idx] +
                            d_D01[d_idx] * d_invT10[d_idx] +
                            d_D02[d_idx] * d_invT20[d_idx])
        d_gradv01[d_idx] = (d_D00[d_idx] * d_invT01[d_idx] +
                            d_D01[d_idx] * d_invT11[d_idx] +
                            d_D02[d_idx] * d_invT21[d_idx])
        d_gradv02[d_idx] = (d_D00[d_idx] * d_invT02[d_idx] +
                            d_D01[d_idx] * d_invT12[d_idx] +
                            d_D02[d_idx] * d_invT22[d_idx])
        d_gradv10[d_idx] = (d_D10[d_idx] * d_invT00[d_idx] +
                            d_D11[d_idx] * d_invT10[d_idx] +
                            d_D12[d_idx] * d_invT20[d_idx])
        d_gradv11[d_idx] = (d_D10[d_idx] * d_invT01[d_idx] +
                            d_D11[d_idx] * d_invT11[d_idx] +
                            d_D12[d_idx] * d_invT21[d_idx])
        d_gradv12[d_idx] = (d_D10[d_idx] * d_invT02[d_idx] +
                            d_D11[d_idx] * d_invT12[d_idx] +
                            d_D12[d_idx] * d_invT22[d_idx])
        d_gradv20[d_idx] = (d_D20[d_idx] * d_invT00[d_idx] +
                            d_D21[d_idx] * d_invT10[d_idx] +
                            d_D22[d_idx] * d_invT20[d_idx])
        d_gradv21[d_idx] = (d_D20[d_idx] * d_invT01[d_idx] +
                            d_D21[d_idx] * d_invT11[d_idx] +
                            d_D22[d_idx] * d_invT21[d_idx])
        d_gradv22[d_idx] = (d_D20[d_idx] * d_invT02[d_idx] +
                            d_D21[d_idx] * d_invT12[d_idx] +
                            d_D22[d_idx] * d_invT22[d_idx])


class VelocityDivergence(Equation):
    def post_loop(self, d_idx, d_divv, d_gradv00, d_gradv11, d_gradv22):
        d_divv[d_idx] = d_gradv00[d_idx] + d_gradv11[d_idx] + d_gradv22[d_idx]


class AcclerationGradient(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_DD00, d_DD01, d_DD02, d_DD10,
                   d_DD11, d_DD12, d_DD20, d_DD21, d_DD22):
        d_DD00[d_idx] = 0.0
        d_DD01[d_idx] = 0.0
        d_DD02[d_idx] = 0.0

        d_DD10[d_idx] = 0.0
        d_DD11[d_idx] = 0.0
        d_DD12[d_idx] = 0.0

        d_DD20[d_idx] = 0.0
        d_DD21[d_idx] = 0.0
        d_DD22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_DD00, d_DD01, d_DD02, d_DD10,
             d_DD11, d_DD12, d_DD20, d_DD21, d_DD22, RIJ, XIJ, d_h,
             d_au, s_au, d_av, s_av, d_aw, s_aw, WDASHI, d_hnu):
        Vb = s_m[s_idx] / s_rho[s_idx]

        if RIJ > 1e-12:
            qi = RIJ / d_h[d_idx]
            tilwij = WDASHI * d_hnu[d_idx] / qi
            barwij = Vb * tilwij

            auij = d_au[d_idx] - s_au[s_idx]
            d_DD00[d_idx] -= auij * XIJ[0] * barwij
            d_DD01[d_idx] -= auij * XIJ[1] * barwij
            d_DD02[d_idx] -= auij * XIJ[2] * barwij

            avij = d_av[d_idx] - s_av[s_idx]
            d_DD10[d_idx] -= avij * XIJ[0] * barwij
            d_DD11[d_idx] -= avij * XIJ[1] * barwij
            d_DD12[d_idx] -= avij * XIJ[2] * barwij

            awij = d_aw[d_idx] - s_aw[s_idx]
            d_DD20[d_idx] -= awij * XIJ[0] * barwij
            d_DD21[d_idx] -= awij * XIJ[1] * barwij
            d_DD22[d_idx] -= awij * XIJ[2] * barwij

    def post_loop(self, d_idx, d_invT00, d_invT01, d_invT02, d_invT10,
                  d_invT11, d_invT12, d_invT20, d_invT21, d_invT22, d_DD00,
                  d_DD01, d_DD02, d_DD10, d_DD11, d_DD12, d_DD20, d_DD21,
                  d_DD22, d_grada00, d_grada01, d_grada02, d_grada10,
                  d_grada11, d_grada12, d_grada20, d_grada21, d_grada22):
        d_grada00[d_idx] = (d_DD00[d_idx] * d_invT00[d_idx] + d_DD01[d_idx] *
                            d_invT10[d_idx] + d_DD02[d_idx] * d_invT20[d_idx])
        d_grada01[d_idx] = (d_DD00[d_idx] * d_invT01[d_idx] + d_DD01[d_idx] *
                            d_invT11[d_idx] + d_DD02[d_idx] * d_invT21[d_idx])
        d_grada02[d_idx] = (d_DD00[d_idx] * d_invT02[d_idx] + d_DD01[d_idx] *
                            d_invT12[d_idx] + d_DD02[d_idx] * d_invT22[d_idx])
        d_grada10[d_idx] = (d_DD10[d_idx] * d_invT00[d_idx] + d_DD11[d_idx] *
                            d_invT10[d_idx] + d_DD12[d_idx] * d_invT20[d_idx])
        d_grada11[d_idx] = (d_DD10[d_idx] * d_invT01[d_idx] + d_DD11[d_idx] *
                            d_invT11[d_idx] + d_DD12[d_idx] * d_invT21[d_idx])
        d_grada12[d_idx] = (d_DD10[d_idx] * d_invT02[d_idx] + d_DD11[d_idx] *
                            d_invT12[d_idx] + d_DD12[d_idx] * d_invT22[d_idx])
        d_grada20[d_idx] = (d_DD20[d_idx] * d_invT00[d_idx] + d_DD21[d_idx] *
                            d_invT10[d_idx] + d_DD22[d_idx] * d_invT20[d_idx])
        d_grada21[d_idx] = (d_DD20[d_idx] * d_invT01[d_idx] + d_DD21[d_idx] *
                            d_invT11[d_idx] + d_DD22[d_idx] * d_invT21[d_idx])
        d_grada22[d_idx] = (d_DD20[d_idx] * d_invT02[d_idx] + d_DD21[d_idx] *
                            d_invT12[d_idx] + d_DD22[d_idx] * d_invT22[d_idx])


class VelocityDivergenceRate(Equation):
    def post_loop(self, d_idx, d_adivv, d_grada00, d_grada11, d_grada22,
                  d_gradv00, d_gradv11, d_gradv22, d_gradv01, d_gradv10,
                  d_gradv02, d_gradv20, d_gradv21, d_gradv12):
        d_adivv[d_idx] = (d_grada00[d_idx] +
                          d_grada11[d_idx] +
                          d_grada22[d_idx] -
                          d_gradv00[d_idx] * d_gradv00[d_idx] -
                          d_gradv11[d_idx] * d_gradv11[d_idx] -
                          d_gradv22[d_idx] * d_gradv22[d_idx] -
                          2.0 * (d_gradv01[d_idx] * d_gradv10[d_idx] +
                                 d_gradv02[d_idx] * d_gradv20[d_idx] +
                                 d_gradv21[d_idx] * d_gradv12[d_idx]))


class TracelessSymmetricStrainRate(Equation):
    def __init__(self, dest, sources, dim):
        self.obydim = 1.0 / dim
        super().__init__(dest, sources)

    def post_loop(self, d_idx, d_gradv00, d_gradv01, d_gradv02, d_gradv10,
                  d_gradv11, d_gradv12, d_gradv20, d_gradv21, d_gradv22,
                  d_S00, d_S10, d_S11, d_S20, d_S21, d_S22, d_divv):
        obydim = self.obydim

        d_S00[d_idx] = d_gradv00[d_idx] - obydim * d_divv[d_idx]

        d_S10[d_idx] = 0.5 * (d_gradv01[d_idx] + d_gradv10[d_idx])
        d_S11[d_idx] = d_gradv11[d_idx] - obydim * d_divv[d_idx]

        d_S20[d_idx] = 0.5 * (d_gradv02[d_idx] + d_gradv20[d_idx])
        d_S21[d_idx] = 0.5 * (d_gradv21[d_idx] + d_gradv12[d_idx])
        d_S22[d_idx] = d_gradv22[d_idx] - obydim * d_divv[d_idx]


class ShockIndicatorR(Equation):

    def initialize(self, d_idx, d_R):
        d_R[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_R, s_divv, WI):
        if s_divv[s_idx] < 0:
            sign = -1.0
        elif s_divv[s_idx] == 0:
            sign = 0.0
        else:
            sign = 1.0
        d_R[d_idx] += sign * s_m[s_idx] * WI

    def post_loop(self, d_idx, d_rho, d_R):
        d_R[d_idx] *= 1 / d_rho[d_idx]


class EOS(Equation):
    def __init__(self, dest, sources, gamma):
        self.gamma = gamma
        self.gammam1 = gamma - 1.0
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_p, d_e, d_cs):
        gamma = self.gamma
        gammam1 = self.gammam1
        d_p[d_idx] = gammam1 * d_e[d_idx] * d_rho[d_idx]
        d_cs[d_idx] = sqrt(gamma * d_p[d_idx] / d_rho[d_idx])


class SignalVelocity(Equation):
    def initialize(self, d_idx, d_vsig):
        d_vsig[d_idx] = 0.0

    def loop_all(
            self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, d_u, d_v, d_w, s_u, s_v,
            s_w, d_cs, s_cs, d_vsig, NBRS, N_NBRS):
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


class FalseDetectionSuppressingLimiterXi(Equation):

    def post_loop(self, d_idx, d_divv, d_R, d_S00, d_S11, d_S22, d_S10, d_S20,
                  d_S21, d_xi):
        omR = 1 - d_R[d_idx]

        num = 2 * omR * omR * omR * omR * d_divv[d_idx]
        num *= num

        # trace(S \cdot S^t)
        trSdotSt = (d_S00[d_idx] ** 2 + d_S10[d_idx] ** 2 + d_S20[d_idx] ** 2 +
                    d_S10[d_idx] ** 2 + d_S11[d_idx] ** 2 + d_S21[d_idx] ** 2 +
                    d_S20[d_idx] ** 2 + d_S21[d_idx] ** 2 + d_S22[d_idx] ** 2)

        den = num + trSdotSt

        if den == 0:
            d_xi[d_idx] = 0.0
        else:
            d_xi[d_idx] = num / den


class NovelShockIndicatorA(Equation):
    def post_loop(self, d_idx, d_adivv, d_xi, d_A):
        d_A[d_idx] = d_xi[d_idx] * max(-d_adivv[d_idx], 0)


class IndividualViscosityLocal(Equation):
    def __init__(self, dest, sources, alphamax):
        self.alphamax = alphamax
        super().__init__(dest, sources)

    def post_loop(self, d_idx, d_h, d_A, d_vsig, d_alphaloc):
        alphamax = self.alphamax
        hsq = d_h[d_idx] * d_h[d_idx]
        num = hsq * d_A[d_idx]
        den = num + d_vsig[d_idx] * d_vsig[d_idx]
        d_alphaloc[d_idx] = alphamax * num / den


class ViscosityDecayTimeScale(Equation):
    def __init__(self, dest, sources, l):
        self.l = l
        super().__init__(dest, sources)

    def post_loop(self, d_idx, d_cs, d_h, d_tau, SPH_KERNEL):
        l = self.l
        fkern = SPH_KERNEL.fkern
        d_tau[d_idx] = d_h[d_idx] * fkern / (l * d_cs[d_idx])


class AdaptIndividualViscosity(Equation):
    def post_loop(self, d_idx, d_alphaloc, d_alpha, d_tau, dt):

        if d_alpha[d_idx] < d_alphaloc[d_idx]:
            d_alpha[d_idx] = d_alphaloc[d_idx]
        else:
            d_alpha[d_idx] = ((d_alphaloc[d_idx] +
                               (d_alpha[d_idx] - d_alphaloc[d_idx]) *
                               exp(-dt / d_tau[d_idx])))


class UpdateGhostProps(Equation):
    def __init__(self, dest, sources=None, dim=2):
        super().__init__(dest, sources)
        self.dim = dim
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_orig_idx, d_p, d_cs, d_tag, d_f, d_h,
                   d_hnurho, d_rho, d_hnu):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_p[d_idx] = d_p[idx]
            d_f[d_idx] = d_f[idx]
            d_h[d_idx] = d_h[idx]
            d_hnurho[d_idx] = d_hnurho[idx]
            d_hnu[d_idx] = d_hnu[idx]
            d_rho[d_idx] = d_rho[idx]


class MomentumAndEnergy(Equation):

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p, s_rho, s_p, d_au, d_av,
             d_aw, VIJ, d_ae, XIJ, d_h, RIJ, s_h, d_f,
             s_f, d_hnurho, s_hnurho, d_m, WDASHI, d_hnu, WDASHJ, s_hnu):
        if RIJ > 1e-12:
            qi = RIJ / d_h[d_idx]
            qj = RIJ / s_h[s_idx]
            tilwij = WDASHI * d_hnu[d_idx] / qi
            tilwji = WDASHJ * s_hnu[s_idx] / qj

            vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

            # denominator
            di = d_hnurho[d_idx] * d_rho[d_idx] * d_h[d_idx] * d_h[d_idx]
            dj = s_hnurho[s_idx] * s_rho[s_idx] * s_h[s_idx] * s_h[s_idx]

            # num by denom
            nbdi = d_p[d_idx] * d_f[d_idx] / di
            nbdj = s_p[s_idx] * s_f[s_idx] / dj

            inparentheses = nbdi * tilwij + nbdj * tilwji

            d_au[d_idx] -= d_m[d_idx] * XIJ[0] * inparentheses
            d_av[d_idx] -= d_m[d_idx] * XIJ[1] * inparentheses
            d_aw[d_idx] -= d_m[d_idx] * XIJ[2] * inparentheses
            d_ae[d_idx] += nbdi * d_m[d_idx] * vijdotxij * tilwij


class ArtificialViscocity(Equation):
    def __init__(self, dest, sources, b):
        self.b = b
        super().__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, d_alpha, s_rho, s_alpha, d_au, d_av,
             d_aw, VIJ, d_ae, XIJ, d_h, RIJ, R2IJ, s_h, d_f,
             s_f, d_hnurho, s_hnurho, d_m, d_cs, s_cs, s_m, EPS, WDASHI,
             d_hnu, WDASHJ, s_hnu):
        b = self.b

        if RIJ > 1e-12:
            qi = RIJ / d_h[d_idx]
            qj = RIJ / s_h[s_idx]
            tilwij = WDASHI * d_hnu[d_idx] / qi
            tilwji = WDASHJ * s_hnu[s_idx] / qj

            vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

            hisq = d_h[d_idx] * d_h[d_idx]
            hjsq = s_h[s_idx] * s_h[s_idx]

            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            # denominator
            di = d_hnurho[d_idx] * hisq
            dj = s_hnurho[s_idx] * hjsq

            # num by denom
            nbdi = d_alpha[d_idx] * d_f[d_idx] / di
            nbdj = s_alpha[s_idx] * s_f[s_idx] / dj

            inparentheses = nbdi * tilwij + nbdj * tilwji

            if vijdotxij < 0:
                hij = 0.5 * (d_h[d_idx] + s_h[s_idx])
                mden = (R2IJ / hisq + R2IJ / hjsq + EPS) * hij
                muij = 2.0 * vijdotxij / mden

                Piij = -muij * (cij - b * muij)
                Piijby2 = 0.5 * Piij

                d_au[d_idx] -= s_m[s_idx] * XIJ[0] * Piijby2 * inparentheses
                d_av[d_idx] -= s_m[s_idx] * XIJ[1] * Piijby2 * inparentheses
                d_aw[d_idx] -= s_m[s_idx] * XIJ[2] * Piijby2 * inparentheses
                d_ae[d_idx] += s_m[s_idx] * vijdotxij * Piijby2 * nbdi * tilwij


class WallBoundary(Equation):
    def initialize(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_div, d_h,
                   d_htmp, d_h0, d_u, d_v, d_w, d_wij):
        d_p[d_idx] = 0.0
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_m[d_idx] = 0.0
        d_rho[d_idx] = 0.0
        d_e[d_idx] = 0.0
        d_cs[d_idx] = 0.0
        d_div[d_idx] = 0.0
        d_wij[d_idx] = 0.0
        d_h[d_idx] = d_h0[d_idx]
        d_htmp[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_e, d_m, d_cs, d_divv, d_h, d_u,
             d_v, d_w, d_wij, d_htmp, s_p, s_rho, s_e, s_m, s_cs, s_h, s_divv,
             s_u, s_v, s_w, WI):
        d_wij[d_idx] += WI
        d_p[d_idx] += s_p[s_idx] * WI
        d_u[d_idx] -= s_u[s_idx] * WI
        d_v[d_idx] -= s_v[s_idx] * WI
        d_w[d_idx] -= s_w[s_idx] * WI
        d_m[d_idx] += s_m[s_idx] * WI
        d_rho[d_idx] += s_rho[s_idx] * WI
        d_e[d_idx] += s_e[s_idx] * WI
        d_cs[d_idx] += s_cs[s_idx] * WI
        d_divv[d_idx] += s_divv[s_idx] * WI
        d_htmp[d_idx] += s_h[s_idx] * WI

    def post_loop(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_divv, d_h, d_u,
                  d_v, d_w, d_wij, d_htmp):
        if (d_wij[d_idx] > 1e-30):
            d_p[d_idx] = d_p[d_idx] / d_wij[d_idx]
            d_u[d_idx] = d_u[d_idx] / d_wij[d_idx]
            d_v[d_idx] = d_v[d_idx] / d_wij[d_idx]
            d_w[d_idx] = d_w[d_idx] / d_wij[d_idx]
            d_m[d_idx] = d_m[d_idx] / d_wij[d_idx]
            d_rho[d_idx] = d_rho[d_idx] / d_wij[d_idx]
            d_e[d_idx] = d_e[d_idx] / d_wij[d_idx]
            d_cs[d_idx] = d_cs[d_idx] / d_wij[d_idx]
            d_divv[d_idx] = d_divv[d_idx] / d_wij[d_idx]
            d_h[d_idx] = d_htmp[d_idx] / d_wij[d_idx]
