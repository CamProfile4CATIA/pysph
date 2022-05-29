"""
References
-----------
    .. [Rosswog2009] Rosswog, Stephan. "Astrophysical smooth particle
        hydrodynamics." New Astronomy Reviews 53, no. 4-6 (2009): 78-104.


    .. [Rosswog2015] Rosswog, Stephan. "Boosting the accuracy of SPH
        techniques: Newtonian and special-relativistic tests." Monthly
        Notices of the Royal Astronomical Society 448, no. 4 (2015):
        3628-3664. https://doi.org/10.1093/mnras/stv225.


    .. [Rosswog2020a] Rosswog, Stephan. "A simple, entropy-based dissipation
        trigger for SPH." The Astrophysical Journal 898, no. 1 (2020): 60.
        https://doi.org/10.3847/1538-4357/ab9a2e.


    .. [Rosswog2020b] Rosswog, Stephan. "The Lagrangian hydrodynamics code
        MAGMA2." Monthly Notices of the Royal Astronomical Society 498, no. 3
        (2020): 4230-4255. https://doi.org/10.1093/mnras/staa2591.

"""
from compyle.types import declare, annotate
from pysph.base.particle_array import get_ghost_tag
from math import *
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.wc.linalg import (augmented_matrix, gj_solve, identity,
                                 mat_mult, mat_vec_mult, dot)

GHOST_TAG = get_ghost_tag()


class NSRSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, hfact, beta=2.0, fkern=1.0,
                 max_density_iterations=250, alphamax=1.0, alphamin=0.1,
                 density_iteration_tolerance=1e-3, has_ghosts=False,
                 eta_crit=0.3, eta_fold=0.2, adaptive_h_scheme='mpm'):
        """
        Newtonian limit of Rosswog's special-relativistic SPH.

        Derivation: [Rosswog2009]_
        Improvements: [Rosswog2015]_
        Limiter: [Rosswog2020a]_
        Summary: [Rosswog2020b]_ (These equations are used in the
        implementation here.)

        Notes
        -----
        Is this exactly in accordance with what is proposed in [Hopkins2015]_ ?
            Not quite.

        What is different then?
            #. Adapting smoothing length using MPM [KP14]_ procedure from
               :class:`SummationDensity
               <pysph.sph.gas_dynamics.basic.SummationDensity>`. From this, the
               grad-h terms removed.
            #. Using :class:`CubicSpline <pysph.base.kernels.CubicSpline>`
               as default kernel.

        Parameters
        ----------
        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries), currently
            not supported
        dim: int
            Dimensionality of the problem.
        gamma: float
            :math:`\\gamma` for Equation of state.
        hfact: float
            :math:`h_{fact}` for smoothing length adaptivity, also referred to
            as kernel_factor in other gas dynamics schemes.
        beta : float, optional
            :math:`\\beta` for artificial viscosity, by default 2.0
        fkern : float, optional
            :math:`f_{kern}`, Factor to scale smoothing length for equivalence
            with classic kernel when using kernel with altered
            `radius_scale` is being used, by default 1.
        max_density_iterations : int, optional
            Maximum number of iterations to run for one density step,
            by default 250.
        density_iteration_tolerance : float, optional
            Maximum difference allowed in two successive density iterations,
            by default 1e-3
        has_ghosts : bool, optional
            if ghost particles (either mirror or periodic) is used, by default
            False
        alphamax : float, optional
            :math:`\\alpha_{av}` for artificial viscosity switch, by default
            1.0
        """

        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.beta = beta
        self.hfact = hfact
        self.density_iteration_tolerance = density_iteration_tolerance
        self.max_density_iterations = max_density_iterations
        self.has_ghosts = has_ghosts
        self.adaptive_h_scheme = adaptive_h_scheme
        self.fkern = fkern
        self.alphamax = alphamax
        self.alphamin = alphamin
        self.eta_crit = eta_crit
        self.eta_fold = eta_fold

    def add_user_options(self, group):
        group.add_argument("--alpha-max", action="store", type=float,
                           dest="alphamax", default=None,
                           help="alpha_max for the artificial viscosity "
                                "switch.")

        group.add_argument("--beta", action="store", type=float, dest="beta",
                           default=None,
                           help="beta for the artificial viscosity.")

        group.add_argument("--gamma", action="store", type=float, dest="gamma",
                           default=None, help="gamma for the state equation.")

    def consume_user_options(self, options):
        vars = ['gamma', 'alphamax', 'beta']
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):

        from pysph.base.kernels import Gaussian
        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        if hasattr(kernel, 'fkern'):
            self.fkern = kernel.fkern
        else:
            self.fkern = 1.0

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = PECStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group

        all_pa = self.fluids + self.solids
        equations = []

        # Find the optimal 'h'

        if self.adaptive_h_scheme == "magma2":
            g1p0 = []
            for fluid in self.fluids:
                g1p0.append(IncreaseSmoothingLength(dest=fluid, sources=None))
            equations.append(Group(equations=g1p0))

            g1p1 = []
            for fluid in self.fluids:
                g1p1.append(UpdateSmoothingLength(dest=fluid, sources=all_pa))
            equations.append(Group(equations=g1p1))
            ioo = True

        else:
            ioo = False

        g1p2 = []
        for fluid in self.fluids:
            g1p2.append(SummationDensity(
                dest=fluid, sources=all_pa, hfact=self.hfact,
                density_iterations=True, dim=self.dim, iterate_only_once=ioo,
                htol=self.density_iteration_tolerance))
            equations.append(Group(
                equations=g1p2, update_nnps=True, iterate=True,
                max_iterations=self.max_density_iterations))

        g2 = []
        for fluid in self.fluids:
            g2.append(
                AuxillaryGradient(dest=fluid, sources=all_pa, dim=self.dim))
            g2.append(
                IdealGasEOS(dest=fluid, sources=None, gamma=self.gamma))
        equations.append(Group(equations=g2))

        g3p1 = []
        for fluid in self.fluids:
            g3p1.append(CorrectionMatrix(dest=fluid, sources=all_pa,
                                         dim=self.dim))
        equations.append(Group(equations=g3p1))

        g3p2 = []
        for fluid in self.fluids:
            g3p2.append(SecondGradient(dest=fluid, sources=all_pa,
                                       dim=self.dim))
            g3p2.append(FirstGradient(dest=fluid, sources=all_pa,
                                      dim=self.dim))
            g3p2.append(EntropyBasedDissipationTrigger(
                dest=fluid, sources=None, alphamax=self.alphamax,
                alphamin=self.alphamin, fkern=self.fkern, l0=log(1e-4),
                l1=log(5e-2), gamma=self.gamma))
        equations.append(Group(equations=g3p2))

        g4 = []
        for solid in self.solids:
            g4.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g4))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(UpdateGhostProps(dest=fluid, sources=None,
                                           dim=self.dim))
            equations.append(Group(equations=gh, real=False))

        g5 = []
        for fluid in self.fluids:
            g5.append(MomentumAndEnergyMI1(dest=fluid, sources=all_pa,
                                           dim=self.dim, beta=self.beta,
                                           fkern=self.fkern,
                                           eta_crit=self.eta_crit,
                                           eta_fold=self.eta_fold))

        equations.append(Group(equations=g5))

        return equations

    def setup_properties(self, particles, clean=True):
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])

        props = ['rho', 'm', 'x', 'y', 'z', 'u', 'v', 'w', 'h', 'cs', 'p', 'e',
                 'au', 'av', 'aw', 'ae', 'pid', 'gid', 'tag', 'dwdh', 'h0',
                 'converged', 'ah', 'arho', 'dt_cfl', 'e0', 'rho0', 'u0', 'v0',
                 'w0', 'x0', 'y0', 'z0']
        more_props = ['drhosumdh', 'n', 'dndh', 'prevn', 'prevdndh',
                      'prevdrhosumdh', 'divv', 'an', 'n0', 'alpha0',
                      'aalpha']
        props.extend(more_props)
        output_props = 'rho p u v w x y z e n divv h alpha'.split(' ')
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            # Guess for number density.
            pa.add_property('n', data=pa.rho / pa.m)
            pa.add_property('s', data=pa.p / (pa.rho**self.gamma))
            pa.add_property('alpha', data=self.alphamin)
            pa.add_property('dv', stride=9)
            pa.add_property('dvaux', stride=9)
            pa.add_property('invdm', stride=9)
            pa.add_property('cm', stride=9)
            pa.add_property('ddv', stride=27)
            pa.add_property('de', stride=3)
            pa.add_property('dde', stride=9)
            pa.add_property('deaux', stride=3)
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('m0 wij htmp'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.set_output_arrays(output_props)


class UpdateSmoothingLength(Equation):
    def _get_helpers_(self):
        return [quicksort]

    def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h,
                 s_m, s_x, s_y, s_z, s_h, NBRS, N_NBRS, SPH_KERNEL):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix(3)')
        rij = declare('matrix(500)')
        nidx = declare('matrix(500, "long")')
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            xij[2] = d_z[d_idx] - s_z[s_idx]
            rij[i] = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            nidx[i] = s_idx
        quicksort(nidx, rij, 0, N_NBRS)
        d_h[d_idx] = rij[10] / SPH_KERNEL.radius_scale


class SummationDensity(Equation):
    def __init__(self, dest, sources, dim, density_iterations=False,
                 iterate_only_once=False, hfact=1.2, htol=1e-6):
        """
        :class:`SummationDensity
        <pysph.sph.gas_dynamics.basic.SummationDensity>` modified to use
         number density and without grad-h terms.

        Ref. Appendix F1 [Hopkins2015]_
        """

        self.density_iterations = density_iterations
        self.iterate_only_once = iterate_only_once
        self.dim = dim
        self.hfact = hfact
        self.htol = htol
        self.equation_has_converged = 1

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_arho, d_drhosumdh, d_n, d_dndh,
                   d_prevn, d_prevdndh, d_prevdrhosumdh, d_an):

        d_rho[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_prevn[d_idx] = d_n[d_idx]
        d_prevdrhosumdh[d_idx] = d_drhosumdh[d_idx]
        d_prevdndh[d_idx] = d_dndh[d_idx]

        d_drhosumdh[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_an[d_idx] = 0.0
        d_dndh[d_idx] = 0.0

        # set the converged attribute for the Equation to True. Within
        # the post-loop, if any particle hasn't converged, this is set
        # to False. The Group can therefore iterate till convergence.
        self.equation_has_converged = 1

    def loop(self, d_idx, s_idx, d_rho, d_arho, d_drhosumdh, s_m, VIJ, WI, DWI,
             GHI, d_n, d_dndh, d_h, d_prevn, d_prevdndh, d_prevdrhosumdh,
             d_an):

        mj = s_m[s_idx]
        vijdotdwij = VIJ[0] * DWI[0] + VIJ[1] * DWI[1] + VIJ[2] * DWI[2]

        # density
        d_rho[d_idx] += mj * WI

        # density accelerations
        # hibynidim = d_h[d_idx] / (d_prevn[d_idx] * self.dim)
        # inbrkti = 1 + d_prevdndh[d_idx] * hibynidim
        # inprthsi = d_prevdrhosumdh[d_idx] * hibynidim
        # fij = 1 - inprthsi / (s_m[s_idx] * inbrkti)

        fij = 1
        vijdotdwij_fij = vijdotdwij * fij
        d_arho[d_idx] += mj * vijdotdwij_fij
        d_an[d_idx] += vijdotdwij_fij

        # gradient of kernel w.r.t h
        d_drhosumdh[d_idx] += mj * GHI
        d_n[d_idx] += WI
        d_dndh[d_idx] += GHI

    def post_loop(self, d_idx, d_h0, d_h, d_ah, d_converged, d_n, d_dndh,
                  d_an):
        # iteratively find smoothing length consistent with the
        if self.density_iterations:
            if not (d_converged[d_idx] == 1):
                hi = d_h[d_idx]
                hi0 = d_h0[d_idx]

                # estimated, without summations
                ni = (self.hfact / hi) ** self.dim
                dndhi = - self.dim * d_n[d_idx] / hi

                # the non-linear function and it's derivative
                func = d_n[d_idx] - ni
                dfdh = d_dndh[d_idx] - dndhi

                # Newton Raphson estimate for the new h
                hnew = hi - func / dfdh

                # Nanny control for h
                if hnew > 1.2 * hi:
                    hnew = 1.2 * hi
                elif hnew < 0.8 * hi:
                    hnew = 0.8 * hi

                # check for convergence
                diff = abs(hnew - hi) / hi0

                if not ((diff < self.htol) or self.iterate_only_once):
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


class IdealGasEOS(Equation):
    def __init__(self, dest, sources, gamma):
        """
        :class:`IdealGasEOS
        <pysph.sph.gas_dynamics.basic.IdealGasEOS>` modified to avoid repeated
        calculations using :meth:`loop() <pysph.sph.equation.Equation.loop()>`.
        Doing the same using :meth:`post_loop()
        <pysph.sph.equation.Equation.loop()>`.
        """
        self.gamma = gamma
        self.gamma1 = gamma - 1.0
        super(IdealGasEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho, d_e, d_cs):
        d_p[d_idx] = self.gamma1 * d_rho[d_idx] * d_e[d_idx]
        d_cs[d_idx] = sqrt(self.gamma * d_p[d_idx] / d_rho[d_idx])


class AuxillaryGradient(Equation):
    def __init__(self, dest, sources, dim):
        """
        First Order consistent velocity gradient and divergence
        """
        self.dim = dim
        self.dimsq = dim * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_mult, augmented_matrix, identity, gj_solve, mat_vec_mult]

    def initialize(self, d_dvaux, d_idx, d_invdm, d_deaux):
        dsi2, i, dim, dimsq = declare('int', 4)
        dimsq = self.dimsq
        dim = self.dim
        dsi2 = dimsq * d_idx
        for i in range(dim):
            d_deaux[dim * d_idx + i] = 0.0

        for i in range(dimsq):
            d_dvaux[dsi2 + i] = 0.0
            d_invdm[dsi2 + i] = 0.0

    def loop(self, d_idx, VIJ, XIJ, d_invdm, DWI, d_dvaux,
             s_m, s_idx, d_deaux, d_e, s_e):
        dsi2, row, col, drowcol, dim, dimsq = declare('int', 6)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = d_idx * dimsq
        eij = d_e[d_idx] - s_e[s_idx]
        for row in range(dim):
            d_deaux[d_idx * dim + row] += s_m[s_idx] * eij * DWI[row]
            for col in range(dim):
                drowcol = dsi2 + row * dim + col
                d_dvaux[drowcol] += s_m[s_idx] * VIJ[row] * DWI[col]
                d_invdm[drowcol] += s_m[s_idx] * XIJ[row] * DWI[col]

    def post_loop(self, d_idx, d_dv, d_divv, d_invdm, d_dvaux, d_deaux):
        dsi2, row, col, rowcol, drowcol, dim, dimsq = declare('int', 7)
        invdm, idmat, dvaux, dvauxpre, dm = declare('matrix(9)', 5)
        auginvdm = declare('matrix(18)')
        deauxpre, deaux = declare('matrix(3)', 2)

        dim = self.dim
        dimsq = dim * dim
        dsi2 = dimsq * d_idx

        for row in range(dim):
            deauxpre[row] = d_deaux[dim * d_idx + row]
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                dvauxpre[rowcol] = d_dvaux[drowcol]
                invdm[rowcol] = d_invdm[drowcol]

        identity(idmat, dim)
        augmented_matrix(invdm, idmat, dim, dim, dim, auginvdm)
        gj_solve(auginvdm, dim, dim, dm)
        mat_mult(dm, dvauxpre, dim, dvaux)
        mat_vec_mult(dm, deauxpre, dim, deaux)

        for row in range(dim):
            d_deaux[d_idx * dim + row] = deaux[row]
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                d_dvaux[drowcol] = dvaux[rowcol]


class FirstGradient(Equation):
    def __init__(self, dest, sources, dim):
        """
        First Order consistent velocity gradient and divergence
        """
        self.dim = dim
        self.dimsq = dim * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_mult, mat_vec_mult]

    def initialize(self, d_dv, d_idx, d_divv, d_de):
        dsi2, i, dim, dimsq = declare('int', 4)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx

        for i in range(dim):
            d_de[dim * d_idx + i] = 0.0

        for i in range(dimsq):
            d_dv[dsi2 + i] = 0.0
        d_divv[d_idx] = 0.0

    def loop(self, d_idx, VIJ, XIJ, d_dv, WI,
             s_m, s_rho, s_idx, d_e, s_e, d_de):
        dsi2, row, col, drowcol, dim, dimsq = declare('int', 6)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = d_idx * dimsq
        mbbyrhob = s_m[s_idx] / s_rho[s_idx]
        eij = d_e[d_idx] - s_e[s_idx]
        for row in range(dim):
            d_de[d_idx * dim + row] += mbbyrhob * eij * XIJ[row] * WI
            for col in range(dim):
                drowcol = dsi2 + row * dim + col
                d_dv[drowcol] += mbbyrhob * VIJ[row] * XIJ[col] * WI

    def post_loop(self, d_idx, d_dv, d_divv, d_cm, d_de):
        dv, dvpre, cm = declare('matrix(9)', 3)

        dsi2, row, col, rowcol, drowcol, dim, dimsq = declare('int', 7)
        depre, de = declare('matrix(3)', 2)
        dim = self.dim
        dimsq = dim * dim
        dsi2 = dimsq * d_idx

        for row in range(dim):
            depre[row] = d_de[dim * d_idx + row]
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                dvpre[rowcol] = d_dv[drowcol]
                cm[rowcol] = d_cm[drowcol]

        mat_mult(cm, dvpre, dim, dv)
        mat_vec_mult(cm, depre, dim, de)

        for row in range(dim):
            d_divv[d_idx] += dv[row * dim + row]
            d_de[d_idx * dim + row] = de[row]
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                d_dv[drowcol] = dv[rowcol]


class SecondGradient(Equation):
    def __init__(self, dest, sources, dim):
        """
        First Order consistent velocity gradient and divergence
        """
        self.dim = dim
        self.dimsq = dim * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_mult]

    def initialize(self, d_ddv, d_idx, d_divv, d_dde):
        dsi3, i, dim, dimcu, blk, row, col = declare('int', 7)
        dsi2, dimsq = declare('int', 2)
        dim = self.dim
        dimsq = dim * dim
        dimcu = dim * dim * dim
        dsi2 = dimsq * d_idx
        dsi3 = dimcu * d_idx
        for i in range(dimcu):
            d_ddv[dsi3 + i] = 0.0
        for i in range(dimsq):
            d_dde[dsi2 + i] = 0.0

    def loop(self, d_idx, VIJ, XIJ, d_dvaux, s_dvaux, WI, d_ddv,
             s_m, s_rho, s_idx, d_e, s_deaux, d_deaux, d_dde, s_de):
        dsi2, row, col, drowcol, dim, dimsq = declare('int', 6)
        blk, dblkrowcol, ssi2, srowcol, rowcol = declare('int', 5)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = d_idx * dimsq
        ssi2 = s_idx * dimsq
        mbbyrhob = s_m[s_idx] / s_rho[s_idx]

        for row in range(dim):
            deij = d_deaux[d_idx * dim + row] - s_deaux[s_idx * dim + row]
            for col in range(dim):
                drowcol = dsi2 + row * dim + col
                d_dde[drowcol] += mbbyrhob * deij * XIJ[col] * WI

        for blk in range(dim):
            for row in range(dim):
                for col in range(dim):
                    dblkrowcol = dsi2 * dim + blk * dimsq + row * dim + col
                    dvij = d_dvaux[dsi2 + blk * dim + row] - \
                           s_dvaux[ssi2 + blk * dim + row]
                    d_ddv[dblkrowcol] += mbbyrhob * dvij * XIJ[col] * WI

    def post_loop(self, d_idx, d_dv, d_divv, d_cm, d_ddv, d_dde):
        ddvpre = declare('matrix(27)')
        ddvpreb, ddvblk, cm, ddepre, dde = declare('matrix(9)', 5)
        dsi2, row, col, rowcol, dim, dimsq = declare('int', 6)
        blk, blkrowcol, dblkrowcol, dsi3, drowcol = declare('int', 5)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        dsi3 = dsi2 * dim

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                ddepre[rowcol] = d_dde[drowcol]
                cm[rowcol] = d_cm[drowcol]

        mat_mult(cm, ddepre, dim, dde)

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                d_dde[dsi2 + rowcol] = dde[rowcol]

        for blk in range(dim):
            for row in range(dim):
                for col in range(dim):
                    rowcol = row * dim + col
                    blkrowcol = blk * dimsq + rowcol
                    dblkrowcol = dsi3 + blkrowcol
                    ddvpre[blkrowcol] = d_ddv[dblkrowcol]

        for blk in range(dim):
            for row in range(dim):
                for col in range(dim):
                    rowcol = row * dim + col
                    blkrowcol = blk * dimsq + rowcol
                    ddvpreb[rowcol] = ddvpre[blkrowcol]
            mat_mult(cm, ddvpreb, dim, ddvblk)
            for row in range(dim):
                for col in range(dim):
                    rowcol = row * dim + col
                    dblkrowcol = dsi3 + blk * dimsq + rowcol
                    d_ddv[dblkrowcol] = ddvblk[rowcol]


class EntropyBasedDissipationTrigger(Equation):
    def __init__(self, dest, sources, alphamax, alphamin, fkern, l0, l1,
                 gamma):
        self.alphamax = alphamax
        self.fkern = fkern
        self.l0 = l0
        self.l1 = l1
        self.gamma = gamma
        self.alphamin = alphamin
        super().__init__(dest, sources)

    def post_loop(self, d_h, d_idx, d_cs, d_divv, d_dv, d_alpha, d_s,
                  d_p, d_rho, dt, d_aalpha):
        snew = d_p[d_idx] / pow(d_rho[d_idx], self.gamma)
        tau = d_h[d_idx] / d_cs[d_idx]
        epsdot = abs(d_s[d_idx] - snew) * tau / (d_s[d_idx] * dt)
        d_s[d_idx] = snew
        l = log(epsdot)
        x = min(max((l - self.l0) / (self.l1 - self.l0), 0), 1)
        sx = 6 * pow(x, 5) - 15 * pow(x, 4) + 10 * pow(x, 3)
        alphades = self.alphamax * sx
        if d_alpha[d_idx]>alphades:
            d_aalpha[d_idx] = -(d_alpha[d_idx] - self.alphamin)/(30*tau)
        else:
            d_alpha[d_idx] = alphades
            d_aalpha[d_idx] = 0.0



class CorrectionMatrix(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        self.dimsq = dim * dim
        super().__init__(dest, sources)

    def _get_helpers_(self):
        return [identity, augmented_matrix, gj_solve]

    def initialize(self, d_cm, d_idx):
        dsi, i, dimsq = declare('int', 3)
        dimsq = self.dimsq
        dsi = dimsq * d_idx
        for i in range(dimsq):
            d_cm[dsi + i] = 0.0

    def loop(self, d_idx, s_m, s_idx, VIJ, DWI, XIJ, d_dv,
             s_rho, d_cm, WI):
        dsi2, row, col, drowcol, dim, dimsq = declare('int', 6)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = d_idx * dimsq
        mbbyrhob = s_m[s_idx] / s_rho[s_idx]
        for row in range(dim):
            for col in range(dim):
                drowcol = dsi2 + row * dim + col
                d_cm[drowcol] += mbbyrhob * XIJ[row] * XIJ[col] * WI

    def post_loop(self, d_idx, d_dv, d_divv, d_cm):
        invcm, cm, idmat = declare('matrix(9)', 3)
        augcm = declare('matrix(18)')
        dsi2, row, col, rowcol, drowcol, dim, dimsq = declare('int', 7)

        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        identity(invcm, dim)
        identity(idmat, dim)

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                invcm[rowcol] = d_cm[drowcol]

        augmented_matrix(invcm, idmat, dim, dim, dim, augcm)
        gj_solve(augcm, dim, dim, cm)

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                d_cm[drowcol] = cm[rowcol]


class MomentumAndEnergyMI1(Equation):
    def _get_helpers_(self):
        return [mat_vec_mult, dot]

    def __init__(self, dest, sources, dim, fkern, eta_crit=0.3, eta_fold=0.2,
                 beta=2.0, alphac=0.05):
        r"""
        Momentum and Energy Equations with artificial viscosity.

        Possible typo in that has been taken care of:

        Instead of Equation F3 [Hopkins2015]_ for evolution of total
        energy sans artificial viscosity and artificial conductivity,

            .. math::
                \frac{\mathrm{d} E_{i}}{\mathrm{~d} t}=\boldsymbol{v}_{i}
                \cdot \frac{\mathrm{d} \boldsymbol{P}_{i}}{\mathrm{~d} t}-
                \sum_{j} m_{i} m_{j}\left(\boldsymbol{v}_{i}-
                \boldsymbol{v}_{j}\right) \cdot\left[\frac{P_{i}}
                {\bar{\rho}_{i}^{2}} f_{i, j} \nabla_{i}
                W_{i j}\left(h_{i}\right)\right],

        it should have been,

            .. math::
                \frac{\mathrm{d} E_{i}}{\mathrm{~d} t}=\boldsymbol{v}_{i}
                \cdot \frac{\mathrm{d} \boldsymbol{P}_{i}}{\mathrm{~d} t}+
                \sum_{j} m_{i} m_{j}\left(\boldsymbol{v}_{i}-
                \boldsymbol{v}_{j}\right) \cdot\left[\frac{P_{i}}
                {\bar{\rho}_{i}^{2}} f_{i, j} \nabla_{i}
                W_{i j}\left(h_{i}\right)\right].

        Specific thermal energy, :math:`u`, would therefore be evolved
        using,

            .. math::
                \frac{\mathrm{d} u_{i}}{\mathrm{~d} t}=
                \sum_{j} m_{j}\left(\boldsymbol{v}_{i}-
                \boldsymbol{v}_{j}\right) \cdot\left[\frac{P_{i}}
                {\bar{\rho}_{i}^{2}} f_{i, j} \nabla_{i}
                W_{i j}\left(h_{i}\right)\right]
        """
        self.beta = beta
        self.dim = dim
        self.fkern = fkern
        self.dimsq = dim * dim
        self.eta_crit = eta_crit
        self.eta_fold = eta_fold
        self.alphac = alphac
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0

        # d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs, d_rho, s_rho,
             d_au, d_av, d_aw, d_ae, XIJ, VIJ, HIJ, d_alpha, s_alpha,
             R2IJ, RHOIJ1, d_h, d_dndh, d_n, d_drhosumdh, s_h, s_dndh, s_n,
             s_drhosumdh, d_cm, s_cm, WI, WJ, d_u, d_v, d_w, s_u, s_v,
             s_w, d_dv, s_dv, d_ddv, s_ddv, d_de, s_de, d_dde, s_dde, d_e,
             s_e):

        # TODO: Make eps a parameter
        eps = 0.01
        epssq = eps * eps

        beta = self.beta

        hi = self.fkern * d_h[d_idx]
        hj = self.fkern * s_h[s_idx]

        # averaged sound speed
        cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

        scm, dcm, idmat = declare('matrix(9)', 3)
        gmi, gmj, etai, etaj, vij, mpinc = declare('matrix(3)', 6)
        dvdel, ddvdeldel = declare('matrix(3)', 2)
        dsi2, ssi2, row, col, blk = declare('int', 5)
        rowcol, drowcol, srowcol, dim, dimsq = declare('int', 5)
        dim = self.dim
        dimsq = self.dimsq
        dsi2 = dimsq * d_idx
        ssi2 = dimsq * s_idx

        for row in range(dim):
            gmi[row] = 0.0
            gmj[row] = 0.0
            etai[row] = XIJ[row] / hi
            etaj[row] = XIJ[row] / hj

        etaisq = dot(etai, etai, dim)
        etajsq = dot(etaj, etaj, dim)

        etaij = sqrt(min(etaisq, etajsq))

        aanum = 0.0
        aaden = 0.0
        for row in range(dim):
            mpinc[row] = 0.5 * XIJ[row]
            for col in range(dim):
                rowcol = row * dim + col
                aanum += d_dv[dsi2 + rowcol] * XIJ[row] * XIJ[col]
                aaden += s_dv[ssi2 + rowcol] * XIJ[row] * XIJ[col]
        aaij = aanum / aaden

        phiijin = min(1.0, 4 * aaij / ((1 + aaij) * (1 + aaij)))
        phiij = max(0.0, phiijin)

        if etaij < self.eta_crit:
            powin = (etaij - self.eta_crit) / self.eta_fold
            phiij = phiij * exp(-powin * powin)

        dedel = 0.0
        ddedel = 0.0
        for row in range(dim):
            ddvdeldel[row] = 0.0
            dvdel[row] = 0.0
            dedel -= (d_de[d_idx * dim + row] +
                      s_de[s_idx * dim + row]) * mpinc[col]
            for col in range(dim):
                rowcol = row * dim + col
                dvdel[row] -= (d_dv[dsi2 + rowcol] +
                               s_dv[ssi2 + rowcol]) * mpinc[col]
                ddedel += (d_dde[dsi2 + rowcol] -
                           s_dde[ssi2 + rowcol]) * mpinc[row] * mpinc[col]

        for blk in range(dim):
            for row in range(dim):
                for col in range(dim):
                    rowcol = row * dim + col
                    ddvdeldel[row] += \
                        (
                                d_ddv[dsi2 * dim + blk * dimsq + rowcol] -
                                s_ddv[ssi2 * dim + blk * dimsq + rowcol]
                        ) * mpinc[col] * mpinc[blk]

        vij[0] = VIJ[0] + phiij * (dvdel[0] + 0.5 * ddvdeldel[0])
        vij[1] = VIJ[1] + phiij * (dvdel[1] + 0.5 * ddvdeldel[1])
        vij[2] = VIJ[2] + phiij * (dvdel[2] + 0.5 * ddvdeldel[2])

        eij = d_e[d_idx] - s_e[s_idx] + phiij * (dedel + 0.5 * ddedel)
        rhoij = 0.5 * (s_rho[s_idx] + d_rho[d_idx])
        pij = d_rho[d_idx] - s_rho[s_idx]
        vsigng = sqrt(abs(pij) / rhoij)

        mui = min(0.0, dot(vij, etai, dim) / (etaisq + epssq))
        muj = min(0.0, dot(vij, etaj, dim) / (etajsq + epssq))

        qi = d_rho[d_idx] * mui * (-d_alpha[d_idx] * d_cs[d_idx] +
                                   beta * mui)
        qj = s_rho[s_idx] * muj * (-s_alpha[s_idx] * s_cs[s_idx] +
                                   beta * muj)

        for row in range(dim):
            for col in range(dim):
                rowcol = row * dim + col
                drowcol = dsi2 + rowcol
                srowcol = ssi2 + rowcol
                gmi[row] -= d_cm[drowcol] * XIJ[col] * WI
                gmj[row] -= s_cm[srowcol] * XIJ[col] * WJ

        mj = s_m[s_idx]

        # p_i/rhoi**2
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        pibrhoi2 = (d_p[d_idx] + qi) / rhoi2

        # pj/rhoj**2
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]
        pjbrhoj2 = (s_p[s_idx] + qj) / rhoj2

        # accelerations for velocity
        d_au[d_idx] -= mj * (pibrhoi2 * gmi[0] + pjbrhoj2 * gmj[0])
        d_av[d_idx] -= mj * (pibrhoi2 * gmi[1] + pjbrhoj2 * gmj[1])
        d_aw[d_idx] -= mj * (pibrhoi2 * gmi[2] + pjbrhoj2 * gmj[2])

        # accelerations for the thermal energy
        vijdotdwi = dot(VIJ, gmi, dim)
        normgmij = sqrt((gmi[0] + gmj[0]) * (gmi[0] + gmj[0]) +
                        (gmi[1] + gmj[1]) * (gmi[1] + gmj[1]) +
                        (gmi[2] + gmj[2]) * (gmi[2] + gmj[2]))

        d_ae[d_idx] -= 0.5 * self.alphac * mj * vsigng * eij * normgmij / rhoij
        d_ae[d_idx] += mj * pibrhoi2 * vijdotdwi


class IncreaseSmoothingLength(Equation):
    def post_loop(self, d_idx, d_h):
        d_h[d_idx] *= 1.10


class WallBoundary(Equation):
    """
        :class:`WallBoundary
        <pysph.sph.gas_dynamics.boundary_equations.WallBoundary>` modified
        for TSPH.

        Most importantly, mass of the boundary particle should never be zero
        since it appears in denominator of fij. This has been addressed.
    """

    def initialize(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_h, d_htmp, d_h0,
                   d_u, d_v, d_w, d_wij, d_n, d_dndh, d_drhosumdh, d_divv,
                   d_m0):
        d_p[d_idx] = 0.0
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_m0[d_idx] = d_m[d_idx]
        d_m[d_idx] = 0.0
        d_rho[d_idx] = 0.0
        d_e[d_idx] = 0.0
        d_cs[d_idx] = 0.0
        d_divv[d_idx] = 0.0
        d_wij[d_idx] = 0.0
        d_h[d_idx] = d_h0[d_idx]
        d_htmp[d_idx] = 0.0
        d_n[d_idx] = 0.0
        d_dndh[d_idx] = 0.0
        d_drhosumdh[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_e, d_m, d_cs, d_divv, d_h, d_u,
             d_v, d_w, d_wij, d_htmp, s_p, s_rho, s_e, s_m, s_cs, s_h, s_divv,
             s_u, s_v, s_w, WI, s_n, d_n, s_dndh, d_dndh, d_drhosumdh,
             s_drhosumdh):
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
        d_n[d_idx] += s_n[s_idx] * WI
        d_dndh[d_idx] += s_dndh[s_idx] * WI
        d_drhosumdh[d_idx] += s_drhosumdh[s_idx] * WI

    def post_loop(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_divv, d_h, d_u,
                  d_v, d_w, d_wij, d_htmp, d_n, d_dndh, d_drhosumdh, d_m0):
        if d_wij[d_idx] > 1e-30:
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
            d_n[d_idx] = d_n[d_idx] / d_wij[d_idx]
            d_dndh[d_idx] = d_dndh[d_idx] / d_wij[d_idx]
            d_drhosumdh[d_idx] = d_drhosumdh[d_idx] / d_wij[d_idx]

        # Secret Sauce
        if d_m[d_idx] < 1e-10:
            d_m[d_idx] = d_m0[d_idx]


class UpdateGhostProps(Equation):
    def __init__(self, dest, dim, sources=None):
        """
        :class:`MPMUpdateGhostProps
        <pysph.sph.gas_dynamics.basic.MPMUpdateGhostProps>` modified
        for TSPH
        """
        super().__init__(dest, sources)
        self.dim = dim
        self.dimsq = dim * dim
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_orig_idx, d_p, d_tag, d_h, d_rho, d_dndh,
                   d_n, d_cm, d_dv, d_dvaux, d_ddv, d_dde, d_de, d_deaux):
        idx, dim, dimsq, row, col, rowcol = declare('int', 6)
        blkrowcol, dsi2, si2 = declare('int', 3)
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            d_p[d_idx] = d_p[idx]
            d_h[d_idx] = d_h[idx]
            d_rho[d_idx] = d_rho[idx]
            d_dndh[d_idx] = d_dndh[idx]
            d_n[d_idx] = d_n[idx]
            dim = self.dim
            dimsq = self.dimsq
            dsi2 = dimsq * d_idx
            si2 = dimsq * idx
            for row in range(dim):
                d_de[d_idx * dim + row] = d_de[idx * dim + row]
                d_deaux[d_idx * dim + row] = d_de[idx * dim + row]
                for col in range(dim):
                    rowcol = row * dim + col
                    d_cm[dsi2 + rowcol] = d_cm[si2 + rowcol]
                    d_dv[dsi2 + rowcol] = d_dv[si2 + rowcol]
                    d_dvaux[dsi2 + rowcol] = d_dvaux[si2 + rowcol]
                    d_dde[dsi2 + rowcol] = d_dde[si2 + rowcol]

            for blk in range(dim):
                for row in range(dim):
                    for col in range(dim):
                        blkrowcol = blk * dimsq + row * dim + col
                        d_ddv[dim * dsi2 + blkrowcol] = \
                            d_ddv[dim * si2 + blkrowcol]


class PECStep(IntegratorStep):
    """Predictor Corrector integrator for Gas-dynamics modified for TSPH"""

    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h, d_u0,
                   d_v0, d_w0, d_u, d_v, d_w, d_e, d_e0, d_h0, d_converged,
                   d_rho, d_rho0, d_n, d_n0, d_alpha, d_alpha0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_e0[d_idx] = d_e[d_idx]

        d_h0[d_idx] = d_h[d_idx]
        d_rho0[d_idx] = d_rho[d_idx]
        d_n0[d_idx] = d_n[d_idx]
        d_alpha0[d_idx] = d_alpha[d_idx]

        # set the converged attribute to 0 at the beginning of a Group
        d_converged[d_idx] = 0

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_e0, d_e, d_au, d_av, d_aw, d_ae, d_rho, d_rho0,
               d_arho, d_h, d_h0, d_ah, dt, d_n, d_n0, d_an, d_alpha,
               d_alpha0, d_aalpha):
        dtb2 = 0.5 * dt

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        # update thermal energy
        d_e[d_idx] = d_e0[d_idx] + dtb2 * d_ae[d_idx]

        # predict density and smoothing lengths for faster
        # convergence. NNPS need not be explicitly updated since it
        # will be called at the end of the predictor stage.
        d_h[d_idx] = d_h0[d_idx] + dtb2 * d_ah[d_idx]
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]
        d_n[d_idx] = d_n0[d_idx] + dtb2 * d_an[d_idx]
        d_alpha[d_idx] = d_alpha0[d_idx] + dtb2 * d_aalpha[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_e0, d_e, d_au, d_av, d_aw, d_ae, dt,
               d_alpha, d_alpha0, d_aalpha):
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_e[d_idx] = d_e0[d_idx] + dt * d_ae[d_idx]
        d_alpha[d_idx] = d_alpha0[d_idx] + dt * d_aalpha[d_idx]


@annotate(fst='int', lst='int', key='doublep', arr='longp')
def quicksort(arr, key, fst=0, lst=3):
    '''
    Sort in-place with QuickSort

    :param key:  the list of numbers to sort
    :param fst: the first index from xs to begin sorting from,
                must be in the range [0, len(arr))
    :param lst: the last index from xs to stop sorting at
                must be in the range [fst, len(arr))
    :return:    nothing, the side effect is that arr[fst, lst] is sorted
    '''
    i, j = declare('int', 2)
    if fst >= lst:
        return

    i, j = fst, lst
    pivot = key[lst]

    while i <= j:
        while key[i] < pivot:
            i += 1
        while key[j] > pivot:
            j -= 1

        if i <= j:
            key[i], key[j] = key[j], key[i]
            arr[i], arr[j] = arr[j], arr[i]
            i, j = i + 1, j - 1
    quicksort(arr, key, fst, j)
    quicksort(arr, key, i, lst)
