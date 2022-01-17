from math import pi

from pysph.sph.equation import Group
from pysph.sph.gas_dynamics.cullen_dehnen.equations import *
from pysph.sph.scheme import Scheme


class CullenDehnenScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, l, b, alphamax, h0=None,
                 fkern=1.0,
                 has_ghosts=False):

        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.l = l
        self.b = b
        self.has_ghosts = has_ghosts
        self.alphamax = alphamax
        self.h0 = h0
        # Since hasattr() or try-except cannot be used inside equations,
        # it is better take care of it here itself.
        self.fkern = fkern

    def add_user_options(self, group):
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None,
            help="Gamma for the state equation."
        )

    def consume_user_options(self, options):
        vars = ['gamma']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):

        from pysph.sph.gas_dynamics.cullen_dehnen.kernel import CubicSplineH1
        if kernel is None:
            kernel = CubicSplineH1(dim=self.dim)
            self.fkern = kernel.fkern

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.gas_dynamics.cullen_dehnen.integrator import (
            KickDriftKickIntegrator, KickDriftKickStep)

        if integrator_cls is not None:
            cls = integrator_cls
        else:
            cls = KickDriftKickIntegrator
        step_cls = KickDriftKickStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        equations = []

        sweep0 = []
        all_pa = self.fluids + self.solids
        dim = self.dim
        for fluid in self.fluids:
            sweep0.append(
                SummationDensity(
                    dest=fluid, sources=all_pa, dim=dim
                )
            )
            sweep0.append(
                Factorf(dest=fluid,
                        sources=all_pa,
                        dim=dim)
            )

        equations.append(Group(equations=sweep0, update_nnps=True))

        adapt = []
        for fluid in self.fluids:
            adapt.append(
                AdjustSmoothingLength(
                    dest=fluid, sources=None, dim=dim)
            )

        equations.append(Group(equations=adapt, update_nnps=True))

        # Strictly, this wall boundary equation is not a part of
        # Cullen Dehnen paper. These are from
        # pysph.sph.gas_dynamics.boundary_equations.WallBoundary
        # with modifications just so that this scheme can be used to run tests
        # with solid walls.
        walleq1 = []
        for solid in self.solids:
            walleq1.append(WallBoundary1(solid, sources=self.fluids,
                                         dim=self.dim))
        equations.append(Group(equations=walleq1))

        sweep1 = []
        for fluid in self.fluids:
            sweep1.append(
                SummationDensity(dest=fluid, sources=all_pa, dim=dim)
            )
            sweep1.append(
                Factorf(dest=fluid,
                        sources=all_pa,
                        dim=dim),
            )
            sweep1.append(
                SmoothingLengthRate(dest=fluid, sources=all_pa,
                                    dim=dim)
            )
            sweep1.append(
                VelocityGradient(dest=fluid, sources=all_pa, dim=dim)
            )
            sweep1.append(
                VelocityDivergence(dest=fluid, sources=None, dim=dim)
            )
            sweep1.append(
                AcclerationGradient(dest=fluid, sources=all_pa,
                                    dim=dim)
            )
            sweep1.append(
                VelocityDivergenceRate(dest=fluid, sources=None, dim=dim)
            )
            sweep1.append(
                TracelessSymmetricStrainRate(dest=fluid, sources=None,
                                             dim=dim)
            )
            sweep1.append(
                ShockIndicatorR(dest=fluid, sources=all_pa)
            )

        equations.append(Group(equations=sweep1, update_nnps=True))

        bwsweeps = []
        for fluid in self.fluids:
            bwsweeps.append(
                EOS(dest=fluid, sources=None, gamma=self.gamma)
            )
            bwsweeps.append(
                SignalVelocity(dest=fluid, sources=all_pa)
            )
            bwsweeps.append(
                FalseDetectionSuppressingLimiterXi(dest=fluid,
                                                   sources=None, dim=dim)
            )
            bwsweeps.append(
                NovelShockIndicatorA(dest=fluid, sources=None)
            )
            bwsweeps.append(
                IndividualViscosityLocal(dest=fluid, sources=None,
                                         alphamax=self.alphamax)
            )
            bwsweeps.append(
                ViscosityDecayTimeScale(dest=fluid, sources=None,
                                        l=self.l, fkern=self.fkern)
            )
            bwsweeps.append(
                AdaptIndividualViscosity(dest=fluid, sources=None)
            )

        equations.append(Group(equations=bwsweeps, update_nnps=True))

        # Strictly, this wall boundary equation is not a part of
        # cullen dehnen paper. These are from
        # pysph.sph.gas_dynamics.boundary_equations.WallBoundary
        # with modifications just so that this scheme can be used to run tests
        # with solid walls.
        walleq2 = []
        for solid in self.solids:
            walleq2.append(WallBoundary2(solid, sources=self.fluids,
                                         dim=self.dim))
        equations.append(Group(equations=walleq2))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(
                    UpdateGhostProps(dest=fluid, sources=None)
                )
            equations.append(Group(equations=gh, real=False))

        sweep2 = []
        for fluid in self.fluids:
            sweep2.append(
                MomentumAndEnergy(dest=fluid, sources=all_pa)
            )
            sweep2.append(
                ArtificialViscocity(dest=fluid, sources=all_pa, b=self.b)
            )
        equations.append(Group(equations=sweep2, update_nnps=True))

        return equations

    def setup_properties(self, particles, clean=True):
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'p', 'rho', 'h', 'e']
        props = ['tag', 'pid', 'gid', 'ah', 'x', 'u', 'z', 'h0', 'e', 'aw',
                 'p', 'v', 'w', 'y', 'ae', 'cs', 'm', 'au', 'rho', 'h', 'av']

        add_props = ['hnurho', 'f', 'ftil', 'et', 'ut', 'vt', 'wt',
                     'divv', 'adivv', 'R', 'vsig', 'xi', 'A',
                     'alphaloc', 'tau', 'alpha', 'ahden', 'hnu']

        props.extend(add_props)

        # For Mh.
        # According to the paper, Mh is supposed to be a global constant and
        # need not be set here per particle. Constant Mh throughout the domain
        # would mean striving for constant mass inside the kernel
        # and would lead to a different number of neighbours
        # for heavy and light particles. Letting Mh be a particle property
        # allows the flexibility to run with different Mh for each
        # particle, making it possible to have variable mass inside the kernel
        # while keeping the number of neighbours constant while also making
        # it possible to go with constant mass inside the kernel as required.

        if self.dim == 1:
            Nh = 5.0
            Vnu = 2.0
        elif self.dim == 2:
            Nh = 13.0
            Vnu = pi
        elif self.dim == 3:
            Nh = 40.0
            Vnu = 4.0 * pi / 3.0

        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            pa.add_property('gradv', stride=9)
            pa.add_property('invtt', stride=9)
            pa.add_property('grada', stride=9)
            pa.add_property('ss', stride=6)
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.add_property('Mh', data=pa.m * Nh / Vnu)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('divv cs wij htmp'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.add_property('Mh', data=max(pa.m) * Nh / Vnu)
            pa.set_output_arrays(output_props)
