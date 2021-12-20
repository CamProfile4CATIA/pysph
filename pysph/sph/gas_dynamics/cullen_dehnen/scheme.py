from pysph.sph.scheme import Scheme, add_bool_argument
from math import pi


class CullenDehnenScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, l, b, alphamax, Mh=None,
                 Nh=None, m=None, h0=None,
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
        if Mh is None and m is not None:
            if Nh is None:
                if self.dim == 1:
                    Nh = 5
                elif self.dim == 2:
                    Nh = 13
                elif self.dim == 3:
                    Nh = 40
            if self.dim == 1:
                Vnu = 2.0
            elif self.dim == 2:
                Vnu = pi
            elif self.dim == 3:
                Vnu = 4.0 * pi / 3.0

            Mh = m * Nh / Vnu
        self.Mh = Mh

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
        if not hasattr(kernel, 'fkern'):
            kernel.fkern = 1.0

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
        from pysph.sph.equation import Group
        from pysph.sph.gas_dynamics.cullen_dehnen.equations import (
            SummationDensity, Factorf,
            AdjustSmoothingLength, SmoothingLengthRate, VelocityGradient,
            VelocityDivergence, AcclerationGradient, VelocityDivergenceRate,
            TracelessSymmetricStrainRate, ShockIndicatorR, EOS, SignalVelocity,
            FalseDetectionSuppressingLimiterXi, NovelShockIndicatorA,
            IndividualViscosityLocal, ViscosityDecayTimeScale,
            AdaptIndividualViscosity, UpdateGhostProps,
            MomentumAndEnergy, ArtificialViscocity, WallBoundary)

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
                VelocityDivergence(dest=fluid, sources=None)
            )
            sweep1.append(
                AcclerationGradient(dest=fluid, sources=all_pa,
                                    dim=dim)
            )
            sweep1.append(
                VelocityDivergenceRate(dest=fluid, sources=None)
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
                                                   sources=None)
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
                                        l=self.l)
            )
            bwsweeps.append(
                AdaptIndividualViscosity(dest=fluid, sources=None)
            )

        equations.append(Group(equations=bwsweeps, update_nnps=True))

        walleq = []
        for solid in self.solids:
            walleq.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=walleq))

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
        from pysph.base.utils import get_particle_array_gasd
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_gasd(name='junk')
        props = list(dummy.properties.keys())
        output_props = dummy.output_property_arrays
        add_props = ['hnurho', 'f', 'ftil', 'et', 'ut', 'vt', 'wt', 'D21',
                     'gradv20', 'invT10', 'invT00', 'invT20', 'invT11',
                     'invT02', 'D01', 'gradv01', 'gradv10', 'D10', 'D02',
                     'gradv21', 'invT22', 'gradv02', 'D00', 'invT21', 'invT01',
                     'D22', 'D12', 'D11', 'invT12', 'gradv12', 'D20',
                     'gradv11', 'gradv00', 'gradv22', 'divv', 'grada21',
                     'grada11', 'grada00', 'grada02', 'grada12', 'DD11',
                     'DD21', 'DD02', 'DD22', 'grada01', 'DD10', 'DD00', 'DD12',
                     'grada10', 'DD20', 'DD01', 'grada20', 'grada22', 'adivv',
                     'S10', 'S22', 'S11', 'S00', 'S20', 'S21', 'R', 'vsig',
                     'xi', 'A', 'alphaloc', 'tau', 'alpha', 'ahden', 'hnu']
        props.extend(add_props)
        if self.dim == 1:
            Nh = 5.0
            Vnu = 2.0
        elif self.dim == 2:
            Nh = 13.0
            Vnu = pi
        elif self.dim == 3:
            Nh = 40
            Vnu = 4.0 * pi / 3.0

        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.add_property('Mh', data=pa.m * Nh / Vnu)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('div cs wij htmp'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.add_property('Mh', data=pa.m * Nh / Vnu)
            pa.set_output_arrays(output_props)
