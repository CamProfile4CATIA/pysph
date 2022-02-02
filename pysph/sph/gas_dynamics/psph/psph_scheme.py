from pysph.sph.scheme import Scheme, add_bool_argument


class PSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, kernel_factor, alpha1=1.0,
                 alpha2=0.0, beta=2.0, update_alpha2=False, fkern=1.0,
                 max_density_iterations=250, alphaav=1.0,
                 density_iteration_tolerance=1e-3, has_ghosts=False):

        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.update_alpha2 = update_alpha2
        self.beta = beta
        self.kernel_factor = kernel_factor
        self.density_iteration_tolerance = density_iteration_tolerance
        self.max_density_iterations = max_density_iterations
        self.has_ghosts = has_ghosts
        self.fkern = fkern
        self.alphaav = alphaav

    def add_user_options(self, group):

        group.add_argument(
            "--alpha1", action="store", type=float, dest="alpha1",
            default=None,
            help="Alpha1 for the artificial viscosity."
        )
        group.add_argument(
            "--beta", action="store", type=float, dest="beta",
            default=None,
            help="Beta for the artificial viscosity."
        )
        group.add_argument(
            "--alpha2", action="store", type=float, dest="alpha2",
            default=None,
            help="Alpha2 for artificial viscosity"
        )
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None,
            help="Gamma for the state equation."
        )

    def consume_user_options(self, options):
        vars = ['gamma', 'alpha2', 'alpha1', 'beta']
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
        from pysph.sph.gas_dynamics.psph.pec_step import PECStep

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
        from pysph.sph.gas_dynamics.basic import (MPMUpdateGhostProps)
        from pysph.sph.gas_dynamics.tsph.equations import (
            MorrisMonaghanSwitch)
        from pysph.sph.gas_dynamics.psph.equations import (
            PSPHSummationDensityAndPressure, GradientKinsfolkC1,
            SignalVelocity, LimiterAndAlphas, MomentumAndEnergy)
        from pysph.sph.gas_dynamics.boundary_equations import WallBoundary

        equations = []
        # Find the optimal 'h'

        g1 = []
        for fluid in self.fluids:
            g1.append(
                PSPHSummationDensityAndPressure(
                    dest=fluid, sources=self.fluids, k=self.kernel_factor,
                    density_iterations=True, dim=self.dim,
                    htol=self.density_iteration_tolerance, gamma=self.gamma
                )
            )

            equations.append(Group(
                equations=g1, update_nnps=True, iterate=True,
                max_iterations=self.max_density_iterations
            ))

        g5 = []
        for fluid in self.fluids:
            g5.append(GradientKinsfolkC1(
                dest=fluid,
                sources=self.fluids + self.solids,
                dim=self.dim))

            g5.append(SignalVelocity(
                dest=fluid,
                sources=self.fluids + self.solids,
            ))

            g5.append(MorrisMonaghanSwitch(
                dest=fluid,
                sources=None,
                alpha1_min=self.alpha1,
                sigma=0.1
            ))
        equations.append(Group(equations=g5))

        g2 = []
        for fluid in self.fluids:
            g2.append(LimiterAndAlphas(
                dest=fluid,
                sources=self.fluids))
        equations.append(Group(equations=g2))


        g3 = []
        for solid in self.solids:
            g3.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g3))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(
                    MPMUpdateGhostProps(dest=fluid, sources=None)
                )
            equations.append(Group(equations=gh, real=False))

        g4 = []
        for fluid in self.fluids:
            g4.append(MomentumAndEnergy(
                dest=fluid, sources=self.fluids + self.solids,
                dim=self.dim,
                beta=self.beta,
                fkern=self.fkern
            ))

        equations.append(Group(equations=g4))

        return equations

    def setup_properties(self, particles, clean=True):
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])

        props = ['rho', 'm', 'x', 'y', 'z', 'u', 'v', 'w', 'h', 'cs', 'p', 'e',
                 'au', 'av', 'aw', 'ae', 'pid', 'gid', 'tag', 'dwdh', 'alpha1',
                 'aalpha1', 'alpha10', 'h0', 'converged', 'ah', 'arho',
                 'dt_cfl', 'e0', 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0']
        more_props = ['drhosumdh', 'n', 'dndh', 'prevn', 'prevdndh',
                      'prevdrhosumdh', 'divv', 'dpsumdh', 'dprevpsumdh', 'an',
                      'adivv', 'trssdsst', 'vsig', 'alpha', 'alpha0', 'xi', "R"]
        props.extend(more_props)
        output_props = []
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            # Guess for number density.
            pa.add_property('n', data=pa.rho / pa.m)
            pa.add_property('gradv', stride=9)
            pa.add_property('invtt', stride=9)
            pa.add_property('grada', stride=9)
            pa.add_property('ss', stride=6)
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('div cs wij htmp'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.set_output_arrays(output_props)