"""Sedov point explosion problem. (7 minutes)

Particles are distributed on concentric circles about the origin with
increasing number of particles with increasing radius. A unit charge
is distributed about the center which gives the initial pressure
disturbance.

"""
# NumPy and standard library imports
import os.path
import numpy

# PySPH base and carray imports
import numpy as np

from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.scheme import GasDScheme, SchemeChooser
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme
from pysph.sph.gas_dynamics.magma2 import MAGMA2Scheme

# PySPH tools
from pysph.tools.uniform_distribution import generate_bcc3D

# Numerical constants
dim = 3
gamma = 5.0 / 3.0
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.1

# scheme constants
alpha1 = 10.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.2


class Sedov3D(Application):
    def initialize(self):
        self.min1d = -0.5
        self.max1d = 0.5

    def add_user_options(self, group):
        group.add_argument(
            "--nparticles", action="store", type=int, dest="nprt",
            default=262144, help="Approximate number of particles in domain"
        )

    def consume_user_options(self):
        self.n_particles = self.options.nprt

    def create_particles(self):
        min1d = self.min1d
        max1d = self.max1d
        n = self.n_particles
        n1d = (0.5 * n) ** (1 / 3)
        a = (self.max1d - self.min1d) / n1d
        dx = 0.5 * a

        min_wb = min1d - 8.0 * dx
        max_wb = max1d + 8.0 * dx

        x, y, z = generate_bcc3D(a, min_wb, max_wb, min_wb, max_wb, min_wb,
                                 max_wb)

        fluid_indices = []
        boundary_indices = []
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            if (xi < min1d) | (xi > max1d) | (yi < min1d) | (yi > max1d) | (
                    zi < min1d) | (zi > max1d):
                boundary_indices.append(i)
            else:
                fluid_indices.append(i)

        m0 = 3.81e-6

        _fluid = gpa(name='fluid', x=x[fluid_indices], y=y[fluid_indices],
                     z=z[fluid_indices], deltax=0, deltay=0, deltaz=0,
                     pouerr=0, n=0, rho=1.0, m=m0,
                     rhodes=1.0, h=4 * kernel_factor * dx, arho=0.0, an=0.0,
                     dndh=0.0, prevn=0.0, prevdndh=0.0, converged=0.0, ah=0.0,
                     h0=0.0)

        _boundary = gpa(name='boundary', x=x[boundary_indices],
                        y=y[boundary_indices], z=z[boundary_indices], rho=1.0,
                        m=m0, rhodes=1.0,
                        h=4 * kernel_factor * dx)

        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import WendlandQuinticC6
        from pysph.sph.gas_dynamics.magma2 import (SettleByArtificialPressure,
                                                   SummationDensityMPMStyle)
        from pysph.sph.basic_equations import SummationDensity
        from pysph.sph.equation import Group
        from pysph.solver.utils import dump

        kernel = WendlandQuinticC6(dim=dim)
        all_pa = ['fluid', 'boundary']
        equations0 = \
            [
                Group(
                    equations=[
                        SummationDensity(dest='fluid', sources=all_pa)],
                    update_nnps=True, iterate=True, max_iterations=250),
                Group(
                    equations=[SettleByArtificialPressure(dest='fluid',
                                                          sources=all_pa)])
            ]
        equations = \
            [
                Group(
                    equations=[
                        SummationDensityMPMStyle(dest='fluid',
                                                 sources=all_pa,
                                                 dim=dim)],
                    update_nnps=True, iterate=True,
                    max_iterations=250),
                Group(
                    equations=[SettleByArtificialPressure(dest='fluid',
                                                          sources=all_pa)])
            ]

        evaluator0 = SPHEvaluator(arrays=[_fluid, _boundary],
                                  equations=equations0, dim=dim, kernel=kernel)

        evaluator = SPHEvaluator(arrays=[_fluid, _boundary],
                                 equations=equations, dim=dim, kernel=kernel)

        out_dir = self.options.output_dir
        # for i in range(250):
        #     evaluator0.evaluate()
        #     _boundary.set(
        #         m=numpy.mean(_fluid.m) * numpy.ones_like(_boundary.x))
        #
        #     dump(filename=f"{out_dir}/settle_{i}",
        #          particles=[_fluid, _boundary],
        #          solver_data={'t': 0, 'dt': 0, 'count': i},
        #          detailed_output=True)
        #
        # _fluid.set(
        #     h0=_fluid.h)
        #
        # for i in range(250, 500):
        #     evaluator.evaluate()
        #     _boundary.set(
        #         m=numpy.mean(_fluid.m) * numpy.ones_like(_boundary.x))
        #     _fluid.set(
        #         converged=numpy.zeros_like(_boundary.x))
        #     _fluid.set(
        #         h0=_fluid.h)
        #     dump(filename=f"{out_dir}/settle_{i}",
        #          particles=[_fluid, _boundary],
        #          solver_data={'t': 0, 'dt': 0, 'count': i},
        #          detailed_output=True)

        x = x[fluid_indices]
        y = y[fluid_indices]
        z = z[fluid_indices]
        rho = 1.0 * np.ones_like(x)
        m = m0 * np.ones_like(x)

        e = numpy.ones_like(x) * 1e-9
        r = numpy.sqrt(x * x + y * y + z * z)
        e[r < (4 * kernel_factor * dx)] = 1 / numpy.mean(
            m[r < (4 * kernel_factor * dx)])
        p = gamma1 * rho * e

        fluid = gpa(name='fluid', x=x, y=y, z=z,
                    rho=rho, p=p, e=e, h=kernel_factor * dx, m=m)

        boundary = gpa(name='boundary', x=_boundary.x, y=_boundary.y,
                       z=_boundary.z, rho=_boundary.rho,
                       p=gamma1 * _boundary.rho * 1e-9,
                       e=1e-9, h=_boundary.h, m=_boundary.m)
        self.scheme.setup_properties([fluid, boundary])

        # set the initial smoothing length proportional to the particle
        # volume
        fluid.h[:] = kernel_factor * (fluid.m / fluid.rho) ** (1. / dim)
        boundary.h[:] = kernel_factor * (boundary.m / boundary.rho) ** (
                    1. / dim)

        print("Sedov's point explosion in 3D with %d particles"
              % (fluid.get_number_of_particles()))

        return [fluid]

    def create_scheme(self):
        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta, adaptive_h_scheme="mpm",
            update_alpha1=True, update_alpha2=True
        )
        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=kernel_factor
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=kernel_factor
        )

        # TODO: Try Make this work with reconstruction order 2.

        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            adaptive_h_scheme='mpm', hfact=1.2)

        s = SchemeChooser(
            default='mpm', mpm=mpm, psph=psph, tsph=tsph, magma2=magma2
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        s.configure_solver(
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=25
        )

    def pre_step(self, solver):
        solver.dump_output()


if __name__ == '__main__':
    app = Sedov3D()
    app.run()
