"""Simulate the classical Sod Shocktube problem in 1D (5 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup_cv import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, GSPHScheme, SchemeChooser
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme
from pysph.sph.gas_dynamics.magma2 import MAGMA2Scheme
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.base.nnps import DomainManager
import numpy
from pysph.base.utils import get_particle_array as gpa

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.15


class SodShockTube(ShockTubeSetup):

    def initialize(self):
        self.xmin = -0.5
        self.xmax = 0.5
        self.x0 = 0.0
        self.rhol = 1.0
        self.rhor = 0.125
        self.pl = 1.0
        self.pr = 0.1
        self.ul = 0.0
        self.ur = 0.0

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=1.5,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=640,
            help="Number of particles in left region"
        )

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        ratio = self.rhor / self.rhol
        self.nr = self.nl * ratio
        self.dxl = 0.5 / self.nl
        self.dxr = 0.5 / self.nr
        self.ml = self.dxl * self.rhol
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx
        self.dt = dt
        self.tf = tf

    def create_particles(self):
        # Boundary particles are not needed as we are mirroring the particles
        # using the domain manager. Hence, bx is set to 0.0.
        f, b = self.generate_particles(
            xmin=self.xmin, xmax=self.xmax, dxl=self.dxl, dxr=self.dxr,
            m=self.ml, pl=self.pl, pr=self.pr, h0=self.h0, bx=0.00,
            gamma1=gamma1, ul=self.ul, ur=self.ur
        )
        self.scheme.setup_properties([f, b])

        xmin = self.xmin
        xmax = self.xmax
        dx = (xmax - xmin) / 720
        pl = self.pl
        pr = self.pr
        h0 = self.h0
        bx = 0.00
        ul = self.ul
        ur = self.ur

        xt = numpy.arange(xmin - bx + 0.5 * dx, xmax, dx)

        left_indices = numpy.where((xt > xmin) & (xt < 0))[0]
        right_indices = numpy.where((xt >= 0) & (xt < xmax))[0]

        x1 = xt[left_indices]
        x2 = xt[right_indices]

        x = numpy.concatenate([x1, x2])

        right_indices = numpy.where(x > 0.0)[0]

        # rho = numpy.ones_like(x) * self.rhol
        # rho[right_indices] = self.rhor
        #
        # p = numpy.ones_like(x) * pl
        # p[right_indices] = pr
        #
        # u = numpy.ones_like(x) * ul
        # u[right_indices] = ur

        rho = (self.rhol - self.rhor) / (1 + numpy.exp(x / dx)) + self.rhor
        p = (self.pl - self.pr) / (1 + numpy.exp(x / dx)) + self.pr
        u = (self.ul - self.ur) / (1 + numpy.exp(x / dx)) + self.ur

        m = dx * rho

        h = numpy.ones_like(x) * h0
        m = numpy.ones_like(x) * m
        e = p / (gamma1 * rho)
        # e = numpy.cos(x) + 2
        wij = numpy.ones_like(x)

        constants = {}
        fluid = gpa(
            constants=constants, name='fluid', x=x, rho=rho, p=p,
            e=e, h=h, m=m, u=u, wij=wij, h0=h.copy()
        )
        self.scheme.setup_properties([fluid])

        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=self.xmin, xmax=self.xmax, mirror_in_x=True,
            n_layers=2
        )

    def configure_scheme(self):
        scheme = self.scheme
        if self.options.scheme in ['gsph', 'mpm']:
            scheme.configure(kernel_factor=self.hdx)
        elif self.options.scheme in ['psph', 'tsph', 'magma2']:
            scheme.configure(hfact=self.hdx)
        scheme.configure_solver(tf=self.tf, dt=self.dt)

    def create_scheme(self):
        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=1.0, k=0.3, eps=0.5, g1=0.2, g2=0.4)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=None, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True,
        )
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=None,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=True, blend_alpha=2.0,
            niter=20, tol=1e-6
        )
        crk = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0,
            nu=0, h0=0, p0=0, gamma=gamma, cl=3
        )

        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=None
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=None, has_ghosts=True
        )

        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=None, has_ghosts=True, ndes=7, formulation='mi2',
            reconstruction_order=2, adaptive_h_scheme='mpm'
        )

        s = SchemeChooser(
            default='adke', adke=adke, mpm=mpm, gsph=gsph, crk=crk, psph=psph,
            tsph=tsph, magma2=magma2)
        return s


if __name__ == '__main__':
    app = SodShockTube()
    app.run()
    app.post_process()
