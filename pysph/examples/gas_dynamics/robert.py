"""Simulate the Robert's problem (1D) (40 seconds).
"""

from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, GSPHScheme, SchemeChooser
from pysph.sph.gas_dynamics.cullen_dehnen.scheme import CullenDehnenScheme
import numpy

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.1

# domain size and discretization parameters
xmin = -0.5
xmax = 0.5


class Robert(ShockTubeSetup):

    def initialize(self):
        self.xmin = -0.5
        self.xmax = 0.5
        self.x0 = 0.0
        self.rhol = 3.86
        self.rhor = 1.0
        self.pl = 10.33
        self.pr = 1.0
        self.ul = -0.39
        self.ur = -3.02

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=2.0,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=1930,
            help="Number of particles in left region"
        )

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        ratio = self.rhor/self.rhol
        self.xb_ratio = 2
        self.nr = ratio*self.nl
        self.dxl = 0.5/self.nl
        self.dxr = 0.5/self.nr
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx

    def create_particles(self):
        f,b =self.generate_particles(xmin=self.xmin*self.xb_ratio,
                                       xmax=self.xmax*self.xb_ratio,
                                       dxl=self.dxl, dxr=self.dxr,
                                       m=self.dxr, pl=self.pl, pr=self.pr,
                                       h0=self.h0, bx=0.03, gamma1=gamma1,
                                       ul=self.ul, ur=self.ur)

        self.scheme.setup_properties([f, b])
        if self.options.scheme == 'cullendehnen':
            # override Mh set by CullenDehnenScheme.setup_properties()
            f.add_property('Mh', data=self.hdx*self.dxl*self.rhol)
        return [f]

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            alpha=1, beta=2.0, k=1.0, eps=0.5, g1=0.5, g2=1.0)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=2.0,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6
        )

        cullendehnen = CullenDehnenScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            l=0.1, alphamax=2.0, b=2.0, has_ghosts=True
        )

        s = SchemeChooser(default='adke', adke=adke, mpm=mpm, gsph=gsph,
                          cullendehnen=cullendehnen)
        return s

    def configure_scheme(self):
        s = self.scheme
        dxl = 0.5/self.nl
        ratio = self.rhor/self.rhol
        nr = ratio*self.nl
        dxr = 0.5/self.nr
        h0 = self.hdx * self.dxr
        kernel_factor = self.options.hdx
        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=True, pfreq=50)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'gsph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'crk':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=1)
        elif self.options.scheme == 'cullendehnen':
            from pysph.base.kernels import Gaussian
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50, kernel=Gaussian(dim=dim))

if __name__ == '__main__':
    app = Robert()
    app.run()
    app.post_process()
