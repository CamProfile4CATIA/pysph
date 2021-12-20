"""Wall-shock problem in 1D (40 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, GSPHScheme, SchemeChooser
from pysph.sph.gas_dynamics.cullen_dehnen.scheme import CullenDehnenScheme

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-6
tf = 0.4

# domain size and discretization parameters
xmin = -0.2
xmax = 0.2


class WallShock(ShockTubeSetup):

    def initialize(self):
        self.xmin = xmin
        self.xmax = xmax
        self.x0 = 0.0
        self.rhol = 1.0
        self.rhor = 1.0
        self.pl = 4e-7
        self.pr = 4e-7
        self.ul = 1.0
        self.ur = -1.0

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=1.5,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=500,
            help="Number of particles in left region"
        )

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        ratio = self.rhor/self.rhol
        self.nr = ratio*self.nl
        self.xb_ratio = 5
        self.dxl = (self.x0 - self.xmin) / self.nl
        self.dxr = (self.xmax - self.x0) / self.nr
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx

    def create_particles(self):
        f,b = self.generate_particles(xmin=self.xmin*self.xb_ratio,
                                       xmax=self.xmax*self.xb_ratio,
                                       dxl=self.dxl, dxr=self.dxr,
                                       m=self.dxl, pl=self.pl,
                                       pr=self.pr, h0=self.h0, bx=0.02,
                                       gamma1=gamma1, ul=self.ul, ur=self.ur)

        if self.options.scheme == 'cullendehnen':
            # override Mh set by CullenDehnenScheme.setup_properties()
            f.add_property('Mh', data=self.hdx*self.dxr*self.rhor)
        return [f,b]

    def create_scheme(self):
        self.dt = dt
        self.tf = tf

        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            alpha=1, beta=1, k=0.7, eps=0.5, g1=0.5, g2=1.0)

        mpm = GasDScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            kernel_factor=1.0,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=2,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=40, tol=1e-6
        )
        cullendehnen = CullenDehnenScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            l=0.1, alphamax=2.0, b=1.0
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
            from pysph.base.kernels import CubicSpline
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50,kernel=CubicSpline(dim=dim))

if __name__ == '__main__':
    app = WallShock()
    app.run()
    app.post_process()
