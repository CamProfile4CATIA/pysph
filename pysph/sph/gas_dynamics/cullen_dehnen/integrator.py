from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep
from math import exp


class KickDriftKickStep(IntegratorStep):

    def stage1(self, dt, d_idx, d_x, d_u, d_y, d_v, d_z, d_w,
               d_au, d_av, d_aw, d_ae, d_e, d_h, d_ah, d_ut, d_vt, d_wt,
               d_et, t):
        dtb2 = 0.5 * dt

        # Initial Kick
        d_ut[d_idx] = d_u[d_idx] + dtb2 * d_au[d_idx]
        d_vt[d_idx] = d_v[d_idx] + dtb2 * d_av[d_idx]
        d_wt[d_idx] = d_w[d_idx] + dtb2 * d_aw[d_idx]
        d_et[d_idx] = d_e[d_idx] + dtb2 * d_ae[d_idx]

        # Full Drift
        d_x[d_idx] = d_x[d_idx] + dt * d_ut[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_vt[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_wt[d_idx]

        # Prediction
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]
        # d_e[d_idx] = d_e[d_idx] * exp(dt * d_ae[d_idx] / d_e[d_idx])
        # d_h[d_idx] = d_h[d_idx] * exp(dt * d_ah[d_idx] / d_h[d_idx])

        # Fix for Noh's problem.
        # Without this if statement, d_e[d_idx] becomes inf!
        # TODO: Find the real cause to do away with this temporary fix.
        if abs(d_ae[d_idx]) >= 1e-20:
            d_e[d_idx] = d_e[d_idx] * exp(dt * d_ae[d_idx] / d_e[d_idx])
            d_h[d_idx] = d_h[d_idx] * exp(dt * d_ah[d_idx] / d_h[d_idx])

    def stage2(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_ae,
               d_e, dt, d_ut, d_vt, d_wt, d_et):

        dtb2 = 0.5 * dt

        # Final Kick
        d_u[d_idx] = d_ut[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_vt[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_wt[d_idx] + dtb2 * d_aw[d_idx]
        d_e[d_idx] = d_et[d_idx] + dtb2 * d_ae[d_idx]


class KickDriftKickIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.stage1()
        self.update_domain()
        self.compute_accelerations()
        self.update_domain()
        self.stage2()

