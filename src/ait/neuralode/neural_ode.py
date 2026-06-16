import jax
import equinox as eqx
import diffrax as dfx

from ..odefn import ODEFn


class NeuralODE(eqx.Module):
    f: ODEFn
    T: float = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(self, f, T=5.0, tol=1e-3, dt0=0.01, max_steps=4096):
        self.f = f
        self.T, self.tol, self.dt0, self.max_steps = T, tol, dt0, max_steps

    def _vector_field(self, t, x, args):
        return self.f(x)

    def _solve_one(self, x0):
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self._vector_field),
            dfx.Tsit5(),
            t0=0.0,
            t1=self.T,
            dt0=self.dt0,
            y0=x0,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=self.tol, atol=self.tol),
            max_steps=self.max_steps,
        )
        nfe = sol.stats["num_steps"]
        return sol.ys[-1], self.T, nfe

    def __call__(self, x):  # x: (B, *shape)
        return jax.vmap(self._solve_one)(x)
