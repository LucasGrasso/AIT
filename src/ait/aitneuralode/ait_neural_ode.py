import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import diffrax as dfx


class AITNeuralODE(eqx.Module):
    f: eqx.Module
    h: eqx.Module
    t_max: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(self, f, h, t_max=5.0, eps=1e-3, tol=1e-3, dt0=0.01, max_steps=4096):
        self.f, self.h = f, h
        self.t_max, self.eps, self.tol = t_max, eps, tol
        self.dt0, self.max_steps = dt0, max_steps

    def _vector_field(self, t, state, args):
        x, A, xbar = state
        hx = jnp.reshape(self.h(x), ())
        return (self.f(x), hx, hx * x)  # dz/dt = [f, h, h·x]

    def _cond(self, t, state, args, **kwargs):
        return (1.0 - self.eps) - state[1]

    def _solve_one(self, x0):
        state0 = (x0, jnp.zeros(()), jnp.zeros_like(x0))
        event = dfx.Event(self._cond, optx.Bisection(rtol=1e-5, atol=1e-5))
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self._vector_field),
            dfx.Tsit5(),
            t0=0.0,
            t1=self.t_max,
            dt0=self.dt0,
            y0=state0,
            event=event,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=self.tol, atol=self.tol),
            max_steps=self.max_steps,
        )
        xT, AT, xbarT = sol.ys[0][-1], sol.ys[1][-1], sol.ys[2][-1]
        return xbarT + (1.0 - AT) * xT, sol.ts[-1]

    def __call__(self, x):
        return jax.vmap(self._solve_one)(x)
