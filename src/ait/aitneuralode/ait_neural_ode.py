import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import diffrax as dfx

from ..odefn import ODEFn, HaltingUnit

class AITNeuralODE(eqx.Module):
    f: ODEFn
    h: HaltingUnit
    t_max: float = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    solver: dfx.AbstractSolver = eqx.field(static=True, default=dfx.Tsit5())
    stepsize_controller: dfx.AbstractStepSizeController = eqx.field(
        static=True, default=dfx.PIDController(rtol=1e-3, atol=1e-3)
    )

    def __init__(
        self,
        f,
        h,
        t_max=5.0,
        tol=1e-3,
        dt0=0.01,
        max_steps=4096,
        solver=None,
        stepsize_controller=None,
    ):
        self.f, self.h = f, h
        self.t_max, self.tol = t_max, tol
        self.dt0, self.max_steps = dt0, max_steps
        if solver is not None:
            self.solver = solver
        if stepsize_controller is not None:
            self.stepsize_controller = stepsize_controller

    def _vector_field(self, t, state, args):
        x, A, xbar = state
        hx = jnp.reshape(self.h(x), ())
        return (self.f(x), hx, hx * x)  # dz/dt = [f, h, h·x]

    def _cond(self, t, y, args, **kwargs):
        return 1.0 - y[1]

    def _solve_one(self, x0):
        state0 = (x0, jnp.zeros(()), jnp.zeros_like(x0))
        event = dfx.Event(self._cond, optx.Newton(rtol=1e-5, atol=1e-5))
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self._vector_field),
            self.solver,
            t0=0.0,
            t1=self.t_max,
            dt0=self.dt0,
            y0=state0,
            event=event,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
            throw=False
        )
        return sol.ys[2][-1], sol.ts[-1]

    def __call__(self, x):
        return jax.vmap(self._solve_one)(x)
