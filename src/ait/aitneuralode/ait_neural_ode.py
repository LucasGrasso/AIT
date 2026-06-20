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
    dense: bool = eqx.field(static=True)
    save_interval: float = eqx.field(static=True)
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
        dense=False,
        save_interval=0.1,
        solver=None,
        stepsize_controller=None,
    ):
        self.f, self.h = f, h
        self.t_max, self.tol = t_max, tol
        self.dt0, self.max_steps = dt0, max_steps
        self.dense, self.save_interval = dense, save_interval
        if solver is not None:
            self.solver = solver
        if stepsize_controller is not None:
            self.stepsize_controller = stepsize_controller

    def _vector_field(self, t, state, args):
        x, A, xbar = state
        hx = jnp.reshape(self.h(t, x, args), ())
        return (self.f(t, x, args), hx, hx * x)  # dz/dt = [f, h, h·x]

    def _cond(self, t, y, args, **kwargs):
        return 1.0 - y[1]

    def _saveat(self):
        if self.dense:
            ts = jnp.arange(0.0, self.t_max, self.save_interval)
            return dfx.SaveAt(ts=ts, t1=True)
        return dfx.SaveAt(t1=True)

    def _solve_one(self, x0, args=None):
        state0 = (x0, jnp.zeros(()), jnp.zeros_like(x0))
        event = dfx.Event(self._cond, optx.Newton(rtol=1e-5, atol=1e-5))
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self._vector_field),
            self.solver,
            t0=0.0,
            t1=self.t_max,
            dt0=self.dt0,
            y0=state0,
            args=args,
            event=event,
            saveat=self._saveat(),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
            throw=False,
        )
        assert sol.ys is not None and sol.ts is not None
        steps = sol.stats["num_steps"]
        if self.dense:
            # ys = (x, A, xbar) trajectories (n_save, *state); points past T* are inf
            return sol.ys, sol.ts, steps
        return sol.ys[2][-1], sol.ts[-1], steps

    def __call__(self, x, args=None):
        if args is None:
            return jax.vmap(self._solve_one)(x)
        return jax.vmap(self._solve_one)(x, args)
