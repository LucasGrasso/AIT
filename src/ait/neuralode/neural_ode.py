import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx

from ..odefn import ODEFn


class NeuralODE(eqx.Module):
    f: ODEFn
    T: float = eqx.field(static=True)
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
        T=5.0,
        tol=1e-3,
        dt0=0.01,
        max_steps=4096,
        dense=False,
        save_interval=0.1,
        solver=None,
        stepsize_controller=None,
    ):
        self.f = f
        self.T, self.tol, self.dt0, self.max_steps = T, tol, dt0, max_steps
        self.dense, self.save_interval = dense, save_interval
        if solver is not None:
            self.solver = solver
        if stepsize_controller is not None:
            self.stepsize_controller = stepsize_controller

    def _vector_field(self, t, x, args):
        return self.f(t, x, args)

    def _saveat(self):
        if self.dense:
            ts = jnp.arange(0.0, self.T, self.save_interval)
            return dfx.SaveAt(ts=ts, t1=True)
        return dfx.SaveAt(t1=True)

    def _solve_one(self, x0, args=None):
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self._vector_field),
            self.solver,
            t0=0.0,
            t1=self.T,
            dt0=self.dt0,
            y0=x0,
            args=args,
            saveat=self._saveat(),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
        )
        assert sol.ys is not None and sol.ts is not None
        steps = sol.stats["num_steps"]
        if self.dense:
            return sol.ys, sol.ts, steps  # ys: (n_save, *state), ts: (n_save,)
        return sol.ys[-1], self.T, steps

    def __call__(self, x, args=None):  # x: (B, *shape)
        if args is None:
            return jax.vmap(self._solve_one)(x)
        return jax.vmap(self._solve_one)(x, args)
