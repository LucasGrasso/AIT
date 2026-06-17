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
        solver=None,
        stepsize_controller=None,
    ):
        self.f = f
        self.T, self.tol, self.dt0, self.max_steps = T, tol, dt0, max_steps
        if solver is not None:
            self.solver = solver
        if stepsize_controller is not None:
            self.stepsize_controller = stepsize_controller

    def _vector_field(self, t, x, args):
        return self.f(x)

    def _solve_one(self, x0):
        sol = dfx.diffeqsolve(
            dfx.ODETerm(self._vector_field),
            self.solver,
            t0=0.0,
            t1=self.T,
            dt0=self.dt0,
            y0=x0,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
        )
        steps = sol.stats["num_f_evals"]
        return sol.ys[-1], self.T, steps

    def __call__(self, x):  # x: (B, *shape)
        return jax.vmap(self._solve_one)(x)
