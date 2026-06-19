import equinox as eqx
from jaxtyping import ArrayLike


class ODEFn(eqx.Module):
    def __call__(self, t, x) -> ArrayLike: ...
