import jax
import jax.numpy as jnp
import equinox as eqx

from ait import AITNeuralODE, NeuralODE
from ait.odefn import ODEFn, HaltingUnit

PAD_ID = 0


class Embedder(eqx.Module):
    emb: eqx.nn.Embedding
    d_model: int = eqx.field(static=True)

    def __init__(self, key, vocab, d_model):
        self.emb = eqx.nn.Embedding(vocab, d_model, key=key)
        self.d_model = d_model

    def __call__(self, tokens):  # (S,) int -> (S, D)
        x = self.emb.weight[tokens]
        return x + self.sinusoidal_pe(x.shape[0], self.d_model)

    @staticmethod
    def sinusoidal_pe(seq_len, d_model):
        pos = jnp.arange(seq_len)[:, None]
        i = jnp.arange(d_model)[None, :]
        angle = pos / jnp.power(10000.0, (2 * (i // 2)) / d_model)
        return jnp.where(i % 2 == 0, jnp.sin(angle), jnp.cos(angle))


class AttnField(ODEFn):
    """Transformer encoder block as the ODE vector field dx/dt = block(x).

    Deterministic (no dropout): the field is the integrand of the ODE and is
    re-evaluated at adaptive times, so stochastic dropout would make the solve
    ill-defined. The residual + LayerNorm keep it bounded.
    """

    attn: eqx.nn.MultiheadAttention
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    def __init__(self, key, d_model, n_heads, d_ff):
        k = jax.random.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(n_heads, d_model, key=k[0])
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)
        self.ff1 = eqx.nn.Linear(d_model, d_ff, key=k[1])
        self.ff2 = eqx.nn.Linear(d_ff, d_model, key=k[2])

    def __call__(self, t, x, args=None):  # x: (S, D); args: (S,) bool key mask
        attn_mask = None
        if args is not None:
            s = x.shape[0]
            attn_mask = jnp.broadcast_to(args[None, :], (s, s))  # attend to real keys
        a = self.attn(x, x, x, mask=attn_mask, inference=True)
        x = jax.vmap(self.ln1)(x + a)
        f = jax.vmap(self.ff2)(jax.nn.relu(jax.vmap(self.ff1)(x)))
        out = jax.vmap(self.ln2)(x + f)
        if args is not None:
            out = out * args[:, None]  # freeze pad rows so they can't perturb the solve
        return out


class AttnHaltUnit(HaltingUnit):
    lin: eqx.nn.Linear

    def __init__(self, key, d_model, h_min=1e-4):
        self.lin = eqx.nn.Linear(d_model, 1, key=key)
        super().__init__(h_min)

    def __call__(self, t, x, args=None):  # x: (S, D) -> scalar; args: (S,) bool mask
        per_tok = jax.nn.softplus(jax.vmap(self.lin)(x))[:, 0]  # (S,)
        if args is None:
            h = jnp.mean(per_tok)
        else:
            m = args.astype(per_tok.dtype)
            h = jnp.sum(per_tok * m) / jnp.clip(jnp.sum(m), 1.0)  # masked mean
        return h + self.h_min


class TokenODEClassifier(eqx.Module):
    """Embed tokens -> AIT/NODE solve over a transformer field -> classify the
    halting-mass-averaged sequence representation. Mirrors ImgODEClassifier."""

    embedder: Embedder
    ode: AITNeuralODE | NeuralODE
    head: eqx.nn.Linear

    def __init__(
        self,
        key,
        vocab,
        model="ait",
        d_model=128,
        n_heads=8,
        d_ff=512,
        t_max=2.0,
        h_min=None,
    ):
        if h_min is None:
            h_min = 1.0 / t_max
        k = jax.random.split(key, 4)
        self.embedder = Embedder(k[0], vocab, d_model)
        f = AttnField(k[1], d_model, n_heads, d_ff)
        if model == "ait":
            self.ode = AITNeuralODE(f, AttnHaltUnit(k[2], d_model, h_min), t_max=t_max)
        else:
            self.ode = NeuralODE(f, T=t_max)
        self.head = eqx.nn.Linear(d_model, vocab, key=k[3])

    def __call__(self, x):  # x: (B, S) int token ids (PAD_ID padded)
        tokens = x.astype(jnp.int32)
        mask = tokens != PAD_ID  # (B, S) bool
        emb = jax.vmap(self.embedder)(tokens) * mask[..., None]  # zero pad positions
        x_out, T, steps = self.ode(emb, mask)  # mask threads into attn + halting
        m = mask[..., None]
        pooled = jnp.sum(x_out * m, axis=1) / jnp.clip(
            jnp.sum(m, axis=1), 1.0
        )  # masked mean over real tokens -> (B, D)
        logits = jax.vmap(self.head)(pooled)
        return logits, T, steps
