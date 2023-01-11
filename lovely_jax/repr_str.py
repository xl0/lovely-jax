# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_repr_str.ipynb.

# %% auto 0
__all__ = ['jax_to_str_common', 'lovely']

# %% ../nbs/00_repr_str.ipynb 5
import warnings
from typing import Union, Optional as O

import numpy as np
import jax, jax.numpy as jnp

from lovely_numpy import np_to_str_common, pretty_str, sparse_join, ansi_color, in_debugger
from lovely_numpy import config as lnp_config

from .utils.config import get_config
from .utils.misc import to_numpy, is_cpu, test_array_repr

# %% ../nbs/00_repr_str.ipynb 8
dtnames =   {   "float16": "f16",
                "float32": "", # Default dtype in jax
                "float64": "f64", 
                "bfloat16": "bf16",
                "uint8": "u8",
                "uint16": "u16",
                "uint32": "u32",
                "uint64": "u64",
                "int8": "i8",
                "int16": "i16",
                "int32": "i32",
                "int64": "i64",
            }

def short_dtype(x: jax.Array) -> str:
    return dtnames.get(x.dtype.name, str(x.dtype))

# %% ../nbs/00_repr_str.ipynb 10
def plain_repr(x: jax.Array):
    "Pick the right function to get a plain repr"
    # assert isinstance(x, np.ndarray), f"expected np.ndarray but got {type(x)}" # Could be a sub-class.
    return x._plain_repr() if hasattr(type(x), "_plain_repr") else repr(x)

# def plain_str(x: torch.Tensor):
#     "Pick the right function to get a plain str."
#     # assert isinstance(x, np.ndarray), f"expected np.ndarray but got {type(x)}"
#     return x._plain_str() if hasattr(type(x), "_plain_str") else str(x)

# %% ../nbs/00_repr_str.ipynb 11
def is_nasty(x: jax.Array):
    """Return true of any `x` values are inf or nan"""
    
    if x.size == 0: return False # min/max don't like zero-lenght arrays
    
    x_min = x.min()
    x_max = x.max()
    
    return jnp.isnan(x_min) or jnp.isinf(x_min) or jnp.isinf(x_max)

# %% ../nbs/00_repr_str.ipynb 13
def jax_to_str_common(x: jax.Array,  # Input
                        color=True,                     # ANSI color highlighting
                        ddof=0):                        # For "std" unbiasing

    if x.size == 0:
        return ansi_color("empty", "grey", color)

    zeros = ansi_color("all_zeros", "grey", color) if jnp.equal(x, 0.).all() and x.size > 1 else None
    pinf = ansi_color("+Inf!", "red", color) if jnp.isposinf(x).any() else None
    ninf = ansi_color("-Inf!", "red", color) if jnp.isneginf(x).any() else None
    nan = ansi_color("NaN!", "red", color) if jnp.isnan(x).any() else None

    attention = sparse_join([zeros,pinf,ninf,nan])
    numel = f"n={x.size}" if x.size > 5 and max(x.shape) != x.size else None

    summary = None
    if not zeros and x.ndim > 0:
        # Calculate stats on good values only.
        # This is memory expensive, don't do it on GPU.
        # We divert to numpy if there are any nasties in the data.
        # gx = x[ jnp.isfinite(x) ]

        minmax = f"x∈[{pretty_str(x.min())}, {pretty_str(x.max())}]" if x.size > 2 else None
        meanstd = f"μ={pretty_str(x.mean())} σ={pretty_str(x.std(ddof=ddof))}" if x.size >= 2 else None
        summary = sparse_join([numel, minmax, meanstd])


    return sparse_join([ summary, attention])

# %% ../nbs/00_repr_str.ipynb 14
def to_str(x: jax.Array,  # Input
            plain: bool=False,
            verbose: bool=False,
            depth=0,
            lvl=0,
            color=None) -> str:

    if plain:
        return plain_repr(x)

    conf = get_config()

    tname = type(x).__name__.split(".")[-1]
    shape = str(list(x.shape)) if x.ndim else None
    type_str = sparse_join([tname, shape], sep="")
    

    dev = f"{x.device().platform}:{x.device().id}"
    dtype = short_dtype(x)
    # grad_fn = t.grad_fn.name() if t.grad_fn else None
    # PyTorch does not want you to know, but all `grad_fn``
    # tensors actuall have `requires_grad=True`` too.
    # grad = "grad" if t.requires_grad else None 
    grad = grad_fn = None


    # For complex tensors, just show the shape / size part for now.
    if not jnp.iscomplexobj(x):
        if color is None: color=conf.color
        if in_debugger(): color = False
        # `lovely-numpy` is used to calculate stats when doing so on GPU would require
        # memory allocation (not float tensors, tensors with bad numbers), or if the
        # data is on CPU (because numpy is faster).
        #
        # Temporarily set the numpy config to match our config for consistency.
        with lnp_config(precision=conf.precision,
                        threshold_min=conf.threshold_min,
                        threshold_max=conf.threshold_max,
                        sci_mode=conf.sci_mode):

            if is_cpu(x) or is_nasty(x):
                common = np_to_str_common(np.array(x), color=color)
            else:
                common = jax_to_str_common(x, color=color)

            vals = pretty_str(x) if 0 < x.size <= 10 else None
            res = sparse_join([type_str, dtype, common, grad, grad_fn, dev, vals])
    else:
        res = plain_repr(x)


    if verbose:
        res += "\n" + plain_repr(x)

    if depth and x.ndim > 1:

        deep_width = min((x.shape[0]), conf.deeper_width) # Print at most this many lines
        deep_lines = [ " "*conf.indent*(lvl+1) + to_str(x[i,:], depth=depth-1, lvl=lvl+1)
                            for i in range(deep_width)] 

        # If we were limited by width, print ...
        if deep_width < x.shape[0]: deep_lines.append(" "*conf.indent*(lvl+1) + "...")

        res += "\n" + "\n".join(deep_lines)

    return res

# %% ../nbs/00_repr_str.ipynb 15
def history_warning():
    "Issue a warning (once) ifw e are running in IPYthon with output cache enabled"

    if "get_ipython" in globals() and get_ipython().cache_size > 0:
        warnings.warn("IPYthon has its output cache enabled. See https://xl0.github.io/lovely-tensors/history.html")

# %% ../nbs/00_repr_str.ipynb 18
class StrProxy():
    def __init__(self, x: jax.Array, plain=False, verbose=False, depth=0, lvl=0, color=None):
        self.x = x
        self.plain = plain
        self.verbose = verbose
        self.depth=depth
        self.lvl=lvl
        self.color=color
        history_warning()
    
    def __repr__(self):
        return to_str(self.x, plain=self.plain, verbose=self.verbose,
                      depth=self.depth, lvl=self.lvl, color=self.color)

    # This is used for .deeper attribute and .deeper(depth=...).
    # The second onthe results in a __call__.
    def __call__(self, depth=1):
        return StrProxy(self.x, depth=depth)

# %% ../nbs/00_repr_str.ipynb 19
def lovely(x: jax.Array, # Tensor of interest
            verbose=False,  # Whether to show the full tensor
            plain=False,    # Just print if exactly as before
            depth=0,        # Show stats in depth
            color=None):    # Force color (True/False) or auto.
    return StrProxy(x, verbose=verbose, plain=plain, depth=depth, color=color)
