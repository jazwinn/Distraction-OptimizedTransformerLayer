"""The optimized transformer implementation, kept out of the harness.

torch_transformer_benchmark.py stays close to the original template and pulls
everything from here:

    config.py    the knobs -- attention backend, kernel choice, CUDA graphs
    cli.py       --attn-backend / --attn-impl / --cuda-graph
    model.py     OptimizedTransformer, the whole forward pass
    layers.py    the submodules it is built from, baseline-name-compatible
    kernels.py   dispatch into csrc/ via kernel_ext, with fallbacks
    graphs.py    CUDA graph capture and replay
    util.py      small shared helpers

That list is reading order, highest level first. The imports run the other
way and form a chain with no cycles in it:

    model -> layers -> kernels -> config
    model -> graphs  -> kernels
    cli   -> config

config.py and util.py import nothing from the package, so either can be
read on its own.

The harness declares

    class UserOptimizedTransformer(OptimizedTransformer, BaselineTransformer)

so the optimized model stays an instance of the baseline class without this
package needing to import the harness.
"""

from . import cli, config, graphs, kernels, layers, util
from .model import OptimizedTransformer

__all__ = ["OptimizedTransformer", "cli", "config", "graphs", "kernels",
           "layers", "util"]
