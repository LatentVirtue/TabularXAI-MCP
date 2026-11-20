"""Public API surface of tabular_xai.

Kept deliberately light: importing this package must not drag in torch or
lightgbm. Use the explicit submodule imports (or the CLI/MCP entrypoints)
for anything heavy.
"""

__version__ = "0.1.0"

MODEL_LIGHTGBM = "lightgbm"
MODEL_PYTORCH_MLP = "pytorch_mlp"

__all__ = ["__version__", "MODEL_LIGHTGBM", "MODEL_PYTORCH_MLP"]
