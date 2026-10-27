"""windturbine: shared code for the CARE to Compare Wind Farm C experiments.

Notebooks orchestrate; this package defines. See docs/ARCHITECTURE.md.
"""

import os

# Keras 3 on PyTorch: TensorFlow >= 2.11 has no GPU support on native Windows.
os.environ.setdefault("KERAS_BACKEND", "torch")
# Deterministic cuBLAS for torch.use_deterministic_algorithms(True).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
