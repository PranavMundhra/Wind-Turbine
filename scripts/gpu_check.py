"""GPU sanity + speed check for the Keras-on-PyTorch training path."""
import sys, time
sys.path.insert(0, r"C:\Users\pooja\Desktop\wind turbine\Wind-Turbine\src")
import numpy as np, windturbine, keras, torch
from windturbine.federated import run_preflight, run_strategy_preflight
from windturbine.config import load_config
from windturbine.models import build_autoencoder, train_epochs, mean_squared_error
from windturbine.models import autoencoder as ae
print("backend", keras.backend.backend(), torch.__version__, torch.cuda.get_device_name(0), flush=True)
print("fedavg preflight", run_preflight()["passed"], flush=True)
print("strategy preflight", run_strategy_preflight()["passed"], flush=True)
cfg = load_config(r"C:\Users\pooja\Desktop\wind turbine\Wind-Turbine\configs\base.yaml")
X = np.random.default_rng(0).normal(size=(64 * 128 * 10 + 1000, 140)).astype(np.float32)
# Graph path must equal the eager path exactly.
results = []
for graph_steps in (10**9, 128):
    ae._GRAPH_STEPS = graph_steps
    m = build_autoencoder(140, cfg, seed=1)
    train_epochs(m, X[:64 * 300], epochs=1, batch_size=64, seed=3)
    results.append(m.get_weights())
ae._GRAPH_STEPS = 128
print("graph==eager max diff", max(float(np.abs(a - b).max()) for a, b in zip(*results)), flush=True)
for dp in (None, 1.0):
    m = build_autoencoder(140, cfg)
    kw = dict(dp_clip_norm=dp, dp_noise_multiplier=1.0) if dp else {}
    train_epochs(m, X[:64 * 300], epochs=1, batch_size=64, seed=0, **kw)
    torch.cuda.synchronize(); t = time.perf_counter()
    train_epochs(m, X, epochs=1, batch_size=64, seed=1, **kw)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    print(f"dp={dp}: {dt / (len(X) / 64) * 1000:.3f} ms/step  mse={mean_squared_error(m, X[:9000]):.4f}", flush=True)
