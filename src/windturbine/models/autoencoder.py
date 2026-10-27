"""Named Keras autoencoder shared by F1-F4 and the matched comparator.

Keras 3 runs on the PyTorch backend (set in windturbine/__init__.py) so training
uses the CUDA GPU on native Windows. Imports stay lazy so data/audit/scoring
remain CPU-package friendly.
"""
from __future__ import annotations

import numpy as np


def build_autoencoder(n_features, cfg, *, seed=None):
    import keras

    if n_features <= 0 or not cfg.hidden_layers or any(int(n) != n or n <= 0 for n in cfg.hidden_layers):
        raise ValueError("Model dimensions must be positive integers")
    if (not np.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0
            or not np.isfinite(cfg.noise_std) or cfg.noise_std < 0):
        raise ValueError("Invalid learning rate or noise level")
    keras.utils.set_random_seed(cfg.seed if seed is None else seed)
    inputs = keras.Input(shape=(n_features,), name="sensors")
    x = inputs
    if cfg.noise_std:
        x = keras.layers.GaussianNoise(cfg.noise_std, name="input_noise")(x)
    names = ("encode1", "encode2", "bottleneck", "decode1", "decode2") if len(cfg.hidden_layers) == 5 else None
    for index, units in enumerate(cfg.hidden_layers):
        name = names[index] if names else f"hidden_{index:02d}"
        x = keras.layers.Dense(int(units), activation="relu", name=name)(x)
    outputs = keras.layers.Dense(n_features, activation="linear", name="output")(x)
    model = keras.Model(inputs, outputs, name="normal_behavior_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss="mse")
    return model


def parameter_contract(model, schema):
    """The ordered layer-variable shape/dtype contract required for aggregation."""
    return {"schema": list(schema), "parameters": [
        {"name": variable.path, "shape": list(variable.shape), "dtype": str(variable.dtype)}
        for variable in model.weights
    ]}


def _dense_stack(model, private=False):
    """Input-noise stddev and the model's Dense layers in order."""
    import keras

    layers = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]
    noise = 0.0
    if layers and isinstance(layers[0], keras.layers.GaussianNoise):
        noise = float(layers.pop(0).stddev)
    if noise and private:
        raise ValueError("DP-SGD supports only a plain Dense stack with biases (set noise_std=0)")
    if not layers or any(not isinstance(layer, keras.layers.Dense) or not layer.use_bias for layer in layers):
        raise ValueError("Training supports only a plain Dense stack with biases")
    if sum(len(layer.trainable_weights) for layer in layers) != len(model.trainable_variables):
        raise ValueError("Found trainable variables outside the Dense stack")
    return noise, layers


def _forward(params, x, n_layers):
    """Dense stack forward pass on raw torch tensors: ReLU hidden layers, linear output."""
    import torch

    inputs, pre = [], []
    for k in range(n_layers):
        inputs.append(x)
        z = torch.addmm(params[2 * k + 1], x, params[2 * k])
        pre.append(z)
        x = torch.relu(z) if k < n_layers - 1 else z
    return x, inputs, pre


_GRAPH_STEPS = 128  # minibatches per CUDA-graph replay
_GRAPHS = {}


def _step(state, batch, noise, hp):
    """One minibatch update on raw torch tensors; shared by eager and CUDA-graph paths.

    state: params, m, v (Adam moments), t (step counter tensor), anchors, total.
    noise: per-parameter standard-normal tensors for DP-SGD, else None.
    """
    import torch

    params, n_layers, rows = state["params"], hp["n_layers"], batch.shape[0]
    if not hp["private"]:
        out, _, _ = _forward(params, batch, n_layers)
        loss = torch.mean(torch.square(batch - out))
        grads = list(torch.autograd.grad(loss, params))
    else:
        out, inputs, pre = _forward(params, batch, n_layers)
        row_losses = torch.mean(torch.square(batch - out), dim=1)
        loss = row_losses.mean()
        outs = torch.autograd.grad(row_losses.sum(), pre)
        clip, scale = hp["clip"], hp["noise_multiplier"] * hp["clip"]
        with torch.no_grad():
            # Exact per-example clipping for a dense stack: example i's kernel gradient
            # is outer(a_i, g_i), so its squared norm is |a_i|^2 |g_i|^2.
            squared = sum(a.square().sum(1) * g.square().sum(1) + g.square().sum(1) for a, g in zip(inputs, outs))
            factors = torch.clamp(clip / torch.clamp(squared.sqrt(), min=1e-12), max=1.0)[:, None]
            grads = []
            for a, g in zip(inputs, outs):
                gf = g * factors
                for clipped in (a.detach().T @ gf, gf.sum(0)):
                    grads.append((clipped + noise[len(grads)] * scale) / rows)
    with torch.no_grad():
        for index, anchor in zip(hp["proximal"], state["anchors"]):
            grads[index] = grads[index] + hp["mu"] * (params[index] - anchor)
        if hp["adam"]:
            b1, b2 = hp["b1"], hp["b2"]
            state["t"].add_(1)
            torch._foreach_lerp_(state["m"], grads, 1 - b1)
            torch._foreach_mul_(state["v"], b2)
            torch._foreach_addcmul_(state["v"], grads, grads, 1 - b2)
            # keras.optimizers.Adam: alpha = lr * sqrt(1 - b2^t) / (1 - b1^t)
            alpha = hp["lr"] * torch.sqrt(1 - torch.pow(b2, state["t"])) / (1 - torch.pow(b1, state["t"]))
            denom = torch._foreach_sqrt(state["v"])
            torch._foreach_add_(denom, hp["eps"])
            update = torch._foreach_div(state["m"], denom)
            torch._foreach_mul_(update, alpha)
            torch._foreach_sub_(params, update)
        else:
            torch._foreach_add_(params, grads, alpha=-hp["lr"])
        state["total"] += loss.detach().double() * rows


def _new_state(shapes, n_anchors, device):
    import torch

    zeros = lambda: [torch.zeros(s, device=device) for s in shapes]
    params = [torch.zeros(s, device=device, requires_grad=True) for s in shapes]
    return {"params": params, "m": zeros(), "v": zeros(), "t": torch.zeros((), device=device),
            "anchors": [torch.zeros(shapes[i], device=device) for i in range(n_anchors)],
            "total": torch.zeros((), dtype=torch.float64, device=device)}


def _graph(key, shapes, batch_size, n_features, hp, device):
    """A captured CUDA graph running _GRAPH_STEPS minibatch updates on static buffers."""
    import torch

    entry = _GRAPHS.get(key)
    if entry is not None:
        return entry
    state = _new_state(shapes, len(hp["proximal"]), device)
    state["anchors"] = [torch.zeros(shapes[i], device=device) for i in hp["proximal"]]
    x = torch.zeros((_GRAPH_STEPS, batch_size, n_features), device=device)
    noise = [torch.zeros((_GRAPH_STEPS, *s), device=device) for s in shapes] if hp["private"] else None

    def body():
        for i in range(_GRAPH_STEPS):
            _step(state, x[i], [n[i] for n in noise] if noise else None, hp)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # warm-up on a side stream, as CUDA-graph capture requires
            body()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    entry = _GRAPHS[key] = (graph, state, x, noise)
    return entry


def train_epochs(model, X, *, epochs, batch_size, seed, mu=0.0, anchor=None, dp_clip_norm=None,
                 dp_noise_multiplier=0.0):
    """Train explicit shuffled minibatches on the Keras model's torch parameters (GPU).

    Keras 3 runs on the PyTorch backend so the RTX GPU works on native Windows
    (TensorFlow >= 2.11 has no Windows GPU support). The step is the one Keras
    takes: mean squared error, then one update of the model's own optimizer
    (Adam, or plain SGD in the preflight gates) with keras.optimizers.Adam's
    formula. Optimizer state lives on model.optimizer, so its lifetime is the
    caller's choice; this function never resets it. Full chunks of
    _GRAPH_STEPS minibatches replay one captured CUDA graph (the model is tiny,
    so per-step Python/launch cost would dominate); the rest run eagerly with
    the same _step code.

    Each epoch visits every row once in a seeded random order; the last batch
    may be smaller than batch_size. FedProx: ``anchor`` maps weight index ->
    broadcast array and ``mu`` scales the proximal pull. DP-SGD: set
    ``dp_clip_norm``; each example's gradient is clipped exactly and Gaussian
    noise noise_multiplier*clip_norm is added to the clipped sum. The reported
    loss is the unclipped data loss without the proximal term.
    """
    import keras
    import torch

    if (isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0
            or isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
        raise ValueError("epochs and batch_size must be positive integers")
    if not len(X) or X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Training matrix must be nonempty, finite, and two-dimensional")
    if not np.isfinite(mu) or mu < 0 or (mu and not anchor):
        raise ValueError("mu must be finite and nonnegative, with an anchor when positive")
    private = dp_clip_norm is not None
    if private and not (np.isfinite(dp_clip_norm) and dp_clip_norm > 0
                        and np.isfinite(dp_noise_multiplier) and dp_noise_multiplier >= 0):
        raise ValueError("DP-SGD needs a positive clip norm and a nonnegative noise multiplier")
    if anchor and len(model.trainable_variables) != len(model.weights):
        raise ValueError("Proximal anchors assume every model weight is trainable")
    noise_std, dense = _dense_stack(model, private)
    params = [variable.value for variable in model.trainable_variables]
    if [id(p) for p in params] != [id(w.value) for layer in dense for w in (layer.kernel, layer.bias)]:
        raise ValueError("Unexpected trainable-variable order")
    device = params[0].device
    proximal = tuple(sorted(anchor)) if mu else ()

    optimizer = model.optimizer
    adam = isinstance(optimizer, keras.optimizers.Adam)
    if adam and (optimizer.amsgrad or optimizer.weight_decay):
        raise ValueError("Only plain Adam is supported")
    if not adam and not (isinstance(optimizer, keras.optimizers.SGD) and not optimizer.momentum
                         and not optimizer.nesterov):
        raise ValueError("Only Adam or momentum-free SGD is supported")
    hp = {"n_layers": len(dense), "private": private, "proximal": proximal, "mu": float(mu), "adam": adam,
          "lr": float(keras.ops.convert_to_numpy(optimizer.learning_rate)),
          "clip": float(dp_clip_norm or 1.0), "noise_multiplier": float(dp_noise_multiplier)}
    if adam:
        hp.update(b1=float(optimizer.beta_1), b2=float(optimizer.beta_2), eps=float(optimizer.epsilon))
    shapes = [tuple(p.shape) for p in params]
    n_rows, n_features = X.shape

    # Eager working state, loaded from the model and its optimizer.
    state = _new_state(shapes, 0, device)
    state["params"] = params
    saved = getattr(optimizer, "_wt_adam", None)
    if saved is not None:
        state["m"], state["v"] = saved
    state["t"].fill_(float(keras.ops.convert_to_numpy(optimizer.iterations)))
    state["anchors"] = [torch.as_tensor(np.asarray(anchor[i]), dtype=params[i].dtype, device=device) for i in proximal]

    n_full = n_rows // batch_size
    use_graph = device.type == "cuda" and not noise_std and n_full >= _GRAPH_STEPS
    if use_graph:
        key = (tuple(shapes), batch_size, n_features, tuple(sorted(hp.items())), str(device))
        graph, gstate, gx, gnoise = _graph(key, shapes, batch_size, n_features, hp, device)
        for dst, src in zip(gstate["anchors"], state["anchors"]):
            dst.copy_(src)

    def run_graph_chunk(chunk):
        with torch.no_grad():
            torch._foreach_copy_(gstate["params"], state["params"])
            torch._foreach_copy_(gstate["m"], state["m"])
            torch._foreach_copy_(gstate["v"], state["v"])
            gstate["t"].copy_(state["t"])
            gstate["total"].zero_()
            gx.copy_(chunk)
            if gnoise:
                for buffer in gnoise:
                    buffer.normal_(generator=generator)
        graph.replay()
        with torch.no_grad():
            torch._foreach_copy_(state["params"], gstate["params"])
            torch._foreach_copy_(state["m"], gstate["m"])
            torch._foreach_copy_(state["v"], gstate["v"])
            state["t"].copy_(gstate["t"])
            state["total"] += gstate["total"]

    rng = np.random.default_rng(seed)
    # A separate stream for DP noise keeps the shuffle order identical with and without DP.
    noise_rng = np.random.default_rng([seed, 1])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_rng.integers(0, 2**62)))
    losses = []
    for _ in range(epochs):
        shuffled = torch.from_numpy(np.ascontiguousarray(X[rng.permutation(n_rows)], dtype=np.float32)).to(device)
        if noise_std:
            # GaussianNoise's role: training-time input jitter, one seeded draw per epoch.
            source = shuffled + noise_std * torch.randn(shuffled.shape, generator=generator, device=device)
        state["total"].zero_()
        first, done = 0, 0
        if use_graph:
            stack = shuffled[:n_full * batch_size].view(n_full, batch_size, n_features)
            while done + _GRAPH_STEPS <= n_full:
                run_graph_chunk(stack[done:done + _GRAPH_STEPS])
                done += _GRAPH_STEPS
            first = done * batch_size
        for start in range(first, n_rows, batch_size):
            batch = shuffled[start:start + batch_size]
            noise = ([torch.randn(s, generator=generator, device=device) for s in shapes] if private else None)
            if noise_std:
                # Input jitter only feeds the model; the reconstruction target stays clean.
                _step_noisy(state, batch, source[start:start + batch_size], hp)
            else:
                _step(state, batch, noise, hp)
        total = float(state["total"])
        if not np.isfinite(total):
            raise ValueError("Nonfinite client training loss")
        losses.append(total / n_rows)
    if adam:
        optimizer._wt_adam = (state["m"], state["v"])
    optimizer.iterations.assign(int(state["t"].item()))
    return float(np.mean(losses))


def _step_noisy(state, batch, source, hp):
    """Plain step with GaussianNoise-style jittered inputs (noise_std > 0, never with DP)."""
    import torch

    params = state["params"]
    out, _, _ = _forward(params, source, hp["n_layers"])
    loss = torch.mean(torch.square(batch - out))
    grads = list(torch.autograd.grad(loss, params))
    with torch.no_grad():
        for index, anchor in zip(hp["proximal"], state["anchors"]):
            grads[index] = grads[index] + hp["mu"] * (params[index] - anchor)
        if hp["adam"]:
            b1, b2 = hp["b1"], hp["b2"]
            state["t"].add_(1)
            torch._foreach_lerp_(state["m"], grads, 1 - b1)
            torch._foreach_mul_(state["v"], b2)
            torch._foreach_addcmul_(state["v"], grads, grads, 1 - b2)
            alpha = hp["lr"] * torch.sqrt(1 - torch.pow(b2, state["t"])) / (1 - torch.pow(b1, state["t"]))
            denom = torch._foreach_sqrt(state["v"])
            torch._foreach_add_(denom, hp["eps"])
            update = torch._foreach_div(state["m"], denom)
            torch._foreach_mul_(update, alpha)
            torch._foreach_sub_(params, update)
        else:
            torch._foreach_add_(params, grads, alpha=-hp["lr"])
        state["total"] += loss.detach().double() * batch.shape[0]


def mean_squared_error(model, X, batch_size=8192):
    import torch

    if not len(X):
        raise ValueError("Validation matrix cannot be empty")
    _, dense = _dense_stack(model)
    params = [w.value for layer in dense for w in (layer.kernel, layer.bias)]
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)).to(params[0].device)
            out, _, _ = _forward(params, batch, len(dense))
            total += float(torch.square(batch.double() - out.double()).mean(dim=1).sum())
    value = total / len(X)
    if not np.isfinite(value):
        raise ValueError("Nonfinite validation loss")
    return value
