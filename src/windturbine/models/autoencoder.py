"""Named TensorFlow autoencoder shared by F1 and its matched comparator.

TensorFlow imports stay lazy so data/audit/scoring remain CPU-package friendly.
"""
from __future__ import annotations

import numpy as np


def build_autoencoder(n_features, cfg, *, seed=None):
    import tensorflow as tf

    if n_features <= 0 or not cfg.hidden_layers or any(int(n) != n or n <= 0 for n in cfg.hidden_layers):
        raise ValueError("Model dimensions must be positive integers")
    if (not np.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0
            or not np.isfinite(cfg.noise_std) or cfg.noise_std < 0):
        raise ValueError("Invalid learning rate or noise level")
    tf.keras.utils.set_random_seed(cfg.seed if seed is None else seed)
    inputs = tf.keras.Input(shape=(n_features,), name="sensors")
    x = inputs
    if cfg.noise_std:
        x = tf.keras.layers.GaussianNoise(cfg.noise_std, name="input_noise")(x)
    names = ("encode1", "encode2", "bottleneck", "decode1", "decode2") if len(cfg.hidden_layers) == 5 else None
    for index, units in enumerate(cfg.hidden_layers):
        name = names[index] if names else f"hidden_{index:02d}"
        x = tf.keras.layers.Dense(int(units), activation="relu", name=name)(x)
    outputs = tf.keras.layers.Dense(n_features, activation="linear", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name="normal_behavior_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss="mse")
    return model


def parameter_contract(model, schema):
    """The ordered layer-variable shape/dtype contract required for aggregation."""
    return {"schema": list(schema), "parameters": [
        {"name": variable.path, "shape": list(variable.shape), "dtype": str(variable.dtype)}
        for variable in model.weights
    ]}


def _dense_stack(model):
    """The model's Dense layers in order; DP-SGD's per-example clipping needs exactly that."""
    import tensorflow as tf

    layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    if not layers or any(not isinstance(layer, tf.keras.layers.Dense) or not layer.use_bias for layer in layers):
        raise ValueError("DP-SGD supports only a plain Dense stack with biases (set noise_std=0)")
    if sum(len(layer.trainable_weights) for layer in layers) != len(model.trainable_variables):
        raise ValueError("DP-SGD found trainable variables outside the Dense stack")
    return layers


def _graph_runner(model, proximal, private):
    """One compiled function that runs a stack of minibatches through the model.

    Looping over batches inside the graph removes the per-step Python/eager and
    host-sync cost of train_on_batch, which dominates for a model this small.
    The plain step is the one Keras takes: mean squared error, then one Adam update.
    ``proximal`` lists variable indices pulled toward an anchor (FedProx adds
    mu*(w - anchor) to their gradients). ``private`` switches to DP-SGD: each
    example's gradient is clipped to clip_norm before summing, then Gaussian noise
    with standard deviation noise_multiplier*clip_norm is added to the sum.
    """
    import tensorflow as tf

    name = "_wt_runner" + "".join(f"_p{index}" for index in proximal) + ("_dp" if private else "")
    runner = getattr(model, name, None)
    if runner is not None:
        return runner
    # Create Adam's slot variables eagerly so tracing never creates variables.
    if not model.optimizer.built:
        model.optimizer.build(model.trainable_variables)
    variables = model.trainable_variables
    dense = _dense_stack(model) if private else []
    position = {id(variable): index for index, variable in enumerate(variables)}

    @tf.autograph.experimental.do_not_convert
    def gradients_for(batch, clip_norm, noise_multiplier, seed, index):
        if not private:
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(batch - model(batch, training=True)))
            return loss, tape.gradient(loss, variables)
        # Exact per-example clipping for a dense stack: example i's kernel gradient
        # is outer(a_i, g_i), so its squared norm is |a_i|^2 |g_i|^2 and the clipped
        # sum is one matmul. This avoids materialising per-example gradients.
        with tf.GradientTape() as tape:
            x, layer_inputs, pre_activations = batch, [], []
            for layer in dense:
                layer_inputs.append(x)
                z = tf.matmul(x, layer.kernel) + layer.bias
                tape.watch(z)
                pre_activations.append(z)
                x = layer.activation(z)
            losses = tf.reduce_mean(tf.square(batch - x), axis=1)
            summed = tf.reduce_sum(losses)
        # Rows are independent, so row i of d(sum of losses)/dz is example i's gradient.
        outputs = tape.gradient(summed, pre_activations)
        squared = tf.add_n([tf.reduce_sum(tf.square(a), axis=1) * tf.reduce_sum(tf.square(g), axis=1)
                            + tf.reduce_sum(tf.square(g), axis=1) for a, g in zip(layer_inputs, outputs)])
        factors = tf.minimum(1.0, clip_norm / tf.maximum(tf.sqrt(squared), 1e-12))[:, None]
        rows = tf.cast(tf.shape(batch)[0], tf.float32)
        batch_seed = tf.random.experimental.stateless_fold_in(seed, index)
        gradients = [None] * len(variables)
        for layer, a, g in zip(dense, layer_inputs, outputs):
            for variable, clipped_sum in ((layer.kernel, tf.matmul(a, g * factors, transpose_a=True)),
                                          (layer.bias, tf.reduce_sum(g * factors, axis=0))):
                slot = position[id(variable)]
                noise = tf.random.stateless_normal(
                    tf.shape(variable), seed=tf.random.experimental.stateless_fold_in(batch_seed, slot),
                    stddev=noise_multiplier * clip_norm)
                gradients[slot] = (clipped_sum + noise) / rows
        return tf.reduce_mean(losses), gradients

    # A fixed signature traces once per model even though the last stack has a
    # smaller batch; F1-F4 build a fresh model every client-round.
    scalar = tf.TensorSpec([], tf.float32)
    signature = [tf.TensorSpec([None, None, model.input_shape[-1]], tf.float32),
                 [tf.TensorSpec(variables[index].shape, variables[index].dtype) for index in proximal],
                 scalar, scalar, scalar, tf.TensorSpec([2], tf.int64)]

    @tf.function(input_signature=signature)
    def run(batches, anchors, mu, clip_norm, noise_multiplier, seed):
        total = tf.constant(0.0, dtype=tf.float32)
        for index in tf.range(tf.shape(batches)[0]):
            batch = batches[index]
            loss, gradients = gradients_for(batch, clip_norm, noise_multiplier, seed, index)
            for anchor, position in zip(anchors, proximal, strict=True):
                gradients[position] = gradients[position] + mu * (variables[position] - anchor)
            model.optimizer.apply_gradients(zip(gradients, variables))
            total += loss * tf.cast(tf.shape(batch)[0], tf.float32)
        return total

    setattr(model, name, run)
    return run


def train_epochs(model, X, *, epochs, batch_size, seed, mu=0.0, anchor=None, dp_clip_norm=None,
                 dp_noise_multiplier=0.0):
    """Train explicit shuffled minibatches without tf.data worker pools.

    The caller decides optimizer lifetime. This function never resets Adam.
    Loss includes every common feature, including locally inactive dimensions.
    Each epoch visits every row once in a seeded random order; the last batch
    may be smaller than batch_size.

    FedProx: ``anchor`` maps weight index -> broadcast array and ``mu`` scales the
    proximal pull toward it. DP-SGD: set ``dp_clip_norm``; the reported loss is
    the unclipped data loss without the proximal term.
    """
    import tensorflow as tf

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
    proximal = tuple(sorted(anchor)) if mu else ()
    run = _graph_runner(model, proximal, private)
    anchors = [tf.constant(anchor[index], dtype=model.trainable_variables[index].dtype) for index in proximal]
    options = (tf.constant(float(mu), tf.float32), tf.constant(float(dp_clip_norm or 1.0), tf.float32),
               tf.constant(float(dp_noise_multiplier), tf.float32))
    rng = np.random.default_rng(seed)
    # A separate stream for DP noise keeps the shuffle order identical with and without DP.
    noise_rng = np.random.default_rng([seed, 1])
    n_rows, n_features = X.shape
    n_full = n_rows // batch_size
    losses = []
    for _ in range(epochs):
        # Shuffle once into RAM so batches are contiguous slices, not memmap gathers.
        shuffled = np.ascontiguousarray(X[rng.permutation(n_rows)], dtype=np.float32)
        total = 0.0
        stacks = []
        if n_full:
            stacks.append(shuffled[:n_full * batch_size].reshape(n_full, batch_size, n_features))
        if n_rows % batch_size:
            stacks.append(shuffled[n_full * batch_size:][None])
        for stack in stacks:
            noise_seed = tf.constant(noise_rng.integers(0, 2**31 - 1, size=2), tf.int64)
            total += float(run(tf.constant(stack), anchors, *options, noise_seed))
        if not np.isfinite(total):
            raise ValueError("Nonfinite client training loss")
        losses.append(total / n_rows)
    return float(np.mean(losses))


def mean_squared_error(model, X, batch_size=8192):
    if not len(X):
        raise ValueError("Validation matrix cannot be empty")
    total = 0.0
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        residual = batch.astype(np.float64) - model(batch, training=False).numpy().astype(np.float64)
        total += float(np.square(residual).mean(axis=1).sum())
    value = total / len(X)
    if not np.isfinite(value):
        raise ValueError("Nonfinite validation loss")
    return value
