"""Autoencoder (243 -> 133 -> 83 -> 20 -> 83 -> 133 -> 243) and training loop.

TODO (roadmap phase R): move ``build_autoencoder`` and ``fit_autoencoder`` from the
arm notebooks. Keep TensorFlow imports inside this subpackage so ``scoring`` and
``audit`` stay importable (and testable in CI) without it.
"""
