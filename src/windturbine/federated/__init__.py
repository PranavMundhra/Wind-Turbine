"""Federated learning (roadmap phases 1-5). Not implemented.

Design decisions already made (see docs/adr/0001):
  * client = ``asset_id`` (22 clients), never an event file;
  * scaler and threshold stay local; encoder shared, decoder fine-tuned locally
    (phase 3); FedProx mu sweep (phase 4); secure aggregation then DP (phase 5);
  * ``notebooks/20_federated_clients.ipynb`` builds the per-client dataset today.
"""
