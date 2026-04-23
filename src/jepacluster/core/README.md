# jepacluster/src/jepacluster/core

This directory contains the fundamental ML components of `jepacluster`.

- **architecture.py**: Definitions for the context encoder, target encoder, and the predictor.
- **trainer.py**: The training loop. Implements the L2 distance loss between predicted and actual latent states and manages the EMA (Exponential Moving Average) updates for the target encoder.
- **dataset.py**: Log-to-Tensor pipeline. Implements the sliding window logic that transforms flat log streams into paired context/target sequences.