# BRViT SEED core modules

This directory contains a module-level reference implementation for the SEED
portion of BRViT. It is intended to disclose the principal method components
without providing a complete experiment package.

Included modules:

- `seed_data.py`: SEED `de_LDS` loading, subject normalization, and 2D mapping.
- `model.py`: frequency-spatial attention, overlapping patch tokens, region
  tokens, patch-region cross-attention, Transformer encoding, and classifier.
- `losses.py`: label-smoothed classification loss and LMMD alignment loss.
- `train.py`: reusable one-epoch training and evaluation functions.

