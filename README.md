# BRViT 

This directory contains a module-level reference implementation of BRViT.

Included modules:

- `seed_data.py`: SEED `de_LDS` loading, subject normalization, and 2D mapping.
- `model.py`: frequency-spatial attention, overlapping patch tokens, region
  tokens, patch-region cross-attention, Transformer encoding, and classifier.
- `losses.py`: label-smoothed classification loss and LMMD alignment loss.
- `train.py`: reusable one-epoch training and evaluation functions.

