"""SEED de_LDS loading and topographic mapping utilities."""

import os
import re

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


CHANNEL_MAP_62 = {
    "FP1": (0, 3), "FPZ": (0, 4), "FP2": (0, 5),
    "AF3": (1, 2), "AF4": (1, 6),
    "F7": (2, 0), "F5": (2, 1), "F3": (2, 2), "F1": (2, 3),
    "FZ": (2, 4), "F2": (2, 5), "F4": (2, 6), "F6": (2, 7), "F8": (2, 8),
    "FT7": (3, 0), "FC5": (3, 1), "FC3": (3, 2), "FC1": (3, 3),
    "FCZ": (3, 4), "FC2": (3, 5), "FC4": (3, 6), "FC6": (3, 7), "FT8": (3, 8),
    "T7": (4, 0), "C5": (4, 1), "C3": (4, 2), "C1": (4, 3),
    "CZ": (4, 4), "C2": (4, 5), "C4": (4, 6), "C6": (4, 7), "T8": (4, 8),
    "TP7": (5, 0), "CP5": (5, 1), "CP3": (5, 2), "CP1": (5, 3),
    "CPZ": (5, 4), "CP2": (5, 5), "CP4": (5, 6), "CP6": (5, 7), "TP8": (5, 8),
    "P7": (6, 0), "P5": (6, 1), "P3": (6, 2), "P1": (6, 3),
    "PZ": (6, 4), "P2": (6, 5), "P4": (6, 6), "P6": (6, 7), "P8": (6, 8),
    "PO7": (7, 1), "PO5": (7, 2), "PO3": (7, 3), "POZ": (7, 4),
    "PO4": (7, 5), "PO6": (7, 6), "PO8": (7, 7),
    "CB1": (8, 2), "O1": (8, 3), "OZ": (8, 4), "O2": (8, 5), "CB2": (8, 6),
}

SEED_CHANNEL_ORDER = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]

TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}


class EEGDataset(Dataset):
    def __init__(self, maps, labels):
        self.maps = torch.as_tensor(maps, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, index):
        return self.maps[index], self.labels[index]


def normalize_per_subject(data):
    """Apply feature-wise Z-score normalization within one subject."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def de_to_topographic_maps(de_features):
    """Convert [N, 62*5] de_LDS features into [N, 5, 9, 9] maps."""
    de_features = np.asarray(de_features, dtype=np.float32)
    if de_features.ndim == 1:
        de_features = de_features[None, :]
    de_features = de_features.reshape(-1, 62, 5)
    maps = np.zeros((len(de_features), 5, 9, 9), dtype=np.float32)
    for channel_index, channel_name in enumerate(SEED_CHANNEL_ORDER):
        row, col = CHANNEL_MAP_62[channel_name]
        maps[:, :, row, col] = de_features[:, channel_index, :]
    return maps


def discover_seed_files(data_dir):
    """Return {(subject_id, zero_based_session_id): mat_path}."""
    grouped = {}
    for filename in os.listdir(data_dir):
        match = re.match(r"^(\d+)_.*\.mat$", filename)
        if match:
            grouped.setdefault(int(match.group(1)), []).append(filename)

    mapping = {}
    for subject_id, filenames in grouped.items():
        for session_id, filename in enumerate(sorted(filenames)):
            mapping[(subject_id, session_id)] = os.path.join(data_dir, filename)
    return mapping


def load_seed_subject(filepath):
    """Load the 15 de_LDS trials of one SEED subject/session file."""
    mat = sio.loadmat(filepath)
    trials, labels = [], []
    for trial_index, raw_label in enumerate(TRIAL_LABELS, start=1):
        key = f"de_LDS{trial_index}"
        if key not in mat:
            raise KeyError(f"Missing required variable: {key}")
        trial = np.asarray(mat[key], dtype=np.float32)
        if trial.ndim != 3:
            raise ValueError(f"Unexpected shape for {key}: {trial.shape}")
        if trial.shape[0] == 62 and trial.shape[2] == 5:
            trial = trial.transpose(1, 0, 2)
        elif not (trial.shape[1] == 62 and trial.shape[2] == 5):
            raise ValueError(f"Unexpected shape for {key}: {trial.shape}")
        trial = trial.reshape(trial.shape[0], 62 * 5)
        trials.append(trial)
        labels.extend([LABEL_MAP[raw_label]] * len(trial))
    return np.concatenate(trials).astype(np.float32), np.asarray(labels, dtype=np.int64)


def load_seed_session(data_dir, session_id=0):
    """Load one SEED session as subject-wise lists; session_id is zero based."""
    mapping = discover_seed_files(data_dir)
    subject_ids = sorted({subject_id for subject_id, sid in mapping if sid == session_id})
    data_by_subject, labels_by_subject = [], []
    for subject_id in subject_ids:
        data, labels = load_seed_subject(mapping[(subject_id, session_id)])
        data_by_subject.append(data)
        labels_by_subject.append(labels)
    return data_by_subject, labels_by_subject
