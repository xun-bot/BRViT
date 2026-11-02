"""Training and evaluation utilities for BRViT."""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from losses import LabelSmoothingCrossEntropy, lmmd_loss


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_batch(inputs, labels, alpha=0.4):
    coefficient = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    coefficient = max(coefficient, 1.0 - coefficient)
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = coefficient * inputs + (1.0 - coefficient) * inputs[permutation]
    return mixed, labels, labels[permutation], coefficient


def lmmd_schedule(epoch, total_epochs, burn_in_epochs=20):
    if epoch <= burn_in_epochs:
        return 0.0
    progress = (epoch - burn_in_epochs) / max(total_epochs - burn_in_epochs, 1)
    return 2.0 / (1.0 + math.exp(-5.0 * progress)) - 1.0


def train_one_epoch(model, source_loader, target_loader, optimizer, device,
                    epoch, total_epochs, lmmd_weight=0.30,
                    burn_in_epochs=20, mixup_alpha=0.4,
                    mixup_probability=0.8, label_smoothing=0.15):
    """Train one epoch with labeled source and unlabeled target samples."""
    model.train()
    criterion = LabelSmoothingCrossEntropy(label_smoothing)
    target_iterator = iter(target_loader)
    running_loss = 0.0

    for source_inputs, source_labels in source_loader:
        try:
            target_inputs, _ = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_inputs, _ = next(target_iterator)

        source_inputs = source_inputs.to(device)
        source_labels = source_labels.to(device)
        target_inputs = target_inputs.to(device)
        optimizer.zero_grad()

        if np.random.random() < mixup_probability:
            mixed, labels_a, labels_b, coefficient = mixup_batch(
                source_inputs, source_labels, mixup_alpha
            )
            mixed_logits = model(mixed)
            classification_loss = (
                coefficient * criterion(mixed_logits, labels_a)
                + (1.0 - coefficient) * criterion(mixed_logits, labels_b)
            )
            _, source_features = model(source_inputs, return_features=True)
        else:
            source_logits, source_features = model(source_inputs, return_features=True)
            classification_loss = criterion(source_logits, source_labels)

        loss = classification_loss
        alignment_scale = lmmd_schedule(epoch, total_epochs, burn_in_epochs)
        if alignment_scale > 0:
            target_logits, target_features = model(target_inputs, return_features=True)
            target_probabilities = F.softmax(target_logits.detach(), dim=1)
            alignment_loss = lmmd_loss(
                source_features, target_features, source_labels, target_probabilities
            )
            loss = loss + lmmd_weight * alignment_scale * alignment_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * source_inputs.size(0)

    return running_loss / len(source_loader.dataset)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)
