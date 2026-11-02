"""Classification and distribution-alignment losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.15):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        log_probabilities = F.log_softmax(logits, dim=-1)
        hard = -log_probabilities.gather(1, targets[:, None]).squeeze(1)
        smooth = -log_probabilities.mean(dim=-1)
        return ((1.0 - self.smoothing) * hard + self.smoothing * smooth).mean()


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    features = torch.cat([source, target], dim=0)
    squared_distance = torch.cdist(features, features).pow(2)
    count = features.size(0)
    bandwidth = squared_distance.detach().sum() / (count * count - count + 1e-8)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    return sum(
        torch.exp(-squared_distance / (bandwidth * kernel_mul ** i + 1e-8))
        for i in range(kernel_num)
    )


def lmmd_loss(source, target, source_labels, target_probabilities,
              n_classes=3, kernel_mul=2.0, kernel_num=5):
    """Class-conditional LMMD using hard source and soft target memberships."""
    source_one_hot = F.one_hot(source_labels, n_classes).float()
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num)
    n_source = source.size(0)
    kernel_ss = kernels[:n_source, :n_source]
    kernel_tt = kernels[n_source:, n_source:]
    kernel_st = kernels[:n_source, n_source:]

    loss = source.new_zeros(())
    valid_classes = 0
    for class_index in range(n_classes):
        source_weight = source_one_hot[:, class_index]
        target_weight = target_probabilities[:, class_index]
        if source_weight.sum() < 1e-4 or target_weight.sum() < 1e-4:
            continue
        source_weight = source_weight / source_weight.sum()
        target_weight = target_weight / target_weight.sum()
        loss = loss + source_weight @ kernel_ss @ source_weight
        loss = loss + target_weight @ kernel_tt @ target_weight
        loss = loss - 2.0 * (source_weight @ kernel_st @ target_weight)
        valid_classes += 1
    return loss / max(valid_classes, 1)
