import torch
import torch.nn as nn


class NSELoss(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                basin_std: torch.Tensor) -> torch.Tensor:
        """
        predictions : (B, lead_times)
        targets     : (B, lead_times)
        basin_std   : (B,)  — per-sample flow std from the basin's full time series
        """
        sq_err = (predictions - targets) ** 2           # (B, lead_times)
        denom = (basin_std.unsqueeze(1) + self.epsilon ) ** 2  # (B, 1) — broadcasts over lead dim
        return (sq_err / denom).mean()


class BatchAwareLossWrapper:
    """
    Uniform callable interface for the training loop: (predictions, targets, batch) → scalar.
    MSE-based losses ignore basin_std; NSELoss extracts it from the batch dict.
    """
    def __init__(self, loss_fn, uses_basin_std: bool = False):
        self._loss_fn = loss_fn
        self._uses_basin_std = uses_basin_std

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor,
                 batch: dict) -> torch.Tensor:
        if self._uses_basin_std:
            basin_std = batch['basin_std'].to(predictions.device)
            return self._loss_fn(predictions, targets, basin_std)
        return self._loss_fn(predictions, targets)
