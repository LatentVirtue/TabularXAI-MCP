"""Residual tabular MLP in PyTorch.

Architecture: Linear stem -> N x (Linear -> LayerNorm -> SiLU -> Dropout + residual
skip) -> linear head. Training: AdamW, cosine LR decay, early stopping on the
fold-validation loss with best-weights restore, CPU/CUDA auto-detection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import NNParams, TaskType


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularNet(nn.Module):
    """Linear -> [Linear->LayerNorm->SiLU->Dropout + skip] * N -> head."""

    def __init__(self, in_dim: int, hidden_dim: int, n_blocks: int, out_dim: int, dropout: float):
        super().__init__()
        self.stem = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class TorchTabularModel:
    """Estimator-like wrapper: fit / predict / predict_proba, picklable."""

    def __init__(
        self,
        task: TaskType,
        params: NNParams,
        seed: int = 42,
        device_preference: str = "auto",
    ):
        self.task = TaskType(task)
        self.params = params
        self.seed = seed
        self.device_preference = device_preference
        self.model: TabularNet | None = None
        self.classes_: list[str] = []
        self.n_classes_: int = 0
        self.input_dim_: int = 0
        self.history: dict[str, list[float]] = {}
        self.n_parameters_: int = 0

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TorchTabularModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = resolve_device(self.device_preference)
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        if self.task is TaskType.REGRESSION:
            out_dim, y_t = 1, torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            loss_fn = nn.MSELoss()
        elif self.task is TaskType.BINARY_CLASSIFICATION:
            out_dim, y_t = 1, torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            out_dim, y_t = self.n_classes_, torch.tensor(y, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()

        self.model = TabularNet(
            self.input_dim_,
            self.params.hidden_dim,
            self.params.n_blocks,
            out_dim,
            self.params.dropout,
        ).to(self.device)
        self.n_parameters_ = sum(p.numel() for p in self.model.parameters())

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.params.epochs)

        Xt = torch.as_tensor(X, device=self.device)
        use_val = X_val is not None and y_val is not None
        if use_val:
            Xv = torch.as_tensor(np.ascontiguousarray(X_val, dtype=np.float32), device=self.device)
            if self.task is TaskType.REGRESSION:
                yv_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            elif self.task is TaskType.BINARY_CLASSIFICATION:
                yv_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            else:
                yv_t = torch.tensor(y_val, dtype=torch.long)
            yv_t = yv_t.to(self.device)

        n = X.shape[0]
        bs = min(self.params.batch_size, max(8, n))
        best_loss = float("inf")
        best_state: dict | None = None
        bad_epochs = 0
        self.history = {"train_loss": [], "val_loss": []}

        for _epoch in range(self.params.epochs):
            self.model.train()
            perm = torch.randperm(n)
            running = 0.0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(Xt[idx]), y_t[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                running += float(loss.detach()) * len(idx)
            sched.step()
            train_loss = running / n
            self.history["train_loss"].append(train_loss)

            if use_val:
                val_loss = self._loss_on(loss_fn, Xv, yv_t)
                self.history["val_loss"].append(val_loss)
                if val_loss < best_loss - 1e-6:
                    best_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.params.patience:
                        break
            else:
                self.history["val_loss"].append(train_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _loss_on(self, loss_fn, Xb: torch.Tensor, yb: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            return float(loss_fn(self.model(Xb), yb))

    # ------------------------------------------------------------- predict
    def _logits(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        out = []
        with torch.no_grad():
            X = np.ascontiguousarray(X, dtype=np.float32)
            for start in range(0, len(X), 8192):
                chunk = torch.as_tensor(X[start : start + 8192], device=self.device)
                out.append(self.model(chunk).float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Binary/multiclass: (n, n_classes). Raises for regression."""
        logits = self._logits(X)
        if self.task is TaskType.BINARY_CLASSIFICATION:
            p = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            return np.stack([1.0 - p, p], axis=1)
        if self.task is TaskType.MULTICLASS_CLASSIFICATION:
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        raise AttributeError("predict_proba is not available for regression.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(X)
        if self.task is TaskType.REGRESSION:
            return logits[:, 0]
        return logits.argmax(axis=1)
