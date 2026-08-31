"""시퀀스 오토인코더 이상탐지.

AutoEncoderModel 과 전제가 같다 — 정상 개체의 표본만 학습하고 재구성 오차를
이상 점수로 쓴다. 다른 점은 입력이 한 행이 아니라 lookback 창이라는 것이다.

    dense AE   (84피처) -> 잠재 -> (84피처)        한 시점의 이상
    LSTM AE    (30일 x N피처) -> 잠재 -> 복원      시간 형태의 이상

한 시점 값은 정상 범위인데 변화 궤적이 이상한 경우를 잡을 수 있는지가 이
모델을 두는 이유다.

입력은 이미 표준화된 상태로 들어온다 (runner 가 시퀀스 계열에 fold 의 train
구간에서 적합시킨 스케일러를 적용한다). 그래서 이 클래스는 스케일러를 따로
들지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseModel
from .sequence import WindowDataset


class LSTMAutoEncoderNet(nn.Module):
    """인코더가 창을 잠재 벡터로 접고, 디코더가 그것을 창 길이로 편다."""

    def __init__(self, n_features: int, lookback: int, params: dict):
        super().__init__()
        hidden = int(params.get("hidden_size", 64))
        latent = int(params.get("latent_dim", 16))
        layers = int(params.get("num_layers", 1))
        self.lookback = lookback
        self.encoder = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        seed = self.from_latent(latent).unsqueeze(1).repeat(1, self.lookback, 1)
        decoded, _ = self.decoder(seed)
        return self.output(decoded)


class SeqAutoEncoderModel(BaseModel):
    family = "sequence"
    name = "lstm_ae"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self.net: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _loader(self, fold, index=None, *, shuffle: bool) -> DataLoader:
        end_index = fold.end_index if index is None else fold.end_index[index]
        y = fold.y if index is None else fold.y[index]
        valid = getattr(fold, "valid_len", None)
        if valid is not None and index is not None:
            valid = valid[index]
        dataset = WindowDataset(fold.matrix, end_index, fold.lookback, y, valid)
        return DataLoader(
            dataset,
            batch_size=int(self.training.get("batch_size", 512)),
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
        )

    def fit(self, train, val) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        healthy = np.flatnonzero(train.healthy_mask())
        if healthy.size == 0:
            raise ValueError("train 구간에 정상 개체의 표본이 없습니다.")
        print(
            f"      [정상만 학습] 표본 {len(train):,} 중 정상 {healthy.size:,} "
            f"({healthy.size / len(train):.2%})"
        )

        self.net = LSTMAutoEncoderNet(
            train.n_features, train.lookback, self.params
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(self.training.get("learning_rate", 1e-3)),
            weight_decay=float(self.training.get("weight_decay", 1e-5)),
        )
        epochs = int(self.training.get("epochs", 20))
        patience = int(self.training.get("early_stopping_patience", 4))
        val_healthy = np.flatnonzero(val.healthy_mask())
        train_loader = self._loader(train, healthy, shuffle=True)

        best, best_epoch, best_state, waited = np.inf, -1, None, 0
        history = []
        for epoch in range(epochs):
            self.net.train()
            total, seen = 0.0, 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(self.net(batch_x), batch_x)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * batch_x.shape[0]
                seen += int(batch_x.shape[0])

            val_loss = float(np.mean(self._errors(val, val_healthy)))
            history.append(
                {"epoch": epoch, "train_loss": total / max(seen, 1), "val_loss": val_loss}
            )
            print(
                f"      epoch {epoch:02d} loss={total / max(seen, 1):.5f} "
                f"val_loss={val_loss:.5f}"
            )
            if np.isfinite(val_loss) and val_loss < best:
                best, best_epoch, waited = val_loss, epoch, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()
                }
            else:
                waited += 1
                if waited >= patience:
                    print(f"      early stopping (patience {patience})")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.fit_info = {
            "n_features": int(train.n_features),
            "lookback": int(train.lookback),
            "best_epoch": best_epoch,
            "best_val_loss": float(best),
            "epochs_run": len(history),
            "device": str(self.device),
            "n_train": len(train),
            "n_train_healthy": int(healthy.size),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
            "warm_started": False,
            "history": history,
        }
        return self.fit_info

    @torch.no_grad()
    def _errors(self, fold, index=None) -> np.ndarray:
        self.net.eval()
        loader = self._loader(fold, index, shuffle=False)
        parts = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            error = (self.net(batch_x) - batch_x).pow(2).mean(dim=(1, 2))
            parts.append(error.float().cpu().numpy())
        return np.concatenate(parts) if parts else np.zeros(0)

    def predict_proba(self, data) -> np.ndarray:
        """재구성 오차. 확률이 아니므로 Brier score 는 이 모델에서 해석하지 않는다."""
        return self._errors(data).astype(np.float64)

    def complexity(self) -> int:
        if self.net is None:
            return 0
        return int(sum(p.numel() for p in self.net.parameters()))

    def _save_weights(self, directory: Path) -> None:
        torch.save(self.net.state_dict(), directory / "model.pt")

    def _load_weights(self, directory: Path) -> None:
        self.net = LSTMAutoEncoderNet(
            int(self.fit_info["n_features"]), int(self.fit_info["lookback"]), self.params
        ).to(self.device)
        self.net.load_state_dict(
            torch.load(directory / "model.pt", map_location=self.device, weights_only=True)
        )
