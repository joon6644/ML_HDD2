"""테이블 입력 MLP.

트리 계열과 같은 평탄화 피처를 받되 학습기가 신경망이다. 시퀀스 모델과
트리 모델 사이의 대조군 역할을 한다 — 시간 구조를 안 쓰면서 비선형만 쓰는 경우.

트리와 달리 결측과 스케일을 직접 처리해야 한다.
  - 스케일러를 이 모델이 들고 있는다. train 구간에서만 적합시킨다.
  - 결측은 표준화 후 0(=평균)으로 채운다. 트리처럼 분기로 다룰 수 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..features.fold import Scaler
from .base import BaseModel
from .sequence import build_criterion


class MLPNet(nn.Module):
    def __init__(self, n_features: int, params: dict):
        super().__init__()
        hidden = [int(h) for h in params.get("hidden_dims", [256, 128, 64])]
        dropout = float(params.get("dropout", 0.2))
        batch_norm = bool(params.get("batch_norm", True))
        layers: list[nn.Module] = []
        dims = [n_features, *hidden]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


class MLPModel(BaseModel):
    """테이블 입력 신경망 공통 학습기.

    net_class 만 바꾸면 다른 구조를 붙일 수 있다. 학습 루프·스케일러·조기
    종료가 같아야 구조 차이만 비교된다.
    """

    family = "tabular"
    name = "mlp"
    net_class = MLPNet

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self.net: nn.Module | None = None
        self.scaler: Scaler | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        # Scaler.transform 이 fill_nan=True 에서 이미 nan_to_num 을 한다.
        # 여기서 한 번 더 부르면 행렬 사본이 하나 더 생긴다 (2,500만 행에서 8GB).
        return self.scaler.transform(X, fill_nan=True).astype(np.float32, copy=False)

    @staticmethod
    def _batches(n: int, size: int, *, shuffle: bool, generator=None):
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, size):
            yield order[start : start + size]

    def fit(self, train, val) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        scaling = self.training.get("scaling", {})
        self.scaler = Scaler(
            scaling.get("method", "standard"),
            scaling.get("clip_quantile", [0.001, 0.999]),
        ).fit(train.X)

        train_x = torch.from_numpy(self._prepare(train.X))
        train_y = torch.from_numpy(train.y.astype(np.float32))
        val_x = torch.from_numpy(self._prepare(val.X))

        self.net = self.net_class(train_x.shape[1], self.params).to(self.device)
        criterion, loss_info = build_criterion(self.training, train.y, self.device)
        criterion = criterion.to(self.device) if hasattr(criterion, "to") else criterion
        print(f"      loss={loss_info}")
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(self.training.get("learning_rate", 1e-3)),
            weight_decay=float(self.training.get("weight_decay", 1e-5)),
        )
        batch_size = int(self.training.get("batch_size", 4096))
        epochs = int(self.training.get("epochs", 30))
        patience = int(self.training.get("early_stopping_patience", 5))
        monitor = str(self.training.get("early_stopping_metric", "val_pr_auc"))
        if monitor == "val_pr_auc":
            monitor_fn = self._pr_auc
        elif monitor == "val_pauc":
            monitor_fn = self._pauc
        else:
            raise ValueError(
                f"알 수 없는 early_stopping_metric: {monitor!r} (val_pr_auc | val_pauc)"
            )
        generator = torch.Generator().manual_seed(self.seed)

        best, best_epoch, best_state, waited = -np.inf, -1, None, 0
        history = []
        for epoch in range(epochs):
            self.net.train()
            total, seen = 0.0, 0
            for index in self._batches(
                len(train_x), batch_size, shuffle=True, generator=generator
            ):
                if index.shape[0] < 2:  # BatchNorm 은 크기 1 배치를 못 쓴다
                    continue
                xb = train_x[index].to(self.device, non_blocking=True)
                yb = train_y[index].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * index.shape[0]
                seen += int(index.shape[0])

            score = self._score(val_x, batch_size)
            val_pr_auc = monitor_fn(val.y, score)
            history.append(
                {"epoch": epoch, "train_loss": total / max(seen, 1), "val_pr_auc": val_pr_auc}
            )
            print(
                f"      epoch {epoch:02d} loss={total / max(seen, 1):.5f} "
                f"val_pr_auc={val_pr_auc:.5f}"
            )
            if np.isfinite(val_pr_auc) and val_pr_auc > best:
                best, best_epoch, waited = val_pr_auc, epoch, 0
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
            "n_features": int(train_x.shape[1]),
            "feature_names": list(train.columns),
            "best_epoch": best_epoch,
            "best_val_pr_auc": float(best),
            "epochs_run": len(history),
            "device": str(self.device),
            "n_train": len(train),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
            "warm_started": False,
            **loss_info,
            "history": history,
        }
        return self.fit_info

    @torch.no_grad()
    def _score(self, x: torch.Tensor, batch_size: int) -> np.ndarray:
        self.net.eval()
        out = np.empty(x.shape[0], dtype=np.float64)
        for index in self._batches(x.shape[0], batch_size, shuffle=False):
            xb = x[index].to(self.device, non_blocking=True)
            out[index.numpy()] = torch.sigmoid(self.net(xb)).float().cpu().numpy()
        return out

    def predict_proba(self, data) -> np.ndarray:
        return self._score(
            torch.from_numpy(self._prepare(data.X)),
            int(self.training.get("batch_size", 4096)),
        )

    @staticmethod
    def _pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score

        if y_true.max() == y_true.min():
            return float("nan")
        return float(average_precision_score(y_true, score))

    @staticmethod
    def _pauc(y_true: np.ndarray, score: np.ndarray, max_fpr: float = 0.05) -> float:
        """FAR <= max_fpr 구간의 부분 AUC. 시퀀스 모델과 같은 감시값."""
        from sklearn.metrics import roc_auc_score

        if y_true.max() == y_true.min():
            return float("nan")
        return float(roc_auc_score(y_true, score, max_fpr=max_fpr))

    def complexity(self) -> int:
        return 0 if self.net is None else int(sum(p.numel() for p in self.net.parameters()))

    def _save_weights(self, directory: Path) -> None:
        torch.save(self.net.state_dict(), directory / "model.pt")
        with (directory / "scaler.json").open("w", encoding="utf-8") as fh:
            json.dump(self.scaler.state_dict(), fh, ensure_ascii=False)

    def _load_weights(self, directory: Path) -> None:
        with (directory / "scaler.json").open("r", encoding="utf-8") as fh:
            self.scaler = Scaler.from_state_dict(json.load(fh))
        self.net = self.net_class(int(self.fit_info["n_features"]), self.params).to(self.device)
        self.net.load_state_dict(
            torch.load(directory / "model.pt", map_location=self.device, weights_only=True)
        )


class TabResNetNet(nn.Module):
    """테이블용 ResNet (Gorishniy et al. 2021).

    잔차 블록을 쌓아 MLP 보다 깊게 간다. 표준 벤치마크에서 MLP 보다 낫다고
    보고된 구조라 "신경망을 더 잘 만들면 트리를 이기는가"의 대조군이다.
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        width = int(params.get("width", 256))
        depth = int(params.get("depth", 4))
        dropout = float(params.get("dropout", 0.2))
        self.stem = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(width),
                    nn.Linear(width, width * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width * 2, width),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.BatchNorm1d(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        out = self.stem(x)
        for block in self.blocks:
            out = out + block(out)
        return self.head(torch.relu(self.norm(out))).squeeze(-1)


class FTTransformerNet(nn.Module):
    """FT-Transformer (Gorishniy et al. 2021).

    피처 하나하나를 토큰으로 만들어 Transformer 에 넣는다. 시퀀스 모델의
    Transformer 가 "시점"을 토큰으로 보는 것과 달리 여기서는 "피처"가 토큰이다.
    시간 구조 없이 피처 간 상호작용만 주의집중으로 다룬다.
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        d_token = int(params.get("d_token", 64))
        depth = int(params.get("depth", 3))
        heads = int(params.get("nhead", 8))
        dropout = float(params.get("dropout", 0.1))
        # 각 피처에 고유 임베딩 + 편향 (수치형 토큰화)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=d_token ** -0.5)
        nn.init.normal_(self.bias, std=d_token ** -0.5)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=heads, dim_feedforward=d_token * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias      # (B, F, d)
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        out = self.encoder(tokens)
        return self.head(self.norm(out[:, 0])).squeeze(-1)


class TabResNetModel(MLPModel):
    net_class = TabResNetNet
    name = "tabresnet"


class FTTransformerModel(MLPModel):
    net_class = FTTransformerNet
    name = "fttransformer"
