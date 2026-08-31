"""오토인코더 이상탐지.

지도학습 모델(LightGBM, GRU …)과 전제가 다르다. 고장 표본을 보고 "고장의
모양"을 배우는 대신, **정상 개체만** 보고 "정상의 모양"을 배운 뒤 거기서
벗어나는 정도를 이상 점수로 쓴다.

    학습  : 창 끝 시점에 고장이 확인되지 않은 디스크의 행만
    점수  : 재구성 오차 ||x - decode(encode(x))||^2  (클수록 이상)
    판정  : 나머지 파이프라인은 그대로. 임곗값도 val 에서 고른다.

이 방식이 이 문제에서 가질 수 있는 이점은 두 가지다.
  - 양성 표본 수에 의존하지 않는다. 한 달 val 양성이 2개뿐인 fold 에서도
    학습 자체는 흔들리지 않는다 (임곗값 선정은 여전히 흔들린다).
  - 학습 때 본 적 없는 고장 유형도 "정상이 아님"으로 잡을 수 있다.

반대로 약점도 분명하다. 정상에서 벗어난다고 다 고장은 아니다. 워크로드 변화,
펌웨어 교체, 온도 환경 변화가 전부 재구성 오차를 올린다.

--- 누출에 대하여 -------------------------------------------------------------

"정상 개체"를 failure_date 가 비어 있는 디스크로 정의하면 안 된다. 확장 창의
train 구간이 t월에 끝나는데 t+5월에 고장 날 디스크를 미리 빼는 것이 되어,
그 시점에 알 수 없는 정보로 학습 표본을 고르는 셈이다. FoldMatrix.healthy_mask
가 "창 끝까지 고장이 확인되지 않았는가"로 판정한다.

--- 결측에 대하여 -------------------------------------------------------------

트리 모델은 결측을 그대로 먹지만 신경망은 못 먹는다. segment 시작부의
diff/rolling 은 NULL 이라 그 칸을 0으로 채우고 학습하면 "변화가 없었다"는
잘못된 신호가 되고, 재구성 오차에도 가짜로 기여한다. 여기서는 결측 칸을
표준화 후 0(=평균)으로 채우되 **손실과 점수 계산에서 그 칸을 제외**한다.
관측 공백이 있는 디스크가 그 이유만으로 이상 판정되는 것을 막는다.

스케일러는 이 모델이 직접 들고 있는다. train 의 정상 행에서만 적합시키며
(fold 밖 통계를 보지 않는다) 저장/복원 대상이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..features.fold import Scaler
from .base import BaseModel


class DenseAutoEncoder(nn.Module):
    """대칭 MLP 오토인코더.

    hidden_dims 를 따라 좁혔다가 latent_dim 을 거쳐 다시 넓힌다. 병목이
    입력보다 좁아야 항등 사상으로 도망가지 않는다.
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        hidden = [int(h) for h in params.get("hidden_dims", [64, 32])]
        latent = int(params.get("latent_dim", 16))
        dropout = float(params.get("dropout", 0.0))
        activation = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[
            str(params.get("activation", "relu")).lower()
        ]
        batch_norm = bool(params.get("batch_norm", True))

        def stack(dims: list[int]) -> list[nn.Module]:
            layers: list[nn.Module] = []
            for in_dim, out_dim in zip(dims, dims[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            return layers

        self.encoder = nn.Sequential(*stack([n_features, *hidden, latent]))
        # 마지막 층은 활성화 없이 원 공간으로 되돌린다.
        decoder = stack([latent, *hidden[::-1]])
        decoder.append(nn.Linear(hidden[0] if hidden else latent, n_features))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoEncoderModel(BaseModel):
    family = "tabular"
    name = "autoencoder"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self.net: nn.Module | None = None
        self.scaler: Scaler | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 입력 준비 -----------------------------------------------------------
    def _prepare(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(표준화된 값, 관측 마스크). 마스크 0인 칸은 손실에서 뺀다."""
        mask = np.isfinite(X)
        scaled = self.scaler.transform(X, fill_nan=True)
        # 표준화 후에도 남는 비유한값(무한대 절단 실패 등)을 한 번 더 막는다.
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return scaled.astype(np.float32, copy=False), mask

    @staticmethod
    def _row_error(
        net: nn.Module, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """관측된 칸에 대해서만 평균 제곱 오차를 낸다 (행마다 하나)."""
        recon = net(x)
        squared = (recon - x).pow(2) * mask
        return squared.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _batches(self, n: int, batch_size: int, *, shuffle: bool, generator=None):
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            yield order[start : start + batch_size]

    # -- 인터페이스 ----------------------------------------------------------
    def fit(self, train, val) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        healthy = train.healthy_mask()
        n_healthy = int(healthy.sum())
        if n_healthy == 0:
            raise ValueError("train 구간에 정상 개체의 행이 하나도 없습니다.")
        print(
            f"      [정상만 학습] train {len(train):,}행 중 정상 {n_healthy:,}행 "
            f"({n_healthy / len(train):.2%}), 고장 확인 디스크의 행 "
            f"{len(train) - n_healthy:,}행 제외"
        )

        # 스케일러는 실제로 학습에 쓰는 행(정상)에서만 적합시킨다.
        scaling = self.training.get("scaling", {})
        self.scaler = Scaler(
            scaling.get("method", "standard"), scaling.get("clip_quantile", [0.001, 0.999])
        ).fit(train.X[healthy])

        train_x, train_mask = self._prepare(train.X[healthy])
        val_healthy = val.healthy_mask()
        val_x, val_mask = self._prepare(val.X[val_healthy])

        n_features = train_x.shape[1]
        self.net = DenseAutoEncoder(n_features, self.params).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(self.training.get("learning_rate", 1e-3)),
            weight_decay=float(self.training.get("weight_decay", 1e-5)),
        )
        batch_size = int(self.training.get("batch_size", 4096))
        epochs = int(self.training.get("epochs", 30))
        patience = int(self.training.get("early_stopping_patience", 5))
        metric = str(self.training.get("early_stopping_metric", "val_loss"))
        if metric not in ("val_loss", "val_pr_auc"):
            raise ValueError(
                f"알 수 없는 early_stopping_metric: {metric!r} (val_loss | val_pr_auc)"
            )

        train_tensor = torch.from_numpy(train_x)
        train_mask_tensor = torch.from_numpy(train_mask.astype(np.float32))
        val_tensor = torch.from_numpy(val_x)
        val_mask_tensor = torch.from_numpy(val_mask.astype(np.float32))
        # 라벨 기반 지표용. val 전체를 매 epoch 다시 표준화하지 않도록 한 번만
        # 만들어 둔다.
        full_x, full_mask = self._prepare(val.X)
        full_tensor = torch.from_numpy(full_x)
        full_mask_tensor = torch.from_numpy(full_mask.astype(np.float32))
        generator = torch.Generator().manual_seed(self.seed)

        best, best_epoch, best_state, waited = np.inf, -1, None, 0
        history = []

        for epoch in range(epochs):
            self.net.train()
            total, seen = 0.0, 0
            for index in self._batches(
                len(train_x), batch_size, shuffle=True, generator=generator
            ):
                # BatchNorm 은 학습 중 크기 1 배치에서 분산을 못 낸다.
                if index.shape[0] < 2:
                    continue
                batch_x = train_tensor[index].to(self.device, non_blocking=True)
                batch_mask = train_mask_tensor[index].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = self._row_error(self.net, batch_x, batch_mask).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * index.shape[0]
                seen += int(index.shape[0])

            # 정상 val 행의 재구성 오차. 라벨을 쓰지 않는 조기 종료 기준이다.
            val_loss = float(
                self._score_tensor(val_tensor, val_mask_tensor, batch_size).mean()
            )
            # 라벨 기반 지표는 기록만 한다 (기본 기준이 val_loss 일 때).
            val_pr_auc = self._pr_auc(
                val.y, self._score_tensor(full_tensor, full_mask_tensor, batch_size)
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / max(seen, 1),
                    "val_loss": val_loss,
                    "val_pr_auc": val_pr_auc,
                }
            )
            print(
                f"      epoch {epoch:02d} loss={total / max(seen, 1):.5f} "
                f"val_loss={val_loss:.5f} val_pr_auc={val_pr_auc:.5f}"
            )

            current = val_loss if metric == "val_loss" else -val_pr_auc
            if np.isfinite(current) and current < best:
                best, best_epoch, waited = current, epoch, 0
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
            "n_features": int(n_features),
            "feature_names": list(train.columns),
            "best_epoch": best_epoch,
            "best_val_loss": float(best if metric == "val_loss" else np.nan),
            "early_stopping_metric": metric,
            "epochs_run": len(history),
            "device": str(self.device),
            "n_train": len(train),
            "n_train_healthy": n_healthy,
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
            "warm_started": False,
            "history": history,
        }
        return self.fit_info

    @torch.no_grad()
    def _score_tensor(
        self, x: torch.Tensor, mask: torch.Tensor, batch_size: int
    ) -> np.ndarray:
        self.net.eval()
        out = np.empty(x.shape[0], dtype=np.float64)
        for index in self._batches(x.shape[0], batch_size, shuffle=False):
            batch_x = x[index].to(self.device, non_blocking=True)
            batch_mask = mask[index].to(self.device, non_blocking=True)
            error = self._row_error(self.net, batch_x, batch_mask)
            out[index.numpy()] = error.float().cpu().numpy()
        return out

    def predict_proba(self, data) -> np.ndarray:
        """재구성 오차를 이상 점수로 돌려준다.

        이름은 인터페이스 때문이고 확률이 아니다. 순위만 의미가 있으므로
        PR-AUC / ROC-AUC / 임곗값 선정은 그대로 성립하지만, Brier score 처럼
        확률 눈금을 전제하는 지표는 이 모델에서 해석하지 말아야 한다.
        """
        scaled, mask = self._prepare(data.X)
        return self._score_tensor(
            torch.from_numpy(scaled),
            torch.from_numpy(mask.astype(np.float32)),
            int(self.training.get("batch_size", 4096)),
        )

    @staticmethod
    def _pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score

        if y_true.max() == y_true.min():
            return float("nan")
        return float(average_precision_score(y_true, score))

    def complexity(self) -> int:
        if self.net is None:
            return 0
        return int(sum(p.numel() for p in self.net.parameters()))

    def _save_weights(self, directory: Path) -> None:
        torch.save(self.net.state_dict(), directory / "model.pt")
        with (directory / "scaler.json").open("w", encoding="utf-8") as fh:
            json.dump(self.scaler.state_dict(), fh, ensure_ascii=False)

    def _load_weights(self, directory: Path) -> None:
        with (directory / "scaler.json").open("r", encoding="utf-8") as fh:
            self.scaler = Scaler.from_state_dict(json.load(fh))
        self.net = DenseAutoEncoder(
            int(self.fit_info["n_features"]), self.params
        ).to(self.device)
        state = torch.load(
            directory / "model.pt", map_location=self.device, weights_only=True
        )
        self.net.load_state_dict(state)
