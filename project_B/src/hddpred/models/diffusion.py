"""확산 모델 기반 이상탐지.

AutoEncoder / LSTM-AE 와 전제가 같다 — 정상 개체의 행만 학습하고, 복원 오차를
이상 점수로 쓴다. 다른 점은 복원 방식이다.

    AutoEncoder   병목을 지나 복원. 병목이 좁아야 항등 사상을 피한다.
    Diffusion     노이즈를 씌웠다 걷어낸다. 병목이 없고, 학습된 것은
                  "정상 데이터의 점수 함수(score function)"다.

    x  --(t단계 노이즈)-->  x_t  --(denoiser)-->  x0_hat
    이상 점수 = || x - x0_hat ||^2

정상 분포에서 온 표본은 노이즈를 걷어내면 원래 자리로 돌아온다. 정상 분포
밖의 표본은 denoiser 가 정상 다양체 쪽으로 끌어당기므로 원래 자리에서 멀어진다.
그 거리가 곧 이상도다. AnoDDPM (Wyatt et al. 2022) 이 영상에서 쓴 방식이고
표 형식에서는 TabDDPM (Kotelnikov et al. 2023) 이 같은 denoiser 구조를 쓴다.

--- 왜 확산인가 -------------------------------------------------------------

오토인코더는 병목 크기가 결과를 좌우한다. 너무 넓으면 이상까지 그대로 복원해
버리고(항등 사상), 너무 좁으면 정상도 복원을 못 한다. 확산은 그 손잡이가
없고 대신 노이즈 수준 t 가 그 역할을 한다 — t 를 바꾸면 "얼마나 큰 규모의
이상을 볼 것인가"가 바뀐다. 작은 t 는 국소적 이상, 큰 t 는 전역 구조 이상.

--- 하지 않는 것 -----------------------------------------------------------

확산을 양성 표본 증강에 쓰는 용법(TabDDPM 의 원래 목적)은 여기서 쓰지 않는다.
합성 표본이 평가 집합에 새면 결과가 통째로 무효가 되고, 선행연구에도 그
의심을 받는 사례가 있다. 증강은 학습 표본 구성의 문제이므로 쓰려면
features.train_sampling 축으로 따로 다뤄야 한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..features.fold import Scaler
from .base import BaseModel


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer 위치 인코딩과 같은 형태의 노이즈 단계 임베딩."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    out = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=1)
    return out


class DenoiserNet(nn.Module):
    """x_t 와 t 를 받아 섞인 노이즈를 예측한다."""

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        width = int(params.get("width", 512))
        depth = int(params.get("depth", 3))
        t_dim = int(params.get("time_dim", 128))
        self.time = nn.Sequential(
            nn.Linear(t_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.t_dim = t_dim
        self.stem = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                    nn.Linear(width, width),
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(width, n_features)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_t) + self.time(_timestep_embedding(t, self.t_dim))
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class DiffusionAnomalyModel(BaseModel):
    family = "tabular"
    name = "diffusion"

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self.net: nn.Module | None = None
        self.scaler: Scaler | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._schedule: dict | None = None

    # -- 확산 일정 -----------------------------------------------------------
    def _build_schedule(self) -> dict:
        steps = int(self.params.get("timesteps", 200))
        betas = torch.linspace(1e-4, 0.02, steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        return {
            "steps": steps,
            "sqrt_ab": alpha_bar.sqrt(),
            "sqrt_1mab": (1.0 - alpha_bar).sqrt(),
        }

    # -- 입력 ---------------------------------------------------------------
    def _prepare(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(표준화 값, 관측 마스크). 결측 칸은 손실과 점수에서 뺀다."""
        mask = np.isfinite(X)
        scaled = self.scaler.transform(X, fill_nan=True)
        return scaled.astype(np.float32, copy=False), mask

    @staticmethod
    def _batches(n: int, size: int, *, shuffle: bool, generator=None):
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, size):
            yield order[start : start + size]

    # -- 인터페이스 ----------------------------------------------------------
    def fit(self, train, val) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        healthy = train.healthy_mask()
        n_healthy = int(healthy.sum())
        if n_healthy == 0:
            raise ValueError("train 구간에 정상 개체의 행이 없습니다.")
        print(
            f"      [정상만 학습] train {len(train):,}행 중 정상 {n_healthy:,}행 "
            f"({n_healthy / len(train):.2%})"
        )

        scaling = self.training.get("scaling", {})
        healthy_X = train.X[healthy]
        self.scaler = Scaler(
            scaling.get("method", "standard"),
            scaling.get("clip_quantile", [0.001, 0.999]),
        ).fit(healthy_X)

        x_np, mask_np = self._prepare(healthy_X)
        del healthy_X
        x_all = torch.from_numpy(x_np)
        mask_all = torch.from_numpy(mask_np.astype(np.float32))

        self._schedule = self._build_schedule()
        self.net = DenoiserNet(x_all.shape[1], self.params).to(self.device)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(self.training.get("learning_rate", 1e-3)),
            weight_decay=float(self.training.get("weight_decay", 1e-5)),
        )
        batch_size = int(self.training.get("batch_size", 4096))
        epochs = int(self.training.get("epochs", 30))
        patience = int(self.training.get("early_stopping_patience", 5))
        generator = torch.Generator().manual_seed(self.seed)

        val_healthy = val.healthy_mask()
        val_x, val_mask = self._prepare(val.X[val_healthy])
        val_t = torch.from_numpy(val_x)
        val_m = torch.from_numpy(val_mask.astype(np.float32))

        best, best_epoch, best_state, waited = np.inf, -1, None, 0
        history = []
        for epoch in range(epochs):
            self.net.train()
            total, seen = 0.0, 0
            for index in self._batches(
                len(x_all), batch_size, shuffle=True, generator=generator
            ):
                xb = x_all[index].to(self.device, non_blocking=True)
                mb = mask_all[index].to(self.device, non_blocking=True)
                t = torch.randint(
                    0, self._schedule["steps"], (xb.shape[0],), device=self.device
                )
                noise = torch.randn_like(xb)
                x_t = (
                    self._schedule["sqrt_ab"][t].unsqueeze(1) * xb
                    + self._schedule["sqrt_1mab"][t].unsqueeze(1) * noise
                )
                predicted = self.net(x_t, t)
                # 결측 칸은 손실에서 뺀다. 관측 공백이 이상 신호가 되지 않도록.
                loss = (((predicted - noise) ** 2) * mb).sum() / mb.sum().clamp(min=1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * index.shape[0]
                seen += int(index.shape[0])

            val_loss = float(self._scores(val_t, val_m, batch_size).mean())
            history.append(
                {"epoch": epoch, "train_loss": total / max(seen, 1), "val_loss": val_loss}
            )
            print(
                f"      epoch {epoch:02d} loss={total / max(seen, 1):.5f} "
                f"val_recon={val_loss:.5f}"
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
            "n_features": int(x_all.shape[1]),
            "feature_names": list(train.columns),
            "timesteps": self._schedule["steps"],
            "eval_t": int(self.params.get("eval_t", 50)),
            "eval_repeats": int(self.params.get("eval_repeats", 4)),
            "best_epoch": best_epoch,
            "best_val_recon": float(best),
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
    def _scores(self, x: torch.Tensor, mask: torch.Tensor, batch_size: int) -> np.ndarray:
        """고정 노이즈 수준에서 한 번 복원한 뒤의 거리.

        여러 번의 노이즈 추출로 평균해 분산을 줄인다. 전체 역확산을 돌리지
        않는 이유는 비용 때문이다 — 표본 17만개에 200단계를 돌리면 3천만 번의
        forward 가 되고, 한 번 복원만으로도 이상도 순위는 충분히 나온다.
        """
        self.net.eval()
        t_eval = int(self.params.get("eval_t", 50))
        repeats = int(self.params.get("eval_repeats", 4))
        sqrt_ab = self._schedule["sqrt_ab"][t_eval]
        sqrt_1mab = self._schedule["sqrt_1mab"][t_eval]
        out = np.empty(x.shape[0], dtype=np.float64)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        for index in self._batches(x.shape[0], batch_size, shuffle=False):
            xb = x[index].to(self.device, non_blocking=True)
            mb = mask[index].to(self.device, non_blocking=True)
            t = torch.full((xb.shape[0],), t_eval, device=self.device, dtype=torch.long)
            accumulated = torch.zeros(xb.shape[0], device=self.device)
            for _ in range(repeats):
                noise = torch.randn(xb.shape, device=self.device, generator=generator)
                x_t = sqrt_ab * xb + sqrt_1mab * noise
                x0_hat = (x_t - sqrt_1mab * self.net(x_t, t)) / sqrt_ab
                error = ((x0_hat - xb) ** 2) * mb
                accumulated += error.sum(dim=1) / mb.sum(dim=1).clamp(min=1.0)
            out[index.numpy()] = (accumulated / repeats).float().cpu().numpy()
        return out

    def predict_proba(self, data) -> np.ndarray:
        """복원 오차. 확률이 아니므로 Brier score 는 이 모델에서 해석하지 않는다."""
        scaled, mask = self._prepare(data.X)
        return self._scores(
            torch.from_numpy(scaled),
            torch.from_numpy(mask.astype(np.float32)),
            int(self.training.get("batch_size", 4096)),
        )

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
        self._schedule = self._build_schedule()
        self.net = DenoiserNet(int(self.fit_info["n_features"]), self.params).to(
            self.device
        )
        self.net.load_state_dict(
            torch.load(directory / "model.pt", map_location=self.device, weights_only=True)
        )
