"""시퀀스 모델 (LSTM, GRU, TCN, Transformer).

입력은 SequenceFold. 윈도우를 미리 복제하지 않고 행렬 하나를 슬라이싱한다.
L=30, 표본 2천만개면 복제본은 수백 GB가 되기 때문이다.

세 신경망 모두 같은 학습 루프를 쓴다. 모델 간 비교에서 학습 조건 차이가
결과에 섞이지 않게 하기 위한 것이다. 차이는 forward 정의뿐이다.

TCN 과 Transformer 는 causal 이다. 시점 t의 출력이 t 이후를 보지 않는다.
tests/test_no_leakage.py 가 이 성질을 직접 검사한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base import BaseModel


# --------------------------------------------------------------------------
# 데이터
# --------------------------------------------------------------------------
class WindowDataset(Dataset):
    """SequenceFold 의 끝 인덱스에서 lookback 길이만큼 거슬러 슬라이싱한다.

    valid_len 이 주어지면 padding 모드다. 실제 행이 lookback 보다 적은 표본은
    앞쪽을 그 segment 의 첫 행으로 복제하고, 마지막 채널에 mask(1=실제,
    0=패딩)를 붙인다. 복제는 그 segment 의 과거 행만 쓰므로 미래를 보지 않는다.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        end_index: np.ndarray,
        lookback: int,
        y,
        valid_len: np.ndarray | None = None,
    ):
        self.matrix = matrix
        self.end_index = end_index
        self.lookback = lookback
        self.valid_len = valid_len
        self.y = np.zeros(len(end_index), np.float32) if y is None else y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.end_index.shape[0])

    def __getitem__(self, index: int):
        end = int(self.end_index[index])
        if self.valid_len is None:
            start = end - self.lookback + 1
            return torch.from_numpy(self.matrix[start : end + 1]), self.y[index]

        n = int(self.valid_len[index])
        real = self.matrix[end - n + 1 : end + 1]
        if n < self.lookback:
            pad = np.repeat(real[:1], self.lookback - n, axis=0)
            window = np.concatenate((pad, real), axis=0)
        else:
            window = real
        mask = np.zeros((self.lookback, 1), dtype=np.float32)
        mask[self.lookback - n :] = 1.0
        return torch.from_numpy(np.concatenate((window, mask), axis=1)), self.y[index]


class GPUWindowBatcher:
    """창 조립을 GPU 인덱싱 한 번으로 처리하는 배치 공급기.

    WindowDataset + DataLoader 는 표본 하나마다 __getitem__ 을 부른다. 2,500만
    표본이면 에폭당 2,500만 번의 파이썬 호출이고, 각 호출이 np.repeat 과
    np.concatenate 두 번을 한다. 실측에서 GPU 사용률이 15% 였고 초/에폭이
    모델을 바꿔도 320~341초로 거의 변하지 않았다 — 신경망이 아니라 조립이
    병목이라는 뜻이다.

    행렬(2,700만 x 18 float32 = 1.9GB)은 GPU 에 통째로 올라간다. 그러면 배치
    하나가 인덱싱 한 번으로 끝난다.

        rows = end[:, None] + arange(-(L-1), 1)      (B, L)
        rows = maximum(rows, first[:, None])          앞쪽 패딩 = 첫 실제 행 복제
        x    = matrix[rows]                           (B, L, F)

    WindowDataset 과 결과가 정확히 같아야 한다:
      - 패딩   WindowDataset 은 real[:1] 을 (L-n) 번 복제한다. real 의 첫 행은
               end-n+1 이므로 행 번호를 first=end-n+1 로 하한 절단한 것과 같다.
      - mask   WindowDataset 은 mask[L-n:] = 1 로 둔다. 위치 j 의 행 번호가
               end-(L-1)+j 이므로 j >= L-n <=> 행 >= first 다. 같은 조건이다.
    tests/test_sequence_batcher.py 가 두 경로를 직접 비교한다.
    """

    def __init__(self, fold, batch_size: int, device, *, shuffle: bool, seed: int = 0):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = device
        self._epoch = 0
        self._seed = int(seed)

        matrix = np.ascontiguousarray(fold.matrix, dtype=np.float32)
        self.matrix = torch.from_numpy(matrix).to(device, non_blocking=True)
        self.end = torch.as_tensor(fold.end_index, dtype=torch.long, device=device)
        self.y = torch.as_tensor(
            np.zeros(len(fold.end_index), np.float32)
            if fold.y is None else fold.y.astype(np.float32),
            device=device,
        )
        self.lookback = int(fold.lookback)
        valid = getattr(fold, "valid_len", None)
        self.padded = valid is not None
        # 패딩 모드가 아니면 전부 lookback 만큼 채워진 것으로 본다.
        self.valid = (
            torch.as_tensor(valid, dtype=torch.long, device=device)
            if self.padded
            else torch.full_like(self.end, self.lookback)
        )
        self.offsets = torch.arange(
            -(self.lookback - 1), 1, dtype=torch.long, device=device
        )

    def __len__(self) -> int:
        n = int(self.end.shape[0])
        return (n + self.batch_size - 1) // self.batch_size

    def gather(self, sel: torch.Tensor) -> torch.Tensor:
        """표본 인덱스 sel 에 대한 (B, L, F[+1]) 텐서."""
        end = self.end[sel]
        rows = end.unsqueeze(1) + self.offsets.unsqueeze(0)
        first = (end - self.valid[sel] + 1).unsqueeze(1)
        window = self.matrix[torch.maximum(rows, first)]
        if not self.padded:
            return window
        mask = (rows >= first).to(window.dtype).unsqueeze(-1)
        return torch.cat((window, mask), dim=-1)

    def __iter__(self):
        n = int(self.end.shape[0])
        if self.shuffle:
            # 에폭마다 다른 순서. 시드를 고정해 재현 가능하게 둔다.
            generator = torch.Generator(device=self.device.type)
            generator.manual_seed(self._seed + self._epoch)
            order = torch.randperm(n, device=self.device, generator=generator)
            self._epoch += 1
        else:
            order = torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            sel = order[start : start + self.batch_size]
            yield self.gather(sel), self.y[sel]


def matrix_fits_on_gpu(fold, device, headroom: float = 0.35) -> bool:
    """행렬을 GPU 에 통째로 올려도 되는가. 활성값이 쓸 자리를 남긴다."""
    if device.type != "cuda":
        return False
    need = int(fold.matrix.size) * 4
    free, _total = torch.cuda.mem_get_info(device)
    return need < free * (1.0 - headroom)


# --------------------------------------------------------------------------
# 신경망
# --------------------------------------------------------------------------
class RNNNet(nn.Module):
    def __init__(self, n_features: int, params: dict):
        super().__init__()
        cell = params.get("cell", "lstm").lower()
        layer = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        hidden = int(params.get("hidden_size", 128))
        layers = int(params.get("num_layers", 2))
        bidirectional = bool(params.get("bidirectional", False))
        self.rnn = layer(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=float(params.get("dropout", 0.0)) if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # 왼쪽에만 padding 한다. 오른쪽(미래)을 보지 않는다.
        out = torch.relu(self.norm1(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        out = self.dropout(out)
        out = torch.relu(self.norm2(self.conv2(nn.functional.pad(out, (self.pad, 0)))))
        out = self.dropout(out)
        return torch.relu(out + self.residual(x))


class TCNNet(nn.Module):
    def __init__(self, n_features: int, params: dict):
        super().__init__()
        channels = list(params.get("channels", [64, 64, 64, 64]))
        kernel = int(params.get("kernel_size", 3))
        dropout = float(params.get("dropout", 0.2))
        blocks, in_ch = [], n_features
        for level, out_ch in enumerate(channels):
            blocks.append(_CausalBlock(in_ch, out_ch, kernel, 2**level, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        out = self.blocks(x.transpose(1, 2))
        return self.head(out[:, :, -1]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, n_features: int, params: dict):
        super().__init__()
        d_model = int(params.get("d_model", 128))
        self.pooling = params.get("pooling", "last")
        self.input = nn.Linear(n_features, d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(params.get("nhead", 4)),
            dim_feedforward=int(params.get("dim_feedforward", 256)),
            dropout=float(params.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True 라 nested tensor 최적화가 어차피 꺼진다. 명시해서
        # 매 생성마다 나오는 경고를 없앤다.
        self.encoder = nn.TransformerEncoder(
            encoder, int(params.get("num_layers", 2)), enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("_positions", torch.arange(4096).unsqueeze(1))
        self.d_model = d_model

    def _positional(self, length: int, device) -> torch.Tensor:
        position = self._positions[:length].to(device).float()
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        encoding = torch.zeros(length, self.d_model, device=device)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        return encoding.unsqueeze(0)

    def forward(self, x):
        length = x.shape[1]
        hidden = self.input(x) + self._positional(length, x.device)
        # causal mask. 시점 t 가 t 이후를 참조하지 못하게 한다.
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=x.device)
        out = self.encoder(hidden, mask=mask, is_causal=True)
        pooled = out.mean(dim=1) if self.pooling == "mean" else out[:, -1, :]
        return self.head(pooled).squeeze(-1)


# --------------------------------------------------------------------------
# 손실
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """이진 focal loss (Lin et al., 2017).

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    쉬운 표본의 기여를 (1-p_t)^gamma 로 눌러서, 압도적으로 많은 쉬운 음성이
    손실을 지배하는 것을 막는다. 이 데이터의 양성 비율이 행 기준 0.065% 라
    그 상황에 해당한다.

    alpha 는 양성 클래스에 곱하는 가중이다.
      - 숫자를 주면 그 값을 쓴다 (논문 기본 0.25).
      - "auto" 면 train 의 1 - 양성비율로 둔다. 양성이 드물수록 커진다.

    pos_weight 와 함께 쓰면 불균형 보정이 두 번 걸린다. Trainer 가 focal 을
    쓸 때 auto_pos_weight 를 무시하는 이유다.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # p_t = 정답 클래스에 대한 예측 확률. exp(-ce) 로 바로 얻는다.
        p_t = torch.exp(-ce)
        loss = ce * (1.0 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


def build_criterion(training: dict, y_train: np.ndarray, device) -> tuple[nn.Module, dict]:
    """설정과 train 라벨로 손실 함수를 만든다. 기록용 정보도 함께 돌려준다."""
    loss_name = str(training.get("loss", "bce")).lower()
    positive_rate = float((y_train == 1).mean()) if y_train.size else 0.0

    if loss_name == "focal":
        alpha = training.get("focal_alpha", 0.25)
        if isinstance(alpha, str) and alpha.lower() == "auto":
            alpha = 1.0 - positive_rate
        alpha = None if alpha is None else float(alpha)
        gamma = float(training.get("focal_gamma", 2.0))
        info = {"loss": "focal", "focal_gamma": gamma, "focal_alpha": alpha}
        if training.get("auto_pos_weight", False):
            info["note"] = "focal 사용으로 auto_pos_weight 무시"
        return FocalLoss(gamma, alpha), info

    if loss_name != "bce":
        raise ValueError(f"알 수 없는 training.loss: {loss_name!r} (bce | focal)")

    pos_weight = None
    info = {"loss": "bce", "pos_weight": None}
    if training.get("auto_pos_weight", False):
        positives = float((y_train == 1).sum())
        negatives = float((y_train == 0).sum())
        weight = negatives / positives if positives else 1.0
        pos_weight = torch.tensor([weight], device=device)
        info["pos_weight"] = weight
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), info


# --------------------------------------------------------------------------
# 공통 학습 루프
# --------------------------------------------------------------------------
class TorchSequenceModel(BaseModel):
    family = "sequence"
    net_class: type[nn.Module] = RNNNet

    def __init__(self, params: dict, training: dict, seed: int = 42):
        super().__init__(params, training, seed)
        self.net: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._warm_state: dict | None = None
        # 하이퍼파라미터 탐색이 매 에폭의 val 점수를 받아 가지치기할 수 있게
        # 하는 훅. (epoch, best_val_pr_auc) 를 받는다. None 이면 학습 경로는
        # 평소와 완전히 같다.
        self.epoch_callback = None

    def warm_start(self, previous) -> bool:
        """직전 fold 의 가중치를 초기값으로 가져온다.

        구조가 다르면(피처 수 변화 등) 조용히 무시하지 않고 fit 에서 걸러
        기록한다. state_dict 만 들고 있다가 _build 직후에 적용한다.
        """
        net = getattr(previous, "net", None)
        if net is None:
            return False
        self._warm_state = {
            key: value.detach().cpu().clone() for key, value in net.state_dict().items()
        }
        return True

    # -- 내부 ---------------------------------------------------------------
    def _loader(self, fold, *, shuffle: bool, batch_size: int | None = None):
        """배치 공급기.

        행렬이 GPU 에 올라가면 창 조립을 GPU 인덱싱으로 한다 (GPUWindowBatcher).
        표본마다 파이썬 __getitem__ 을 부르는 경로가 사라져 조립 병목이 없어진다.
        안 올라가면 기존 DataLoader 로 떨어진다. 두 경로의 배치 내용은 같다.
        training.gpu_batching 을 false 로 두면 강제로 옛 경로를 쓴다.
        """
        size = int(batch_size or self.training.get("batch_size", 512))
        if self.training.get("gpu_batching", True) and matrix_fits_on_gpu(
            fold, self.device
        ):
            return GPUWindowBatcher(
                fold, size, self.device, shuffle=shuffle, seed=self.seed
            )
        if shuffle and self.device.type == "cuda":
            print(
                f"      [batcher] 행렬 {fold.matrix.nbytes / 2**30:.1f}GB 가 GPU 에 "
                "안 들어가 DataLoader 로 떨어진다 (조립이 병목이 된다)."
            )
        dataset = WindowDataset(
            fold.matrix,
            fold.end_index,
            fold.lookback,
            fold.y,
            getattr(fold, "valid_len", None),
        )
        return DataLoader(
            dataset,
            batch_size=size,
            shuffle=shuffle,
            num_workers=int(self.training.get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _build(self, n_features: int) -> nn.Module:
        torch.manual_seed(self.seed)
        return self.net_class(n_features, self.params).to(self.device)

    @staticmethod
    def _pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score

        if y_true.max() == y_true.min():
            return float("nan")
        return float(average_precision_score(y_true, score))

    @staticmethod
    def _pauc(y_true: np.ndarray, score: np.ndarray, max_fpr: float = 0.05) -> float:
        """FAR <= max_fpr 구간의 부분 AUC (McClish 표준화).

        Optuna 목적함수와 같은 모양의 값이다. 다만 목적함수는 디스크 단위
        월별 창에서 재고 이건 행 단위 val 전체에서 잰다 — 같은 값이 아니라
        같은 관심 구간을 본다는 뜻이다.
        """
        from sklearn.metrics import roc_auc_score

        if y_true.max() == y_true.min():
            return float("nan")
        return float(roc_auc_score(y_true, score, max_fpr=max_fpr))

    # -- 인터페이스 ---------------------------------------------------------
    def fit(self, train, val) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.net = self._build(train.n_features)
        warm_started = False
        if self._warm_state is not None:
            current = self.net.state_dict()
            same_shape = all(
                key in current and current[key].shape == value.shape
                for key, value in self._warm_state.items()
            )
            if same_shape:
                self.net.load_state_dict(self._warm_state)
                warm_started = True
                print("      warm start: 직전 fold 가중치에서 이어서 학습")
            else:
                print("      warm start 불가: 구조가 달라 새로 초기화")

        criterion, loss_info = build_criterion(self.training, train.y, self.device)
        criterion = criterion.to(self.device) if hasattr(criterion, "to") else criterion
        print(f"      loss={loss_info}")
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(self.training.get("learning_rate", 1e-3)),
            weight_decay=float(self.training.get("weight_decay", 0.0)),
        )
        use_amp = bool(self.training.get("amp", True)) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        clip = float(self.training.get("grad_clip", 0.0))

        train_loader = self._loader(train, shuffle=True)
        patience = int(self.training.get("early_stopping_patience", 5))
        epochs = int(self.training.get("epochs", 30))
        # 어떤 val 지표로 최적 에폭을 고를지. 둘 다 클수록 좋다.
        monitor = str(self.training.get("early_stopping_metric", "val_pr_auc"))
        if monitor == "val_pr_auc":
            monitor_fn = self._pr_auc
        elif monitor == "val_pauc":
            monitor_fn = self._pauc
        else:
            raise ValueError(
                f"알 수 없는 early_stopping_metric: {monitor!r} "
                "(val_pr_auc | val_pauc)"
            )

        best_score, best_epoch, best_state, waited = -np.inf, -1, None, 0
        history = []

        for epoch in range(epochs):
            self.net.train()
            total_loss, seen = 0.0, 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(self.net(batch_x), batch_y)
                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item()) * batch_y.shape[0]
                seen += int(batch_y.shape[0])

            val_score = monitor_fn(val.y, self.predict_proba(val))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(seen, 1),
                    monitor: val_score,
                }
            )
            print(
                f"      epoch {epoch:02d} loss={total_loss / max(seen, 1):.5f} "
                f"{monitor}={val_score:.5f}"
            )

            if np.isfinite(val_score) and val_score > best_score:
                best_score, best_epoch, waited = val_score, epoch, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()
                }
            else:
                waited += 1
                if waited >= patience:
                    print(f"      early stopping (patience {patience})")
                    break

            # 하이퍼파라미터 탐색용 훅. 매 에폭의 val 점수를 밖으로 넘겨서,
            # 가망 없는 조합을 끝까지 학습하지 않고 끊을 수 있게 한다.
            # 설정하지 않으면 아무 일도 하지 않으므로 평소 학습 경로는 그대로다.
            if self.epoch_callback is not None:
                self.epoch_callback(epoch, float(best_score))

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.fit_info = {
            "n_features": int(train.n_features),
            "lookback": int(train.lookback),
            "best_epoch": best_epoch,
            "best_val_pr_auc": float(best_score),
            "early_stopping_metric": monitor,
            "epochs_run": len(history),
            "device": str(self.device),
            "n_train": len(train),
            "n_val": len(val),
            "train_positive_rate": train.positive_rate,
            "warm_started": warm_started,
            **loss_info,
            "history": history,
        }
        return self.fit_info

    @torch.no_grad()
    def predict_proba(self, data) -> np.ndarray:
        self.net.eval()
        loader = self._loader(data, shuffle=False)
        out = np.empty(len(data), dtype=np.float64)
        cursor = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            logits = self.net(batch_x).float()
            probability = torch.sigmoid(logits).cpu().numpy()
            out[cursor : cursor + probability.shape[0]] = probability
            cursor += probability.shape[0]
        return out

    def complexity(self) -> int:
        if self.net is None:
            return 0
        return int(sum(p.numel() for p in self.net.parameters()))

    def _save_weights(self, directory: Path) -> None:
        torch.save(self.net.state_dict(), directory / "model.pt")

    def _load_weights(self, directory: Path) -> None:
        n_features = int(self.fit_info["n_features"])
        self.net = self._build(n_features)
        state = torch.load(
            directory / "model.pt", map_location=self.device, weights_only=True
        )
        self.net.load_state_dict(state)


class RNNModel(TorchSequenceModel):
    net_class = RNNNet
    name = "rnn"


class TCNModel(TorchSequenceModel):
    net_class = TCNNet
    name = "tcn"


class TransformerModel(TorchSequenceModel):
    net_class = TransformerNet
    name = "transformer"


class CNN1DNet(nn.Module):
    """평범한 1D CNN. TCN 과의 차이는 dilation 과 residual 이 없다는 것이다.

    lookback 창 전체가 관측 시점 이전 구간이므로 창 안에서 미래를 보는 문제는
    없다. 그래도 causal padding 을 유지해 TCN 과 조건을 맞춘다 (마지막 시점의
    표현이 그 시점까지만 참조한다).
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        channels = [int(c) for c in params.get("channels", [64, 128, 64])]
        kernel = int(params.get("kernel_size", 5))
        dropout = float(params.get("dropout", 0.2))
        self.pad = kernel - 1
        blocks: list[nn.Module] = []
        in_ch = n_features
        for out_ch in channels:
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel))
            blocks.append(nn.BatchNorm1d(out_ch))
            blocks.append(nn.ReLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.pooling = str(params.get("pooling", "last"))
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        out = x.transpose(1, 2)
        for layer in self.blocks:
            if isinstance(layer, nn.Conv1d):
                out = layer(nn.functional.pad(out, (self.pad, 0)))
            else:
                out = layer(out)
        pooled = out.mean(dim=2) if self.pooling == "mean" else out[:, :, -1]
        return self.head(pooled).squeeze(-1)


class CNN1DModel(TorchSequenceModel):
    net_class = CNN1DNet
    name = "cnn1d"


# --------------------------------------------------------------------------
# 시계열 분류 표준 아키텍처
#
# Fawaz et al. (2019) "Deep learning for time series classification: a review"
# 가 벤치마크로 정리한 계열이다. TCN / Transformer 와 달리 창 안에서 대칭
# padding 을 쓰지만 누출이 아니다 — lookback 창 전체가 관측 시점 t 이전
# 구간이고, 창의 끝이 곧 t 이기 때문이다. 창 밖을 보는 경로가 없다.
# --------------------------------------------------------------------------
class FCNNet(nn.Module):
    """Fully Convolutional Network. conv 3층 + global average pooling.

    시계열 분류에서 오래 쓰인 강한 기준선이다. pooling 이 전체 구간을 평균해서
    특정 시점에 의존하지 않는다.
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        channels = [int(c) for c in params.get("channels", [128, 256, 128])]
        kernels = [int(k) for k in params.get("kernels", [7, 5, 3])]
        layers, in_ch = [], n_features
        for out_ch, k in zip(channels, kernels):
            layers += [
                nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        out = self.body(x.transpose(1, 2))
        return self.head(out.mean(dim=2)).squeeze(-1)


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels=(7, 5, 3)):
        super().__init__()
        layers, ch = [], in_ch
        for i, k in enumerate(kernels):
            layers += [nn.Conv1d(ch, out_ch, k, padding=k // 2), nn.BatchNorm1d(out_ch)]
            if i < len(kernels) - 1:
                layers.append(nn.ReLU())
            ch = out_ch
        self.body = nn.Sequential(*layers)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        return torch.relu(self.body(x) + self.shortcut(x))


class ResNet1DNet(nn.Module):
    """1D ResNet. 잔차 블록 3개 + GAP. FCN 과 함께 표준 기준선이다."""

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        channels = [int(c) for c in params.get("channels", [64, 128, 128])]
        blocks, in_ch = [], n_features
        for out_ch in channels:
            blocks.append(_ResidualBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        out = self.blocks(x.transpose(1, 2))
        return self.head(out.mean(dim=2)).squeeze(-1)


class _InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, filters: int, kernels=(9, 19, 39), bottleneck: int = 32):
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        )
        ch = bottleneck if in_ch > 1 else in_ch
        self.convs = nn.ModuleList(
            [nn.Conv1d(ch, filters, k, padding=k // 2, bias=False) for k in kernels]
        )
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, filters, 1, bias=False),
        )
        self.norm = nn.BatchNorm1d(filters * (len(kernels) + 1))

    def forward(self, x):
        bottled = self.bottleneck(x)
        parts = [conv(bottled) for conv in self.convs] + [self.pool_conv(x)]
        return torch.relu(self.norm(torch.cat(parts, dim=1)))


class InceptionTimeNet(nn.Module):
    """InceptionTime. 서로 다른 커널 크기를 병렬로 두어 여러 시간 척도를 본다.

    SMART 는 급변(재할당 섹터 점프)과 완만한 추세(온도, 전원인가 시간)가
    섞여 있어 다중 척도가 맞는 구조다.
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        depth = int(params.get("depth", 6))
        filters = int(params.get("filters", 32))
        blocks, in_ch, residuals = [], n_features, []
        for d in range(depth):
            blocks.append(_InceptionBlock(in_ch, filters))
            in_ch = filters * 4
            residuals.append(d % 3 == 2)
        self.blocks = nn.ModuleList(blocks)
        self.residuals = residuals
        self.shortcuts = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(n_features if d < 3 else filters * 4, filters * 4, 1),
                              nn.BatchNorm1d(filters * 4))
                if flag else nn.Identity()
                for d, flag in enumerate(residuals)
            ]
        )
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        out = x.transpose(1, 2)
        residual = out
        for block, flag, shortcut in zip(self.blocks, self.residuals, self.shortcuts):
            out = block(out)
            if flag:
                out = torch.relu(out + shortcut(residual))
                residual = out
        return self.head(out.mean(dim=2)).squeeze(-1)


class LSTMFCNNet(nn.Module):
    """LSTM-FCN. 순환 분기와 합성곱 분기를 이어 붙인다 (Karim et al. 2018).

    LSTM 이 장기 의존을, FCN 이 국소 형태를 담당한다. 두 계열을 합치면
    나아지는지 보는 대조군이다.
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        hidden = int(params.get("hidden_size", 128))
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(float(params.get("dropout", 0.2)))
        self.fcn = FCNNet(n_features, params)
        self.fcn.head = nn.Identity()
        fcn_out = int(params.get("channels", [128, 256, 128])[-1])
        self.head = nn.Linear(hidden + fcn_out, 1)

    def forward(self, x):
        rnn_out, _ = self.lstm(x)
        rnn_feat = self.dropout(rnn_out[:, -1, :])
        conv_feat = self.fcn.body(x.transpose(1, 2)).mean(dim=2)
        return self.head(torch.cat([rnn_feat, conv_feat], dim=1)).squeeze(-1)


class AttentionRNNNet(nn.Module):
    """GRU + attention pooling. 마지막 시점만 쓰는 대신 시점 가중합을 쓴다.

    HDD 연구에서 자주 쓰이는 변형이고, 어느 시점이 판단에 기여했는지를
    가중치로 볼 수 있다.
    """

    def __init__(self, n_features: int, params: dict):
        super().__init__()
        hidden = int(params.get("hidden_size", 128))
        layers = int(params.get("num_layers", 2))
        cell = {"lstm": nn.LSTM, "gru": nn.GRU}[str(params.get("cell", "gru")).lower()]
        self.rnn = cell(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=float(params.get("dropout", 0.2)) if layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        weights = torch.softmax(self.attention(out), dim=1)
        return self.head((out * weights).sum(dim=1)).squeeze(-1)


class FCNModel(TorchSequenceModel):
    net_class = FCNNet
    name = "fcn"


class ResNet1DModel(TorchSequenceModel):
    net_class = ResNet1DNet
    name = "resnet1d"


class InceptionTimeModel(TorchSequenceModel):
    net_class = InceptionTimeNet
    name = "inceptiontime"


class LSTMFCNModel(TorchSequenceModel):
    net_class = LSTMFCNNet
    name = "lstmfcn"


class AttentionRNNModel(TorchSequenceModel):
    net_class = AttentionRNNNet
    name = "attnrnn"
