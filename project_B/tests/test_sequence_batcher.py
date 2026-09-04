"""GPUWindowBatcher 가 WindowDataset 과 같은 창을 만드는지 확인한다.

창 조립을 GPU 인덱싱으로 옮기면서 패딩과 mask 규칙을 다시 구현했다. 두 경로가
어긋나면 시퀀스 모델의 입력이 조용히 달라지므로 여기서 직접 비교한다.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hddpred.models.sequence import GPUWindowBatcher, WindowDataset


def make_fold(n_rows=200, n_features=5, lookback=14, n_samples=64, padded=True, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    end_index = rng.integers(0, n_rows, size=n_samples)
    end_index = np.sort(end_index).astype(np.int64)
    y = rng.integers(0, 2, size=n_samples).astype(np.float32)
    valid_len = None
    if padded:
        # 실제 행 수는 1..lookback, 단 end 앞에 그만큼 행이 있어야 한다.
        valid_len = np.minimum(
            rng.integers(1, lookback + 1, size=n_samples), end_index + 1
        ).astype(np.int32)
    else:
        # 패딩이 없으면 lookback 을 다 채울 수 있는 위치만 쓴다.
        end_index = end_index[end_index >= lookback - 1]
        y = y[: len(end_index)]
    return SimpleNamespace(
        matrix=matrix, end_index=end_index, lookback=lookback, y=y, valid_len=valid_len
    )


@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize("lookback", [1, 7, 14, 30])
def test_batcher_matches_dataset(padded, lookback):
    fold = make_fold(lookback=lookback, padded=padded)
    device = torch.device("cpu")
    batcher = GPUWindowBatcher(fold, batch_size=16, device=device, shuffle=False)

    dataset = WindowDataset(
        fold.matrix, fold.end_index, fold.lookback, fold.y, fold.valid_len
    )
    expected = torch.stack([dataset[i][0] for i in range(len(dataset))])
    got = batcher.gather(torch.arange(len(dataset)))

    assert got.shape == expected.shape
    assert torch.equal(got, expected)


def test_labels_follow_selection():
    fold = make_fold()
    batcher = GPUWindowBatcher(fold, batch_size=8, device=torch.device("cpu"), shuffle=False)
    seen_x, seen_y = [], []
    for xb, yb in batcher:
        seen_x.append(xb)
        seen_y.append(yb)
    assert torch.equal(torch.cat(seen_y), torch.from_numpy(fold.y))
    assert torch.cat(seen_x).shape[0] == len(fold.end_index)


def test_shuffle_covers_every_sample_once():
    fold = make_fold(n_samples=100)
    batcher = GPUWindowBatcher(
        fold, batch_size=16, device=torch.device("cpu"), shuffle=True, seed=42
    )
    labels = torch.cat([yb for _x, yb in batcher])
    assert labels.shape[0] == len(fold.end_index)
    # 순서만 바뀌고 구성은 같아야 한다.
    assert torch.equal(labels.sort().values, torch.from_numpy(fold.y).sort().values)


def test_shuffle_order_changes_between_epochs():
    fold = make_fold(n_samples=100)
    batcher = GPUWindowBatcher(
        fold, batch_size=16, device=torch.device("cpu"), shuffle=True, seed=42
    )
    first = torch.cat([yb for _x, yb in batcher])
    second = torch.cat([yb for _x, yb in batcher])
    assert not torch.equal(first, second)
