"""Single-layer multi-task LSTM (PyTorch) over a short window of daily channels.

One network with three heads predicts log(vol_fwd/vol_21), drawdown logit and the standardised
5% return quantile. Windows are sliced from seq.npy by (pid, day_idx), so rows are exactly the
panel rows every other model sees.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.models import prereg
from src.models.base import empty_preds, var_scale

WINDOW = 21
STATIC_R3 = ("naive_vol", "ewma_vol")  # Round 3 static inputs (and the old-checkpoint default)
STATIC_R4 = STATIC_R3 + tuple(prereg.MODEL_FEATURES)  # Round 4 (B2d): + HAR and GARCH forecasts


def window_indices(day_idx: torch.Tensor, window: int = WINDOW) -> torch.Tensor:
    """Row positions of each window; the LAST element is day t itself (no look-ahead)."""
    return day_idx[:, None] + torch.arange(-window + 1, 1, device=day_idx.device)


def pinball_t(y: torch.Tensor, q: torch.Tensor, alpha: float = config.VAR_ALPHA) -> torch.Tensor:
    """Mean pinball loss for tensors."""
    d = y - q
    return torch.maximum(alpha * d, (alpha - 1) * d).mean()


class _Net(nn.Module):
    def __init__(self, n_channels: int, n_static: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[-1], s], dim=1))


class LSTMModel:
    """LSTM baseline-beater; trains with Adam, early stopping on the inner validation block."""

    name = "lstm"

    def __init__(self, seq: np.ndarray | None = None, max_epochs: int = 30, patience: int = 5,
                 batch_size: int = 1024, hidden: int = 32, lr: float = 1e-3,
                 seed: int = config.SEED, static_cols: tuple[str, ...] = STATIC_R3):
        self.seq_np = seq if seq is not None else np.load(config.DATA_PROCESSED / "seq.npy")
        self.max_epochs, self.patience = max_epochs, patience
        self.batch_size, self.hidden, self.lr, self.seed = batch_size, hidden, lr, seed
        self.static_cols = tuple(static_cols)  # log-transformed vol inputs of the head
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # channels: port_ret, mkt_ret, |port_ret|, VIX/100, VIX_chg
    def _channels(self, seq: np.ndarray) -> np.ndarray:
        r, m, vix, dv = seq[..., 0], seq[..., 1], seq[..., 2], seq[..., 3]
        return np.stack([r, m, np.abs(r), vix / 100.0, dv], axis=-1).astype(np.float32)

    def _static(self, df: pd.DataFrame) -> np.ndarray:
        s = np.log(np.maximum(df[list(self.static_cols)].to_numpy(dtype=float), 1e-3))
        return ((s - self.s_mu) / self.s_sd).astype(np.float32)

    def _batch(self, pid, day, static):
        idx = window_indices(day)
        x = (self.X[pid[:, None], idx] - self.mu) / self.sd
        return x, static

    def _tensors(self, df: pd.DataFrame, with_y: bool):
        pid_np = df.index.get_level_values("pid").to_numpy().copy()
        pid = torch.as_tensor(pid_np, device=self.device)
        day = torch.as_tensor(df["day_idx"].to_numpy().astype(np.int64), device=self.device)
        static = torch.as_tensor(self._static(df), device=self.device)
        if not with_y:
            return pid, day, static, None
        scale = var_scale(df["ewma_vol"].to_numpy())
        y = np.column_stack([(df["y_logratio"] - self.y_mu) / self.y_sd, df["dd_flag"],
                             df["r_next"] / scale]).astype(np.float32)
        return pid, day, static, torch.as_tensor(y, device=self.device)

    def _loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = nn.functional.mse_loss(out[:, 0], y[:, 0])
        bce = nn.functional.binary_cross_entropy_with_logits(out[:, 1], y[:, 1])
        return mse + bce + 5.0 * pinball_t(y[:, 2], out[:, 2])

    def _epoch_losses(self, data) -> np.ndarray:
        """Mean validation loss of each head: (MSE vol, BCE drawdown, pinball VaR)."""
        pid, day, static, y = data
        self.net.eval()
        total = np.zeros(3)
        with torch.no_grad():
            for i in range(0, len(pid), 8192):
                sl = slice(i, i + 8192)
                x, s = self._batch(pid[sl], day[sl], static[sl])
                o = self.net(x, s)
                part = (nn.functional.mse_loss(o[:, 0], y[sl][:, 0]),
                        nn.functional.binary_cross_entropy_with_logits(o[:, 1], y[sl][:, 1]),
                        pinball_t(y[sl][:, 2], o[:, 2]))
                total += np.array([t.item() for t in part]) * len(y[sl])
        return total / len(pid)

    def _setup(self, train: pd.DataFrame) -> int:
        """Fit scalers on `train` rows only and stage the sequence tensor; returns n_channels."""
        chan = self._channels(self.seq_np)
        last = int(train["day_idx"].max())  # statistics from the training period only
        pids = np.unique(train.index.get_level_values("pid").to_numpy())  # and training pids
        flat = chan[pids, : last + 1].reshape(-1, chan.shape[-1])
        self.mu = torch.as_tensor(flat.mean(0), device=self.device)
        self.sd = torch.as_tensor(flat.std(0) + 1e-8, device=self.device)
        self.X = torch.as_tensor(chan, device=self.device)
        s = np.log(np.maximum(train[list(self.static_cols)].to_numpy(dtype=float), 1e-3))
        self.s_mu, self.s_sd = s.mean(0), s.std(0) + 1e-8
        self.y_mu, self.y_sd = float(train["y_logratio"].mean()), float(train["y_logratio"].std())
        return chan.shape[-1]

    def _train(self, train: pd.DataFrame, val: pd.DataFrame | None = None,
               epochs: int | None = None, snap_at: list[int] | None = None) -> None:
        """Train a fresh network on the summed multi-task loss.

        With `val`: each head keeps the weights of ITS best validation epoch (the heads share a
        trunk but their curves differ: a composite loss let the vol and drawdown heads decide
        for the VaR head), training stops once no head has improved for `patience` epochs, and
        `best_epochs` records the three epochs. Without: exactly `epochs` epochs, snapshotting
        head k's weights after epoch `snap_at[k]` (used by refit).
        """
        config.set_seeds(self.seed)
        torch.backends.cudnn.deterministic = True
        n_ch = self._setup(train)
        tr = self._tensors(train, True)
        va = self._tensors(val, True) if val is not None else None
        self.net = _Net(n_ch, len(self.static_cols), self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        best, states, best_ep, bad = np.full(3, np.inf), [None] * 3, [0] * 3, 0
        n = len(tr[0])
        for ep in range(epochs if epochs is not None else self.max_epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            for i in range(0, n, self.batch_size):
                b = perm[i : i + self.batch_size]
                x, s_ = self._batch(tr[0][b], tr[1][b], tr[2][b])
                opt.zero_grad()
                self._loss(self.net(x, s_), tr[3][b]).backward()
                opt.step()
            if va is not None:
                v = self._epoch_losses(va)
                improved = v < best - 1e-5
                snap = copy.deepcopy(self.net.state_dict()) if improved.any() else None
                for k in np.flatnonzero(improved):
                    best[k], best_ep[k], states[k] = v[k], ep + 1, snap
                bad = 0 if improved.any() else bad + 1
                if bad >= self.patience:
                    break
            elif snap_at is not None:
                if ep + 1 in snap_at:
                    snap = copy.deepcopy(self.net.state_dict())
                    for k, e in enumerate(snap_at):
                        if e == ep + 1:
                            states[k] = snap
        if va is not None and any(st is None for st in states):
            raise RuntimeError("LSTM validation loss never improved (NaN or inf); "
                               "check the inputs")
        if va is not None:
            self.best_epochs = best_ep
        self.states = states

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Fit scalers on the training period only, then train with per-head early stopping."""
        assert val is not None, "LSTMModel needs a validation block for early stopping"
        self._train(train, val)

    def refit(self, train: pd.DataFrame) -> None:
        """Retrain from scratch on `train` (fit + val), running as many epochs as the slowest
        head needed and keeping each head's weights from its own chosen epoch.

        The epoch counts are not rescaled for the larger set.
        """
        self._train(train, None, epochs=max(self.best_epochs), snap_at=list(self.best_epochs))

    def _heads(self, x: torch.Tensor, s: torch.Tensor) -> np.ndarray:
        """Raw head outputs (n, 3): head k comes from the network state saved for head k."""
        self.net.eval()
        out = np.empty((len(x), 3))
        by_state: dict[int, list[int]] = {}
        for k, st in enumerate(self.states):
            by_state.setdefault(id(st), []).append(k)
        with torch.no_grad():
            for heads in by_state.values():
                self.net.load_state_dict(self.states[heads[0]])
                for i in range(0, len(x), 8192):
                    o = self.net(x[i : i + 8192], s[i : i + 8192]).cpu().numpy()
                    out[i : i + 8192, heads] = o[:, heads]
        return out

    def _to_preds(self, r: np.ndarray, naive_vol: np.ndarray, ewma_vol: np.ndarray, index):
        out = empty_preds(index)
        out["vol_hat"] = naive_vol * np.exp(r[:, 0] * self.y_sd + self.y_mu)
        out["dd_hat"] = 1 / (1 + np.exp(-r[:, 1]))
        out["var_hat"] = r[:, 2] * var_scale(ewma_vol)
        return out

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict vol_hat, dd_hat and var_hat for panel rows (windows read from seq.npy)."""
        pid, day, static, _ = self._tensors(X, False)
        x, s = self._batch(pid, day, static)
        r = self._heads(x, s)
        return self._to_preds(r, X["naive_vol"].to_numpy(), X["ewma_vol"].to_numpy(dtype=float),
                              X.index)

    def predict_window(self, window: np.ndarray, static: dict[str, float]) -> dict:
        """Predict for one arbitrary portfolio from its last WINDOW days of (port_ret, mkt_ret,
        VIX, VIX_chg), shape (WINDOW, 4), and its static vol inputs {column: value} (at least
        this model's `static_cols`). Used by the app for user portfolios."""
        chan = torch.as_tensor(self._channels(window[None]), device=self.device)
        x = (chan - self.mu) / self.sd
        df = pd.DataFrame({c: [float(static[c])] for c in self.static_cols})
        s = torch.as_tensor(self._static(df), device=self.device)
        out = self._to_preds(self._heads(x, s), np.array([float(static["naive_vol"])]),
                             np.array([float(static["ewma_vol"])]), pd.RangeIndex(1))
        return out.iloc[0].to_dict()

    def save(self, path) -> None:
        """Save network weights and the train-only scalers (not the raw sequence data)."""
        torch.save(
            {"states": self.states, "best_epochs": list(getattr(self, "best_epochs", [])),
             "mu": self.mu.cpu(), "sd": self.sd.cpu(),
             "s_mu": self.s_mu, "s_sd": self.s_sd, "y_mu": self.y_mu, "y_sd": self.y_sd,
             "hidden": self.hidden, "n_channels": int(self.mu.numel()),
             "static_cols": list(self.static_cols)}, path)

    @classmethod
    def load(cls, path) -> LSTMModel:
        """Rebuild a model for prediction from `save()` output."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cols = tuple(ck.get("static_cols", STATIC_R3))  # pre-Round-4 files had two inputs
        m = cls(seq=np.zeros((1, 1, 4), dtype=np.float32), hidden=ck["hidden"], static_cols=cols)
        m.mu, m.sd = ck["mu"].to(m.device), ck["sd"].to(m.device)
        m.s_mu, m.s_sd, m.y_mu, m.y_sd = ck["s_mu"], ck["s_sd"], ck["y_mu"], ck["y_sd"]
        m.net = _Net(ck["n_channels"], len(cols), ck["hidden"]).to(m.device)
        m.states = ck["states"] if "states" in ck else [ck["state"]] * 3  # pre-round-3 files
        m.best_epochs = list(ck.get("best_epochs", []))
        return m
