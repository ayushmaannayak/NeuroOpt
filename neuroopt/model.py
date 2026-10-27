"""Embedding + hierarchical LSTM throughput regressor (Ithemal-style), and the
LSTM cost oracle that plugs into the selector.

tokens of each instruction -> token LSTM -> instruction vector
instruction vectors        -> block LSTM -> linear -> log(cycles / iteration)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .decode import Instr, block_bytes
from .tokenizer import MAX_TOKENS, Vocab, block_tokens


class HierLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hidden: int = 128):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, emb=emb, hidden=hidden)
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.tok_lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.ins_lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, ids, ntok, ninstr):
        B, I, T = ids.shape
        mask = ntok > 0
        flat, lens = ids[mask], ntok[mask]                      # (N, T), (N,)
        out, _ = self.tok_lstm(self.emb(flat))
        h = out[torch.arange(flat.shape[0]), lens - 1]         # last real token
        inst = h.new_zeros(B, I, h.shape[-1])
        inst[mask] = h
        out2, _ = self.ins_lstm(inst)
        hb = out2[torch.arange(B), ninstr - 1]                 # last real instruction
        return self.head(hb).squeeze(-1)                       # log cycles/iteration


class Encoded:
    """Blocks pre-encoded as padded int arrays so batching is cheap."""

    def __init__(self, token_blocks: list[list[list[str]]], vocab: Vocab, y=None):
        self.n = np.array([len(b) for b in token_blocks], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.n)])
        rows = [ins for b in token_blocks for ins in vocab.encode(b)]
        self.ids = np.zeros((len(rows), MAX_TOKENS), dtype=np.int64)
        self.ntok = np.zeros(len(rows), dtype=np.int64)
        for k, ins in enumerate(rows):
            self.ids[k, :len(ins)] = ins
            self.ntok[k] = len(ins)
        self.y = None if y is None else np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.n)

    def batch(self, idx):
        idx = np.asarray(idx)
        I = int(self.n[idx].max())
        T = 1
        B = len(idx)
        ids = np.zeros((B, I, MAX_TOKENS), dtype=np.int64)
        ntok = np.zeros((B, I), dtype=np.int64)
        for j, k in enumerate(idx):
            a, b = self.offsets[k], self.offsets[k + 1]
            ids[j, : b - a] = self.ids[a:b]
            ntok[j, : b - a] = self.ntok[a:b]
        T = max(T, int(ntok.max()))
        return (torch.from_numpy(ids[:, :, :T]), torch.from_numpy(ntok), torch.from_numpy(self.n[idx]))

    def batches(self, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
        """Length-bucketed batches: sort by size, chunk, optionally shuffle chunks."""
        order = np.argsort(self.n, kind="stable")
        if shuffle:
            # jitter within similar lengths so batches differ between epochs
            noise = rng.random(len(order))
            order = np.lexsort((noise, self.n))
        chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        if shuffle:
            rng.shuffle(chunks)
        return chunks


@torch.no_grad()
def predict(model: HierLSTM, data: Encoded, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = np.zeros(len(data), dtype=np.float64)
    for idx in data.batches(batch_size, shuffle=False):
        out[idx] = torch.exp(model(*data.batch(idx))).numpy()
    return out


def save(model: HierLSTM, vocab: Vocab, path: Path, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": model.cfg, "vocab": vocab.itos, "meta": meta}, path)


def load(path: Path) -> tuple[HierLSTM, Vocab, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = HierLSTM(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, Vocab(ck["vocab"]), ck.get("meta", {})


class LSTMOracle:
    """Predicted cycles/iteration from the trained model. Drop-in replacement
    for the static table: same ``score`` interface, nothing else changes."""

    def __init__(self, path: Path):
        self.model, self.vocab, self.meta = load(path)
        self.with_registers = self.meta.get("with_registers", True)
        self.name = "lstm"
        self._cache: dict[bytes, float] = {}
        torch.set_num_threads(max(1, torch.get_num_threads()))

    def score(self, block: list[Instr]) -> float:
        return self.score_many([block])[0]

    def score_many(self, blocks: list[list[Instr]]) -> list[float]:
        keys = [block_bytes(b) for b in blocks]
        todo = [(k, b) for k, b in zip(keys, blocks) if k not in self._cache and b]
        if todo:
            enc = Encoded([block_tokens(b, self.with_registers) for _, b in todo], self.vocab)
            for (k, _), v in zip(todo, predict(self.model, enc)):
                self._cache[k] = float(v)
        return [self._cache.get(k, 0.0) for k in keys]


def write_json(path: Path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))
