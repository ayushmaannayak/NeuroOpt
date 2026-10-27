"""M2: train the embedding + LSTM throughput regressor on BHive labels."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.metrics import prediction_report  # noqa: E402
from neuroopt.model import Encoded, HierLSTM, predict, save  # noqa: E402
from neuroopt.tokenizer import Vocab  # noqa: E402


def load_split(arch: str, split_key: str, with_registers: bool):
    data = {"train": [], "val": [], "test": []}
    with gzip.open(ROOT / "data" / "processed" / f"bhive_{arch}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            s = r[split_key]
            if s in data:
                toks = r["instruction_tokens"]
                if not with_registers:
                    from neuroopt.decode import decode_block
                    from neuroopt.tokenizer import block_tokens
                    toks = block_tokens(decode_block(r["hex_bytes"]), with_registers=False)
                data[s].append((toks, r["target_y"]))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="skl")
    ap.add_argument("--split", default="split", choices=["split", "split_family"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training data (ablation)")
    ap.add_argument("--no-registers", action="store_true", help="tokenization ablation")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    data = load_split(args.arch, args.split, not args.no_registers)
    if args.frac < 1.0:
        keep = rng.permutation(len(data["train"]))[: int(len(data["train"]) * args.frac)]
        data["train"] = [data["train"][k] for k in keep]
    vocab = Vocab.build([t for t, _ in data["train"]])
    enc = {k: Encoded([t for t, _ in v], vocab, [y for _, y in v]) for k, v in data.items()}
    print(f"loaded in {time.time() - t0:.0f}s: " + ", ".join(f"{k}={len(v)}" for k, v in enc.items())
          + f", vocab={len(vocab)}", flush=True)

    model = HierLSTM(len(vocab), args.emb, args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(enc["train"].batches(args.batch, False))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    tr = enc["train"]
    log_y = torch.from_numpy(np.log(tr.y))
    history, best, best_state = [], float("inf"), None
    for ep in range(1, args.epochs + 1):
        model.train()
        t1, tot, nb = time.time(), 0.0, 0
        for idx in tr.batches(args.batch, True, rng):
            pred = model(*tr.batch(idx))
            loss = torch.nn.functional.l1_loss(pred, log_y[idx])   # |log p - log y| ~ relative error
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            if nb % 250 == 0:
                print(f"  epoch {ep} batch {nb}: loss {tot / nb:.4f} ({time.time() - t1:.0f}s)", flush=True)
        val = prediction_report(predict(model, enc["val"]), enc["val"].y)
        history.append(dict(epoch=ep, train_loss=tot / nb, **{f"val_{k}": v for k, v in val.items()},
                            seconds=time.time() - t1))
        print(f"epoch {ep:2d} loss {tot / nb:.4f}  val MAPE {val['mape']:.2f}%  "
              f"spearman {val['spearman']:.4f}  ({time.time() - t1:.0f}s)", flush=True)
        if val["mape"] < best:
            best = val["mape"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = prediction_report(predict(model, enc["test"]), enc["test"].y)
    meta = dict(vars(args), with_registers=not args.no_registers, n_train=len(tr),
                best_val_mape=best, test=test, history=history, minutes=(time.time() - t0) / 60)
    out = ROOT / "models" / f"lstm_{args.tag}.pt"
    save(model, vocab, out, meta)
    (ROOT / "results" / f"m2_train_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print("test:", json.dumps(test), "->", out)


if __name__ == "__main__":
    main()
