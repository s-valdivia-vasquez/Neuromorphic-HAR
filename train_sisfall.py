import os
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.spike import SCN6_2


def get_args():
    p = argparse.ArgumentParser(
        description="Train one dual-head spiking CNN from cached event-based IMU data."
    )

    # Paths
    p.add_argument("--data-dir", default="data", help="Folder with train/val/test splits.")
    p.add_argument("--out-dir", default="runs", help="Folder where results are saved.")
    p.add_argument("--run", default="scn6_train", help="Run name inside out-dir.")

    # Execution
    p.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--wd", type=float, default=1e-4, help="Adam weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    p.add_argument("--plateau-patience", type=int, default=5, help="LR scheduler patience.")
    p.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor.")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")

    # Loss
    p.add_argument("--lambda-h1", type=float, default=1.0, help="Weight for global head loss.")
    p.add_argument("--lambda-h2", type=float, default=0.5, help="Weight for static-refinement loss.")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy smoothing for Head 1.")
    p.add_argument("--static-idx", type=int, default=2, help="Static class index in Head 1 labels.")
    p.add_argument("--use-ambiguous", action="store_true", help="Use ambiguous static windows in Head 2.")

    # LIF neuron
    p.add_argument("--tau", type=float, default=0.75, help="LIF membrane decay.")
    p.add_argument("--thresh", type=float, default=0.5, help="LIF firing threshold.")
    p.add_argument("--hard-reset", action="store_true", help="Use hard reset instead of soft reset.")

    # Input
    p.add_argument("--n-ch", type=int, default=6, help="Number of IMU channels.")
    p.add_argument("--n-classes", type=int, default=3, help="Number of Head 1 classes.")

    # Event branch
    p.add_argument("--conv-ch", type=int, nargs=3, default=[32, 64, 64], help="Channels of the three conv blocks.")
    p.add_argument("--kernels", type=int, nargs=3, default=[32, 8, 8], help="Conv1D kernel sizes.")
    p.add_argument("--strides", type=int, nargs=3, default=[8, 2, 1], help="Conv1D strides.")
    p.add_argument("--pool-kernels", type=int, nargs=3, default=[2, 2, 2], help="MaxPool1D window sizes.")
    p.add_argument("--pool-strides", type=int, nargs=3, default=[2, 2, 2], help="MaxPool1D strides.")
    p.add_argument("--pool-paddings", type=int, nargs=3, default=[0, 0, 0], help="MaxPool1D paddings.")
    p.add_argument("--p-drop", type=float, default=0.35, help="Dropout in the event branch.")
    p.add_argument("--merge-polarities", action="store_true", help="Merge positive and negative event polarities.")

    # Offset branch
    p.add_argument("--offset-hidden", type=int, default=8, help="Hidden units in the offset branch.")
    p.add_argument("--head-rate-scale", type=float, default=8.0, help="Scale applied before readout heads.")

    return p.parse_args()


class EventSet(Dataset):
    def __init__(self, d, n_ch=6):
        self.x = torch.from_numpy(np.asarray(d["x_ev"], dtype=np.float32, order="C"))
        self.o = torch.from_numpy(np.asarray(d["offset"], dtype=np.float32, order="C"))
        self.y1 = torch.from_numpy(np.asarray(d["y_h1"], dtype=np.int64))
        self.y2 = torch.from_numpy(np.asarray(d["y_h2"], dtype=np.float32))
        self.w2 = torch.from_numpy(np.asarray(d["w_h2"], dtype=np.float32))

        n = len(self.y1)
        if self.x.shape[0] != n:
            raise ValueError(f"x_ev and y_h1 have different lengths: {self.x.shape[0]} != {n}")
        if self.o.shape != (n, n_ch):
            raise ValueError(f"offset must have shape ({n}, {n_ch}), got {tuple(self.o.shape)}")

    def __len__(self):
        return len(self.y1)

    def __getitem__(self, i):
        return {
            "x": self.x[i],
            "offset": self.o[i],
            "y1": self.y1[i],
            "y2": self.y2[i],
            "w2": self.w2[i],
        }


def load_split(root, split):
    p = os.path.join(root, split)
    names = ["x_ev.npy", "offset.npy", "y_h1.npy", "y_h2.npy", "w_h2.npy"]

    if not os.path.isdir(p):
        raise FileNotFoundError(f"Missing split directory: {p}")

    for name in names:
        fp = os.path.join(p, name)
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing file: {fp}")

    return {
        "x_ev": np.load(os.path.join(p, "x_ev.npy"), mmap_mode="r"),
        "offset": np.load(os.path.join(p, "offset.npy")),
        "y_h1": np.load(os.path.join(p, "y_h1.npy")),
        "y_h2": np.load(os.path.join(p, "y_h2.npy")),
        "w_h2": np.load(os.path.join(p, "w_h2.npy")),
    }


def pos_weight_h2(d, static_idx):
    y1 = np.asarray(d["y_h1"])
    y2 = np.asarray(d["y_h2"])
    m = (y1 == static_idx) & (~np.isclose(y2, 0.5))

    if not np.any(m):
        return 1.0

    pos = int(np.sum(np.isclose(y2[m], 1.0)))
    neg = int(np.sum(np.isclose(y2[m], 0.0)))

    return 1.0 if pos == 0 else float(neg / max(pos, 1))


def get_out(model, x, off):
    y = model(x, offset=off)
    return y[0] if isinstance(y, tuple) else y


def loss_fn(out, y1, y2, w2, cfg, pw):
    z1 = out["head1"]
    z2 = out["head2"]

    if z2.ndim > 1 and z2.shape[-1] == 1:
        z2 = z2.squeeze(-1)

    l1 = F.cross_entropy(
        z1,
        y1,
        label_smoothing=cfg.label_smoothing,
    )

    m = y1 == cfg.static_idx

    if not cfg.use_ambiguous:
        m &= ~torch.isclose(y2, torch.tensor(0.5, device=y2.device, dtype=y2.dtype))

    m &= w2 > 0

    if int(m.sum().item()) > 0:
        pwt = torch.tensor(pw, device=z2.device, dtype=z2.dtype)
        bce = F.binary_cross_entropy_with_logits(
            z2[m],
            y2[m],
            reduction="none",
            pos_weight=pwt,
        )
        l2 = (bce * w2[m]).sum() / (w2[m].sum() + 1e-8)
    else:
        l2 = z2.sum() * 0.0

    return cfg.lambda_h1 * l1 + cfg.lambda_h2 * l2, l1, l2


def run_epoch(model, dl, opt, dev, cfg, pw, scaler=None):
    train = opt is not None
    model.train(train)

    loss_sum = 0.0
    l1_sum = 0.0
    l2_sum = 0.0
    h1_ok = 0
    h2_ok = 0
    n = 0
    n_h2 = 0

    amp = bool(cfg.amp and dev.type == "cuda")

    for b in dl:
        x = b["x"].to(dev, non_blocking=True).float()
        off = b["offset"].to(dev, non_blocking=True).float()
        y1 = b["y1"].to(dev, non_blocking=True).long()
        y2 = b["y2"].to(dev, non_blocking=True).float()
        w2 = b["w2"].to(dev, non_blocking=True).float()

        if train:
            opt.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp):
                out = get_out(model, x, off)
                loss, l1, l2 = loss_fn(out, y1, y2, w2, cfg, pw)

            if train:
                scaler.scale(loss).backward()

                if cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                scaler.step(opt)
                scaler.update()

        bs = x.size(0)
        n += bs

        loss_sum += float(loss.detach().item()) * bs
        l1_sum += float(l1.detach().item()) * bs
        l2_sum += float(l2.detach().item()) * bs

        h1_ok += int((out["head1"].argmax(1) == y1).sum().item())

        z2 = out["head2"]
        if z2.ndim > 1 and z2.shape[-1] == 1:
            z2 = z2.squeeze(-1)

        m = y1 == cfg.static_idx
        if not cfg.use_ambiguous:
            m &= ~torch.isclose(y2, torch.tensor(0.5, device=y2.device, dtype=y2.dtype))
        m &= w2 > 0

        if int(m.sum().item()) > 0:
            pred2 = (torch.sigmoid(z2[m]) >= 0.5).float()
            h2_ok += int((pred2 == y2[m]).sum().item())
            n_h2 += int(m.sum().item())

    return {
        "loss": loss_sum / max(n, 1),
        "loss_h1": l1_sum / max(n, 1),
        "loss_h2": l2_sum / max(n, 1),
        "acc_h1": h1_ok / max(n, 1),
        "acc_h2": h2_ok / max(n_h2, 1) if n_h2 > 0 else float("nan"),
        "n": n,
        "n_h2": n_h2,
    }


def save_json(x, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(x, f, indent=2)


def main():
    cfg = get_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{cfg.gpu}")
        torch.cuda.set_device(dev)
        torch.cuda.manual_seed_all(cfg.seed)
    else:
        dev = torch.device("cpu")

    run_dir = os.path.join(cfg.out_dir, cfg.run)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    met_dir = os.path.join(run_dir, "metrics")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    save_json(vars(cfg), os.path.join(run_dir, "config.json"))

    # Load data
    tr = load_split(cfg.data_dir, "train")
    va = load_split(cfg.data_dir, "val")
    te = load_split(cfg.data_dir, "test")

    print(f"train: {len(tr['y_h1'])}")
    print(f"val  : {len(va['y_h1'])}")
    print(f"test : {len(te['y_h1'])}")

    pw = pos_weight_h2(tr, cfg.static_idx)
    print(f"head2_pos_weight: {pw:.4f}")

    # Build loaders
    pin = dev.type == "cuda"

    dl_tr = DataLoader(
        EventSet(tr, cfg.n_ch),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin,
        drop_last=False,
    )

    dl_va = DataLoader(
        EventSet(va, cfg.n_ch),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=pin,
        drop_last=False,
    )

    dl_te = DataLoader(
        EventSet(te, cfg.n_ch),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=pin,
        drop_last=False,
    )

    # Build model
    model = SCN6_2(
        n_channels=cfg.n_ch,
        n_classes_head1=cfg.n_classes,
        backbone=False,
        tau=cfg.tau,
        thresh=cfg.thresh,
        soft_reset=not cfg.hard_reset,
        merge_polarities=cfg.merge_polarities,
        event_scale=1.0,
        len_sw=None,
        time_steps=None,
        out_channels=64,
        conv_channels=tuple(cfg.conv_ch),
        use_bn=True,
        p_drop=cfg.p_drop,
        bias=False,
        conv_kernel_sizes=tuple(cfg.kernels),
        conv_strides=tuple(cfg.strides),
        conv_paddings=None,
        pool_kernel_sizes=tuple(cfg.pool_kernels),
        pool_strides=tuple(cfg.pool_strides),
        pool_paddings=tuple(cfg.pool_paddings),
        offset_dim=cfg.n_ch,
        offset_hidden=cfg.offset_hidden,
        offset_scale=1.0,
        offset_drop=0.0,
        offset_use_bn=True,
        spiking_heads=False,
        head_rate_scale=cfg.head_rate_scale,
    ).to(dev)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"device: {dev}")
    print(f"params: {n_par:,}")

    # Train model
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.plateau_patience,
        min_lr=cfg.min_lr,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and dev.type == "cuda"))

    hist = []
    best = float("inf")
    bad = 0
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        ts = time.time()

        mt = run_epoch(model, dl_tr, opt, dev, cfg, pw, scaler=scaler)
        mv = run_epoch(model, dl_va, None, dev, cfg, pw)

        sch.step(mv["loss"])
        lr = opt.param_groups[0]["lr"]

        row = {
            "epoch": ep,
            "lr": lr,
            "train": mt,
            "val": mv,
            "time_sec": time.time() - ts,
            "end": datetime.now().strftime("%H:%M:%S"),
        }

        hist.append(row)
        save_json(hist, os.path.join(met_dir, "history.json"))

        if mv["loss"] < best:
            best = mv["loss"]
            bad = 0

            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best,
                    "config": vars(cfg),
                    "history": hist,
                },
                ckpt_path,
            )

            ckpt_msg = " | ckpt"
        else:
            bad += 1
            ckpt_msg = ""

        h2_tr = f"{mt['acc_h2']:.4f}" if np.isfinite(mt["acc_h2"]) else "NA"
        h2_va = f"{mv['acc_h2']:.4f}" if np.isfinite(mv["acc_h2"]) else "NA"

        print(
            f"[{ep:03d}/{cfg.epochs:03d}] "
            f"tr_loss={mt['loss']:.4f} tr_h1={mt['acc_h1']:.4f} tr_h2={h2_tr} | "
            f"va_loss={mv['loss']:.4f} va_h1={mv['acc_h1']:.4f} va_h2={h2_va} | "
            f"lr={lr:.2e}{ckpt_msg}"
        )

        if bad >= cfg.patience:
            print(f"early_stop: epoch={ep}, best_val_loss={best:.4f}")
            break

    # Test model
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])

    ms = run_epoch(model, dl_te, None, dev, cfg, pw)
    ms["best_val_loss"] = best
    ms["total_time_sec"] = time.time() - t0

    save_json(ms, os.path.join(met_dir, "test_metrics.json"))

    print("test")
    print(json.dumps(ms, indent=2))


if __name__ == "__main__":
    main()