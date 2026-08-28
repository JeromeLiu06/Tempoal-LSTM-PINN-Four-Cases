
import os
import csv
import json
import math
import time
import platform
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# Generic utilities
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 ** 2)


def get_cpu_name() -> str:
    name = platform.processor()
    if name:
        return name
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def get_hardware_info(device: str) -> Dict[str, Any]:
    info = {
        "device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_name": get_cpu_name(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.version.cuda is not None else "None",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": "None",
        "gpu_total_memory_MB": 0.0,
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        idx = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = p.name
        info["gpu_total_memory_MB"] = p.total_memory / (1024.0 ** 2)
    return info


def peak_gpu_memory(device: str) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.startswith("cuda"):
        return (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            torch.cuda.max_memory_reserved() / (1024.0 ** 2),
        )
    return 0.0, 0.0


def sample_uniform(n: int, low: float, high: float, device: str, dtype: torch.dtype):
    return low + (high - low) * torch.rand(n, 1, device=device, dtype=dtype)


def split_indices(n: int, ratio: float, seed: int, device: str):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    p = torch.randperm(n, generator=g, device=device)
    ntr = max(1, min(int(ratio * n), n - 1))
    return p[:ntr], p[ntr:]


def deterministic_minibatch(bundle: Tuple[torch.Tensor, ...], batch: int, seed: int):
    n = bundle[0].shape[0]
    g = torch.Generator(device=bundle[0].device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=bundle[0].device)[:min(batch, n)]
    return tuple(a[idx] for a in bundle)


def compute_metrics_np(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred).reshape(-1)
    ref = np.asarray(ref).reshape(-1)
    diff = pred - ref
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))
    rel_l2 = float(l2 / (np.linalg.norm(ref) + 1e-12))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "L2": l2, "RelL2": rel_l2,
        "MaxAbsError": max_abs, "MeanAbsError": mean_abs,
    }


def compute_metrics_torch(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    diff = pred.detach() - ref.detach()
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse)
    l2 = torch.linalg.norm(diff.reshape(-1))
    rel = l2 / (torch.linalg.norm(ref.detach().reshape(-1)) + 1e-12)
    return {
        "MSE": float(mse.item()),
        "RMSE": float(rmse.item()),
        "MAE": float(torch.mean(torch.abs(diff)).item()),
        "L2": float(l2.item()),
        "RelL2": float(rel.item()),
        "MaxAbsError": float(torch.max(torch.abs(diff)).item()),
        "MeanAbsError": float(torch.mean(torch.abs(diff)).item()),
    }


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None):
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_history_csv(path: str, history: Dict[str, List[Any]]):
    ensure_dir(os.path.dirname(path))
    keys = list(history.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(history[keys[0]]) if keys else 0
        for i in range(n):
            w.writerow([history[k][i] for k in keys])


def save_json(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def time_tag(t: float) -> str:
    return f"t{str(round(float(t), 4)).replace('.', 'p')}"


# =========================
# Corrected m=4 sequential architecture
# =========================

class LSTMPINNModel(nn.Module):
    def __init__(self, out_dim: int, hidden: int, lstm_layers: int):
        super().__init__()
        self.embed = nn.Linear(3, hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        z = self.embed(seq)
        with torch.backends.cudnn.flags(enabled=False):
            z, _ = self.lstm(z)
        return self.head(z)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # The last token is the current target point; previous tokens supply context.
        return self.forward_sequence(seq)[:, -1, :]


def grad_last(u: torch.Tensor, seq: torch.Tensor, coord: int) -> torch.Tensor:
    """
    Derivative of the TARGET output w.r.t. the TARGET token coordinate.
    This avoids mixing derivatives of different sequence positions.
    """
    g = torch.autograd.grad(
        u, seq,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    return g[:, -1, coord:coord+1]


def build_temporal_context(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    m: int,
    dt: float,
    t0: float,
) -> torch.Tensor:
    # Same spatial point, monotonically increasing time history ending at t.
    offsets = torch.arange(m - 1, -1, -1, device=x.device, dtype=x.dtype).view(1, m, 1) * dt
    th = torch.clamp(t[:, None, :] - offsets, min=t0)
    return torch.cat(
        [
            x[:, None, :].expand(-1, m, -1),
            y[:, None, :].expand(-1, m, -1),
            th,
        ],
        dim=2,
    )


def _part1by1(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint32)
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def morton_order_unit_square(x: torch.Tensor, y: torch.Tensor, bits: int = 16) -> torch.Tensor:
    xn = np.clip(x.detach().cpu().numpy().reshape(-1), 0.0, 1.0)
    yn = np.clip(y.detach().cpu().numpy().reshape(-1), 0.0, 1.0)
    scale = float((1 << bits) - 1)
    xi = (xn * scale).astype(np.uint32)
    yi = (yn * scale).astype(np.uint32)
    key = _part1by1(xi) | (_part1by1(yi) << 1)
    return torch.as_tensor(np.argsort(key), device=x.device, dtype=torch.long)


def build_spatial_contexts(x: torch.Tensor, y: torch.Tensor, m: int) -> torch.Tensor:
    """
    Steady-case pseudo-spatial context.
    Every original point is retained as a target exactly once.
    """
    order = morton_order_unit_square(x, y)
    xs, ys = x[order], y[order]
    n = xs.shape[0]
    ids = torch.arange(n, device=x.device)
    offs = torch.arange(m - 1, -1, -1, device=x.device)
    ctx = torch.clamp(ids[:, None] - offs[None, :], min=0)
    seq_sorted = torch.stack(
        [xs[ctx, 0], ys[ctx, 0], torch.zeros_like(xs[ctx, 0])],
        dim=2,
    )
    inv = torch.empty_like(order)
    inv[order] = torch.arange(n, device=x.device)
    return seq_sorted[inv]


def make_output_dirs(out_dir: str):
    for sub in [
        "", "field_data", "metrics", "cost", "loss",
        "figures/reference", "figures/prediction",
        "figures/error", "figures/loss",
    ]:
        ensure_dir(os.path.join(out_dir, sub))


def save_map(data: np.ndarray, extent, title: str, path: str):
    plt.figure(figsize=(5.6, 4.6))
    plt.imshow(data.T, origin="lower", extent=extent, aspect="auto")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_loss_plots(out_dir: str, history: Dict[str, List[Any]]):
    if not history.get("iter"):
        return
    specs = [
        ("loss_total_train", "loss_total_val", "Total loss", "loss_total_curve.png"),
        ("loss_pde_train", "loss_pde_val", "PDE loss", "loss_pde_curve.png"),
        ("loss_mom_train", "loss_mom_val", "Momentum loss", "loss_mom_curve.png"),
        ("loss_div_train", "loss_div_val", "Divergence loss", "loss_div_curve.png"),
        ("loss_ic_train", "loss_ic_val", "IC loss", "loss_ic_curve.png"),
        ("loss_bc_train", "loss_bc_val", "BC loss", "loss_bc_curve.png"),
        ("loss_data_train", "loss_data_val", "Data loss", "loss_data_curve.png"),
        ("RelL2_val", None, "Validation RelL2", "rel_l2_curve.png"),
    ]
    for tr, va, title, fn in specs:
        if tr not in history or not history[tr]:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(history["iter"], history[tr], label=tr)
        if va is not None and va in history:
            plt.plot(history["iter"], history[va], label=va)
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figures", "loss", fn), dpi=220)
        plt.close()


def save_model_info(cfg, model: nn.Module, output_fields: str, sequence_note: str):
    path = os.path.join(cfg.out_dir, "model_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Case: {cfg.case_name}\n")
        f.write(f"Model: {cfg.model_name}\n")
        f.write("Architecture: Linear(3,H) + 2-layer LSTM + MLP head\n")
        f.write(f"Input token: complete coordinate vector (x,y,t*)\n")
        f.write(f"Sequence length: {cfg.sequence_length}\n")
        f.write(f"Sequence semantics: {sequence_note}\n")
        f.write(f"Output fields: {output_fields}\n")
        f.write(f"Hidden size: {cfg.hidden}\n")
        f.write(f"LSTM layers: {cfg.lstm_layers}\n")
        f.write(f"Trainable parameters: {count_trainable_parameters(model)}\n")
        f.write(f"Total parameters: {count_total_parameters(model)}\n")
        f.write(f"Parameter size MB: {parameter_size_mb(model):.6f}\n")
        f.write("Optimizer: Adam\n")
        f.write(f"Learning rate: {cfg.lr}\n")
        f.write(f"Iterations: {cfg.iters}\n")
        f.write(f"Seed: {cfg.seed}\n")


def save_cost_outputs(
    cfg, model, history, iter_times, total_wall_time, best_iter, best_rel_l2, final_rel_l2
):
    hw = get_hardware_info(cfg.device)
    peak_alloc, peak_reserved = peak_gpu_memory(cfg.device)
    row = {
        "case": cfg.case_name,
        "model": cfg.model_name,
        "device": cfg.device,
        "trainable_parameters": count_trainable_parameters(model),
        "total_parameters": count_total_parameters(model),
        "parameter_size_MB": parameter_size_mb(model),
        "total_iterations_configured": cfg.iters,
        "total_iterations_completed": history["iter"][-1] if history["iter"] else 0,
        "total_wall_time_sec": total_wall_time,
        "total_wall_time_min": total_wall_time / 60.0,
        "mean_time_per_iter_sec": float(np.mean(iter_times)) if iter_times else 0.0,
        "median_time_per_iter_sec": float(np.median(iter_times)) if iter_times else 0.0,
        "gpu_name": hw["gpu_name"],
        "gpu_count": hw["gpu_count"],
        "cuda_available": hw["cuda_available"],
        "cuda_version": hw["cuda_version"],
        "torch_version": hw["torch_version"],
        "cpu_name": hw["cpu_name"],
        "gpu_total_memory_MB": hw["gpu_total_memory_MB"],
        "peak_gpu_memory_allocated_MB": peak_alloc,
        "peak_gpu_memory_reserved_MB": peak_reserved,
        "best_iter": best_iter,
        "best_RelL2": best_rel_l2,
        "final_RelL2": final_rel_l2,
        "sequence_length": cfg.sequence_length,
    }
    write_csv(os.path.join(cfg.out_dir, "cost", "model_cost_summary.csv"), [row])

    rows = []
    for th in cfg.rel_l2_thresholds:
        reached = False
        ii = -1
        tt = -1.0
        for it, rel, tm in zip(
            history.get("iter", []),
            history.get("RelL2_val", []),
            history.get("time_elapsed_sec", []),
        ):
            if np.isfinite(rel) and rel <= th:
                reached = True
                ii = int(it)
                tt = float(tm)
                break
        rows.append({
            "case": cfg.case_name,
            "model": cfg.model_name,
            "metric": "RelL2",
            "threshold": th,
            "reached": reached,
            "iter_to_threshold": ii,
            "time_to_threshold_sec": tt,
        })
    write_csv(
        os.path.join(cfg.out_dir, "cost", "threshold_cost.csv"),
        rows,
        ["case", "model", "metric", "threshold", "reached",
         "iter_to_threshold", "time_to_threshold_sec"],
    )


def checkpoint_payload(cfg, model, optimizer, iteration, metric_name, metric_value):
    return {
        "iter": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        metric_name: metric_value,
        "config": asdict(cfg),
    }



# =========================
# 1. Configuration
# =========================

@dataclass
class NS2DConfig:
    case_name: str = "TGV"
    model_name: str = "LSTM-PINN-temporal-m4"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    nu: float = 0.1

    x0: float = 0.0
    x1: float = 2.0 * math.pi
    y0: float = 0.0
    y1: float = 2.0 * math.pi
    t0: float = 0.0
    t1: float = 5.0

    n_col: int = 20000
    n_ic: int = 2400
    n_bcx: int = 2400
    n_bcy: int = 2400
    train_ratio: float = 0.7

    iters: int = 80000
    lr: float = 1e-4
    batch_col: int = 1024
    batch_ic: int = 256
    batch_bcx: int = 256
    batch_bcy: int = 256

    val_batch_col: int = 512
    val_batch_ic: int = 256
    val_batch_bcx: int = 256
    val_batch_bcy: int = 256

    eval_every: int = 500
    grad_clip: float = 1.0

    w_mom: float = 1.0
    w_div: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 1.0

    hidden: int = 192
    lstm_layers: int = 2
    sequence_length: int = 4
    temporal_dt: float = 0.125

    eval_nx: int = 80
    eval_ny: int = 80
    eval_times: Tuple[float, ...] = (0.0, 2.5, 5.0)
    rel_l2_thresholds: Tuple[float, ...] = (1e-2, 1e-3)

    out_dir: str = "./outputs/TGV/LSTM-PINN-temporal-m4"
    seed: int = 0


# =========================
# 2. Taylor-Green manufactured reference
# =========================

def uvp_mms(x, y, t):
    e1 = torch.exp(-t)
    e2 = torch.exp(-2.0 * t)
    u = torch.sin(x) * torch.cos(y) * e1
    v = -torch.cos(x) * torch.sin(y) * e1
    p = 0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * e2
    return u, v, p


def forcing_mms(x, y, t, nu):
    e1 = torch.exp(-t)
    fu = (2.0 * nu - 1.0) * e1 * torch.sin(x) * torch.cos(y)
    fv = (1.0 - 2.0 * nu) * e1 * torch.cos(x) * torch.sin(y)
    return fu, fv


# =========================
# 3. Frozen dataset
# =========================

def build_dataset(cfg: NS2DConfig):
    # All points are generated ONCE after set_seed(cfg.seed).
    x = sample_uniform(cfg.n_col, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    y = sample_uniform(cfg.n_col, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    t = sample_uniform(cfg.n_col, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
    seq = build_temporal_context(x, y, t, cfg.sequence_length, cfg.temporal_dt, cfg.t0)
    tr, va = split_indices(cfg.n_col, cfg.train_ratio, cfg.seed + 101, cfg.device)
    col_tr = (seq[tr],)
    col_va = (seq[va],)

    xic = sample_uniform(cfg.n_ic, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    yic = sample_uniform(cfg.n_ic, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    tic = torch.full_like(xic, cfg.t0)
    seq_ic = build_temporal_context(xic, yic, tic, cfg.sequence_length, cfg.temporal_dt, cfg.t0)
    uic, vic, pic = uvp_mms(xic, yic, tic)
    tr, va = split_indices(cfg.n_ic, cfg.train_ratio, cfg.seed + 202, cfg.device)
    ic_tr = (seq_ic[tr], uic[tr].detach(), vic[tr].detach(), pic[tr].detach())
    ic_va = (seq_ic[va], uic[va].detach(), vic[va].detach(), pic[va].detach())

    yb = sample_uniform(cfg.n_bcx, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    tb = sample_uniform(cfg.n_bcx, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
    xl = torch.full_like(yb, cfg.x0)
    xr = torch.full_like(yb, cfg.x1)
    sl = build_temporal_context(xl, yb, tb, cfg.sequence_length, cfg.temporal_dt, cfg.t0)
    sr = build_temporal_context(xr, yb, tb, cfg.sequence_length, cfg.temporal_dt, cfg.t0)
    tr, va = split_indices(cfg.n_bcx, cfg.train_ratio, cfg.seed + 303, cfg.device)
    bcx_tr = (sl[tr], sr[tr])
    bcx_va = (sl[va], sr[va])

    xb = sample_uniform(cfg.n_bcy, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    tb = sample_uniform(cfg.n_bcy, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
    y0 = torch.full_like(xb, cfg.y0)
    y1 = torch.full_like(xb, cfg.y1)
    sb = build_temporal_context(xb, y0, tb, cfg.sequence_length, cfg.temporal_dt, cfg.t0)
    st = build_temporal_context(xb, y1, tb, cfg.sequence_length, cfg.temporal_dt, cfg.t0)
    tr, va = split_indices(cfg.n_bcy, cfg.train_ratio, cfg.seed + 404, cfg.device)
    bcy_tr = (sb[tr], st[tr])
    bcy_va = (sb[va], st[va])

    return {
        "col_tr": col_tr, "col_va": col_va,
        "ic_tr": ic_tr, "ic_va": ic_va,
        "bcx_tr": bcx_tr, "bcx_va": bcx_va,
        "bcy_tr": bcy_tr, "bcy_va": bcy_va,
    }


# =========================
# 4. Losses
# =========================

def pde_parts(cfg, model, seq):
    seq = seq.clone().detach().requires_grad_(True)
    out = model(seq)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    ut = grad_last(u, seq, 2)
    vt = grad_last(v, seq, 2)
    ux = grad_last(u, seq, 0)
    uy = grad_last(u, seq, 1)
    vx = grad_last(v, seq, 0)
    vy = grad_last(v, seq, 1)
    px = grad_last(p, seq, 0)
    py = grad_last(p, seq, 1)

    uxx = grad_last(ux, seq, 0)
    uyy = grad_last(uy, seq, 1)
    vxx = grad_last(vx, seq, 0)
    vyy = grad_last(vy, seq, 1)

    x = seq[:, -1, 0:1]
    y = seq[:, -1, 1:2]
    t = seq[:, -1, 2:3]
    fu, fv = forcing_mms(x, y, t, cfg.nu)

    ru = ut + u * ux + v * uy + px - cfg.nu * (uxx + uyy) - fu
    rv = vt + u * vx + v * vy + py - cfg.nu * (vxx + vyy) - fv
    div = ux + vy

    mom = torch.mean(ru ** 2) + torch.mean(rv ** 2)
    div_loss = torch.mean(div ** 2)
    return cfg.w_mom * mom + cfg.w_div * div_loss, mom, div_loss


def ic_loss(model, bundle):
    seq, ur, vr, pr = bundle
    o = model(seq)
    return (
        torch.mean((o[:, 0:1] - ur) ** 2) +
        torch.mean((o[:, 1:2] - vr) ** 2) +
        torch.mean((o[:, 2:3] - pr) ** 2)
    )


def periodic_loss(model, bundle):
    sa, sb = bundle
    a = model(sa)
    b = model(sb)
    return torch.mean((a - b) ** 2)


def data_parts(model, seq):
    o = model(seq)
    x = seq[:, -1, 0:1]
    y = seq[:, -1, 1:2]
    t = seq[:, -1, 2:3]
    ur, vr, pr = uvp_mms(x, y, t)
    sr = torch.sqrt(ur ** 2 + vr ** 2)
    sp = torch.sqrt(o[:, 0:1] ** 2 + o[:, 1:2] ** 2)
    lu = torch.mean((o[:, 0:1] - ur) ** 2)
    lv = torch.mean((o[:, 1:2] - vr) ** 2)
    lp = torch.mean((o[:, 2:3] - pr) ** 2)
    ls = torch.mean((sp - sr) ** 2)
    return lu + lv + lp + ls, lu, lv, lp, ls


def total_loss(cfg, model, col, ic, bcx, bcy):
    lpde, lmom, ldiv = pde_parts(cfg, model, col[0])
    lic = ic_loss(model, ic)
    lbc = periodic_loss(model, bcx) + periodic_loss(model, bcy)
    ldata, lu, lv, lp, ls = data_parts(model, col[0])
    total = lpde + cfg.w_ic * lic + cfg.w_bc * lbc
    return total, {
        "pde": lpde, "mom": lmom, "div": ldiv, "ic": lic, "bc": lbc,
        "data": ldata, "data_u": lu, "data_v": lv, "data_p": lp, "data_speed": ls,
    }


# =========================
# 5. Validation / evaluation
# =========================

@torch.no_grad()
def evaluate_val_metrics(model, col_va):
    seq = col_va[0]
    o = model(seq)
    x = seq[:, -1, 0:1]
    y = seq[:, -1, 1:2]
    t = seq[:, -1, 2:3]
    ur, vr, pr = uvp_mms(x, y, t)
    sr = torch.sqrt(ur ** 2 + vr ** 2)
    sp = torch.sqrt(o[:, 0:1] ** 2 + o[:, 1:2] ** 2)
    return compute_metrics_torch(
        torch.cat([o[:, 0:1], o[:, 1:2], o[:, 2:3], sp], dim=1),
        torch.cat([ur, vr, pr, sr], dim=1),
    )


@torch.no_grad()
def eval_on_grid(cfg, model):
    xs = torch.linspace(cfg.x0, cfg.x1, cfg.eval_nx, device=cfg.device, dtype=cfg.dtype)
    ys = torch.linspace(cfg.y0, cfg.y1, cfg.eval_ny, device=cfg.device, dtype=cfg.dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    X = XX.reshape(-1, 1)
    Y = YY.reshape(-1, 1)
    grid = {"x": xs.cpu().numpy(), "y": ys.cpu().numpy(), "times": np.array(cfg.eval_times)}

    for ti in cfg.eval_times:
        T = torch.full_like(X, float(ti))
        seq = build_temporal_context(X, Y, T, cfg.sequence_length, cfg.temporal_dt, cfg.t0)
        pred = model(seq)
        ur, vr, pr = uvp_mms(X, Y, T)
        sr = torch.sqrt(ur ** 2 + vr ** 2)
        sp = torch.sqrt(pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2)
        tag = time_tag(ti)
        vals = {
            "u_ref": ur, "v_ref": vr, "p_ref": pr, "speed_ref": sr,
            "u_pred": pred[:, 0:1], "v_pred": pred[:, 1:2],
            "p_pred": pred[:, 2:3], "speed_pred": sp,
        }
        for k, v in vals.items():
            grid[f"{k}_{tag}"] = v.view(cfg.eval_nx, cfg.eval_ny).cpu().numpy()
        for fld in ["u", "v", "p", "speed"]:
            grid[f"{fld}_err_{tag}"] = grid[f"{fld}_pred_{tag}"] - grid[f"{fld}_ref_{tag}"]
    return grid


def save_grid_outputs(cfg, grid, history):
    np.savez(os.path.join(cfg.out_dir, "grid.npz"), **grid)
    fields = ["u", "v", "p", "speed"]
    rows = []
    all_pred, all_ref = [], []
    XX, YY = np.meshgrid(grid["x"], grid["y"], indexing="ij")
    extent = [cfg.x0, cfg.x1, cfg.y0, cfg.y1]

    for ti in cfg.eval_times:
        tag = time_tag(ti)
        folder = os.path.join(cfg.out_dir, "field_data", tag)
        ensure_dir(folder)
        for fld in fields:
            pred = grid[f"{fld}_pred_{tag}"]
            ref = grid[f"{fld}_ref_{tag}"]
            err = pred - ref
            arr = np.column_stack([
                XX.reshape(-1), YY.reshape(-1), pred.reshape(-1), ref.reshape(-1),
                err.reshape(-1), np.abs(err).reshape(-1)
            ])
            np.savetxt(
                os.path.join(folder, f"{fld}_field_data_{tag}.txt"),
                arr, fmt="%.10e", delimiter="\t",
                header="x\ty\tPrediction\tReference\tError\tAbsError", comments=""
            )
            mm = compute_metrics_np(pred, ref)
            rows.append({"case": cfg.case_name, "model": cfg.model_name, "time": ti, "field": fld, **mm})
            all_pred.append(pred.reshape(-1))
            all_ref.append(ref.reshape(-1))

            save_map(ref, extent, f"Reference {fld}, {tag}",
                     os.path.join(cfg.out_dir, "figures", "reference", f"{fld}_reference_{tag}.png"))
            save_map(pred, extent, f"Prediction {fld}, {tag}",
                     os.path.join(cfg.out_dir, "figures", "prediction", f"{fld}_prediction_{tag}.png"))
            save_map(np.abs(err), extent, f"|{fld} error|, {tag}",
                     os.path.join(cfg.out_dir, "figures", "error", f"{fld}_error_{tag}.png"))

    write_csv(os.path.join(cfg.out_dir, "metrics", "field_metrics.csv"), rows)
    overall = compute_metrics_np(np.concatenate(all_pred), np.concatenate(all_ref))
    write_csv(os.path.join(cfg.out_dir, "metrics", "overall_metrics.csv"),
              [{"case": cfg.case_name, "model": cfg.model_name, **overall}])
    save_loss_plots(cfg.out_dir, history)
    return overall


# =========================
# 6. History
# =========================

def make_history():
    keys = [
        "iter", "loss_total_train", "loss_total_val",
        "loss_pde_train", "loss_pde_val",
        "loss_mom_train", "loss_mom_val",
        "loss_div_train", "loss_div_val",
        "loss_ic_train", "loss_ic_val",
        "loss_bc_train", "loss_bc_val",
        "loss_data_train", "loss_data_val",
        "loss_data_u_train", "loss_data_u_val",
        "loss_data_v_train", "loss_data_v_val",
        "loss_data_p_train", "loss_data_p_val",
        "loss_data_speed_train", "loss_data_speed_val",
        "MSE_val", "RMSE_val", "MAE_val", "L2_val", "RelL2_val",
        "MaxAbsError_val", "MeanAbsError_val", "time_elapsed_sec",
    ]
    return {k: [] for k in keys}


def append_history(history, row):
    for k in history:
        history[k].append(row.get(k, 0.0))


# =========================
# 7. Training
# =========================

def train(cfg: NS2DConfig):
    make_output_dirs(cfg.out_dir)
    set_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    data = build_dataset(cfg)
    model = LSTMPINNModel(3, cfg.hidden, cfg.lstm_layers).to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    save_model_info(
        cfg, model, "u, v, p",
        "true temporal sequence at fixed (x,y); target is final token"
    )
    save_json(os.path.join(cfg.out_dir, "config.json"), asdict(cfg))
    save_json(os.path.join(cfg.out_dir, "environment.json"), get_hardware_info(cfg.device))

    history = make_history()
    iter_times = []
    best_rel = float("inf")
    best_iter = -1
    t_start = time.time()

    print(f"[INFO] Case: {cfg.case_name} | Model: {cfg.model_name}")
    print(f"[INFO] Sequence length: {cfg.sequence_length}")
    print(f"[INFO] Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"[INFO] Output: {os.path.abspath(cfg.out_dir)}")

    for it in range(1, cfg.iters + 1):
        st = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)

        col = deterministic_minibatch(data["col_tr"], cfg.batch_col, cfg.seed + it)
        ic = deterministic_minibatch(data["ic_tr"], cfg.batch_ic, cfg.seed + 10000 + it)
        bcx = deterministic_minibatch(data["bcx_tr"], cfg.batch_bcx, cfg.seed + 20000 + it)
        bcy = deterministic_minibatch(data["bcy_tr"], cfg.batch_bcy, cfg.seed + 30000 + it)

        loss, parts = total_loss(cfg, model, col, ic, bcx, bcy)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at iteration {it}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        iter_times.append(time.time() - st)

        if it == 1 or it % cfg.eval_every == 0:
            model.eval()

            colv = deterministic_minibatch(data["col_va"], cfg.val_batch_col, cfg.seed + 40000 + it)
            icv = deterministic_minibatch(data["ic_va"], cfg.val_batch_ic, cfg.seed + 50000 + it)
            bcxv = deterministic_minibatch(data["bcx_va"], cfg.val_batch_bcx, cfg.seed + 60000 + it)
            bcyv = deterministic_minibatch(data["bcy_va"], cfg.val_batch_bcy, cfg.seed + 70000 + it)
            lv, pv = total_loss(cfg, model, colv, icv, bcxv, bcyv)
            vm = evaluate_val_metrics(model, colv)

            row = {
                "iter": it,
                "loss_total_train": float(loss.detach().item()),
                "loss_total_val": float(lv.detach().item()),
                "loss_pde_train": float(parts["pde"].detach().item()),
                "loss_pde_val": float(pv["pde"].detach().item()),
                "loss_mom_train": float(parts["mom"].detach().item()),
                "loss_mom_val": float(pv["mom"].detach().item()),
                "loss_div_train": float(parts["div"].detach().item()),
                "loss_div_val": float(pv["div"].detach().item()),
                "loss_ic_train": float(parts["ic"].detach().item()),
                "loss_ic_val": float(pv["ic"].detach().item()),
                "loss_bc_train": float(parts["bc"].detach().item()),
                "loss_bc_val": float(pv["bc"].detach().item()),
                "loss_data_train": float(parts["data"].detach().item()),
                "loss_data_val": float(pv["data"].detach().item()),
                "loss_data_u_train": float(parts["data_u"].detach().item()),
                "loss_data_u_val": float(pv["data_u"].detach().item()),
                "loss_data_v_train": float(parts["data_v"].detach().item()),
                "loss_data_v_val": float(pv["data_v"].detach().item()),
                "loss_data_p_train": float(parts["data_p"].detach().item()),
                "loss_data_p_val": float(pv["data_p"].detach().item()),
                "loss_data_speed_train": float(parts["data_speed"].detach().item()),
                "loss_data_speed_val": float(pv["data_speed"].detach().item()),
                **{f"{k}_val": v for k, v in vm.items()},
                "time_elapsed_sec": time.time() - t_start,
            }
            append_history(history, row)
            write_history_csv(os.path.join(cfg.out_dir, "loss", "train_history.csv"), history)

            if vm["RelL2"] < best_rel:
                best_rel = vm["RelL2"]
                best_iter = it
                torch.save(
                    checkpoint_payload(cfg, model, opt, it, "best_rel_l2", best_rel),
                    os.path.join(cfg.out_dir, "best_model.pt"),
                )

            print(
                f"[{it:06d}] loss_tr={row['loss_total_train']:.3e} "
                f"loss_va={row['loss_total_val']:.3e} "
                f"RelL2={vm['RelL2']:.3e} elapsed={row['time_elapsed_sec']/60:.2f} min"
            )

    total_wall_time = time.time() - t_start
    torch.save(
        checkpoint_payload(
            cfg, model, opt,
            history["iter"][-1] if history["iter"] else 0,
            "final_rel_l2_val",
            history["RelL2_val"][-1] if history["RelL2_val"] else None,
        ),
        os.path.join(cfg.out_dir, "final_model.pt"),
    )

    model.eval()
    grid = eval_on_grid(cfg, model)
    overall = save_grid_outputs(cfg, grid, history)
    save_cost_outputs(cfg, model, history, iter_times, total_wall_time, best_iter, best_rel, overall["RelL2"])

    print(f"[DONE] Total wall time: {total_wall_time/60:.2f} min")
    print(f"[DONE] Best RelL2: {best_rel:.6e} at iter {best_iter}")
    return model, history


def main():
    cfg = NS2DConfig()
    train(cfg)


if __name__ == "__main__":
    main()
