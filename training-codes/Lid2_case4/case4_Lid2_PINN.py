

from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


SOURCE_REPOSITORY = "JeromeLiu06/Temporal-Modeling-in-Physics-Informed-Neural-Networks-for-Transient-and-Coupled-Fluid-Dynamics"
SOURCE_MAIN_COMMIT = "1e9079a590f85bea18771d6a6f9b567b12442d03"
SOURCE_CASE2_FILE = "ns2d_cavity_pinn_rebuild_outputs.py"


@dataclass
class CavityConfig:
    case_name: str = "Case4-Lid2-Re100"
    model_name: str = "PINN"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Standard steady incompressible lid-driven cavity: Re=100, nu=1/Re.
    nu: float = 0.01
    re: float = 100.0

    x0: float = 0.0
    x1: float = 1.0
    y0: float = 0.0
    y1: float = 1.0

    n_col: int = 22000
    n_bc_bottom: int = 2200
    n_bc_top: int = 2200
    n_bc_left: int = 2200
    n_bc_right: int = 2200
    train_ratio: float = 0.7

    iters: int = 80000
    lr: float = 1e-4
    batch_col: int = 1024
    batch_bc: int = 256
    val_batch_col: int = 512
    val_batch_bc: int = 256
    eval_every: int = 500
    grad_clip: float = 1.0

    w_mom: float = 1.0
    w_div: float = 1.0
    w_bc: float = 10.0

    # Exact Case2 vanilla PINN settings.
    hidden: int = 128
    num_hidden_layers: int = 8

    eval_nx: int = 120
    eval_ny: int = 120
    rel_l2_thresholds: Tuple[float, float] = (1e-2, 1e-3)

    reference_path: str = "./CFD参考解/case4_Lid2_Re100/final_reference/case4_fem_Re100_N512.npz"
    out_dir: str = "./outputs/Case4-Lid2-Re100/PINN"
    seed: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def get_cpu_name() -> str:
    name = platform.processor()
    if name:
        return name
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "Unknown CPU"


def get_hardware_info(device: str) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "device": device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_name": get_cpu_name(),
        "gpu_name": "None",
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_available": bool(cuda_available),
        "cuda_version": torch.version.cuda or "None",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "gpu_total_memory_MB": 0.0,
    }
    if cuda_available and device.startswith("cuda"):
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_total_memory_MB"] = props.total_memory / (1024 ** 2)
    return info


def peak_gpu_memory(device: str) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.startswith("cuda"):
        return (
            torch.cuda.max_memory_allocated() / (1024 ** 2),
            torch.cuda.max_memory_reserved() / (1024 ** 2),
        )
    return 0.0, 0.0


def save_config_and_environment(cfg: CavityConfig) -> None:
    ensure_dir(cfg.out_dir)
    config = {key: jsonable(value) for key, value in asdict(cfg).items()}
    config.update({
        "source_repository": SOURCE_REPOSITORY,
        "source_main_commit": SOURCE_MAIN_COMMIT,
        "source_case2_file": SOURCE_CASE2_FILE,
    })
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.out_dir, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(get_hardware_info(cfg.device), handle, indent=2, ensure_ascii=False)


def sample_uniform(n: int, low: float, high: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, 1, device=device, dtype=dtype)


def split_indices(n: int, ratio: float, seed: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if n < 2:
        return torch.arange(n, device=device), torch.zeros(0, dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator, device=device)
    n_train = max(1, min(int(ratio * n), n - 1))
    return perm[:n_train], perm[n_train:]


def build_dataset(cfg: CavityConfig) -> Dict[str, Tuple[torch.Tensor, ...]]:
    x_col = sample_uniform(cfg.n_col, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    y_col = sample_uniform(cfg.n_col, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    train_idx, val_idx = split_indices(cfg.n_col, cfg.train_ratio, cfg.seed + 101, cfg.device)

    def make_bc(n: int, side_seed: int, side: str):
        if side == "bottom":
            x = sample_uniform(n, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
            y = torch.full_like(x, cfg.y0)
        elif side == "top":
            x = sample_uniform(n, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
            y = torch.full_like(x, cfg.y1)
        elif side == "left":
            y = sample_uniform(n, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
            x = torch.full_like(y, cfg.x0)
        elif side == "right":
            y = sample_uniform(n, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
            x = torch.full_like(y, cfg.x1)
        else:
            raise ValueError(f"Unknown boundary side: {side}")

        # No pressure target is created or supervised at any wall.
        u_target = torch.ones_like(x) if side == "top" else torch.zeros_like(x)
        v_target = torch.zeros_like(x)
        tr, va = split_indices(n, cfg.train_ratio, side_seed, cfg.device)
        return (
            (x[tr], y[tr], u_target[tr], v_target[tr]),
            (x[va], y[va], u_target[va], v_target[va]),
        )

    bottom_tr, bottom_va = make_bc(cfg.n_bc_bottom, cfg.seed + 201, "bottom")
    top_tr, top_va = make_bc(cfg.n_bc_top, cfg.seed + 202, "top")
    left_tr, left_va = make_bc(cfg.n_bc_left, cfg.seed + 203, "left")
    right_tr, right_va = make_bc(cfg.n_bc_right, cfg.seed + 204, "right")
    return {
        "col_tr": (x_col[train_idx], y_col[train_idx]),
        "col_va": (x_col[val_idx], y_col[val_idx]),
        "bc_bottom_tr": bottom_tr,
        "bc_bottom_va": bottom_va,
        "bc_top_tr": top_tr,
        "bc_top_va": top_va,
        "bc_left_tr": left_tr,
        "bc_left_va": left_va,
        "bc_right_tr": right_tr,
        "bc_right_va": right_va,
    }


# Exact model classes copied from the Case2 source named above.
class BaselineModel(nn.Module):


    def __init__(self, cfg: CavityConfig):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(2, cfg.hidden))
        layers.append(nn.Tanh())
        for _ in range(cfg.num_hidden_layers - 1):
            layers.append(nn.Linear(cfg.hidden, cfg.hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(cfg.hidden, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def grad1(output: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        coordinate,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplacian_2d(output: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    output_x = grad1(output, x)
    output_y = grad1(output, y)
    output_xx = grad1(output_x, x)
    output_yy = grad1(output_y, y)
    return output_x, output_y, output_xx + output_yy


def boundary_loss_from_bundle(model: BaselineModel, bundle: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    x, y, u_target, v_target = bundle
    u_pred, v_pred, _ = model(x, y)
    return torch.mean((u_pred - u_target) ** 2) + torch.mean((v_pred - v_target) ** 2)





def batch_losses(
    cfg: CavityConfig,
    model: BaselineModel,
    col: Tuple[torch.Tensor, torch.Tensor],
    bc_bottom: Tuple[torch.Tensor, ...],
    bc_top: Tuple[torch.Tensor, ...],
    bc_left: Tuple[torch.Tensor, ...],
    bc_right: Tuple[torch.Tensor, ...],
):
    with torch.enable_grad():
        x_col, y_col = col
        x = x_col.clone().detach().requires_grad_(True)
        y = y_col.clone().detach().requires_grad_(True)

        u, v, p = model(x, y)
        u_x, u_y, lap_u = laplacian_2d(u, x, y)
        v_x, v_y, lap_v = laplacian_2d(v, x, y)
        p_x = grad1(p, x)
        p_y = grad1(p, y)

        # Standard steady incompressible Navier-Stokes, with no manufactured forcing.
        r_u = u * u_x + v * u_y + p_x - cfg.nu * lap_u
        r_v = u * v_x + v * v_y + p_y - cfg.nu * lap_v
        r_div = u_x + v_y

        loss_mom = torch.mean(r_u ** 2) + torch.mean(r_v ** 2)
        loss_div = torch.mean(r_div ** 2)
        loss_pde = loss_mom + loss_div

        loss_bc = (
            boundary_loss_from_bundle(model, bc_bottom)
            + boundary_loss_from_bundle(model, bc_top)
            + boundary_loss_from_bundle(model, bc_left)
            + boundary_loss_from_bundle(model, bc_right)
        )
        loss = cfg.w_mom * loss_mom + cfg.w_div * loss_div + cfg.w_bc * loss_bc
        return loss, {"pde": loss_pde, "mom": loss_mom, "div": loss_div, "bc": loss_bc}


def minibatch2(a: torch.Tensor, b: torch.Tensor, batch: int, seed: int):
    generator = torch.Generator(device=a.device)
    generator.manual_seed(seed)
    idx = torch.randperm(a.shape[0], generator=generator, device=a.device)[:min(batch, a.shape[0])]
    return a[idx], b[idx]


def minibatch4(a, b, c, d, batch: int, seed: int):
    generator = torch.Generator(device=a.device)
    generator.manual_seed(seed)
    idx = torch.randperm(a.shape[0], generator=generator, device=a.device)[:min(batch, a.shape[0])]
    return a[idx], b[idx], c[idx], d[idx]


METRIC_COLUMNS = ["MSE", "RMSE", "MAE", "L2", "RelL2", "MaxAbsError", "MeanAbsError"]


def compute_metrics_np(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref_flat = np.asarray(ref, dtype=np.float64).reshape(-1)
    diff = pred_flat - ref_flat
    mse = float(np.mean(diff ** 2))
    l2 = float(np.linalg.norm(diff))
    return {
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(np.mean(np.abs(diff))),
        "L2": l2,
        "RelL2": float(l2 / (np.linalg.norm(ref_flat) + 1e-12)),
        "MaxAbsError": float(np.max(np.abs(diff))),
        "MeanAbsError": float(np.mean(np.abs(diff))),
    }


def _reference_points(x: np.ndarray, y: np.ndarray, field: np.ndarray):
    x = np.asarray(x)
    y = np.asarray(y)
    field = np.asarray(field)
    if x.ndim == 1 and y.ndim == 1 and field.size == x.size * y.size:
        xx, yy = np.meshgrid(x, y, indexing="ij")
        if field.shape == (x.size, y.size):
            values = field
        elif field.shape == (y.size, x.size):
            values = field.T
        else:
            values = field.reshape(x.size, y.size)
        return xx.reshape(-1), yy.reshape(-1), values.reshape(-1)
    if x.shape == field.shape and y.shape == field.shape:
        return x.reshape(-1), y.reshape(-1), field.reshape(-1)
    if x.size == field.size and y.size == field.size:
        return x.reshape(-1), y.reshape(-1), field.reshape(-1)
    raise ValueError(
        f"Cannot align reference coordinates x{x.shape}, y{y.shape} with field{field.shape}."
    )


def load_fem_reference(cfg: CavityConfig, allow_missing: bool = True) -> Optional[Dict[str, np.ndarray]]:
    path = os.path.abspath(cfg.reference_path)
    if not os.path.isfile(path):
        if allow_missing:
            return None
        raise FileNotFoundError(
            "FEM reference is required for final field metrics but was not found: " + path
        )

    try:
        from scipy.interpolate import griddata
    except ImportError as exc:
        raise ImportError("SciPy is required to interpolate the FEM reference.") from exc

    with np.load(path) as data:
        missing = [key for key in ("x", "y", "u", "v", "p") if key not in data]
        if missing:
            raise KeyError(f"FEM reference {path} is missing keys: {missing}")
        x_src = np.asarray(data["x"])
        y_src = np.asarray(data["y"])
        raw_fields = {field: np.asarray(data[field]) for field in ("u", "v", "p")}

    xs = np.linspace(cfg.x0, cfg.x1, cfg.eval_nx)
    ys = np.linspace(cfg.y0, cfg.y1, cfg.eval_ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    interpolated: Dict[str, np.ndarray] = {"x": xs, "y": ys}
    for field, values in raw_fields.items():
        px, py, pv = _reference_points(x_src, y_src, values)
        finite = np.isfinite(px) & np.isfinite(py) & np.isfinite(pv)
        if np.count_nonzero(finite) < 3:
            raise ValueError(f"FEM reference field {field!r} has fewer than three finite points.")
        points = np.column_stack([px[finite], py[finite]])
        target = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        values_linear = griddata(points, pv[finite], target, method="linear")
        missing_mask = ~np.isfinite(values_linear)
        if np.any(missing_mask):
            values_linear[missing_mask] = griddata(
                points, pv[finite], target[missing_mask], method="nearest"
            )
        interpolated[field + "_ref"] = values_linear.reshape(cfg.eval_nx, cfg.eval_ny)

    # Pressure gauge is removed before every comparison.
    interpolated["p_ref"] = interpolated["p_ref"] - np.mean(interpolated["p_ref"])
    interpolated["speed_ref"] = np.sqrt(interpolated["u_ref"] ** 2 + interpolated["v_ref"] ** 2)
    interpolated["reference_path"] = np.asarray(path)
    return interpolated


@torch.no_grad()
def predict_grid(cfg: CavityConfig, model: BaselineModel, reference: Optional[Dict[str, np.ndarray]] = None):
    xs = torch.linspace(cfg.x0, cfg.x1, cfg.eval_nx, device=cfg.device, dtype=cfg.dtype)
    ys = torch.linspace(cfg.y0, cfg.y1, cfg.eval_ny, device=cfg.device, dtype=cfg.dtype)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.reshape(-1, 1)
    y_flat = yy.reshape(-1, 1)
    u_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    p_parts: List[torch.Tensor] = []
    for start in range(0, x_flat.shape[0], 4096):
        u, v, p = model(x_flat[start:start + 4096], y_flat[start:start + 4096])
        u_parts.append(u)
        v_parts.append(v)
        p_parts.append(p)
    u_pred = torch.cat(u_parts).reshape(cfg.eval_nx, cfg.eval_ny).cpu().numpy()
    v_pred = torch.cat(v_parts).reshape(cfg.eval_nx, cfg.eval_ny).cpu().numpy()
    p_raw = torch.cat(p_parts).reshape(cfg.eval_nx, cfg.eval_ny).cpu().numpy()
    p_pred = p_raw - np.mean(p_raw)
    grid: Dict[str, np.ndarray] = {
        "x": xs.cpu().numpy(),
        "y": ys.cpu().numpy(),
        "u_pred": u_pred,
        "v_pred": v_pred,
        "p_pred_raw": p_raw,
        "p_pred": p_pred,
        "speed_pred": np.sqrt(u_pred ** 2 + v_pred ** 2),
        "reference_available": np.asarray(reference is not None),
    }
    if reference is not None:
        for key in ("u_ref", "v_ref", "p_ref", "speed_ref", "reference_path"):
            grid[key] = reference[key]
        for field in ("u", "v", "p", "speed"):
            grid[field + "_err"] = grid[field + "_pred"] - grid[field + "_ref"]
    return grid


def overall_grid_metrics(grid: Dict[str, np.ndarray]) -> Dict[str, float]:
    pred = np.concatenate([grid[field + "_pred"].reshape(-1) for field in ("u", "v", "p", "speed")])
    ref = np.concatenate([grid[field + "_ref"].reshape(-1) for field in ("u", "v", "p", "speed")])
    return compute_metrics_np(pred, ref)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(path: str, history: Dict[str, List[Any]]) -> None:
    keys = list(history)
    rows = [{key: history[key][i] for key in keys} for i in range(len(history[keys[0]]))] if keys else []
    write_csv(path, rows, keys)


def save_field_data(cfg: CavityConfig, grid: Dict[str, np.ndarray]) -> None:
    xx, yy = np.meshgrid(grid["x"], grid["y"], indexing="ij")
    out_dir = os.path.join(cfg.out_dir, "field_data", "steady")
    ensure_dir(out_dir)
    has_ref = bool(grid["reference_available"])
    for field in ("u", "v", "p", "speed"):
        pred = grid[field + "_pred"]
        if has_ref:
            ref = grid[field + "_ref"]
            err = pred - ref
            values = np.column_stack([
                xx.reshape(-1), yy.reshape(-1), pred.reshape(-1), ref.reshape(-1),
                err.reshape(-1), np.abs(err).reshape(-1),
            ])
            header = "x\ty\tPrediction\tReference\tError\tAbsError"
        else:
            values = np.column_stack([xx.reshape(-1), yy.reshape(-1), pred.reshape(-1)])
            header = "x\ty\tPrediction"
        np.savetxt(
            os.path.join(out_dir, field + "_field_data.txt"), values,
            fmt="%.10e", delimiter="\t", header=header, comments="",
        )


def save_metrics(cfg: CavityConfig, grid: Dict[str, np.ndarray]) -> Dict[str, float]:
    has_ref = bool(grid["reference_available"])
    status = "complete" if has_ref else "deferred_reference_missing"
    field_rows: List[Dict[str, Any]] = []
    for field in ("u", "v", "p", "speed"):
        row: Dict[str, Any] = {
            "case": cfg.case_name, "model": cfg.model_name, "time": "steady",
            "field": field, "status": status, "reference_path": os.path.abspath(cfg.reference_path),
        }
        metrics = compute_metrics_np(grid[field + "_pred"], grid[field + "_ref"]) if has_ref else {
            key: float("nan") for key in METRIC_COLUMNS
        }
        row.update(metrics)
        field_rows.append(row)
    fieldnames = ["case", "model", "time", "field", "status", "reference_path"] + METRIC_COLUMNS
    write_csv(os.path.join(cfg.out_dir, "field_metrics.csv"), field_rows, fieldnames)

    overall = overall_grid_metrics(grid) if has_ref else {key: float("nan") for key in METRIC_COLUMNS}
    overall_row: Dict[str, Any] = {
        "case": cfg.case_name, "model": cfg.model_name, "status": status,
        "reference_path": os.path.abspath(cfg.reference_path),
    }
    overall_row.update(overall)
    write_csv(
        os.path.join(cfg.out_dir, "overall_metrics.csv"), [overall_row],
        ["case", "model", "status", "reference_path"] + METRIC_COLUMNS,
    )
    if not has_ref:
        with open(os.path.join(cfg.out_dir, "EVALUATION_DEFERRED.txt"), "w", encoding="utf-8") as handle:
            handle.write(
                "Final field metrics and RelL2 evaluation were deferred because the FEM reference was not found.\n"
                f"Expected reference: {os.path.abspath(cfg.reference_path)}\n"
                "No synthetic or manufactured reference was substituted. Re-run evaluation after the file exists.\n"
            )
    return overall


def save_plots(cfg: CavityConfig, grid: Dict[str, np.ndarray], history: Dict[str, List[Any]]) -> None:
    extent = [grid["x"][0], grid["x"][-1], grid["y"][0], grid["y"][-1]]

    def save_map(values: np.ndarray, title: str, path: str) -> None:
        ensure_dir(os.path.dirname(path))
        plt.figure(figsize=(5.6, 4.6))
        plt.imshow(values.T, origin="lower", extent=extent, aspect="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    has_ref = bool(grid["reference_available"])
    for field in ("u", "v", "p", "speed"):
        save_map(grid[field + "_pred"], f"Prediction {field}", os.path.join(
            cfg.out_dir, "figures", "prediction", field + "_prediction.png"
        ))
        if has_ref:
            save_map(grid[field + "_ref"], f"FEM reference {field}", os.path.join(
                cfg.out_dir, "figures", "reference", field + "_reference.png"
            ))
            save_map(np.abs(grid[field + "_err"]), f"|{field} error|", os.path.join(
                cfg.out_dir, "figures", "error", field + "_error.png"
            ))

    loss_pairs = [
        ("loss_total_train", "loss_total_val", "Total loss", "loss_total_curve.png"),
        ("loss_pde_train", "loss_pde_val", "PDE loss", "loss_pde_curve.png"),
        ("loss_mom_train", "loss_mom_val", "Momentum loss", "loss_mom_curve.png"),
        ("loss_div_train", "loss_div_val", "Divergence loss", "loss_div_curve.png"),
        ("loss_bc_train", "loss_bc_val", "Boundary loss", "loss_bc_curve.png"),
        ("RelL2_val", None, "FEM-grid RelL2", "rel_l2_curve.png"),
    ]
    for train_key, val_key, title, filename in loss_pairs:
        values = np.asarray(history.get(train_key, []), dtype=float)
        iterations = np.asarray(history.get("iter", []), dtype=float)
        finite = np.isfinite(values) & (values > 0)
        if not np.any(finite):
            continue
        ensure_dir(os.path.join(cfg.out_dir, "figures", "loss"))
        plt.figure(figsize=(7, 4))
        plt.plot(iterations[finite], values[finite], label=train_key)
        if val_key is not None:
            val_values = np.asarray(history.get(val_key, []), dtype=float)
            val_finite = np.isfinite(val_values) & (val_values > 0)
            if np.any(val_finite):
                plt.plot(iterations[val_finite], val_values[val_finite], label=val_key)
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "figures", "loss", filename), dpi=220)
        plt.close()


def save_model_info(cfg: CavityConfig, model: nn.Module) -> None:
    with open(os.path.join(cfg.out_dir, "model_info.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Case: {cfg.case_name}\nModel: {cfg.model_name}\n")
        handle.write("Architecture: Vanilla fully connected PINN; hidden=128; hidden_layers=8; activation=tanh\n")
        handle.write("Input dimension: 2 (x, y)\nOutput dimension: 3 (u, v, p)\n")
        handle.write(f"Trainable parameters: {count_trainable_parameters(model)}\n")
        handle.write(f"Total parameters: {count_total_parameters(model)}\n")
        handle.write(f"Parameter size MB: {parameter_size_mb(model):.6f}\n")
        handle.write(f"Optimizer: Adam\nLearning rate: {cfg.lr}\nIterations: {cfg.iters}\n")
        handle.write(f"Source main commit: {SOURCE_MAIN_COMMIT}\nSource Case2 file: {SOURCE_CASE2_FILE}\n")


def make_history() -> Dict[str, List[Any]]:
    keys = [
        "iter", "loss_total_train", "loss_total_val", "loss_pde_train", "loss_pde_val",
        "loss_mom_train", "loss_mom_val", "loss_div_train", "loss_div_val",
        "loss_bc_train", "loss_bc_val", "MSE_val", "RMSE_val", "MAE_val", "L2_val",
        "RelL2_val", "MaxAbsError_val", "MeanAbsError_val", "reference_available",
        "time_elapsed_sec",
    ]
    return {key: [] for key in keys}


def append_history(history: Dict[str, List[Any]], row: Dict[str, Any]) -> None:
    for key in history:
        history[key].append(row.get(key, float("nan")))


def build_threshold_rows(cfg: CavityConfig, history: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for threshold in cfg.rel_l2_thresholds:
        reached = False
        hit_iter = -1
        hit_time = -1.0
        for iteration, rel_l2, elapsed in zip(
            history.get("iter", []), history.get("RelL2_val", []), history.get("time_elapsed_sec", [])
        ):
            if np.isfinite(rel_l2) and rel_l2 <= threshold:
                reached, hit_iter, hit_time = True, int(iteration), float(elapsed)
                break
        rows.append({
            "case": cfg.case_name, "model": cfg.model_name, "metric": "RelL2",
            "threshold": threshold, "reached": reached, "iter_to_threshold": hit_iter,
            "time_to_threshold_sec": hit_time,
            "status": "complete" if any(np.isfinite(history.get("RelL2_val", []))) else "deferred_reference_missing",
        })
    return rows


def save_cost_outputs(
    cfg: CavityConfig,
    model: nn.Module,
    history: Dict[str, List[Any]],
    iter_times: List[float],
    total_wall_time_sec: float,
    best_iter: int,
    best_rel_l2: Optional[float],
    final_rel_l2: Optional[float],
    best_selection_metric: str,
) -> None:
    hardware = get_hardware_info(cfg.device)
    peak_allocated, peak_reserved = peak_gpu_memory(cfg.device)
    summary = {
        "case": cfg.case_name,
        "model": cfg.model_name,
        "device": cfg.device,
        "trainable_parameters": count_trainable_parameters(model),
        "total_parameters": count_total_parameters(model),
        "parameter_size_MB": parameter_size_mb(model),
        "total_iterations_configured": cfg.iters,
        "total_iterations_completed": history["iter"][-1] if history["iter"] else 0,
        "total_wall_time_sec": total_wall_time_sec,
        "total_wall_time_min": total_wall_time_sec / 60.0,
        "mean_time_per_iter_sec": float(np.mean(iter_times)) if iter_times else float("nan"),
        "median_time_per_iter_sec": float(np.median(iter_times)) if iter_times else float("nan"),
        "gpu_name": hardware["gpu_name"],
        "gpu_count": hardware["gpu_count"],
        "cuda_available": hardware["cuda_available"],
        "cuda_version": hardware["cuda_version"],
        "torch_version": hardware["torch_version"],
        "cpu_name": hardware["cpu_name"],
        "gpu_total_memory_MB": hardware["gpu_total_memory_MB"],
        "peak_gpu_memory_allocated_MB": peak_allocated,
        "peak_gpu_memory_reserved_MB": peak_reserved,
        "best_iter": best_iter,
        "best_selection_metric": best_selection_metric,
        "best_RelL2": best_rel_l2 if best_rel_l2 is not None else float("nan"),
        "final_RelL2": final_rel_l2 if final_rel_l2 is not None else float("nan"),
        "reference_status": "complete" if final_rel_l2 is not None else "deferred_reference_missing",
    }
    summary_path = os.path.join(cfg.out_dir, "cost", "model_cost_summary.csv")
    write_csv(summary_path, [summary], list(summary))
    threshold_rows = build_threshold_rows(cfg, history)
    threshold_fields = [
        "case", "model", "metric", "threshold", "reached", "iter_to_threshold",
        "time_to_threshold_sec", "status",
    ]
    threshold_path = os.path.join(cfg.out_dir, "cost", "threshold_cost.csv")
    write_csv(threshold_path, threshold_rows, threshold_fields)
    shutil.copyfile(threshold_path, os.path.join(cfg.out_dir, "threshold_cost.csv"))


def save_all_outputs(
    cfg: CavityConfig,
    model: nn.Module,
    grid: Dict[str, np.ndarray],
    history: Dict[str, List[Any]],
    iter_times: List[float],
    total_wall_time_sec: float,
    best_iter: int,
    best_rel_l2: Optional[float],
    best_selection_metric: str,
) -> Dict[str, float]:
    for subdir in (
        "checkpoints", "field_data/steady", "cost", "figures/reference", "figures/prediction",
        "figures/error", "figures/loss",
    ):
        ensure_dir(os.path.join(cfg.out_dir, subdir))
    np.savez(os.path.join(cfg.out_dir, "grid.npz"), **grid)
    write_history_csv(os.path.join(cfg.out_dir, "train_history.csv"), history)
    save_field_data(cfg, grid)
    overall = save_metrics(cfg, grid)
    save_plots(cfg, grid, history)
    final_rel_l2 = overall["RelL2"] if np.isfinite(overall["RelL2"]) else None
    save_cost_outputs(
        cfg, model, history, iter_times, total_wall_time_sec, best_iter,
        best_rel_l2, final_rel_l2, best_selection_metric,
    )
    return overall


def _checkpoint_payload(
    cfg: CavityConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_rel_l2: Optional[float],
    best_val_loss: float,
    best_selection_metric: str,
) -> Dict[str, Any]:
    return {
        "iter": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_rel_l2": best_rel_l2,
        "best_val_loss": best_val_loss,
        "best_selection_metric": best_selection_metric,
        "config": {key: jsonable(value) for key, value in asdict(cfg).items()},
        "source_repository": SOURCE_REPOSITORY,
        "source_main_commit": SOURCE_MAIN_COMMIT,
        "source_case2_file": SOURCE_CASE2_FILE,
    }


def train(cfg: CavityConfig) -> BaselineModel:
    if cfg.re <= 0 or not math.isclose(cfg.nu, 1.0 / cfg.re, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Case4 requires nu=1/Re; received nu={cfg.nu}, Re={cfg.re}.")
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "checkpoints"))
    set_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    model = BaselineModel(cfg).to(device=cfg.device, dtype=cfg.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    data = build_dataset(cfg)
    reference = load_fem_reference(cfg, allow_missing=True)
    save_config_and_environment(cfg)
    save_model_info(cfg, model)

    history = make_history()
    iter_times: List[float] = []
    best_score = float("inf")
    best_iter = -1
    best_rel_l2: Optional[float] = None
    best_val_loss = float("inf")
    best_selection_metric = "FEM-grid RelL2" if reference is not None else "validation total loss"
    best_path = os.path.join(cfg.out_dir, "checkpoints", "best_checkpoint.pt")
    final_path = os.path.join(cfg.out_dir, "checkpoints", "final_checkpoint.pt")

    print(f"[INFO] Case: {cfg.case_name} | Model: {cfg.model_name}")
    print(f"[INFO] Device: {cfg.device} | Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"[INFO] Output directory: {os.path.abspath(cfg.out_dir)}")
    if reference is None:
        print(f"[WARN] FEM reference not found: {os.path.abspath(cfg.reference_path)}")
        print("[WARN] Training will continue; field metrics and RelL2 remain deferred. No reference is fabricated.")
    else:
        print(f"[INFO] FEM reference loaded and interpolated: {os.path.abspath(cfg.reference_path)}")

    start_time = time.time()
    completed_iter = 0
    for iteration in range(1, cfg.iters + 1):
        if torch.cuda.is_available() and cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        iter_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        col_tr = minibatch2(*data["col_tr"], cfg.batch_col, cfg.seed + iteration)
        bottom_tr = minibatch4(*data["bc_bottom_tr"], cfg.batch_bc, cfg.seed + 10000 + iteration)
        top_tr = minibatch4(*data["bc_top_tr"], cfg.batch_bc, cfg.seed + 20000 + iteration)
        left_tr = minibatch4(*data["bc_left_tr"], cfg.batch_bc, cfg.seed + 30000 + iteration)
        right_tr = minibatch4(*data["bc_right_tr"], cfg.batch_bc, cfg.seed + 40000 + iteration)
        loss, parts = batch_losses(cfg, model, col_tr, bottom_tr, top_tr, left_tr, right_tr)
        if not torch.isfinite(loss):
            print(f"[iter {iteration}] Non-finite loss detected; training stopped.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if torch.cuda.is_available() and cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        iter_times.append(time.time() - iter_start)
        completed_iter = iteration

        if iteration == 1 or iteration % cfg.eval_every == 0:
            model.eval()
            col_va = minibatch2(*data["col_va"], cfg.val_batch_col, cfg.seed + 50000 + iteration)
            bottom_va = minibatch4(*data["bc_bottom_va"], cfg.val_batch_bc, cfg.seed + 60000 + iteration)
            top_va = minibatch4(*data["bc_top_va"], cfg.val_batch_bc, cfg.seed + 70000 + iteration)
            left_va = minibatch4(*data["bc_left_va"], cfg.val_batch_bc, cfg.seed + 80000 + iteration)
            right_va = minibatch4(*data["bc_right_va"], cfg.val_batch_bc, cfg.seed + 90000 + iteration)
            val_loss_t, val_parts_t = batch_losses(
                cfg, model, col_va, bottom_va, top_va, left_va, right_va
            )
            train_parts = {key: float(value.detach().item()) for key, value in parts.items()}
            val_parts = {key: float(value.detach().item()) for key, value in val_parts_t.items()}
            val_loss = float(val_loss_t.detach().item())

            if reference is None and os.path.isfile(cfg.reference_path):
                reference = load_fem_reference(cfg, allow_missing=False)
                best_selection_metric = "FEM-grid RelL2"
                best_score = float("inf")
                print(f"[INFO] FEM reference became available at iter {iteration}; RelL2 evaluation enabled.")

            metrics = {key: float("nan") for key in METRIC_COLUMNS}
            if reference is not None:
                metrics = overall_grid_metrics(predict_grid(cfg, model, reference))
            rel_l2 = metrics["RelL2"]
            selection_score = rel_l2 if np.isfinite(rel_l2) else val_loss
            if selection_score < best_score:
                best_score = selection_score
                best_iter = iteration
                best_val_loss = val_loss
                best_rel_l2 = float(rel_l2) if np.isfinite(rel_l2) else None
                torch.save(
                    _checkpoint_payload(
                        cfg, model, optimizer, iteration, best_rel_l2,
                        best_val_loss, best_selection_metric,
                    ),
                    best_path,
                )

            elapsed = time.time() - start_time
            row = {
                "iter": iteration,
                "loss_total_train": float(loss.detach().item()),
                "loss_total_val": val_loss,
                "loss_pde_train": train_parts["pde"],
                "loss_pde_val": val_parts["pde"],
                "loss_mom_train": train_parts["mom"],
                "loss_mom_val": val_parts["mom"],
                "loss_div_train": train_parts["div"],
                "loss_div_val": val_parts["div"],
                "loss_bc_train": train_parts["bc"],
                "loss_bc_val": val_parts["bc"],
                "reference_available": reference is not None,
                "time_elapsed_sec": elapsed,
            }
            row.update({key + "_val": metrics[key] for key in METRIC_COLUMNS})
            append_history(history, row)
            print(
                f"[{iteration:6d}] loss_tr={float(loss.detach().item()):.3e} "
                f"loss_va={val_loss:.3e} RelL2={rel_l2:.3e} elapsed={elapsed / 60:.1f} min"
            )

    total_wall_time_sec = time.time() - start_time
    if not os.path.isfile(best_path):
        best_iter = completed_iter
        torch.save(
            _checkpoint_payload(
                cfg, model, optimizer, completed_iter, None, float("nan"), best_selection_metric
            ),
            best_path,
        )

    if reference is None and os.path.isfile(cfg.reference_path):
        reference = load_fem_reference(cfg, allow_missing=False)
    model.eval()
    final_grid = predict_grid(cfg, model, reference)
    final_metrics = overall_grid_metrics(final_grid) if reference is not None else {
        key: float("nan") for key in METRIC_COLUMNS
    }
    final_payload = _checkpoint_payload(
        cfg, model, optimizer, completed_iter, best_rel_l2, best_val_loss, best_selection_metric
    )
    final_payload["final_rel_l2"] = (
        float(final_metrics["RelL2"]) if np.isfinite(final_metrics["RelL2"]) else None
    )
    torch.save(final_payload, final_path)
    overall = save_all_outputs(
        cfg, model, final_grid, history, iter_times, total_wall_time_sec,
        best_iter, best_rel_l2, best_selection_metric,
    )
    print(f"[DONE] Total wall time: {total_wall_time_sec / 60:.2f} min")
    if np.isfinite(overall["RelL2"]):
        print(f"[DONE] Final overall FEM-grid RelL2: {overall['RelL2']:.6e}")
    else:
        print("[DEFERRED] Final field metrics/RelL2 require the FEM reference; no reference was fabricated.")
    print(f"[DONE] Outputs saved to: {os.path.abspath(cfg.out_dir)}")
    return model


def main() -> None:
    train(CavityConfig())


if __name__ == "__main__":
    main()
