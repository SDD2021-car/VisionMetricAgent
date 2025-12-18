# src/metrics_backend.py
from __future__ import annotations
import os
import numpy as np
from typing import Dict, Any, List, Tuple

# ✅ 引入你现有的计算函数
# 假设文件名是 psnr_ssim_all.py，且里面有 SSIMs_PSNRs 函数
from psnr_ssim_all import SSIMs_PSNRs


SUPPORTED = {"psnr", "ssim", "rmse", "lpips", "cw_ssim", "fsim", "sam"}

def _stats(values) -> Dict[str, Any]:
    """Compute mean/var/min/max/std/n from a 1D array-like."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]  # 防止 inf/nan 影响统计（可选）
    if arr.size == 0:
        return {"n": 0, "mean": None, "var": None, "min": None, "max": None, "std": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "var": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }

def evaluate_dirs(
    gt_dir: str,
    gen_dir: str,
    metrics: List[str],
    im_res: Tuple[int, int] = (256, 256),
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate selected metrics for ONE folder pair (gt_dir, gen_dir).
    Returns:
        {
          "psnr": {"n":..., "mean":..., "var":..., "min":..., "max":..., "std":...},
          "ssim": {...},
          ...
        }
    """
    metrics = [m.lower() for m in metrics]
    for m in metrics:
        if m not in SUPPORTED:
            raise ValueError(f"Unsupported metric: {m}. Supported: {sorted(SUPPORTED)}")

    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    if not os.path.isdir(gen_dir):
        raise FileNotFoundError(f"gen_dir not found: {gen_dir}")

    # 1) 调用你现有的函数，一次性拿到所有逐图数组
    SSIM_measures, PSNR_measures, RMSE_measures, cw_SSIM_measures, LPIPS_measures, FSIM_measures, SAM_measures = \
        SSIMs_PSNRs(gt_dir, gen_dir, im_res=im_res)

    # 2) 建立 “指标名 -> 数组” 的映射
    mapping = {
        "ssim": SSIM_measures,
        "psnr": PSNR_measures,
        "rmse": RMSE_measures,
        "cw_ssim": cw_SSIM_measures,
        "lpips": LPIPS_measures,
        "fsim": FSIM_measures,
        "sam": SAM_measures,
    }

    # 3) 只挑选用户指定 metrics，并计算统计
    out: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        out[m] = _stats(mapping[m])

    return out

def save_multi_report_txt(
    path: str,
    session_pairs: Dict[str, Dict[str, str]],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    metrics_order: List[str] | None = None,
) -> str:
    """
    Save multi-pair results to ONE txt.
    session_pairs: name -> {"gt_dir":..., "gen_dir":...}
    results: name -> metric -> stats
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # 可选：指定输出指标顺序
    if metrics_order is None:
        # 取第一对里的指标顺序
        any_pair = next(iter(results.values()), {})
        metrics_order = list(any_pair.keys())

    lines = []
    lines.append("=== Multi-Pair Evaluation Report ===")
    lines.append("")

    for pair_name, pair_info in session_pairs.items():
        if pair_name not in results:
            continue
        lines.append(f"=== PAIR: {pair_name} ===")
        lines.append(f"GT_DIR : {pair_info['gt_dir']}")
        lines.append(f"GEN_DIR: {pair_info['gen_dir']}")
        for m in metrics_order:
            if m not in results[pair_name]:
                continue
            s = results[pair_name][m]
            lines.append(
                f"{m.upper():8s} | n={s['n']} "
                f"mean={_fmt(s['mean'])} var={_fmt(s['var'])} "
                f"min={_fmt(s['min'])} max={_fmt(s['max'])} std={_fmt(s['std'])}"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path

def _fmt(x):
    if x is None:
        return "None"
    return f"{x:.6f}"
