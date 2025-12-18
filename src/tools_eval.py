# src/tools_eval.py
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any

from langchain.tools import tool

from session_state import (
    SUPPORTED,
    add_pair as _add_pair_state,
    list_pairs as _list_pairs_state,
    get_pairs as _get_pairs_state,
    has_pairs as _has_pairs,
    set_last_results as _set_last_results,
    get_last_results as _get_last_results,
    peek_pending as _peek_pending,
    pop_pending as _pop_pending,
)
from metrics_backend import evaluate_dirs, save_multi_report_txt


def _norm_metrics(metrics: List[str]) -> List[str]:
    m = [x.strip().lower() for x in metrics if x and x.strip()]
    # 去重但保序
    seen = set()
    out = []
    for x in m:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@tool
def add_pair(name: str, gt_dir: str, gen_dir: str, auto_continue: bool = True) -> Dict[str, Any]:
    """
    Add an evaluation folder pair with a name.
    If auto_continue=True and there is a pending request, continue evaluation using backend directly.
    """
    # 1) 校验路径
    if not os.path.isdir(gt_dir):
        return {"ok": False, "error": f"gt_dir not found: {gt_dir}"}
    if not os.path.isdir(gen_dir):
        return {"ok": False, "error": f"gen_dir not found: {gen_dir}"}

    # 2) 写入 session
    info = _add_pair_state(name=name, gt_dir=gt_dir, gen_dir=gen_dir, overwrite=True)

    response: Dict[str, Any] = {
        "ok": True,
        "pair": {"name": info.name, "gt_dir": info.gt_dir, "gen_dir": info.gen_dir},
        "num_pairs": len(_get_pairs_state(None)),
        "continued": False,
    }

    # 3) auto-continue（纯 backend：不调用任何 Tool）
    if auto_continue:
        pending = _peek_pending()
        if pending and pending.metrics:
            req = _pop_pending()

            # 3.1 选定要算的 pairs（None 表示全部）
            selected_pairs = _get_pairs_state(req.pair_names)  # name -> PairInfo

            # 3.2 逐 pair 调 backend 计算
            results = {}
            for pair_name, p in selected_pairs.items():
                stats = evaluate_dirs(
                    gt_dir=p.gt_dir,
                    gen_dir=p.gen_dir,
                    metrics=req.metrics,
                    im_res=(req.im_size, req.im_size),
                )
                results[pair_name] = stats

            # 3.3 写入 last_results（供 save_last_report 使用）
            _set_last_results(metrics=req.metrics, im_size=req.im_size, results=results)

            # 3.4 返回给 agent（agent 可以直接展示）
            response["continued"] = True
            response["continued_eval"] = {
                "ok": True,
                "metrics": req.metrics,
                "im_size": req.im_size,
                "evaluated_pairs": list(results.keys()),
                "results": results,
            }

    return response


@tool
def list_pairs() -> Dict[str, Any]:
    """List all stored evaluation pairs."""
    pairs = _list_pairs_state()
    return {
        "ok": True,
        "pairs": [{"name": p.name, "gt_dir": p.gt_dir, "gen_dir": p.gen_dir} for p in pairs],
        "num_pairs": len(pairs),
    }


@tool
def eval_pairs(metrics: List[str], pair_names: Optional[List[str]] = None, im_size: int = 256) -> Dict[str, Any]:
    """
    Evaluate metrics for multiple pairs.
    - metrics: e.g. ["psnr","ssim","lpips"]
    - pair_names: None => evaluate all pairs
    """
    metrics = _norm_metrics(metrics)
    if not metrics:
        return {"ok": False, "error": "No metrics provided."}

    # 校验指标合法
    bad = [m for m in metrics if m not in SUPPORTED]
    if bad:
        return {"ok": False, "error": f"Unsupported metrics: {bad}. Supported={sorted(SUPPORTED)}"}

    # 选择 pairs
    if not _has_pairs():
        # 这里不直接 set_pending（pending 通常由 Agent 决策触发）
        # 但如果你想工具层兜底，也可以在这里 set_pending
        return {"ok": False, "error": "No pairs added yet. Use add_pair first."}

    try:
        selected = _get_pairs_state(pair_names)  # name -> PairInfo
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # 逐 pair 计算
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, p in selected.items():
        stats = evaluate_dirs(
            gt_dir=p.gt_dir,
            gen_dir=p.gen_dir,
            metrics=metrics,
            im_res=(im_size, im_size),
        )
        results[name] = stats

    # 存 last_results
    _set_last_results(metrics=metrics, im_size=im_size, results=results)

    return {
        "ok": True,
        "metrics": metrics,
        "im_size": im_size,
        "evaluated_pairs": list(results.keys()),
        "results": results,
    }


@tool
def save_last_report(txt_path: str) -> Dict[str, Any]:
    """Save last evaluation results (multi-pair) to one txt."""
    lr = _get_last_results()
    if not lr:
        return {"ok": False, "error": "No last_results. Run eval_pairs first."}

    # 把 PairInfo 转成 name -> {"gt_dir","gen_dir"} 供 backend 写报告
    pairs = _get_pairs_state(None)
    session_pairs = {
        name: {"gt_dir": p.gt_dir, "gen_dir": p.gen_dir}
        for name, p in pairs.items()
        if name in lr.results
    }

    saved_to = save_multi_report_txt(
        path=txt_path,
        session_pairs=session_pairs,
        results=lr.results,
        metrics_order=lr.metrics,
    )
    return {"ok": True, "saved_to": saved_to}
