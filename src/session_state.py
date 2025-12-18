# src/session_state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import time

SUPPORTED = {"psnr", "ssim", "rmse", "lpips", "cw_ssim", "fsim", "sam"}

@dataclass
class PairInfo:
    name: str
    gt_dir: str
    gen_dir: str
    created_at: float

@dataclass
class PendingRequest:
    metrics: Optional[List[str]] = None
    pair_names: Optional[List[str]] = None  # None -> all
    im_size: int = 256
    reason: str = ""

@dataclass
class LastResults:
    metrics: List[str]
    im_size: int
    results: Dict[str, Dict[str, Dict[str, Any]]]  # pair -> metric -> stats
    created_at: float

SESSION: Dict[str, Any] = {
    "pairs": {},          # name -> PairInfo
    "pending": None,      # PendingRequest or None
    "last_results": None, # LastResults or None
}

# -------- pairs ops --------
def has_pairs() -> bool:
    return len(SESSION["pairs"]) > 0

def add_pair(name: str, gt_dir: str, gen_dir: str, overwrite: bool = True) -> PairInfo:
    pairs: Dict[str, PairInfo] = SESSION["pairs"]
    if (name in pairs) and (not overwrite):
        raise ValueError(f"pair '{name}' already exists")
    info = PairInfo(name=name, gt_dir=gt_dir, gen_dir=gen_dir, created_at=time.time())
    pairs[name] = info
    return info

def remove_pair(name: str) -> None:
    pairs: Dict[str, PairInfo] = SESSION["pairs"]
    if name in pairs:
        del pairs[name]

def list_pairs() -> List[PairInfo]:
    return list(SESSION["pairs"].values())

def get_pairs(pair_names: Optional[List[str]] = None) -> Dict[str, PairInfo]:
    pairs: Dict[str, PairInfo] = SESSION["pairs"]
    if pair_names is None:
        return pairs
    missing = [n for n in pair_names if n not in pairs]
    if missing:
        raise ValueError(f"unknown pair names: {missing}")
    return {n: pairs[n] for n in pair_names}

# -------- pending ops --------
def set_pending(metrics: List[str], pair_names: Optional[List[str]], im_size: int, reason: str) -> PendingRequest:
    req = PendingRequest(metrics=metrics, pair_names=pair_names, im_size=im_size, reason=reason)
    SESSION["pending"] = req
    return req

def peek_pending() -> Optional[PendingRequest]:
    return SESSION.get("pending")

def pop_pending() -> Optional[PendingRequest]:
    req = SESSION.get("pending")
    SESSION["pending"] = None
    return req

def clear_pending() -> None:
    SESSION["pending"] = None

# -------- last results ops --------
def set_last_results(metrics: List[str], im_size: int, results: Dict[str, Any]) -> LastResults:
    lr = LastResults(metrics=metrics, im_size=im_size, results=results, created_at=time.time())
    SESSION["last_results"] = lr
    return lr

def get_last_results() -> Optional[LastResults]:
    return SESSION.get("last_results")

def clear_last_results() -> None:
    SESSION["last_results"] = None
