"""Shared helpers for the experiment scripts."""

import json
import os
import platform
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
COST_MODEL = RESULTS / "cost_model.json"

RESULTS.mkdir(exist_ok=True)


def environment_info() -> dict:
    import networkx
    import numba
    import scipy
    import torch

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "numba": numba.__version__,
        "torch": torch.__version__,
    }
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass
    return info


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def mean_ci(values, confidence=0.95):
    """Mean, std and Student-t confidence interval half-width."""
    from scipy import stats

    values = np.asarray(values, dtype=np.float64)
    n = values.size
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, half


def median(values):
    return float(statistics.median(values))
