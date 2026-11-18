"""Results aggregation module for cross-keep-rate and multi-seed evaluations."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Any
import numpy as np


def aggregate_multiseed_records(per_seed_csv_path: str) -> List[Dict[str, Any]]:
    """Aggregate per-seed evaluation records and compute mean, std, and paired delta."""
    if not os.path.isfile(per_seed_csv_path):
        raise FileNotFoundError(f"Per-seed CSV not found: {per_seed_csv_path}")

    with open(per_seed_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        m = r["model_id"]
        by_model.setdefault(m, []).append(r)

    results: List[Dict[str, Any]] = []

    for m_id, model_rows in by_model.items():
        seeds = [int(r["seed"]) for r in model_rows]
        top1_vals = [float(r["top1_accuracy"]) for r in model_rows]
        macro_vals = [float(r["macro_f1"]) for r in model_rows]
        weighted_vals = [float(r["weighted_f1"]) for r in model_rows]
        gas_vals = [float(r["gas_station_recall"]) for r in model_rows]
        city_vals = [float(r["city_street_recall"]) for r in model_rows]
        tunnel_vals = [float(r["tunnel_f1"]) for r in model_rows]

        top1_m, top1_s = float(np.mean(top1_vals)), float(np.std(top1_vals))
        macro_m, macro_s = float(np.mean(macro_vals)), float(np.std(macro_vals))
        weighted_m, weighted_s = float(np.mean(weighted_vals)), float(np.std(weighted_vals))
        gas_m, gas_s = float(np.mean(gas_vals)), float(np.std(gas_vals))
        city_m, city_s = float(np.mean(city_vals)), float(np.std(city_vals))
        tunnel_m, tunnel_s = float(np.mean(tunnel_vals)), float(np.std(tunnel_vals))

        results.append({
            "model_id": m_id,
            "num_seeds": len(seeds),
            "seeds": str(sorted(seeds)),
            "top1_accuracy_mean": round(top1_m, 2),
            "top1_accuracy_std": round(top1_s, 2),
            "top1_accuracy_str": f"{top1_m:.2f}% ± {top1_s:.2f}%",
            "macro_f1_mean": round(macro_m, 2),
            "macro_f1_std": round(macro_s, 2),
            "macro_f1_str": f"{macro_m:.2f}% ± {macro_s:.2f}%",
            "weighted_f1_mean": round(weighted_m, 2),
            "weighted_f1_std": round(weighted_s, 2),
            "weighted_f1_str": f"{weighted_m:.2f}% ± {weighted_s:.2f}%",
            "city_street_recall_mean": round(city_m, 2),
            "city_street_recall_std": round(city_s, 2),
            "gas_station_recall_mean": round(gas_m, 2),
            "gas_station_recall_std": round(gas_s, 2),
            "tunnel_f1_mean": round(tunnel_m, 2),
            "tunnel_f1_std": round(tunnel_s, 2),
        })

    # Compute paired delta (CMPR minus CB) if both exist
    if "K07-CB" in by_model and "K07-CB-CMPR" in by_model:
        cb_rows = {int(r["seed"]): r for r in by_model["K07-CB"]}
        cmpr_rows = {int(r["seed"]): r for r in by_model["K07-CB-CMPR"]}
        common_seeds = sorted(list(set(cb_rows.keys()) & set(cmpr_rows.keys())))

        delta_top1 = [float(cmpr_rows[s]["top1_accuracy"]) - float(cb_rows[s]["top1_accuracy"]) for s in common_seeds]
        delta_macro = [float(cmpr_rows[s]["macro_f1"]) - float(cb_rows[s]["macro_f1"]) for s in common_seeds]
        delta_weighted = [float(cmpr_rows[s]["weighted_f1"]) - float(cb_rows[s]["weighted_f1"]) for s in common_seeds]
        delta_city = [float(cmpr_rows[s]["city_street_recall"]) - float(cb_rows[s]["city_street_recall"]) for s in common_seeds]
        delta_gas = [float(cmpr_rows[s]["gas_station_recall"]) - float(cb_rows[s]["gas_station_recall"]) for s in common_seeds]
        delta_tunnel = [float(cmpr_rows[s]["tunnel_f1"]) - float(cb_rows[s]["tunnel_f1"]) for s in common_seeds]

        d_top1_m, d_top1_s = float(np.mean(delta_top1)), float(np.std(delta_top1))
        d_macro_m, d_macro_s = float(np.mean(delta_macro)), float(np.std(delta_macro))
        d_weighted_m, d_weighted_s = float(np.mean(delta_weighted)), float(np.std(delta_weighted))
        d_city_m, d_city_s = float(np.mean(delta_city)), float(np.std(delta_city))
        d_gas_m, d_gas_s = float(np.mean(delta_gas)), float(np.std(delta_gas))
        d_tunnel_m, d_tunnel_s = float(np.mean(delta_tunnel)), float(np.std(delta_tunnel))

        results.append({
            "model_id": "DELTA (K07-CB-CMPR minus K07-CB)",
            "num_seeds": len(common_seeds),
            "seeds": str(common_seeds),
            "top1_accuracy_mean": round(d_top1_m, 2),
            "top1_accuracy_std": round(d_top1_s, 2),
            "top1_accuracy_str": f"{'+' if d_top1_m >= 0 else ''}{d_top1_m:.2f}% ± {d_top1_s:.2f}%",
            "macro_f1_mean": round(d_macro_m, 2),
            "macro_f1_std": round(d_macro_s, 2),
            "macro_f1_str": f"{'+' if d_macro_m >= 0 else ''}{d_macro_m:.2f}% ± {d_macro_s:.2f}%",
            "weighted_f1_mean": round(d_weighted_m, 2),
            "weighted_f1_std": round(d_weighted_s, 2),
            "weighted_f1_str": f"{'+' if d_weighted_m >= 0 else ''}{d_weighted_m:.2f}% ± {d_weighted_s:.2f}%",
            "city_street_recall_mean": round(d_city_m, 2),
            "city_street_recall_std": round(d_city_s, 2),
            "gas_station_recall_mean": round(d_gas_m, 2),
            "gas_station_recall_std": round(d_gas_s, 2),
            "tunnel_f1_mean": round(d_tunnel_m, 2),
            "tunnel_f1_std": round(d_tunnel_s, 2),
        })

    return results
