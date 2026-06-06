import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

Event = Tuple[float, float]


@dataclass
class ClinicalEvalConfig:
    window_step_s: float = 1.0
    threshold: float = 0.5
    moving_avg_size: int = 5
    min_event_duration_s: float = 0.0
    merge_gap_s: float = 30.0
    pre_ictal_tolerance_s: float = 0.0
    post_ictal_tolerance_s: float = 0.0
    max_event_duration_s: Optional[float] = None
    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalEvaluator:
    def __init__(self, config: ClinicalEvalConfig):
        self.cfg = config

    def _safe_array(self, x):
        x = np.asarray(x, dtype=np.float32)
        return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    def _moving_average(self, x: np.ndarray) -> np.ndarray:
        k = max(1, int(self.cfg.moving_avg_size))
        if k == 1:
            return x
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(x, kernel, mode="same")

    def _binarize(self, probs: np.ndarray) -> np.ndarray:
        return (probs >= float(self.cfg.threshold)).astype(np.uint8)

    def _binary_to_events(self, binary: np.ndarray) -> List[Event]:
        events: List[Event] = []
        in_event = False
        start_idx = 0

        for i, val in enumerate(binary):
            if val == 1 and not in_event:
                in_event = True
                start_idx = i
            elif val == 0 and in_event:
                events.append((start_idx * self.cfg.window_step_s, i * self.cfg.window_step_s))
                in_event = False

        if in_event:
            events.append((start_idx * self.cfg.window_step_s, len(binary) * self.cfg.window_step_s))

        return events

    def _filter_short_events(self, events: List[Event]) -> List[Event]:
        if self.cfg.min_event_duration_s is None or self.cfg.min_event_duration_s <= 0:
            return events
        return [(s, e) for s, e in events if (e - s) >= self.cfg.min_event_duration_s]

    def _merge_close_events(self, events: List[Event]) -> List[Event]:
        if not events:
            return events

        if self.cfg.merge_gap_s is None or self.cfg.merge_gap_s <= 0:
            return sorted(events, key=lambda x: x[0])

        events = sorted(events, key=lambda x: x[0])
        merged = [events[0]]

        for s, e in events[1:]:
            ls, le = merged[-1]
            if (s - le) < self.cfg.merge_gap_s:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))

        return merged

    def _split_long_events(self, events: List[Event]) -> List[Event]:
        if self.cfg.max_event_duration_s is None or self.cfg.max_event_duration_s <= 0:
            return events

        out: List[Event] = []
        for s, e in events:
            if (e - s) <= self.cfg.max_event_duration_s:
                out.append((s, e))
            else:
                cur = s
                while cur < e:
                    nxt = min(cur + self.cfg.max_event_duration_s, e)
                    out.append((cur, nxt))
                    cur = nxt
        return out

    def _apply_tolerance_to_reference(self, ref_events: List[Event]) -> List[Event]:
        out = []
        for s, e in ref_events:
            out.append((
                max(0.0, s - self.cfg.pre_ictal_tolerance_s),
                e + self.cfg.post_ictal_tolerance_s
            ))
        return out

    @staticmethod
    def _overlap(ev1: Event, ev2: Event) -> float:
        s1, e1 = ev1
        s2, e2 = ev2
        return max(0.0, min(e1, e2) - max(s1, s2))

    def _match_events(self, pred_events: List[Event], ref_events: List[Event]) -> Dict[str, Any]:
        matched_ref = set()
        matched_pred = set()
        latencies = []

        tolerant_refs = self._apply_tolerance_to_reference(ref_events)

        for i, pred in enumerate(pred_events):
            best_j = None
            best_overlap = 0.0

            for j, ref_tol in enumerate(tolerant_refs):
                if j in matched_ref:
                    continue
                ov = self._overlap(pred, ref_tol)
                if ov > best_overlap:
                    best_overlap = ov
                    best_j = j

            if best_j is not None and best_overlap > 0:
                matched_ref.add(best_j)
                matched_pred.add(i)
                latencies.append(pred[0] - ref_events[best_j][0])

        tp = len(matched_ref)
        fp = len(pred_events) - len(matched_pred)
        fn = len(ref_events) - len(matched_ref)

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "latencies": latencies,
        }

    def postprocess_predictions(self, y_pred_prob: np.ndarray) -> List[Event]:
        probs = self._safe_array(y_pred_prob)
        probs = self._moving_average(probs)
        binary = self._binarize(probs)
        events = self._binary_to_events(binary)
        events = self._filter_short_events(events)
        events = self._merge_close_events(events)
        events = self._split_long_events(events)
        return events

    def evaluate_record(self, y_pred_prob: np.ndarray, ref_events: List[Event], record_duration_s: float) -> Dict[str, Any]:
        pred_events = self.postprocess_predictions(y_pred_prob)
        m = self._match_events(pred_events, ref_events)

        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        sens = tp / (tp + fn + self.cfg.eps)
        prec = tp / (tp + fp + self.cfg.eps)
        f1 = 2 * sens * prec / (sens + prec + self.cfg.eps)
        fa_h = fp / (record_duration_s / 3600.0 + self.cfg.eps)

        lat = m["latencies"]

        return {
            "n_ref_events": len(ref_events),
            "n_pred_events": len(pred_events),
            "tp_event": tp,
            "fp_event": fp,
            "fn_event": fn,
            "event_sensitivity": sens,
            "event_precision": prec,
            "event_f1": f1,
            "fa_per_hour": fa_h,
            "mean_latency_s": float(np.mean(lat)) if lat else None,
            "median_latency_s": float(np.median(lat)) if lat else None,
            "pred_events": pred_events,
        }

    def evaluate_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_tp = total_fp = total_fn = 0
        total_duration_s = 0.0
        all_lat = []
        per_record = []

        for rec in records:
            r = self.evaluate_record(
                rec["y_pred_prob"],
                rec["ref_events"],
                rec["record_duration_s"],
            )
            per_record.append(r)
            total_tp += r["tp_event"]
            total_fp += r["fp_event"]
            total_fn += r["fn_event"]
            total_duration_s += rec["record_duration_s"]

            if r["mean_latency_s"] is not None:
                all_lat.extend(self._match_events(r["pred_events"], rec["ref_events"])["latencies"])

        sens = total_tp / (total_tp + total_fn + self.cfg.eps)
        prec = total_tp / (total_tp + total_fp + self.cfg.eps)
        f1 = 2 * sens * prec / (sens + prec + self.cfg.eps)
        fa_h = total_fp / (total_duration_s / 3600.0 + self.cfg.eps)

        return {
            "n_records": len(records),
            "tp_event": total_tp,
            "fp_event": total_fp,
            "fn_event": total_fn,
            "event_sensitivity": sens,
            "event_precision": prec,
            "event_f1": f1,
            "fa_per_hour": fa_h,
            "mean_latency_s": float(np.mean(all_lat)) if all_lat else None,
            "median_latency_s": float(np.median(all_lat)) if all_lat else None,
            "per_record": per_record,
            "config": self.cfg.to_dict(),
        }


def clinical_score(metrics: Dict[str, Any], fa_penalty: float = 0.10, min_sensitivity: float = 0.80) -> float:
    sens = float(metrics["event_sensitivity"])
    fa_h = float(metrics["fa_per_hour"])
    f1 = float(metrics["event_f1"])

    if sens < min_sensitivity:
        return -1e9 + sens

    return (2.0 * sens) + (1.0 * f1) - (fa_penalty * fa_h)


def extract_record_id(filepath: str) -> str:
    name = Path(filepath).stem
    m = re.match(r"^(.*)_win\d+$", name)
    return m.group(1) if m else name


def load_index_csv(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row["filepath"].strip()
            lbl = int(row["label"])
            rows.append({"filepath": fp, "label": lbl, "record_id": extract_record_id(fp)})
    return rows


def windows_to_events(
    labels: List[int],
    window_step_s: float = 1.0,
    min_event_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> List[Event]:
    events = []
    in_event = False
    start_idx = 0

    for i, val in enumerate(labels):
        if val == 1 and not in_event:
            in_event = True
            start_idx = i
        elif val == 0 and in_event:
            events.append((start_idx * window_step_s, i * window_step_s))
            in_event = False

    if in_event:
        events.append((start_idx * window_step_s, len(labels) * window_step_s))

    if min_event_duration_s > 0:
        events = [(s, e) for s, e in events if (e - s) >= min_event_duration_s]

    if merge_gap_s > 0 and events:
        events = sorted(events, key=lambda x: x[0])
        merged = [events[0]]
        for s, e in events[1:]:
            ls, le = merged[-1]
            if s - le < merge_gap_s:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        events = merged

    return events


def group_index_by_record(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["record_id"], []).append(row)
    for rid in grouped:
        grouped[rid] = sorted(grouped[rid], key=lambda r: r["filepath"])
    return grouped


def build_reference_events_from_index(
    csv_path: str,
    window_step_s: float = 1.0,
    min_event_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> Dict[str, Dict]:
    rows = load_index_csv(csv_path)
    grouped = group_index_by_record(rows)
    out = {}

    for rid, rec_rows in grouped.items():
        labels = [r["label"] for r in rec_rows]
        ref_events = windows_to_events(
            labels,
            window_step_s=window_step_s,
            min_event_duration_s=min_event_duration_s,
            merge_gap_s=merge_gap_s,
        )
        out[rid] = {
            "record_id": rid,
            "n_windows": len(labels),
            "record_duration_s": len(labels) * window_step_s,
            "ref_events": ref_events,
            "labels": labels,
            "filepaths": [r["filepath"] for r in rec_rows],
        }

    return out


def save_reference_events_json(ref_dict: Dict[str, Dict], out_path: str) -> None:
    serializable = {
        rid: {
            "record_id": v["record_id"],
            "n_windows": v["n_windows"],
            "record_duration_s": v["record_duration_s"],
            "ref_events": [list(ev) for ev in v["ref_events"]],
        }
        for rid, v in ref_dict.items()
    }
    Path(out_path).write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prediction_array(pt_path: str):
    try:
        import torch

        obj = torch.load(pt_path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ["y_pred_prob", "probs", "preds", "logits"]:
                if key in obj:
                    arr = obj[key]
                    break
            else:
                arr = next(iter(obj.values()))
        else:
            arr = obj

        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()

        return np.asarray(arr)

    except Exception:
        return np.load(pt_path)


if __name__ == "__main__":
    print("Module ready. Import these functions/classes in your training or evaluation pipeline.")