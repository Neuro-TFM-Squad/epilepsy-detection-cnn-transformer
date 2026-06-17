import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from src.utils.postprocessing import PostProcessConfig, PredictionPostProcessor

Event = Tuple[float, float]


@dataclass
class ClinicalEvalConfig:
    pre_ictal_tolerance_s: float = 0.0
    post_ictal_tolerance_s: float = 0.0
    min_overlap_s: float = 0.0
    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalEvaluator:
    def __init__(
        self,
        eval_config: ClinicalEvalConfig,
        postprocess_config: Optional[PostProcessConfig] = None,
    ):
        self.eval_cfg = eval_config
        self.post_cfg = postprocess_config
        self.postprocessor = (
            PredictionPostProcessor(postprocess_config)
            if postprocess_config is not None else None
        )

    def _apply_tolerance_to_reference(self, ref_events: List[Event]) -> List[Event]:
        out = []
        for s, e in ref_events:
            out.append((
                max(0.0, float(s) - self.eval_cfg.pre_ictal_tolerance_s),
                float(e) + self.eval_cfg.post_ictal_tolerance_s
            ))
        return out

    @staticmethod
    def _event_overlap(ev1: Event, ev2: Event) -> float:
        s1, e1 = ev1
        s2, e2 = ev2
        return max(0.0, min(e1, e2) - max(s1, s2))

    def _match_events_reference_centered(
        self,
        pred_events: List[Event],
        ref_events: List[Event],
    ) -> Dict[str, Any]:
        tolerant_refs = self._apply_tolerance_to_reference(ref_events)

        used_pred_indices = set()
        matched_ref_indices = set()
        matches = []

        for ref_idx, (ref, ref_tol) in enumerate(zip(ref_events, tolerant_refs)):
            overlapping_pred = []
            for pred_idx, pred in enumerate(pred_events):
                overlap = self._event_overlap(pred, ref_tol)
                if overlap >= self.eval_cfg.min_overlap_s and overlap > 0.0:
                    overlapping_pred.append((pred_idx, pred, overlap))

            if len(overlapping_pred) == 0:
                continue

            matched_ref_indices.add(ref_idx)

            overlapping_pred = sorted(overlapping_pred, key=lambda x: x[1][0])
            for pred_idx, pred, _ in overlapping_pred:
                used_pred_indices.add(pred_idx)

            first_pred_idx, first_pred, _ = overlapping_pred[0]
            ref_start, ref_end = ref
            pred_start, pred_end = first_pred

            total_overlap = float(sum(x[2] for x in overlapping_pred))

            matches.append({
                "ref_idx": ref_idx,
                "matched_pred_indices": [x[0] for x in overlapping_pred],
                "n_matched_pred_fragments": len(overlapping_pred),
                "first_pred_idx": first_pred_idx,
                "pred_start_s": float(pred_start),
                "pred_end_s": float(pred_end),
                "pred_duration_s": float(pred_end - pred_start),
                "ref_start_s": float(ref_start),
                "ref_end_s": float(ref_end),
                "ref_duration_s": float(ref_end - ref_start),
                "total_overlap_s": total_overlap,
                "latency_to_onset_s": float(pred_start - ref_start),
            })

        unmatched_ref_indices = [j for j in range(len(ref_events)) if j not in matched_ref_indices]
        unmatched_pred_indices = [i for i in range(len(pred_events)) if i not in used_pred_indices]

        return {
            "tp": len(matches),
            "fp": len(unmatched_pred_indices),
            "fn": len(unmatched_ref_indices),
            "matches": matches,
            "matched_ref_indices": sorted(list(matched_ref_indices)),
            "used_pred_indices": sorted(list(used_pred_indices)),
            "unmatched_ref_indices": unmatched_ref_indices,
            "unmatched_pred_indices": unmatched_pred_indices,
        }

    def _build_event_table(
        self,
        pred_events: List[Event],
        ref_events: List[Event],
        match_info: Dict[str, Any],
        record_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = []

        matched_ref = set(match_info["matched_ref_indices"])
        used_pred = set(match_info["used_pred_indices"])

        for m in match_info["matches"]:
            rows.append({
                "record_id": record_id,
                "event_type": "TP",
                "ref_idx": m["ref_idx"],
                "matched_pred_indices": json.dumps(m["matched_pred_indices"]),
                "n_matched_pred_fragments": m["n_matched_pred_fragments"],
                "pred_start_s": m["pred_start_s"],
                "pred_end_s": m["pred_end_s"],
                "pred_duration_s": m["pred_duration_s"],
                "ref_start_s": m["ref_start_s"],
                "ref_end_s": m["ref_end_s"],
                "ref_duration_s": m["ref_duration_s"],
                "total_overlap_s": m["total_overlap_s"],
                "latency_to_onset_s": m["latency_to_onset_s"],
            })

        for pred_idx, (s, e) in enumerate(pred_events):
            if pred_idx in used_pred:
                continue
            rows.append({
                "record_id": record_id,
                "event_type": "FP",
                "ref_idx": None,
                "matched_pred_indices": json.dumps([pred_idx]),
                "n_matched_pred_fragments": 1,
                "pred_start_s": float(s),
                "pred_end_s": float(e),
                "pred_duration_s": float(e - s),
                "ref_start_s": None,
                "ref_end_s": None,
                "ref_duration_s": None,
                "total_overlap_s": 0.0,
                "latency_to_onset_s": None,
            })

        for ref_idx, (s, e) in enumerate(ref_events):
            if ref_idx in matched_ref:
                continue
            rows.append({
                "record_id": record_id,
                "event_type": "FN",
                "ref_idx": ref_idx,
                "matched_pred_indices": json.dumps([]),
                "n_matched_pred_fragments": 0,
                "pred_start_s": None,
                "pred_end_s": None,
                "pred_duration_s": None,
                "ref_start_s": float(s),
                "ref_end_s": float(e),
                "ref_duration_s": float(e - s),
                "total_overlap_s": 0.0,
                "latency_to_onset_s": None,
            })

        rows = sorted(
            rows,
            key=lambda r: (
                r["record_id"] if r["record_id"] is not None else "",
                float("inf") if r["ref_start_s"] is None else r["ref_start_s"],
                float("inf") if r["pred_start_s"] is None else r["pred_start_s"],
            )
        )
        return rows

    def events_from_probabilities(
        self,
        y_pred_prob,
        return_intermediates: bool = False,
    ):
        if self.postprocessor is None:
            raise ValueError("Se requiere postprocess_config para evaluar desde probabilidades.")
        return self.postprocessor.process(y_pred_prob, return_intermediates=return_intermediates)

    def evaluate_record_from_events(
        self,
        pred_events: List[Event],
        ref_events: List[Event],
        record_duration_s: float,
        record_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        pred_events = [(float(s), float(e)) for s, e in pred_events]
        ref_events = [(float(s), float(e)) for s, e in ref_events]

        match_info = self._match_events_reference_centered(pred_events, ref_events)

        tp = match_info["tp"]
        fp = match_info["fp"]
        fn = match_info["fn"]

        sens = tp / (tp + fn + self.eval_cfg.eps)
        prec = tp / (tp + fp + self.eval_cfg.eps)
        f1 = 2.0 * sens * prec / (sens + prec + self.eval_cfg.eps)
        fa_per_hour = fp / (record_duration_s / 3600.0 + self.eval_cfg.eps)

        latencies = [m["latency_to_onset_s"] for m in match_info["matches"]]
        event_rows = self._build_event_table(
            pred_events=pred_events,
            ref_events=ref_events,
            match_info=match_info,
            record_id=record_id,
        )

        return {
            "record_id": record_id,
            "n_ref_events": len(ref_events),
            "n_pred_events": len(pred_events),
            "tp_event": tp,
            "fp_event": fp,
            "fn_event": fn,
            "event_sensitivity": sens,
            "event_precision": prec,
            "event_f1": f1,
            "fa_per_hour": fa_per_hour,
            "mean_latency_s": float(np.mean(latencies)) if latencies else None,
            "median_latency_s": float(np.median(latencies)) if latencies else None,
            "latencies_s": latencies,
            "pred_events": pred_events,
            "ref_events": ref_events,
            "matches": match_info["matches"],
            "event_rows": event_rows,
        }

    def evaluate_record(
        self,
        y_pred_prob,
        ref_events: List[Event],
        record_duration_s: float,
        record_id: Optional[str] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        processed = self.events_from_probabilities(y_pred_prob, return_intermediates=return_intermediates)
        pred_events = processed["events_final"] if return_intermediates else processed

        result = self.evaluate_record_from_events(
            pred_events=pred_events,
            ref_events=ref_events,
            record_duration_s=record_duration_s,
            record_id=record_id,
        )

        if return_intermediates:
            result["postprocess"] = processed

        return result

    def evaluate_dataset(
        self,
        records: List[Dict[str, Any]],
        use_probabilities: bool = True,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_duration_s = 0.0

        per_record = []
        all_latencies = []
        all_event_rows = []

        for rec in records:
            record_id = rec.get("record_id", None)
            ref_events = rec["ref_events"]
            record_duration_s = float(rec["record_duration_s"])

            if use_probabilities:
                result = self.evaluate_record(
                    y_pred_prob=rec["y_pred_prob"],
                    ref_events=ref_events,
                    record_duration_s=record_duration_s,
                    record_id=record_id,
                    return_intermediates=return_intermediates,
                )
            else:
                result = self.evaluate_record_from_events(
                    pred_events=rec["pred_events"],
                    ref_events=ref_events,
                    record_duration_s=record_duration_s,
                    record_id=record_id,
                )

            per_record.append(result)
            total_tp += result["tp_event"]
            total_fp += result["fp_event"]
            total_fn += result["fn_event"]
            total_duration_s += record_duration_s
            all_latencies.extend(result["latencies_s"])
            all_event_rows.extend(result["event_rows"])

        sens = total_tp / (total_tp + total_fn + self.eval_cfg.eps)
        prec = total_tp / (total_tp + total_fp + self.eval_cfg.eps)
        f1 = 2.0 * sens * prec / (sens + prec + self.eval_cfg.eps)
        fa_per_hour = total_fp / (total_duration_s / 3600.0 + self.eval_cfg.eps)

        return {
            "n_records": len(records),
            "tp_event": total_tp,
            "fp_event": total_fp,
            "fn_event": total_fn,
            "event_sensitivity": sens,
            "event_precision": prec,
            "event_f1": f1,
            "fa_per_hour": fa_per_hour,
            "mean_latency_s": float(np.mean(all_latencies)) if all_latencies else None,
            "median_latency_s": float(np.median(all_latencies)) if all_latencies else None,
            "per_record": per_record,
            "event_rows": all_event_rows,
            "eval_config": self.eval_cfg.to_dict(),
            "postprocess_config": self.post_cfg.to_dict() if self.post_cfg is not None else None,
        }


def clinical_score(
    metrics: Dict[str, Any],
    fa_penalty: float = 0.10,
    min_sensitivity: float = 0.0,
    latency_penalty: float = 0.0,
) -> float:
    sens = float(metrics["event_sensitivity"])
    fa_h = float(metrics["fa_per_hour"])
    f1 = float(metrics["event_f1"])
    latency = metrics.get("median_latency_s", None)

    if sens < min_sensitivity:
        return -1e9 + sens

    score = (2.0 * sens) + (1.0 * f1) - (fa_penalty * fa_h)

    if latency_penalty > 0.0 and latency is not None:
        score -= latency_penalty * max(0.0, float(latency))

    return score