from pathlib import Path
import json
from itertools import product

import numpy as np
import pandas as pd

from src.utils.evaluador_clinico import (
    ClinicalEvalConfig,
    ClinicalEvaluator,
    clinical_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_CSV = PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s" / "chbmit_index_val_56.csv"
PREDICTIONS_CSV = PROJECT_ROOT / "experiments" / "predictions" / "dwt" / "def" / "predictions_per_window.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "clinical_eval" / "dwt" / "def"

VAL_PATIENTS = ("chb05", "chb06")
WINDOW_STEP_S = 1.0

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]
MOVING_AVG_SIZES = [1, 3]
MIN_EVENT_DURATIONS = [1.0, 3.0, 5.0]
MERGE_GAPS = [0.0, 3.0, 5.0]
MAX_EVENT_DURATIONS = [None, 60.0]

MIN_GLOBAL_SENSITIVITY = 0.80
FA_PENALTY = 0.10


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_ref_events_from_group(group: pd.DataFrame):
    ref_events = []
    in_event = False
    ev_start = None
    ev_end = None

    for _, row in group.iterrows():
        label = int(row["label"])
        start_s = float(row["start_sec"])
        end_s = float(row["end_sec"])

        if label == 1 and not in_event:
            in_event = True
            ev_start = start_s
            ev_end = end_s
        elif label == 1 and in_event:
            ev_end = end_s
        elif label == 0 and in_event:
            ref_events.append((ev_start, ev_end))
            in_event = False
            ev_start, ev_end = None, None

    if in_event:
        ref_events.append((ev_start, ev_end))

    return ref_events


def load_index_records(csv_path: Path, target_patients=VAL_PATIENTS):
    df = pd.read_csv(csv_path)

    required = {
        "patient_id", "record_id", "filepath",
        "window_index", "start_sec", "end_sec", "label"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en INDEX_CSV: {missing}")

    df = df[df["patient_id"].isin(target_patients)].copy()
    df["record_id"] = df["record_id"].astype(str).str.replace(".edf", "", regex=False).str.strip()
    df["window_index"] = df["window_index"].astype(int)
    df = df.sort_values(["patient_id", "record_id", "window_index"]).reset_index(drop=True)

    records = {}
    for record_id, group in df.groupby("record_id", sort=False):
        group = group.sort_values("window_index").reset_index(drop=True)
        first_row = group.iloc[0]

        records[record_id] = {
            "patient_id": str(first_row["patient_id"]),
            "record_id": str(record_id),
            "source_file": str(first_row["filepath"]),
            "ref_events": build_ref_events_from_group(group),
            "n_windows_ref": int(len(group)),
            "window_index_ref": group["window_index"].to_numpy(dtype=np.int32),
        }

    return records


def load_predictions_per_window(pred_csv_path: Path, target_patients=VAL_PATIENTS):
    if not pred_csv_path.exists():
        raise FileNotFoundError(f"No existe PREDICTIONS_CSV: {pred_csv_path}")

    df = pd.read_csv(pred_csv_path)

    required = {"record_id", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en predictions_per_window.csv: {missing}")

    if "patient_id" in df.columns:
        df = df[df["patient_id"].isin(target_patients)].copy()

    df["record_id"] = df["record_id"].astype(str).str.replace(".edf", "", regex=False).str.strip()
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    if "window_index" in df.columns:
        df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
        df = df.dropna(subset=["window_index"]).copy()
        df["window_index"] = df["window_index"].astype(int)
        df = df.sort_values(["record_id", "window_index"]).reset_index(drop=True)
    else:
        df = df.sort_values(["record_id"]).reset_index(drop=True)

    return df


def save_reference_events_json(ref_dict, out_path: Path):
    serializable = {}
    for rid, info in ref_dict.items():
        serializable[rid] = {
            "patient_id": info["patient_id"],
            "record_id": info["record_id"],
            "source_file": info["source_file"],
            "ref_events": [list(ev) for ev in info["ref_events"]],
            "n_windows_ref": info["n_windows_ref"],
        }

    out_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def list_prediction_record_ids(pred_df: pd.DataFrame):
    return sorted(pred_df["record_id"].astype(str).unique().tolist())


def debug_alignment(ref_dict, pred_df: pd.DataFrame, max_show: int = 20):
    ref_ids = sorted(ref_dict.keys())
    pred_ids = list_prediction_record_ids(pred_df)

    ref_set = set(ref_ids)
    pred_set = set(pred_ids)

    common_ids = sorted(ref_set & pred_set)
    missing_pred = sorted(ref_set - pred_set)
    extra_pred = sorted(pred_set - ref_set)

    print("\n=== DEBUG ALIGNMENT ===")
    print(f"Referencia record_ids : {len(ref_ids)}")
    print(f"Predicción record_ids : {len(pred_ids)}")
    print(f"Coinciden            : {len(common_ids)}")
    print(f"Sin predicción       : {len(missing_pred)}")
    print(f"Pred extras          : {len(extra_pred)}")

    print("\nPrimeros record_ids de referencia:")
    print(ref_ids[:max_show])

    print("\nPrimeros record_ids de predicción:")
    print(pred_ids[:max_show])

    if missing_pred:
        print("\nRecord_ids sin predicción:")
        print(missing_pred[:max_show])

    if extra_pred:
        print("\nRecord_ids extra en predicciones:")
        print(extra_pred[:max_show])

    print("\nResumen de eventos de referencia:")
    positive_records = []
    for rid in ref_ids:
        n_events = len(ref_dict[rid]["ref_events"])
        if n_events > 0:
            positive_records.append((rid, n_events, ref_dict[rid]["patient_id"]))
    print(f"Registros con al menos 1 evento real: {len(positive_records)}")
    print(positive_records[:max_show])


def normalize_prediction_array(pred):
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.squeeze(pred)
    pred = np.ravel(pred)
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    pred = np.clip(pred, 0.0, 1.0)
    return pred.astype(np.float32)


def build_dataset_records(ref_dict, pred_df: pd.DataFrame):
    dataset_records = []
    skipped_records = []

    print("\nConstruyendo dataset_records desde predictions_per_window.csv...")

    for record_id, record_info in ref_dict.items():
        group = pred_df[pred_df["record_id"] == record_id].copy()

        if len(group) == 0:
            skipped_records.append(record_id)
            print(f"[SKIP] {record_id} sin filas en predictions_per_window.csv")
            continue

        if "window_index" in group.columns:
            group = group.sort_values("window_index").reset_index(drop=True)

        y_pred_prob = normalize_prediction_array(group["prob"].to_numpy(dtype=np.float32))
        ref_events = record_info["ref_events"]
        record_duration_s = float(len(y_pred_prob)) * WINDOW_STEP_S

        n_ref = record_info["n_windows_ref"]
        n_pred = len(y_pred_prob)
        pmax = float(np.max(y_pred_prob))
        pmean = float(np.mean(y_pred_prob))
        p95 = float(np.percentile(y_pred_prob, 95))
        pgt05 = int((y_pred_prob >= 0.5).sum())
        print(
            f"[MATCH] {record_id} | "
            f"n_probs={n_pred} | "
            f"n_ref_windows={n_ref} | "
            f"n_ref_events={len(ref_events)} | "
            f"duration_s={record_duration_s}"

            f"[STATS] {record_id} | "
            f"min={float(np.min(y_pred_prob)):.4f} "
            f"max={pmax:.4f} "
            f"mean={pmean:.4f} "
            f"p95={p95:.4f} "
            f">=0.5={pgt05}"
        )

        if n_pred != n_ref:
            print(
                f"[WARN] {record_id} desajuste ventanas: "
                f"pred={n_pred} vs ref={n_ref}"
            )

        dataset_records.append({
            "record_id": record_id,
            "prediction_file": str(PREDICTIONS_CSV),
            "source_file": record_info["source_file"],
            "y_pred_prob": y_pred_prob,
            "ref_events": ref_events,
            "record_duration_s": record_duration_s,
            "n_windows_ref": n_ref,
            "n_windows_pred": n_pred,
        })

    return dataset_records, skipped_records


def clamp_non_negative_latency(metrics: dict):
    if metrics["mean_latency_s"] is not None:
        metrics["mean_latency_s"] = max(0.0, float(metrics["mean_latency_s"]))
    if metrics["median_latency_s"] is not None:
        metrics["median_latency_s"] = max(0.0, float(metrics["median_latency_s"]))
    return metrics


def evaluate_with_config(dataset_records, config: ClinicalEvalConfig):
    evaluator = ClinicalEvaluator(config)
    global_metrics = evaluator.evaluate_dataset(dataset_records)
    global_metrics = clamp_non_negative_latency(global_metrics)

    per_record_rows = []
    for rec in dataset_records:
        result = evaluator.evaluate_record(
            y_pred_prob=rec["y_pred_prob"],
            ref_events=rec["ref_events"],
            record_duration_s=rec["record_duration_s"],
        )

        per_record_rows.append({
            "record_id": rec["record_id"],
            "prediction_file": rec["prediction_file"],
            "source_file": rec["source_file"],
            "n_windows_ref": rec["n_windows_ref"],
            "n_windows_pred": rec["n_windows_pred"],
            "record_duration_s": rec["record_duration_s"],
            "n_ref_events": result["n_ref_events"],
            "n_pred_events": result["n_pred_events"],
            "tp_event": result["tp_event"],
            "fp_event": result["fp_event"],
            "fn_event": result["fn_event"],
            "event_sensitivity": result["event_sensitivity"],
            "event_precision": result["event_precision"],
            "event_f1": result["event_f1"],
            "fa_per_hour": result["fa_per_hour"],
            "mean_latency_s": result["mean_latency_s"],
            "median_latency_s": result["median_latency_s"],
        })

    return global_metrics, per_record_rows


def search_best_global_config(dataset_records):
    grid_results = []
    best = None

    total = (
        len(THRESHOLDS)
        * len(MOVING_AVG_SIZES)
        * len(MIN_EVENT_DURATIONS)
        * len(MERGE_GAPS)
        * len(MAX_EVENT_DURATIONS)
    )
    print(f"\nProbando {total} configuraciones globales de postprocesado...")

    idx = 0
    for thr, ma, min_dur, gap, max_dur in product(
        THRESHOLDS,
        MOVING_AVG_SIZES,
        MIN_EVENT_DURATIONS,
        MERGE_GAPS,
        MAX_EVENT_DURATIONS,
    ):
        idx += 1
        config = ClinicalEvalConfig(
            window_step_s=WINDOW_STEP_S,
            threshold=thr,
            moving_avg_size=ma,
            min_event_duration_s=min_dur,
            merge_gap_s=gap,
            pre_ictal_tolerance_s=0.0,
            post_ictal_tolerance_s=0.0,
            max_event_duration_s=max_dur,
        )

        global_metrics, _ = evaluate_with_config(dataset_records, config)
        score = clinical_score(
            global_metrics,
            fa_penalty=FA_PENALTY,
            min_sensitivity=MIN_GLOBAL_SENSITIVITY,
        )

        row = {
            "threshold": thr,
            "moving_avg_size": ma,
            "min_event_duration_s": min_dur,
            "merge_gap_s": gap,
            "max_event_duration_s": max_dur,
            "event_sensitivity": global_metrics["event_sensitivity"],
            "event_precision": global_metrics["event_precision"],
            "event_f1": global_metrics["event_f1"],
            "fa_per_hour": global_metrics["fa_per_hour"],
            "tp_event": global_metrics["tp_event"],
            "fp_event": global_metrics["fp_event"],
            "fn_event": global_metrics["fn_event"],
            "score": score,
        }
        grid_results.append(row)

        if best is None:
            best = row.copy()
            best["config_obj"] = config
        else:
            best_key = (best["score"], -best["fa_per_hour"], best["event_sensitivity"])
            new_key = (row["score"], -row["fa_per_hour"], row["event_sensitivity"])
            if new_key > best_key:
                best = row.copy()
                best["config_obj"] = config

        print(
            f"[{idx:03d}/{total}] "
            f"thr={thr:.2f} ma={ma} min_dur={min_dur:.0f} gap={gap:.0f} max_dur={max_dur} | "
            f"sens={row['event_sensitivity']:.4f} "
            f"f1={row['event_f1']:.4f} "
            f"fa/h={row['fa_per_hour']:.4f} "
            f"score={row['score']:.4f}"
        )

    grid_df = pd.DataFrame(grid_results).sort_values(
        by=["score", "event_sensitivity", "fa_per_hour", "event_f1"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    return best, grid_df


def save_global_metrics(global_metrics: dict, out_csv: Path):
    row = {
        "n_records": global_metrics["n_records"],
        "tp_event": global_metrics["tp_event"],
        "fp_event": global_metrics["fp_event"],
        "fn_event": global_metrics["fn_event"],
        "event_sensitivity": global_metrics["event_sensitivity"],
        "event_precision": global_metrics["event_precision"],
        "event_f1": global_metrics["event_f1"],
        "fa_per_hour": global_metrics["fa_per_hour"],
        "mean_latency_s": global_metrics["mean_latency_s"],
        "median_latency_s": global_metrics["median_latency_s"],
    }

    cfg = global_metrics.get("config", {})
    for k, v in cfg.items():
        row[f"cfg_{k}"] = v

    pd.DataFrame([row]).to_csv(out_csv, index=False)


def save_per_record_metrics(rows: list, out_csv: Path):
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    ensure_dir(OUTPUT_DIR)

    print(f"INDEX_CSV       : {INDEX_CSV}")
    print(f"PREDICTIONS_CSV : {PREDICTIONS_CSV}")
    print(f"OUTPUT_DIR      : {OUTPUT_DIR}")

    ref_dict = load_index_records(INDEX_CSV, target_patients=VAL_PATIENTS)
    pred_df = load_predictions_per_window(PREDICTIONS_CSV, target_patients=VAL_PATIENTS)

    debug_alignment(ref_dict, pred_df, max_show=30)
    save_reference_events_json(ref_dict, OUTPUT_DIR / "reference_events.json")

    dataset_records, skipped_records = build_dataset_records(ref_dict, pred_df)

    (OUTPUT_DIR / "skipped_records.json").write_text(
        json.dumps(skipped_records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if len(dataset_records) == 0:
        raise RuntimeError("No hay registros con predicción para evaluar.")

    best, grid_df = search_best_global_config(dataset_records)
    grid_df.to_csv(OUTPUT_DIR / "postprocessing_grid_search.csv", index=False)

    best_config = best["config_obj"]
    global_metrics, per_record_rows = evaluate_with_config(dataset_records, best_config)

    save_global_metrics(global_metrics, OUTPUT_DIR / "metrics_global.csv")
    save_per_record_metrics(per_record_rows, OUTPUT_DIR / "metrics_per_record.csv")

    (OUTPUT_DIR / "metrics_global.json").write_text(
        json.dumps(global_metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )
    (OUTPUT_DIR / "best_postprocessing_config.json").write_text(
        json.dumps(best_config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== MEJOR CONFIG GLOBAL ===")
    print(best_config)

    print("\n=== RESULTADOS GLOBALES ===")
    print(f"n_records : {global_metrics['n_records']}")
    print(f"TP events : {global_metrics['tp_event']}")
    print(f"FP events : {global_metrics['fp_event']}")
    print(f"FN events : {global_metrics['fn_event']}")
    print(f"Sensibilidad evento: {global_metrics['event_sensitivity']:.4f}")
    print(f"Precisión evento : {global_metrics['event_precision']:.4f}")
    print(f"F1 evento : {global_metrics['event_f1']:.4f}")
    print(f"FA/h : {global_metrics['fa_per_hour']:.4f}")
    print(f"Latencia media (s) : {global_metrics['mean_latency_s']}")
    print(f"Latencia mediana(s): {global_metrics['median_latency_s']}")


if __name__ == "__main__":
    main()