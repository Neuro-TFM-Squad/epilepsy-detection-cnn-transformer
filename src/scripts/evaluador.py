"""
run_clinical_eval.py

Evaluador clínico universal para CHB-MIT usando metadata por registro.

Versión 3:
- lee metadata_chbmit_etiquetado.csv,
- reconstruye record_id desde la columna `archivo`,
- usa la columna `crisis` como eventos de referencia,
- busca predicciones por registro,
- hace barrido GLOBAL de postprocesado,
- elige una única configuración global para todos los registros,
- guarda resultados globales, por registro y la configuración óptima.
"""

from pathlib import Path
import json
import ast
from itertools import product

import numpy as np
import pandas as pd

from src.utils.evaluador_clinico import (
    ClinicalEvalConfig,
    ClinicalEvaluator,
    clinical_score,
    load_prediction_array,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV = PROJECT_ROOT / "src" / "utils" / "metadata_chbmit_etiquetado.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "experiments" / "predictions" / "spatio_temporal_cnn" / "def"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "clinical_eval" / "spatio_temporal_cnn" / "def"
WINDOW_STEP_S = 1.0

# Rejilla global de postprocesado
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85]
MOVING_AVG_SIZES = [1, 3, 5]
MIN_EVENT_DURATIONS = [5.0, 10.0, 15.0]
MERGE_GAPS = [0.0, 5.0, 10.0]
MAX_EVENT_DURATIONS = [None, 60.0, 120.0]

# Criterio clínico
MIN_GLOBAL_SENSITIVITY = 0.80
FA_PENALTY = 0.10


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_record_id_from_archivo(x: str) -> str:
    stem = Path(str(x)).stem
    return stem.replace("+", "")


def parse_crisis_column(value):
    if pd.isna(value):
        return []

    s = str(value).strip()
    if s in ("", "[]", "nan", "None"):
        return []

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, tuple):
            parsed = [parsed]

        out = []
        for ev in parsed:
            if isinstance(ev, (list, tuple)) and len(ev) == 2:
                start, end = float(ev[0]), float(ev[1])
                if end < start:
                    start, end = end, start
                out.append((start, end))
        return out
    except Exception:
        return []


def load_metadata_records(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"paciente", "archivo", "crisis", "total_crisis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en metadata: {missing}")

    df["record_id"] = df["archivo"].apply(extract_record_id_from_archivo)
    records = {}

    for _, row in df.iterrows():
        record_id = row["record_id"]
        ref_events = parse_crisis_column(row["crisis"])
        records[record_id] = {
            "patient_id": row["paciente"],
            "record_id": record_id,
            "source_file": row["archivo"],
            "ref_events": ref_events,
            "total_crisis": int(row["total_crisis"]),
        }

    return records


def save_reference_events_json(ref_dict, out_path: Path):
    serializable = {}
    for rid, info in ref_dict.items():
        serializable[rid] = {
            "patient_id": info["patient_id"],
            "record_id": info["record_id"],
            "source_file": info["source_file"],
            "ref_events": [list(ev) for ev in info["ref_events"]],
            "total_crisis": info["total_crisis"],
        }
    out_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def find_prediction_file(record_id: str, predictions_dir: Path):
    for ext in (".pt", ".npy", ".npz"):
        candidate = predictions_dir / f"{record_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def normalize_prediction_array(pred):
    pred = np.asarray(pred)
    pred = np.squeeze(pred)
    pred = np.ravel(pred)
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    return pred.astype(np.float32)


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


def clamp_non_negative_latency(metrics: dict):
    if metrics["mean_latency_s"] is not None:
        metrics["mean_latency_s"] = max(0.0, float(metrics["mean_latency_s"]))
    if metrics["median_latency_s"] is not None:
        metrics["median_latency_s"] = max(0.0, float(metrics["median_latency_s"]))
    return metrics


def build_dataset_records(ref_dict):
    dataset_records = []
    skipped_records = []

    print("Buscando predicciones por registro...")
    for record_id, record_info in ref_dict.items():
        pred_file = find_prediction_file(record_id, PREDICTIONS_DIR)
        if pred_file is None:
            skipped_records.append(record_id)
            continue

        y_pred_prob = normalize_prediction_array(load_prediction_array(str(pred_file)))
        ref_events = record_info["ref_events"]
        record_duration_s = float(len(y_pred_prob)) * WINDOW_STEP_S

        dataset_records.append({
            "record_id": record_id,
            "prediction_file": str(pred_file),
            "source_file": record_info["source_file"],
            "y_pred_prob": y_pred_prob,
            "ref_events": ref_events,
            "record_duration_s": record_duration_s,
        })

    return dataset_records, skipped_records


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
            "n_windows_pred": len(rec["y_pred_prob"]),
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
    print(f"Probando {total} configuraciones globales de postprocesado...")

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

        # Orden: mejor score, luego menor FA/h, luego mayor sensibilidad
        key = (score, -global_metrics["fa_per_hour"], global_metrics["event_sensitivity"])
        if best is None or key > best[0]:
            best = (key, config, global_metrics)

        print(
            f"[{idx:03d}/{total}] "
            f"thr={thr:.2f} ma={ma} min_dur={min_dur:.0f} gap={gap:.0f} max_dur={max_dur} | "
            f"sens={global_metrics['event_sensitivity']:.4f} "
            f"f1={global_metrics['event_f1']:.4f} "
            f"fa/h={global_metrics['fa_per_hour']:.4f} "
            f"score={score:.4f}"
        )

    grid_df = pd.DataFrame(grid_results).sort_values(
        by=["score", "event_sensitivity", "fa_per_hour", "event_f1"],
        ascending=[False, False, True, False],
    )
    return best[1], best[2], grid_df


def main():
    ensure_dir(OUTPUT_DIR)

    print("Reconstruyendo eventos de referencia desde metadata_chbmit_etiquetado.csv...")
    ref_dict = load_metadata_records(INDEX_CSV)

    ref_json_path = OUTPUT_DIR / "reference_events.json"
    save_reference_events_json(ref_dict, ref_json_path)
    print(f"Eventos de referencia guardados en: {ref_json_path}")

    dataset_records, skipped_records = build_dataset_records(ref_dict)

    if len(dataset_records) == 0:
        print("No hay registros válidos para evaluar.")
        print("Revisa el directorio de predicciones y el formato de los archivos.")
        return

    print(f"Registros evaluados: {len(dataset_records)}")
    print(f"Registros omitidos: {len(skipped_records)}")

    best_config, best_global_metrics, grid_df = search_best_global_config(dataset_records)

    print("\n=== MEJOR CONFIG GLOBAL ===")
    print(best_config)

    final_global_metrics, per_record_rows = evaluate_with_config(dataset_records, best_config)
    final_global_metrics = clamp_non_negative_latency(final_global_metrics)

    per_record_csv = OUTPUT_DIR / "metrics_per_record.csv"
    global_csv = OUTPUT_DIR / "metrics_global.csv"
    global_json = OUTPUT_DIR / "metrics_global.json"
    skipped_json = OUTPUT_DIR / "skipped_records.json"
    grid_csv = OUTPUT_DIR / "postprocessing_grid_search.csv"
    best_cfg_json = OUTPUT_DIR / "best_postprocessing_config.json"

    save_per_record_metrics(per_record_rows, per_record_csv)
    save_global_metrics(final_global_metrics, global_csv)

    grid_df.to_csv(grid_csv, index=False)
    skipped_json.write_text(json.dumps(skipped_records, indent=2, ensure_ascii=False), encoding="utf-8")
    global_json.write_text(json.dumps(final_global_metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    best_cfg_json.write_text(json.dumps(best_config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESULTADOS GLOBALES ===")
    print(f"n_records : {final_global_metrics['n_records']}")
    print(f"TP events : {final_global_metrics['tp_event']}")
    print(f"FP events : {final_global_metrics['fp_event']}")
    print(f"FN events : {final_global_metrics['fn_event']}")
    print(f"Sensibilidad evento: {final_global_metrics['event_sensitivity']:.4f}")
    print(f"Precisión evento : {final_global_metrics['event_precision']:.4f}")
    print(f"F1 evento : {final_global_metrics['event_f1']:.4f}")
    print(f"FA/h : {final_global_metrics['fa_per_hour']:.4f}")
    print(f"Latencia media (s) : {final_global_metrics['mean_latency_s']}")
    print(f"Latencia mediana(s): {final_global_metrics['median_latency_s']}")

    print("\n=== CONFIG ÓPTIMA ===")
    for k, v in best_config.to_dict().items():
        print(f"{k}: {v}")

    print("\nArchivos generados:")
    print(f"- {per_record_csv}")
    print(f"- {global_csv}")
    print(f"- {global_json}")
    print(f"- {grid_csv}")
    print(f"- {best_cfg_json}")
    print(f"- {ref_json_path}")
    print(f"- {skipped_json}")


if __name__ == "__main__":
    main()