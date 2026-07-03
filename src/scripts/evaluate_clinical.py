from pathlib import Path
import argparse
import json
from itertools import product

import numpy as np
import pandas as pd

from src.utils.postprocessing import PostProcessConfig
from src.utils.evaluador_clinico_def import (
    ClinicalEvalConfig,
    ClinicalEvaluator,
    clinical_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación clínica unificada y robusta.")

    parser.add_argument("--predictions_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--index_csv",
        default=str(PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s" / "chbmit_index_val_56.csv"),
        type=str,
    )

    parser.add_argument("--tune_patients", nargs="+", default=["chb05"])
    parser.add_argument("--test_patients", nargs="+", default=["chb06"])

    parser.add_argument("--window_step_s", type=float, default=1.0)
    parser.add_argument("--use_hysteresis_grid", action="store_true")

    parser.add_argument("--min_sensitivity", type=float, default=0.0)
    parser.add_argument("--fa_penalty", type=float, default=0.10)
    parser.add_argument("--latency_penalty", type=float, default=0.0)

    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--max_intermediate_records", type=int, default=5)

    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def normalize_record_id(x):
    return str(x).strip().replace(".edf", "")


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
            ev_end = max(ev_end, end_s)
        elif label == 0 and in_event:
            ref_events.append((ev_start, ev_end))
            in_event = False
            ev_start, ev_end = None, None

    if in_event:
        ref_events.append((ev_start, ev_end))

    return ref_events


def load_index_records(csv_path: Path, target_patients):
    df = pd.read_csv(csv_path)

    required = {
        "patient_id", "record_id", "filepath",
        "window_index", "start_sec", "end_sec", "label"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en index_csv: {missing}")

    df = df[df["patient_id"].isin(target_patients)].copy()
    df["record_id"] = df["record_id"].astype(str).map(normalize_record_id)
    df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
    df = df.dropna(subset=["window_index"]).copy()
    df["window_index"] = df["window_index"].astype(int)

    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["start_sec", "end_sec"]).copy()

    df = df.sort_values(["patient_id", "record_id", "window_index"]).reset_index(drop=True)

    records = {}
    for record_id, group in df.groupby("record_id", sort=False):
        group = group.sort_values("window_index").reset_index(drop=True)
        first_row = group.iloc[0]

        record_start_s = float(group["start_sec"].min())
        record_end_s = float(group["end_sec"].max())
        record_duration_s = max(0.0, record_end_s - record_start_s)

        records[record_id] = {
            "patient_id": str(first_row["patient_id"]),
            "record_id": str(record_id),
            "source_file": str(first_row["filepath"]),
            "ref_events": build_ref_events_from_group(group),
            "n_windows_ref": int(len(group)),
            "record_start_s": record_start_s,
            "record_end_s": record_end_s,
            "record_duration_s": record_duration_s,
            "window_index_ref": group["window_index"].to_numpy(dtype=np.int32),
            "start_sec_ref": group["start_sec"].to_numpy(dtype=np.float32),
            "end_sec_ref": group["end_sec"].to_numpy(dtype=np.float32),
        }

    return records


def load_predictions_per_window(pred_csv_path: Path, target_patients):
    df = pd.read_csv(pred_csv_path)

    required = {"record_id", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en predictions_per_window.csv: {missing}")

    if "patient_id" in df.columns:
        df = df[df["patient_id"].isin(target_patients)].copy()

    df["record_id"] = df["record_id"].astype(str).map(normalize_record_id)
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    if "window_index" in df.columns:
        df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
        df = df.dropna(subset=["window_index"]).copy()
        df["window_index"] = df["window_index"].astype(int)

    if "start_sec" in df.columns:
        df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    if "end_sec" in df.columns:
        df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")

    sort_cols = [c for c in ["record_id", "window_index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.sort_values(["record_id"]).reset_index(drop=True)

    return df


def load_prediction_array(pt_path: str):
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
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

    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def find_prediction_file(record_id: str, predictions_dir: Path):
    for ext in (".pt", ".npy", ".npz"):
        p = predictions_dir / f"{record_id}{ext}"
        if p.exists():
            return p
    return None


def build_dataset_records_from_csv(ref_dict, pred_df):
    dataset_records = []
    skipped_records = []

    for record_id, ref_info in ref_dict.items():
        group = pred_df[pred_df["record_id"] == record_id].copy()
        if len(group) == 0:
            skipped_records.append(record_id)
            continue

        if "window_index" in group.columns:
            group = group.sort_values("window_index").reset_index(drop=True)

        y_pred_prob = group["prob"].to_numpy(dtype=np.float32)
        y_pred_prob = np.nan_to_num(y_pred_prob, nan=0.0, posinf=1.0, neginf=0.0)
        y_pred_prob = np.clip(y_pred_prob, 0.0, 1.0)

        if "start_sec" in group.columns and "end_sec" in group.columns:
            valid_time = group["start_sec"].notna() & group["end_sec"].notna()
            if valid_time.any():
                record_start_s = float(group.loc[valid_time, "start_sec"].min())
                record_end_s = float(group.loc[valid_time, "end_sec"].max())
                record_duration_s = max(0.0, record_end_s - record_start_s)
            else:
                record_duration_s = ref_info["record_duration_s"]
        else:
            record_duration_s = ref_info["record_duration_s"]

        dataset_records.append({
            "record_id": record_id,
            "patient_id": ref_info["patient_id"],
            "prediction_source": "predictions_per_window.csv",
            "source_file": ref_info["source_file"],
            "y_pred_prob": y_pred_prob,
            "ref_events": ref_info["ref_events"],
            "record_duration_s": record_duration_s,
            "n_windows_ref": ref_info["n_windows_ref"],
            "n_windows_pred": int(len(y_pred_prob)),
        })

    return dataset_records, skipped_records


def build_dataset_records_from_pt(ref_dict, predictions_dir: Path):
    dataset_records = []
    skipped_records = []

    for record_id, ref_info in ref_dict.items():
        pred_file = find_prediction_file(record_id, predictions_dir)
        if pred_file is None:
            skipped_records.append(record_id)
            continue

        y_pred_prob = load_prediction_array(str(pred_file))

        dataset_records.append({
            "record_id": record_id,
            "patient_id": ref_info["patient_id"],
            "prediction_source": str(pred_file),
            "source_file": ref_info["source_file"],
            "y_pred_prob": y_pred_prob,
            "ref_events": ref_info["ref_events"],
            "record_duration_s": ref_info["record_duration_s"],
            "n_windows_ref": ref_info["n_windows_ref"],
            "n_windows_pred": int(len(y_pred_prob)),
        })

    return dataset_records, skipped_records


def build_dataset_records(ref_dict, predictions_dir: Path, target_patients):
    pred_csv = predictions_dir / "predictions_per_window.csv"
    if pred_csv.exists():
        pred_df = load_predictions_per_window(pred_csv, target_patients=target_patients)
        return build_dataset_records_from_csv(ref_dict, pred_df)
    return build_dataset_records_from_pt(ref_dict, predictions_dir)


def build_postprocessing_grid(use_hysteresis_grid: bool, window_step_s: float):
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    smoothing_sizes = [1, 3, 5]
    min_event_durations = [1.0, 3.0, 5.0]
    merge_gaps = [0.0, 3.0, 5.0]
    smoothings = ["moving_average"]

    grid = []

    for smoothing, smoothing_size, threshold, min_dur, merge_gap in product(
        smoothings, smoothing_sizes, thresholds, min_event_durations, merge_gaps
    ):
        grid.append(
            PostProcessConfig(
                window_step_s=window_step_s,
                smoothing=smoothing,
                smoothing_size=smoothing_size,
                threshold=threshold,
                use_hysteresis=False,
                threshold_on=threshold,
                threshold_off=max(0.0, threshold - 0.10),
                min_event_duration_s=min_dur,
                merge_gap_s=merge_gap,
                max_event_duration_s=None,
            )
        )

    if use_hysteresis_grid:
        threshold_pairs = [(0.40, 0.20), (0.50, 0.30), (0.55, 0.35), (0.60, 0.40)]
        for smoothing, smoothing_size, (thr_on, thr_off), min_dur, merge_gap in product(
            smoothings, smoothing_sizes, threshold_pairs, min_event_durations, merge_gaps
        ):
            grid.append(
                PostProcessConfig(
                    window_step_s=window_step_s,
                    smoothing=smoothing,
                    smoothing_size=smoothing_size,
                    threshold=thr_on,
                    use_hysteresis=True,
                    threshold_on=thr_on,
                    threshold_off=thr_off,
                    min_event_duration_s=min_dur,
                    merge_gap_s=merge_gap,
                    max_event_duration_s=None,
                )
            )

    return grid


def evaluate_with_config(dataset_records, eval_cfg, post_cfg, return_intermediates=False):
    evaluator = ClinicalEvaluator(eval_config=eval_cfg, postprocess_config=post_cfg)
    return evaluator.evaluate_dataset(
        dataset_records,
        use_probabilities=True,
        return_intermediates=return_intermediates,
    )


def search_best_postprocessing(
    dataset_records,
    eval_cfg,
    use_hysteresis_grid,
    window_step_s,
    fa_penalty,
    min_sensitivity,
    latency_penalty,
):
    grid = build_postprocessing_grid(use_hysteresis_grid, window_step_s)

    rows = []
    best = None

    for post_cfg in grid:
        metrics = evaluate_with_config(dataset_records, eval_cfg, post_cfg, return_intermediates=False)
        score = clinical_score(
            metrics,
            fa_penalty=fa_penalty,
            min_sensitivity=min_sensitivity,
            latency_penalty=latency_penalty,
        )

        row = {
            "smoothing": post_cfg.smoothing,
            "smoothing_size": post_cfg.smoothing_size,
            "threshold": post_cfg.threshold,
            "use_hysteresis": post_cfg.use_hysteresis,
            "threshold_on": post_cfg.threshold_on,
            "threshold_off": post_cfg.threshold_off,
            "min_event_duration_s": post_cfg.min_event_duration_s,
            "merge_gap_s": post_cfg.merge_gap_s,
            "event_sensitivity": metrics["event_sensitivity"],
            "event_precision": metrics["event_precision"],
            "event_f1": metrics["event_f1"],
            "fa_per_hour": metrics["fa_per_hour"],
            "mean_latency_s": metrics["mean_latency_s"],
            "median_latency_s": metrics["median_latency_s"],
            "tp_event": metrics["tp_event"],
            "fp_event": metrics["fp_event"],
            "fn_event": metrics["fn_event"],
            "score": score,
        }
        rows.append(row)

        if best is None:
            best = {"score": score, "metrics": metrics, "post_cfg": post_cfg}
        else:
            best_key = (
                best["score"],
                best["metrics"]["event_sensitivity"],
                -best["metrics"]["fa_per_hour"],
                best["metrics"]["event_f1"],
            )
            new_key = (
                score,
                metrics["event_sensitivity"],
                -metrics["fa_per_hour"],
                metrics["event_f1"],
            )
            if new_key > best_key:
                best = {"score": score, "metrics": metrics, "post_cfg": post_cfg}

    grid_df = pd.DataFrame(rows).sort_values(
        by=["score", "event_sensitivity", "fa_per_hour", "event_f1"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    return best, grid_df


def build_per_record_table(metrics: dict):
    return pd.DataFrame([{
        "record_id": r["record_id"],
        "n_ref_events": r["n_ref_events"],
        "n_pred_events": r["n_pred_events"],
        "tp_event": r["tp_event"],
        "fp_event": r["fp_event"],
        "fn_event": r["fn_event"],
        "event_sensitivity": r["event_sensitivity"],
        "event_precision": r["event_precision"],
        "event_f1": r["event_f1"],
        "fa_per_hour": r["fa_per_hour"],
        "mean_latency_s": r["mean_latency_s"],
        "median_latency_s": r["median_latency_s"],
    } for r in metrics["per_record"]])


def build_global_table(metrics: dict):
    row = {
        "n_records": metrics["n_records"],
        "tp_event": metrics["tp_event"],
        "fp_event": metrics["fp_event"],
        "fn_event": metrics["fn_event"],
        "event_sensitivity": metrics["event_sensitivity"],
        "event_precision": metrics["event_precision"],
        "event_f1": metrics["event_f1"],
        "fa_per_hour": metrics["fa_per_hour"],
        "mean_latency_s": metrics["mean_latency_s"],
        "median_latency_s": metrics["median_latency_s"],
    }
    for k, v in metrics.get("eval_config", {}).items():
        row[f"eval_{k}"] = v
    post_cfg = metrics.get("postprocess_config", None)
    if post_cfg is not None:
        for k, v in post_cfg.items():
            row[f"post_{k}"] = v
    return pd.DataFrame([row])


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_reference_events_json(ref_dict, out_path: Path):
    serializable = {
        rid: {
            "patient_id": info["patient_id"],
            "record_id": info["record_id"],
            "source_file": info["source_file"],
            "ref_events": [list(ev) for ev in info["ref_events"]],
            "n_windows_ref": info["n_windows_ref"],
            "record_duration_s": info["record_duration_s"],
        }
        for rid, info in ref_dict.items()
    }
    save_json(serializable, out_path)


def maybe_save_intermediates(dataset_records, eval_cfg, post_cfg, output_dir: Path, max_records: int = 5):
    inter_dir = output_dir / "intermediates"
    ensure_dir(inter_dir)

    evaluator = ClinicalEvaluator(eval_config=eval_cfg, postprocess_config=post_cfg)
    for rec in dataset_records[:max_records]:
        result = evaluator.evaluate_record(
            y_pred_prob=rec["y_pred_prob"],
            ref_events=rec["ref_events"],
            record_duration_s=rec["record_duration_s"],
            record_id=rec["record_id"],
            return_intermediates=True,
        )
        pp = result["postprocess"]
        rid = rec["record_id"]

        np.save(inter_dir / f"{rid}_probs.npy", pp["probs"])
        np.save(inter_dir / f"{rid}_smoothed_probs.npy", pp["smoothed_probs"])
        np.save(inter_dir / f"{rid}_binary.npy", pp["binary"])

        save_json(
            {
                "record_id": rid,
                "config": pp["config"],
                "events_raw": [list(x) for x in pp["events_raw"]],
                "events_filtered": [list(x) for x in pp["events_filtered"]],
                "events_merged": [list(x) for x in pp["events_merged"]],
                "events_final": [list(x) for x in pp["events_final"]],
                "matches": result["matches"],
                "ref_events": [list(x) for x in result["ref_events"]],
            },
            inter_dir / f"{rid}_intermediates.json"
        )


def main():
    args = parse_args()

    predictions_dir = resolve_path(args.predictions_dir)
    output_dir = resolve_path(args.output_dir)
    index_csv = resolve_path(args.index_csv)
    ensure_dir(output_dir)

    tune_ref = load_index_records(index_csv, tuple(args.tune_patients))
    test_ref = load_index_records(index_csv, tuple(args.test_patients))

    save_reference_events_json(tune_ref, output_dir / "reference_events_tune.json")
    save_reference_events_json(test_ref, output_dir / "reference_events_test.json")

    tune_records, tune_skipped = build_dataset_records(tune_ref, predictions_dir, tuple(args.tune_patients))
    test_records, test_skipped = build_dataset_records(test_ref, predictions_dir, tuple(args.test_patients))

    save_json(tune_skipped, output_dir / "skipped_records_tune.json")
    save_json(test_skipped, output_dir / "skipped_records_test.json")

    if len(tune_records) == 0:
        raise RuntimeError("No hay registros de tuning con predicciones.")
    if len(test_records) == 0:
        raise RuntimeError("No hay registros de test con predicciones.")

    eval_cfg = ClinicalEvalConfig(
        pre_ictal_tolerance_s=0.0,
        post_ictal_tolerance_s=0.0,
        min_overlap_s=0.0,
    )

    best, grid_df = search_best_postprocessing(
        dataset_records=tune_records,
        eval_cfg=eval_cfg,
        use_hysteresis_grid=args.use_hysteresis_grid,
        window_step_s=args.window_step_s,
        fa_penalty=args.fa_penalty,
        min_sensitivity=args.min_sensitivity,
        latency_penalty=args.latency_penalty,
    )

    best_post_cfg = best["post_cfg"]

    tune_metrics = evaluate_with_config(tune_records, eval_cfg, best_post_cfg, return_intermediates=False)
    test_metrics = evaluate_with_config(test_records, eval_cfg, best_post_cfg, return_intermediates=False)

    grid_df.to_csv(output_dir / "postprocessing_grid_search_tune.csv", index=False)
    build_global_table(tune_metrics).to_csv(output_dir / "metrics_global_tune.csv", index=False)
    build_per_record_table(tune_metrics).to_csv(output_dir / "metrics_per_record_tune.csv", index=False)
    pd.DataFrame(tune_metrics["event_rows"]).to_csv(output_dir / "event_rows_tune.csv", index=False)

    build_global_table(test_metrics).to_csv(output_dir / "metrics_global_test.csv", index=False)
    build_per_record_table(test_metrics).to_csv(output_dir / "metrics_per_record_test.csv", index=False)
    pd.DataFrame(test_metrics["event_rows"]).to_csv(output_dir / "event_rows_test.csv", index=False)

    save_json(best_post_cfg.to_dict(), output_dir / "best_postprocessing_config.json")
    save_json(eval_cfg.to_dict(), output_dir / "clinical_eval_config.json")
    save_json(tune_metrics, output_dir / "metrics_global_tune.json")
    save_json(test_metrics, output_dir / "metrics_global_test.json")

    if args.save_intermediates:
        maybe_save_intermediates(
            dataset_records=test_records,
            eval_cfg=eval_cfg,
            post_cfg=best_post_cfg,
            output_dir=output_dir,
            max_records=args.max_intermediate_records,
        )

    print("\n=== MEJOR CONFIGURACIÓN (TUNE) ===")
    print(best_post_cfg)

    print("\n=== RESULTADOS TEST ===")
    print(f"n_records: {test_metrics['n_records']}")
    print(f"TP events: {test_metrics['tp_event']}")
    print(f"FP events: {test_metrics['fp_event']}")
    print(f"FN events: {test_metrics['fn_event']}")
    print(f"Sensibilidad evento: {test_metrics['event_sensitivity']:.4f}")
    print(f"Precisión evento: {test_metrics['event_precision']:.4f}")
    print(f"F1 evento: {test_metrics['event_f1']:.4f}")
    print(f"FA/h: {test_metrics['fa_per_hour']:.4f}")
    print(f"Latencia media (s): {test_metrics['mean_latency_s']}")
    print(f"Latencia mediana (s): {test_metrics['median_latency_s']}")


if __name__ == "__main__":
    main()


# python src/scripts/evaluate_clinical.py --predictions_dir experiments/predictions/hibrido/defNuevo --output_dir experiments/clinical_eval/hibrido/robusto --tune_patients chb05 --test_patients chb06 --use_hysteresis_grid
# python src/scripts/evaluate_clinical.py --predictions_dir experiments/predictions/hibrido/defNuevo --output_dir experiments/clinical_eval/hibrido/robusto_inv --tune_patients chb06 --test_patients chb05 --use_hysteresis_grid
# python src/scripts/evaluate_clinical.py --predictions_dir experiments/predictions/attention_cnn/defNuevo --output_dir experiments/clinical_eval/attention_cnn/robusto --tune_patients chb05 --test_patients chb06 --use_hysteresis_grid --save_intermediates --max_intermediate_records 3