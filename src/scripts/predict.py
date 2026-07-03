from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.seizureDatasetMulticanal import SeizureDatasetMultichannel
from src.datasets.dwt_dataset import DWTSeizureDataset

from src.models.hibrido import SpatioTemporalCNN
from src.models.eegnet_model import EEGNet
from src.models.attention import AttentionCNN
from src.models.dwt_mlp_model import DWTMLP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL_PATIENTS = ("chb05", "chb06")
DEFAULT_INDEX_CSV = str(
    PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s" / "chbmit_index.csv"
)
DEFAULT_DWT_DATA_ROOT = str(
    PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "dataset_chbmit_dwt_18ch_bin"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generar predicciones por registro y por ventana para evaluación clínica."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["spatio_temporal_cnn", "attention_cnn", "eegnet", "dwt"]
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--val_patients",
        nargs="+",
        default=list(VAL_PATIENTS),
        help="Pacientes de validación a inferir. Ej: --val_patients chb05 chb06"
    )
    parser.add_argument(
        "--index_csv",
        default=DEFAULT_INDEX_CSV,
        type=str,
        help="Ruta al CSV de índice para modelos de señal cruda. Por defecto chbmit_index.csv (todos los pacientes)."
    )
    parser.add_argument(
        "--dwt_data_root",
        default=DEFAULT_DWT_DATA_ROOT,
        type=str,
        help="Directorio raíz del dataset DWT (contiene chbmit_dwt_index.csv, .bin)."
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def normalize_record_id_from_filepath(x):
    stem = Path(str(x)).stem
    return stem.rsplit("_win", 1)[0] if "_win" in stem else stem


def build_subset_dataframe(index_df: pd.DataFrame, target_patients):
    df = index_df.copy()

    if "patient_id" not in df.columns:
        if "filepath" in df.columns:
            df["patient_id"] = df["filepath"].apply(lambda x: Path(str(x)).stem.split("_")[0])
        else:
            raise ValueError("No existe columna 'patient_id' ni 'filepath' para inferirla.")

    df = df[df["patient_id"].isin(target_patients)].copy().reset_index(drop=True)

    if "record_id" not in df.columns:
        if "filepath" in df.columns:
            df["record_id"] = df["filepath"].apply(normalize_record_id_from_filepath)
        else:
            raise ValueError("No existe columna 'record_id' ni 'filepath' para inferirla.")
    else:
        df["record_id"] = (
            df["record_id"]
            .astype(str)
            .str.replace(".edf", "", regex=False)
            .str.strip()
        )

    if "window_index" in df.columns:
        df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
        df = df.dropna(subset=["window_index"]).copy()
        df["window_index"] = df["window_index"].astype(int)

    sort_cols = [c for c in ["patient_id", "record_id", "window_index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def load_raw_checkpoint(checkpoint_path: Path, device: str):
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_model(model_name: str, checkpoint_data=None):
    if model_name == "spatio_temporal_cnn":
        return SpatioTemporalCNN(
            in_channels=18,
            window_size=256,
            temporal_filters=16,
            spatial_filters=32,
            num_classes=1,
            temporal_kernel=15,
            dropout_p=0.3,
            spatial_dropout_p=0.1,
            temporal_norm="instance",
        )

    if model_name == "attention_cnn":
        return AttentionCNN(
            in_channels=18,
            num_classes=1,
            dropout=0.4
        )

    if model_name == "eegnet":
        return EEGNet(
            in_channels=18,
            window_size=256,
            num_classes=1
        )

    if model_name == "dwt":
        if checkpoint_data is None:
            raise ValueError("Para DWT necesitas pasar checkpoint_data al construir el modelo.")

        cfg = checkpoint_data.get("config", {})
        input_dim = checkpoint_data.get("input_dim", None)
        if input_dim is None:
            raise ValueError("El checkpoint DWT no contiene 'input_dim'.")

        hidden_dims = cfg.get("hidden_dims", (256, 128))
        dropout = cfg.get("dropout", 0.25)

        if isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)

        return DWTMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    raise ValueError(f"Modelo no soportado: {model_name}")


def load_checkpoint_into_model(model, checkpoint_data, device: str):
    state = checkpoint_data
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_inference_bundle(model_name: str, checkpoint_data, target_patients, args=None):
    if model_name == "dwt":
        dwt_root = Path(args.dwt_data_root) if args else (
            PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "dataset_chbmit_dwt_18ch_bin"
        )
        if not dwt_root.is_absolute():
            dwt_root = PROJECT_ROOT / dwt_root
        csv_path = dwt_root / "chbmit_dwt_index.csv"
        signals_bin = dwt_root / "chbmit_dwt_signals.bin"
        labels_bin = dwt_root / "chbmit_dwt_labels.bin"

        index_df = pd.read_csv(csv_path)
        subset_df = build_subset_dataframe(index_df, target_patients).reset_index(drop=True)

        if "global_idx" not in subset_df.columns:
            raise ValueError("El índice DWT necesita la columna 'global_idx'.")

        global_indices = subset_df["global_idx"].astype(int).tolist()

        feature_dim = checkpoint_data.get("feature_dim", None)
        if feature_dim is None:
            cfg = checkpoint_data.get("config", {})
            feature_dim = cfg.get("feature_dim", 5184)

        num_samples = len(index_df)

        dataset = DWTSeizureDataset(
            signals_path=signals_bin,
            labels_path=labels_bin,
            index_csv_path=csv_path,
            indices=global_indices,
            num_samples=num_samples,
            feature_dim=feature_dim,
            normalize=True,
        )

        local_indices = list(range(len(global_indices)))
        return subset_df, dataset, local_indices

    # Modelos de señal cruda: usa el índice completo para obtener posiciones globales correctas
    data_root = PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s"
    signals_bin = data_root / "chbmit_signals.bin"
    labels_bin = data_root / "chbmit_labels.bin"

    index_csv = Path(args.index_csv) if args else (data_root / "chbmit_index.csv")
    if not index_csv.is_absolute():
        index_csv = PROJECT_ROOT / index_csv

    index_df = pd.read_csv(index_csv)

    # Preservar posiciones originales en el binario ANTES de filtrar
    mask = index_df["patient_id"].isin(target_patients)
    global_indices = index_df.index[mask].tolist()
    subset_df = build_subset_dataframe(index_df, target_patients).reset_index(drop=True)
    num_samples_full = len(index_df)

    dataset = SeizureDatasetMultichannel(
        signals_path=str(signals_bin),
        labels_path=str(labels_bin),
        augment=False,
        sequence_length=1,
        indices=global_indices,
        num_samples=num_samples_full,
    )

    local_indices = list(range(len(subset_df)))
    return subset_df, dataset, local_indices


@torch.no_grad()
def predict_subset(model, dataset, indices, batch_size, num_workers, device):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    probs = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        logits = model(x)
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(batch_probs)

    return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


def save_predictions_per_record(subset_df: pd.DataFrame, output_dir: Path):
    if "window_index" in subset_df.columns:
        grouped = subset_df.sort_values(["record_id", "window_index"]).reset_index(drop=True)
    else:
        grouped = subset_df.sort_values(["record_id"]).reset_index(drop=True)

    for record_id, group in grouped.groupby("record_id", sort=False):
        probs = group["prob"].to_numpy(dtype=np.float32)
        out_path = output_dir / f"{record_id}.pt"

        payload = {
            "record_id": str(record_id),
            "y_pred_prob": probs,
            "n_windows": int(len(group)),
        }

        if "patient_id" in group.columns:
            payload["patient_id"] = str(group["patient_id"].iloc[0])

        if "window_index" in group.columns:
            payload["window_index"] = group["window_index"].to_numpy(dtype=np.int32)

        torch.save(payload, out_path)

        print(
            f"[SAVE_PT] {record_id} | "
            f"n_windows={len(probs)} | "
            f"min={float(probs.min()):.4f} | "
            f"max={float(probs.max()):.4f} | "
            f"mean={float(probs.mean()):.4f}"
        )


def save_predictions_per_window_csv(subset_df: pd.DataFrame, output_dir: Path):
    cols = [
        c for c in [
            "patient_id", "record_id", "filepath", "window_index",
            "start_sec", "end_sec", "label", "prob"
        ] if c in subset_df.columns
    ]

    out_csv = output_dir / "predictions_per_window.csv"
    df_out = subset_df[cols].copy()

    if "window_index" in df_out.columns:
        df_out = df_out.sort_values(["record_id", "window_index"]).reset_index(drop=True)
    else:
        df_out = df_out.sort_values(["record_id"]).reset_index(drop=True)

    df_out.to_csv(out_csv, index=False)
    print(f"[SAVE_CSV] {out_csv}")


def print_prediction_summary(subset_df: pd.DataFrame):
    print("\n=== RESUMEN PREDICCIONES ===")
    print(f"Filas índice/predicción: {len(subset_df)}")
    print(f"Registros únicos: {subset_df['record_id'].nunique()}")

    counts = subset_df.groupby("record_id").size()
    print("\nVentanas por record_id (primeros 20):")
    print(counts.head(20).to_dict())

    if "label" in subset_df.columns:
        positives = subset_df.groupby("record_id")["label"].sum()
        positives = positives[positives > 0]
        print("\nRegistros con ventanas positivas reales:")
        print(positives.to_dict())


def main():
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    ensure_dir(output_dir)

    print(f"Modelo: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Val patients: {tuple(args.val_patients)}")

    checkpoint_data = load_raw_checkpoint(checkpoint_path, args.device)

    subset_df, dataset, local_indices = build_inference_bundle(
        model_name=args.model,
        checkpoint_data=checkpoint_data,
        target_patients=tuple(args.val_patients),
        args=args,
    )

    model = build_model(args.model, checkpoint_data=checkpoint_data if args.model == "dwt" else None)
    model = load_checkpoint_into_model(model, checkpoint_data, args.device)

    probs = predict_subset(
        model=model,
        dataset=dataset,
        indices=local_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    if len(probs) != len(subset_df):
        raise RuntimeError(
            f"Desajuste entre predicciones y filas del índice: "
            f"{len(probs)} probs vs {len(subset_df)} filas."
        )

    subset_df = subset_df.copy()
    subset_df["prob"] = probs.astype(np.float32)

    print_prediction_summary(subset_df)
    save_predictions_per_record(subset_df, output_dir)
    save_predictions_per_window_csv(subset_df, output_dir)

    print("\nPredicciones guardadas correctamente por registro y por ventana.")


if __name__ == "__main__":
    main()


#python src/scripts/predict.py --model attention_cnn --checkpoint models/attention_best.pth --output_dir experiments/predictions/attention/def
#python src/scripts/predict.py --model dwt --checkpoint models/dwt_mlp_best_pr_auc.pth --output_dir experiments/predictions/dwt/def