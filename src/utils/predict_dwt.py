from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.seizureDatasetMulticanal import SeizureDatasetMultichannel
from src.models.baseline_model_multicanal import SpatioTemporalCNN
from src.models.eegnet_model import EEGNet
from src.models.attention import AttentionCNN
from src.models.dwt_mlp_model import DWTMLP
from src.datasets.dwt_dataset import DWTSeizureDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL_PATIENTS = ("chb05", "chb06")


def parse_args():
    parser = argparse.ArgumentParser(description="Generar predicciones por registro para evaluación clínica.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["spatio_temporal_cnn", "attention_cnn", "eegnet", "dwt"]
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_record_id_from_filepath(x):
    stem = Path(str(x)).stem
    return stem.rsplit("_win", 1)[0] if "_win" in stem else stem


def build_subset_dataframe(index_df, target_patients):
    df = index_df.copy()

    if "patient_id" not in df.columns:
        if "filepath" in df.columns:
            df["patient_id"] = df["filepath"].apply(lambda x: str(x).split("_")[0])
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
        df["window_index"] = df["window_index"].astype(int)

    return df


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


def load_raw_checkpoint(checkpoint_path: Path, device: str):
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_checkpoint(model, checkpoint_data, device: str):
    state = checkpoint_data
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_subset(model, dataset, indices, batch_size, num_workers, device):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
    )

    probs = []
    for batch in loader:
        features = batch[0].to(device, non_blocking=True)
        logits = model(features)
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(batch_probs)

    return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


def save_predictions_per_record(subset_df, output_dir: Path):
    grouped = subset_df.sort_values(["record_id", "window_index"] if "window_index" in subset_df.columns else ["record_id"])

    for record_id, group in grouped.groupby("record_id", sort=False):
        probs = group["prob"].to_numpy(dtype=np.float32)
        out_path = output_dir / f"{record_id}.pt"

        torch.save(
            {
                "record_id": record_id,
                "y_pred_prob": probs,
                "n_windows": int(len(group)),
                "patient_id": str(group["patient_id"].iloc[0]) if "patient_id" in group.columns else None,
                "window_index": group["window_index"].to_numpy(dtype=np.int32) if "window_index" in group.columns else None,
            },
            out_path
        )
        print(f"[SAVE_PT] {record_id} -> {len(probs)} ventanas")


def save_predictions_per_window_csv(subset_df, output_dir: Path):
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


def print_prediction_summary(subset_df):
    print("\n=== RESUMEN PREDICCIONES ===")
    print(f"Filas índice/predicción: {len(subset_df)}")
    print(f"Registros únicos       : {subset_df['record_id'].nunique()}")

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

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    ensure_dir(output_dir)

    print(f"Modelo : {args.model}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output dir : {output_dir}")
    print(f"Device : {args.device}")

    checkpoint_data = load_raw_checkpoint(checkpoint_path, args.device)

    if args.model == "dwt":
        data_root = PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "dataset_chbmit_dwt_18ch_bin"
        csv_path = data_root / "chbmit_dwt_index.csv"
        signals_bin = data_root / "chbmit_dwt_signals.bin"
        labels_bin = data_root / "chbmit_dwt_labels.bin"

        index_df = pd.read_csv(csv_path)
        subset_df = build_subset_dataframe(index_df, VAL_PATIENTS).reset_index(drop=True)

        if "global_idx" not in subset_df.columns:
            raise ValueError("El índice DWT necesita la columna 'global_idx'.")

        val_idx = subset_df["global_idx"].astype(int).tolist()

        feature_dim = checkpoint_data.get("feature_dim", None)
        if feature_dim is None:
            cfg = checkpoint_data.get("config", {})
            feature_dim = cfg.get("feature_dim", 5184)

        num_samples = len(index_df)

        val_dataset = DWTSeizureDataset(
            signals_path=signals_bin,
            labels_path=labels_bin,
            index_csv_path=csv_path,
            indices=val_idx,
            num_samples=num_samples,
            feature_dim=feature_dim,
            normalize=True,
        )

        model = build_model(args.model, checkpoint_data=checkpoint_data)
        model = load_checkpoint(model, checkpoint_data, args.device)

        local_indices = list(range(len(val_idx)))

    else:
        data_root = PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s"
        csv_path = data_root / "chbmit_index_val_56.csv"
        signals_bin = data_root / "chbmit_signals.bin"
        labels_bin = data_root / "chbmit_labels.bin"

        index_df = pd.read_csv(csv_path)
        subset_df = build_subset_dataframe(index_df, VAL_PATIENTS).reset_index(drop=True)

        global_indices = subset_df.index.to_list()
        num_samples_full = len(index_df)

        val_dataset = SeizureDatasetMultichannel(
            signals_path=str(signals_bin),
            labels_path=str(labels_bin),
            augment=False,
            sequence_length=1,
            indices=global_indices,
            num_samples=num_samples_full,
        )

        model = build_model(args.model, checkpoint_data=None)
        model = load_checkpoint(model, checkpoint_data, args.device)

        local_indices = list(range(len(subset_df)))

    probs = predict_subset(
        model=model,
        dataset=val_dataset,
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
    subset_df["prob"] = probs

    print_prediction_summary(subset_df)
    save_predictions_per_record(subset_df, output_dir)
    save_predictions_per_window_csv(subset_df, output_dir)

    print("Predicciones guardadas correctamente por registro y por ventana.")


if __name__ == "__main__":
    main()