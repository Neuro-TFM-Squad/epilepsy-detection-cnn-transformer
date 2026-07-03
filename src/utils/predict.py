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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_PATIENTS = ("chb05", "chb06")


def parse_args():
    parser = argparse.ArgumentParser(description="Generar predicciones por registro para evaluación clínica.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["spatio_temporal_cnn", "attention_cnn", "eegnet"]
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(output_dir_arg: str) -> Path:
    p = Path(output_dir_arg)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def build_index_dataframe(csv_path: Path):
    df = pd.read_csv(csv_path)

    required = {
        "patient_id", "record_id", "filepath", "window_index",
        "start_sec", "end_sec", "label"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en chbmit_index.csv: {missing}")

    if "global_idx" not in df.columns:
        df = df.reset_index(drop=True)
        df["global_idx"] = np.arange(len(df), dtype=np.int64)

    if "patient_id" not in df.columns or df["patient_id"].isna().any():
        df["patient_id"] = df["filepath"].apply(lambda x: Path(str(x)).stem.split("_")[0])

    return df


def build_model(model_name: str):
    if model_name == "spatio_temporal_cnn":
        model = SpatioTemporalCNN(
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
    elif model_name == "attention_cnn":
        model = AttentionCNN(
            in_channels=18,
            num_classes=1,
            dropout=0.4
        )
    elif model_name == "eegnet":
        model = EEGNet(
            in_channels=18,
            window_size=256,
            num_classes=1
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return model


def load_checkpoint(model, checkpoint_path: Path, device: str):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    probs = []
    for signals, _ in loader:
        signals = signals.to(device, non_blocking=True)
        logits = model(signals)
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(batch_probs)

    return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


def save_predictions_per_record(val_df, probs, output_dir: Path):
    if len(val_df) != len(probs):
        raise ValueError(
            f"Desajuste entre ventanas y probabilidades: "
            f"len(val_df)={len(val_df)} vs len(probs)={len(probs)}"
        )

    val_df = val_df.copy().reset_index(drop=True)
    val_df["prob"] = probs.astype(np.float32)

    for record_id, group in val_df.groupby("record_id", sort=False):
        out_path = output_dir / f"{record_id}.pt"
        arr = group["prob"].to_numpy(dtype=np.float32)

        torch.save(
            {
                "record_id": record_id,
                "y_pred_prob": arr,
            },
            out_path
        )

        print(
            f"[SAVE] {record_id} | "
            f"n_windows={len(arr)} | "
            f"min={float(arr.min()):.4f} "
            f"max={float(arr.max()):.4f} "
            f"mean={float(arr.mean()):.4f}"
        )


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = resolve_output_dir(args.output_dir)

    ensure_dir(output_dir)

    print(f"Modelo     : {args.model}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {args.device}")

    csv_path = PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s" / "chbmit_index.csv"
    signals_bin = PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s" / "chbmit_signals.bin"
    labels_bin = PROJECT_ROOT / "data" / "CHBMIT" / "processed" / "ventana1s" / "chbmit_labels.bin"

    index_df = build_index_dataframe(csv_path)

    subset_df = index_df[index_df["patient_id"].isin(("chb01", "chb02", "chb03", "chb04",
                                                      "chb05", "chb06", "chb07", "chb08",
                                                      "chb09", "chb10"))].copy()

    subset_df = subset_df.sort_values(["patient_id", "record_id", "window_index"]).reset_index(drop=True)

    val_df = subset_df[subset_df["patient_id"].isin(VAL_PATIENTS)].copy().reset_index(drop=True)
    val_indices = val_df["global_idx"].tolist()

    print(f"Total ventanas subset : {len(subset_df)}")
    print(f"Total ventanas val    : {len(val_df)}")
    print(f"Pacientes val         : {VAL_PATIENTS}")

    val_dataset = SeizureDatasetMultichannel(
        signals_path=str(signals_bin),
        labels_path=str(labels_bin),
        augment=False,
        sequence_length=1
    )

    model = build_model(args.model)
    model = load_checkpoint(model, checkpoint_path, args.device)

    probs = predict_subset(
        model=model,
        dataset=val_dataset,
        indices=val_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    print(f"Probabilidades generadas: {len(probs)}")

    save_predictions_per_record(val_df, probs, output_dir)

    print("Predicciones guardadas correctamente por registro.")


if __name__ == "__main__":
    main()


#python -m src.utils.predict --model eegnet --checkpoint .\models\eegnet_clinical_best_def.pth --output_dir .\experiments\predictions\eegnet\def
#python -m src.utils.predict --model attention_cnn --checkpoint .\models\attention_cnn_clinical_best.pth --output_dir .\experiments\predictions\attention_cnn\def
#python -m src.utils.predict --model spatio_temporal_cnn --checkpoint .\models\spatio_temporal_cnn_clinical_best.pth --output_dir .\experiments\predictions\spatio_temporal_cnn\def