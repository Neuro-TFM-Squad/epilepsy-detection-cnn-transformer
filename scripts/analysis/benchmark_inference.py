# Resolve the repository for both module and direct script execution.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import time
import json
import numpy as np
import torch
import pywt

from src.models.hibrido import SpatioTemporalCNN
from src.models.eegnet_model import EEGNet
from src.models.attention import AttentionCNN
from src.models.dwt_mlp_model import DWTMLP

N_WARMUP = 20
N_REPEATS = 200
BATCH_THROUGHPUT = 256


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_wrapped(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    return ck, state


def build_raw_signal_models():
    models = {}

    eegnet = EEGNet(in_channels=18, window_size=256, num_classes=1)
    _, state = load_wrapped("models/eegnet_8pat_best.pth")
    eegnet.load_state_dict(state)
    models["EEGNet-8pat"] = eegnet

    att = AttentionCNN(in_channels=18, num_classes=1, dropout=0.4)
    _, state = load_wrapped("models/attention_cnn_clinical_best.pth")
    att.load_state_dict(state)
    models["AttentionCNN"] = att

    hib = SpatioTemporalCNN(
        in_channels=18, window_size=256, temporal_filters=16, spatial_filters=32,
        num_classes=1, temporal_kernel=15, dropout_p=0.3, spatial_dropout_p=0.1,
        temporal_norm="instance",
    )
    _, state = load_wrapped("models/spatio_temporal_cnn_clinical_best_def.pth")
    hib.load_state_dict(state)
    models["Hibrido"] = hib

    return models


def build_dwt_model():
    ck, state = load_wrapped("models/dwt_mlp_best_pr_auc.pth")
    input_dim = ck["input_dim"]
    cfg = ck.get("config", {})
    hidden_dims = tuple(cfg.get("hidden_dims", (256, 128)))
    dropout = cfg.get("dropout", 0.25)
    model = DWTMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state)
    return model, input_dim


def benchmark_forward(model, input_shape, device, n_warmup=N_WARMUP, n_repeats=N_REPEATS, batch=1):
    model = model.to(device)
    model.eval()
    x = torch.randn((batch,) + input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times) * 1000.0  # ms
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "std_ms": float(times.std()),
        "p95_ms": float(np.percentile(times, 95)),
    }


def benchmark_dwt_feature_extraction(n_repeats=N_REPEATS, wavelet="db4", level=5):
    rng = np.random.default_rng(42)
    window = rng.standard_normal((18, 256)).astype(np.float32)

    for _ in range(N_WARMUP):
        for canal in window:
            pywt.wavedec(canal, wavelet, level=level)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        coefs_ventana = []
        for canal in window:
            coeficientes = pywt.wavedec(canal, wavelet, level=level)
            coefs_canal = np.concatenate(coeficientes)
            coefs_ventana.append(coefs_canal)
        _ = np.concatenate(coefs_ventana).astype(np.float32)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times) * 1000.0
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "std_ms": float(times.std()),
        "p95_ms": float(np.percentile(times, 95)),
    }


def file_size_mb(path):
    import os
    return os.path.getsize(path) / (1024 * 1024)


def main():
    results = {}

    raw_models = build_raw_signal_models()
    dwt_model, dwt_input_dim = build_dwt_model()

    checkpoint_paths = {
        "EEGNet-8pat": "models/eegnet_8pat_best.pth",
        "AttentionCNN": "models/attention_cnn_clinical_best.pth",
        "Hibrido": "models/spatio_temporal_cnn_clinical_best_def.pth",
        "DWT-MLP": "models/dwt_mlp_best_pr_auc.pth",
    }

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA no disponible, solo CPU")

    for name, model in raw_models.items():
        results[name] = {
            "n_params": count_params(model),
            "checkpoint_size_mb": round(file_size_mb(checkpoint_paths[name]), 3),
        }
        for device in devices:
            print(f"Benchmarking {name} on {device} (batch=1)...")
            results[name][f"latency_batch1_{device}"] = benchmark_forward(
                model, (18, 256), device, batch=1
            )
            print(f"Benchmarking {name} on {device} (batch={BATCH_THROUGHPUT})...")
            results[name][f"throughput_batch{BATCH_THROUGHPUT}_{device}"] = benchmark_forward(
                model, (18, 256), device, batch=BATCH_THROUGHPUT, n_repeats=50
            )

    results["DWT-MLP"] = {
        "n_params": count_params(dwt_model),
        "checkpoint_size_mb": round(file_size_mb(checkpoint_paths["DWT-MLP"]), 3),
        "input_dim": dwt_input_dim,
    }
    for device in devices:
        print(f"Benchmarking DWT-MLP on {device} (batch=1)...")
        results["DWT-MLP"][f"latency_batch1_{device}"] = benchmark_forward(
            dwt_model, (dwt_input_dim,), device, batch=1
        )
        print(f"Benchmarking DWT-MLP on {device} (batch={BATCH_THROUGHPUT})...")
        results["DWT-MLP"][f"throughput_batch{BATCH_THROUGHPUT}_{device}"] = benchmark_forward(
            dwt_model, (dwt_input_dim,), device, batch=BATCH_THROUGHPUT, n_repeats=50
        )

    print("Benchmarking DWT wavelet feature extraction (CPU, pywt)...")
    results["DWT-MLP"]["dwt_feature_extraction_cpu"] = benchmark_dwt_feature_extraction()

    with open("experiments/benchmark/inference_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== RESUMEN ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
