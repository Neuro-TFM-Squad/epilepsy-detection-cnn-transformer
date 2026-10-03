# Resolve the repository for both module and direct script execution.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
import inspect
import importlib

import torch.nn as nn
from graphviz import Digraph


MODELS_DIR = _PROJECT_ROOT / "src/models"
OUTPUT_DIR = _PROJECT_ROOT / "experiments/figures/architectures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "Conv1d": "#c8e6c9",
    "Conv2d": "#c8e6c9",
    "BatchNorm1d": "#d1e3ff",
    "BatchNorm2d": "#d1e3ff",
    "ReLU": "#ffe0b2",
    "GELU": "#ffe0b2",
    "MaxPool1d": "#ffcdd2",
    "MaxPool2d": "#ffcdd2",
    "AvgPool1d": "#f8bbd0",
    "AdaptiveAvgPool1d": "#e1bee7",
    "Dropout": "#eeeeee",
    "Linear": "#f5f5f5",
}


def get_leaf_layers(model):
    layers = []

    for module in model.modules():

        children = list(module.children())

        if len(children) == 0:
            layers.append(module)

    return layers


def layer_label(layer):

    if isinstance(layer, nn.Conv1d):
        return (
            f"Conv1d\n"
            f"{layer.in_channels}→{layer.out_channels}\n"
            f"k={layer.kernel_size[0]}"
        )

    if isinstance(layer, nn.Linear):
        return (
            f"Linear\n"
            f"{layer.in_features}→{layer.out_features}"
        )

    return layer.__class__.__name__


for file in MODELS_DIR.glob("*.py"):

    module_name = f"src.models.{file.stem}"

    try:
        module = importlib.import_module(module_name)

    except Exception as e:
        print(f"ERROR importando {file.name}: {e}")
        continue

    for name, cls in inspect.getmembers(module, inspect.isclass):

        if not issubclass(cls, nn.Module):
            continue

        if cls is nn.Module:
            continue

        try:
            model = cls()

        except Exception as e:
            print(f"No se pudo instanciar {name}: {e}")
            continue

        layers = get_leaf_layers(model)

        g = Digraph(format="svg")
        g.attr(rankdir="LR")
        g.attr(nodesep="0.15")
        g.attr(ranksep="0.20")

        g.node(
            "input",
            "Input",
            shape="box",
            style="rounded,filled",
            fillcolor="#fff8dc",
        )

        previous = "input"

        for i, layer in enumerate(layers):

            layer_type = layer.__class__.__name__

            g.node(
                str(i),
                layer_label(layer),
                shape="box",
                style="rounded,filled",
                fillcolor=COLORS.get(layer_type, "#ffffff"),
            )

            g.edge(previous, str(i))

            previous = str(i)

        g.node(
            "output",
            "Output",
            shape="box",
            style="rounded,filled",
            fillcolor="#fff8dc",
        )

        g.edge(previous, "output")

        g.render(
            OUTPUT_DIR / name,
            cleanup=True,
        )

        print(f"Generado: {name}.svg")