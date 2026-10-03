# Detección de crisis epilépticas cross-patient en EEG

Repositorio de un TFM terminado de IA aplicada a biomedicina. Compara EEGNet,
una CNN espacio-temporal con inicialización temporal desde Bonn, una CNN con
atención ligera y DWT + MLP sobre CHB-MIT. La evaluación prioriza sensibilidad
por evento, falsas alarmas por hora, latencia y F1 clínico.

El resultado principal es la dificultad de generalizar entre pacientes. Bonn es
auxiliar para preentrenamiento; CHB-MIT es el dominio clínico principal. Los
resultados negativos y las versiones históricas se conservan.

## Orientación rápida

```text
src/                 datasets, arquitecturas y evaluación reutilizable
scripts/             training, preprocessing, evaluation, analysis y legado
notebooks/           exploration, xai y legacy
data/                BONN, CHBMIT y metadata versionable
models/              checkpoints entrenados
experiments/         clinical_eval, predictions, xai, benchmark y logs
docs/                thesis, figures, reference, notes y reorganization
```

[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md): contexto científico.
[CURRENT_STATE.md](CURRENT_STATE.md): estado y limitaciones.
[ARCHITECTURE.md](ARCHITECTURE.md): etapas, comandos y artefactos.

## Entorno y datos

Python 3.10, como en el entorno histórico. Desde la raíz:

```powershell
conda env create -f environment.yml
conda activate tfm_eeg
```

Alternativa: un entorno Python 3.10 y `python -m pip install -r requirements.txt`.
Las dependencias directas conservan las versiones observadas; no se ha probado
la instalación en un entorno nuevo. Para usar la variante CUDA 12.1 histórica:

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Obtención y estructura en [data/README.md](data/README.md).
Los datasets locales ocupan aproximadamente 272 GiB y no se versionan.

## Ejecución

Desde la raíz: preprocesado → entrenamiento → probabilidades por ventana →
postprocesado ajustado en tune → evaluación clínica en otros pacientes → análisis.

```powershell
python -m scripts.evaluation.predict --help
python -m scripts.evaluation.evaluate_clinical --help
```

Entrenamientos: `python -m scripts.training.train_eegnet_8pat`,
`scripts.training.train_attention`, `scripts.training.train_hibrido` y
`scripts.training.train_dwt`. Consulta sus `TrainConfig` antes de ejecutarlos:
escriben checkpoints con nombres ya existentes. No son pruebas ligeras.

## Resultados y memoria

- [Índice de resultados](experiments/README.md): distingue etapas y discrepancias.
- `models/`: ocho checkpoints; los pesos se conservan sin cambios.
- La memoria y los documentos editoriales se conservan en local; no se publican.
- [Notebooks](notebooks/README.md): exploración y XAI histórico.
- [Auditoría](docs/reorganization/README.md): movimientos, validación y deuda.

Los JSON ampliados conservan 52 crisis; la memoria final incorpora una corrección
de chb12 y reporta 62. No mezcles cifras de ambas versiones.
