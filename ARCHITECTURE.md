# Pipeline técnico y componentes

Ejecutar desde la raíz. `src` conserva su nombre para mantener imports de modelos
y utilidades. Los ejecutables admiten `python -m scripts.<etapa>.<nombre>` y
llamada directa. Los hiperparámetros permanecen en `TrainConfig`.

## Etapas

| Etapa | Ejecutable o módulo | Entrada → salida |
|---|---|---|
| Bonn | `scripts.preprocessing.preprocessing_bonn` | TXT A–E → `.pt` e Índice |
| CHB-MIT | `scripts.preprocessing.preprocessing_chbmit` | EDF + `data/metadata/metadata_chbmit_etiquetado.csv` → binarios e Índice |
| DWT base | `scripts.preprocessing.raw_to_dwt_chbmit` | EDF chb01–10 → coeficientes db4, nivel 5, binarios e Índice |
| DWT ampliado | `scripts.preprocessing.preprocess_dwt_nuevos` | EDF chb11/12/13/15 → dataset separado |
| Preentrenamiento | `scripts.training.train_bonn` | Bonn procesado → checkpoint Bonn |
| Entrenamiento | `scripts.training.train_*` | CHB-MIT → checkpoints con configuración |
| Inferencia | `scripts.evaluation.predict` | checkpoint + datos → `.pt` por registro y CSV por ventana |
| Clínica | `scripts.evaluation.evaluate_clinical` | probabilidades + referencias → búsqueda tune, configuración y métricas tune/test |
| XAI | `scripts.analysis.xai_analysis` | checkpoints + chb05 → atribuciones `.npy` locales y figuras en `experiments/xai/` |
| Coste | `scripts.analysis.benchmark_inference` | checkpoints → `experiments/benchmark/inference_benchmark.json` |
| Diagramas | `scripts.analysis.graficar_modelos` | clases → `experiments/figures/architectures/` |

`src/datasets` implementa Dataset de PyTorch; `src/models` las arquitecturas.
`src/utils/postprocessing.py` transforma scores en eventos;
`evaluador_clinico_def.py` realiza matching y métricas.
`evaluador_clinico.py` es anterior y sigue usado por entrenamientos: no fusionar
ambos evaluadores sin estudiar sus diferencias.

## Ejecución

Preprocesado y entrenamiento pueden ser costosos y sobrescribir archivos. Algunos
preprocesadores históricos ejecutan trabajo al importar: no importarlos para
comprobar sintaxis.

```powershell
python -m scripts.preprocessing.preprocessing_bonn
python -m scripts.training.train_bonn
python -m scripts.training.train_hibrido
```

Ejemplo de flujo con EEGNet de ocho pacientes, usando nuevas salidas:

```powershell
python -m scripts.evaluation.predict --model eegnet --checkpoint models/eegnet_8pat_best.pth --output_dir experiments/predictions/reproduction/eegnet_8pat --val_patients chb05 chb06
python -m scripts.evaluation.evaluate_clinical --predictions_dir experiments/predictions/reproduction/eegnet_8pat --output_dir experiments/clinical_eval/reproduction/eegnet_8pat --tune_patients chb05 --test_patients chb06
```

Los defaults de búsqueda no garantizan reproducir el punto operativo histórico;
consultar los JSON y parámetros de cada ejecución. Para ampliar test, aportar
`--index_csv` con todos los pacientes, inferir con `--val_patients` y seleccionar
`--test_patients` explícitamente. DWT ampliado usa `--dwt_data_root` con sus
binarios e Índice; ese dataset nuevo no incluye chb05 y no basta por sí solo para
generar tune y test conjuntamente. No concatenar índices/binarios ingenuamente.

## Checkpoints

| Entrenamiento | Salida actual en `models/` |
|---|---|
| `train_bonn` | `baseline_cnn_bonn_clean.pth` |
| `train_eegnet` | `eegnet_pr_auc_best.pth` (22 pacientes) |
| `train_eegnet_8pat` | `eegnet_8pat_best.pth` |
| `train_attention` | `attention_cnn_clinical_best.pth` |
| `train_hibrido` | `spatio_temporal_cnn_clinical_best_def.pth` |
| `train_dwt` | `dwt_mlp_v2.pth` |

Se conservan también `eegnet_clinical_best_def.pth` y `dwt_mlp_best_pr_auc.pth`.
El segundo se usa en XAI y benchmark existentes aunque el entrenamiento actual
escribe v2. Una carpeta «final» no identifica por sí sola checkpoint/anotaciones.

## Material histórico local

`scripts/legacy/` conserva evaluadores anteriores y predictores precursores.
Fuera de Git, `scripts/thesis_legacy/` conserva inserciones/correcciones Word con rutas
actualizadas a `docs/thesis/` y `backups/`. Algunos documentos intermedios
referidos ya faltaban antes de reorganizar. Ejecutarlos puede sobrescribir la
memoria; no sirven como regeneración general.
`docs/reference/` conserva aportaciones/revisiones. `docs/notes/` es evidencia
histórica, no instrucciones vigentes para agentes.
