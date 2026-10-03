# Datos locales y anotaciones

Datos originales/procesados locales: aproximadamente 272 GiB. No se han movido
ni alterado. Solo `metadata/` y este documento son versionables.

```text
data/
├── BONN/raw/set_A … set_E/       TXT originales
├── BONN/processed/              ventanas .pt + bonn_index.csv
├── CHBMIT/raw/chb01 … chb24/     EDF y summaries
├── CHBMIT/processed/
│   ├── ventana1s/               señales/labels .bin e índices CSV
│   ├── ventana2s/               variante histórica
│   ├── dataset_chbmit_dwt_18ch_bin/
│   └── dataset_chbmit_dwt_nuevos/
└── metadata/                    CSV de anotaciones compartidas
```

Obtén CHB-MIT de PhysioNet (https://physionet.org/content/chbmit/) y Bonn de la
fuente de la Universidad de Bonn referenciada en la memoria. Conserva pacientes,
summaries y conjuntos A–E. No se incorpora descargador ni datos externos nuevos.

Generadores en `scripts/preprocessing/`. Bonn local usa también `.TXT`, pero el
preprocesador filtra `.txt`: revisar antes de regenerar. No se ha cambiado ese
comportamiento histórico. No se identificó un generador equivalente para la
variante de ventana de 2 s; se preserva como artefacto histórico.

Los dos CSV de `metadata/` proceden de `src/utils/`, conservados byte a byte.
índices generados pueden contener rutas absolutas antiguas: revisar columnas al
cambiar de máquina. Algunos consumidores usan nombres y binarios, otros rutas.

La memoria final incorpora diez crisis adicionales de chb12 respecto a las
referencias guardadas. Antes de regenerar métricas, reconciliar CSV, summaries
oficiales y referencias por registro. No se han corregido datos científicos.
