# Notebooks históricos

| Carpeta | Contenido |
|---|---|
| `exploration/` | Bonn: EDA/entrenamiento exploratorio; CHB-MIT: una única celda, prácticamente vacío |
| `xai/` | EEGNet/híbrido, con y sin sufijo CHBMIT; SHAP, Grad-CAM y Grad-CAM++ |
| `legacy/` | `XAI.ipynb`: original con tres conflictos merge; no es JSON válido |

Se mantienen nombres y salidas. Los seis notebooks válidos incluyen una celda
que encuentra la raíz y fija el directorio de trabajo; sus rutas de datos/modelos
están actualizadas. Arrancar Jupyter desde la raíz o un subdirectorio del repo.

Los notebooks XAI definen modelos propios y usan particiones aleatorias por
ventana: son exploratorios, no el protocolo clínico cross-patient principal.
Las variantes híbridas referencian un checkpoint EEGNet histórico; conservar esa
inconsistencia y revisar compatibilidad antes de ejecutar. No se han ejecutado
ni limpiado outputs. SHAP no está instalado en el entorno inspeccionado ni forma
parte de dependencias principales; esas celdas necesitan instalación compatible.

El XAI principal scriptado está en `scripts/analysis/xai_analysis.py` y utiliza
Gradientes Integrados. El notebook conflictivo se conserva íntegro, sin elegir
entre GradientExplainer y DeepExplainer ni descartar una de las ramas.
