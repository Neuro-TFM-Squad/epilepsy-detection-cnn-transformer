# Estado actual

TFM terminado según el usuario. Contiene preprocesado, entrenamiento, inferencia,
evaluación clínica y XAI. La memoria final y el material editorial se conservan
solo en local; el repositorio público incluye código, documentación y resultados.

- Comparación base: EEGNet de 8 pacientes, AttentionCNN, híbrido y DWT-MLP.
  Ninguno detecta crisis en la prueba inicial chb06. EEGNet de 22 pacientes es
  referencia histórica, con distinta cobertura de entrenamiento.
- Ampliación: tune chb05, test chb11/chb12/chb13/chb15. Los JSON existentes usan
  52 crisis y sensibilidades 3,85 %, 3,85 %, 5,77 % y 7,69 %, respectivamente.
  La memoria final corrige diez crisis omitidas de chb12: 62 crisis y
  sensibilidades 3,2 %, 3,2 %, 4,8 % y 6,5 %. Los JSON no se han sobrescrito.
  Detalle en [experiments/README.md](experiments/README.md).
- Secundarios: `final_v2`, EEGNet-22pat, ventana de 2 s, diagnósticos de
  entrenamiento y notebooks con particiones por ventana.
- Legado: `scripts/legacy/`, `scripts/thesis_legacy/` y
  `notebooks/legacy/XAI.ipynb` (conflictos Git originales sin resolver).

Limitaciones: falta una exportación clínica corregida alineada con la memoria;
los índices/configuraciones históricos contienen rutas de máquina; no se ha
probado reinstalación completa ni repetido entrenamiento. Los scripts editoriales
son operaciones sobre versiones concretas de Word, no un pipeline general.

Continuación útil: reconciliar anotaciones/resultados en una exportación nueva
trazable y fijar un manifiesto de checkpoints y particiones.
Auditoría: [docs/reorganization/README.md](docs/reorganization/README.md).
