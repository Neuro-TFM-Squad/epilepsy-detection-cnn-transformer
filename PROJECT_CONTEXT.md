# Contexto científico estable

## Problema y alcance

TFM terminado sobre detección de crisis epilépticas en EEG con generalización
cross-patient. Compara enfoques y analiza sus limitaciones con una evaluación
clínicamente interpretable, sin perseguir SOTA ni validar un dispositivo listo
para uso asistencial.

CHB-MIT es el dominio principal de entrenamiento y evaluación clínica. Bonn
aporta preentrenamiento del bloque temporal del híbrido; no se combinan ambos
dominios en el entrenamiento clínico final.

## Enfoques implementados

- EEGNet: baseline compacto; entrenamientos de 22 y de 8 pacientes.
- `SpatioTemporalCNN`: CNN temporal y espacial, transferencia desde `BaselineCNN`
  de Bonn, congelación inicial y descongelación temporal explícita.
- `AttentionCNN`: convoluciones con atención SE ligera.
- DWT + `DWTMLP`: coeficientes wavelet y clasificación con MLP.

No hay un CNN-Transformer en la implementación principal. La interpretabilidad
scriptada utiliza Gradientes Integrados; los notebooks exploran también SHAP,
Grad-CAM y Grad-CAM++. XAI analiza decisiones y no acredita localización clínica
del foco epiléptico.

## Protocolo y criterios

El pipeline principal usa ventanas de 1 s, 256 Hz y 18 derivaciones CHB-MIT.
La partición base entrena con chb01–04 y chb07–10 y valida con chb05/chb06.
EEGNet de 22 pacientes incluye chb11–15: evaluarlo sobre esos sujetos no sería
cross-patient. La variante de 8 pacientes permite la comparación ampliada.

La evaluación clínica ajusta el postprocesado en tune y aplica la configuración
a test con pacientes distintos. Hay evaluaciones en ambas direcciones chb05/chb06
y una ampliación a chb11, chb12, chb13 y chb15.

Sensibilidad por evento, FA/h, latencia y F1 clínico tienen prioridad sobre
accuracy de ventana. El criterio clínico compuesto penaliza falsas alarmas.
No todos los checkpoints históricos se seleccionaron así: EEGNet y DWT incluyen
selección por PR-AUC; atención e híbrido incluyen criterio clínico. Conservar esas
diferencias como parte del experimento.

## Límites

Preservar resultados negativos, separación de pacientes, calibración y
postprocesado documentados. No ampliar modelos ni infraestructura sin petición
expresa. La baja sensibilidad y la variabilidad inter-paciente son hallazgos,
no evidencia de utilidad clínica probada.

Mapa técnico: [ARCHITECTURE.md](ARCHITECTURE.md).
Estado y resultados: [CURRENT_STATE.md](CURRENT_STATE.md).
